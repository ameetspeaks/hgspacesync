import os
import logging
import json
from fastapi import APIRouter, HTTPException, Header
from pydantic import BaseModel
from typing import List, Optional
import google.generativeai as genai
from supabase import create_client, Client
from calc import calculate_birth_chart, get_daily_transits
from utils import parse_ai_json, safe_db_operation, format_error_response, get_chart_summary

router = APIRouter()
logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

genai.configure(api_key=GEMINI_KEY)
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# --- DATA MODELS ---
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    user_id: Optional[str] = None 
    dob: str
    time: str
    lat: float
    lon: float
    tz: float
    history: List[Message] = [] 
    current_message: str
    target_language: str = "Auto" # Auto, English, Hindi, Hinglish

# --- HELPER: RAG SEARCH (OPTIONAL) ---
def search_scriptures(query_text):
    """
    Searches the 'vedic_scriptures' vector table for relevant Shlokas.
    Fails gracefully if feature is not enabled in DB.
    """
    try:
        # Check if user wants RAG (Skip to save latency if not critical)
        # 1. Embed the user's question
        embedding = genai.embed_content(
            model="models/text-embedding-004",
            content=query_text,
            task_type="retrieval_query"
        )
        
        # 2. RPC Call to Supabase (match_scriptures function)
        res = supabase.rpc(
            'match_scriptures', 
            {
                'query_embedding': embedding['embedding'], 
                'match_threshold': 0.5, 
                'match_count': 2
            }
        ).execute()
        
        if res.data:
            return "\n".join([f"📖 [{item['source_book']}]: {item['content']}" for item in res.data])
        return "No specific scripture citation found."
        
    except Exception as e:
        # This is expected if the vector table/function doesn't exist yet.
        # We log a warning but DO NOT crash the chat.
        logger.warning(f"RAG Search skipped: {e}", exc_info=True)
        return ""

# --- SYSTEM PROMPT ---
SYSTEM_INSTRUCTION = """
You are 'Jyotish AI', a warm, empathetic, and deeply insightful Vedic Astrologer who combines ancient wisdom with modern psychological understanding.

**YOUR PERSONALITY:**
- You are like a wise, caring friend who happens to be an expert astrologer
- You speak with warmth, empathy, and genuine concern for the user's wellbeing
- You balance Vedic wisdom with practical, psychological insights
- You acknowledge emotions and validate feelings before offering astrological guidance
- You use natural, conversational language - never robotic or overly formal
- You remember context from previous messages and build on the conversation

**GREETING PROTOCOL:**
- If this is the FIRST message in the conversation (no history), ALWAYS start with a warm, personalized greeting
- Reference their chart elements naturally (e.g., "I see your Moon in Cancer - you're someone who feels deeply...")
- Make them feel seen and understood from the first interaction
- Examples: "Namaste! I'm here to guide you through the cosmic patterns in your life..." or "Hello! I've been looking at your chart, and I can sense..."

**PSYCHOLOGICAL APPROACH:**
- Always acknowledge the emotional/psychological dimension of their question
- Use phrases like "I understand this might feel..." or "It's completely natural to feel..."
- Connect astrological insights to real-life psychological patterns
- Validate their concerns before offering solutions
- Use empathetic language: "I can sense your concern about...", "This must be weighing on you..."

**PERSONALIZATION:**
- Reference specific chart elements naturally in conversation (Moon sign, Ascendant, dominant planets)
- Connect their current question to their chart's unique patterns
- Use their chart to explain WHY they might be experiencing something
- Make connections between their personality (from chart) and their current situation

**CORE DATA SOURCES:**
1. **User Chart:** Use the provided D1 (Birth) and D9 (Navamsa) charts for deep personalization.
2. **Transits:** Use the provided 'Current Transits' to explain timing/triggering of events.
3. **Scriptures:** If 'Scripture Reference' is provided, cite it naturally (e.g., "As the ancient text Phala Deepika teaches us...").

**LANGUAGE PROTOCOL:**
- **Auto:** Detect user's language and match it naturally.
- **Hinglish:** Use natural Indian conversational style (e.g., "Saturn ka transit tough hai, par result milega - don't worry, you've got this!").
- **Hindi:** Use warm, conversational Devanagari.
- **English:** Use warm, conversational English with occasional Sanskrit terms when appropriate.

**CONVERSATION FLOW:**
- If user shares a problem: Acknowledge → Validate → Analyze (Chart + Transits) → Explain → Offer Solutions
- If user asks a question: Greet (if first) → Answer with chart context → Provide deeper insights → Suggest next steps
- Always end with an open-ended question or invitation to explore deeper

**OUTPUT FORMAT (JSON ONLY):**
{
  "reply": "Your warm, personalized, psychologically-aware response. Use Markdown for formatting. Include emojis sparingly for warmth (🌙, ⭐, 🙏, etc.).",
  "action_cards": [
     { "title": "Remedy/Action", "subtitle": "Short description", "value": "HIDDEN_PROMPT_FOR_NEXT_TURN" }
  ]
}

**ACTION CARDS LOGIC:**
- If user asks about a problem: Generate 3 cards - 1. Deep Dive Question (to understand better), 2. Modern Remedy (practical/psychological), 3. Vedic Remedy (traditional/spiritual)
- If user asks a general question: Generate 2-3 cards with related topics they might want to explore
- Make card titles warm and inviting, not clinical

**CRITICAL RULES:**
- NEVER be cold or robotic - always show warmth and empathy
- ALWAYS personalize responses using chart data
- ALWAYS acknowledge emotions before diving into astrology
- Use "you" and "your" to make it personal
- Ask follow-up questions to show genuine interest
- Remember: You're not just an AI - you're a compassionate guide
"""

# --- ENDPOINT ---
@router.post("/ask")
def chat_with_jyotish(req: ChatRequest):
    try:
        # 1. GET CHART CONTEXT (DB or Calc) - Enhanced for personalization
        chart_context = ""
        chart_details = {}
        is_first_message = len(req.history) == 0
        
        if req.user_id:
            db_res = safe_db_operation(
                lambda: supabase.table("user_kundli_data")
                    .select("ai_summary_text, chart_data, ascendant_sign, moon_sign")
                    .eq("user_id", req.user_id)
                    .execute(),
                "Failed to fetch chart from DB"
            )
            if db_res and db_res.data:
                chart_context = db_res.data[0].get('ai_summary_text', '')
                chart_details = {
                    'ascendant': db_res.data[0].get('ascendant_sign', ''),
                    'moon_sign': db_res.data[0].get('moon_sign', ''),
                    'chart_data': db_res.data[0].get('chart_data', [])
                }

        if not chart_context:
            raw = calculate_birth_chart(req.dob, req.time, req.lat, req.lon, req.tz)
            chart_context = get_chart_summary(raw)
            if isinstance(raw, dict):
                chart_details = {
                    'ascendant': raw.get('key_points', {}).get('ascendant', ''),
                    'moon_sign': raw.get('key_points', {}).get('moon_sign', ''),
                    'chart_data': raw.get('raw_json', [])
                }

        # 2. GET DYNAMIC DATA (Transits & Scriptures)
        try:
            current_transits = get_daily_transits() # From calc.py
        except Exception as e:
            logger.warning(f"Failed to get transits: {e}")
            current_transits = "Planetary positions unavailable"
        
        # Try RAG search (will return empty string if disabled)
        scripture_context = search_scriptures(req.current_message)

        # 3. BUILD MODEL
        model = genai.GenerativeModel("gemini-2.0-flash", system_instruction=SYSTEM_INSTRUCTION)
        
        # 4. HISTORY
        gemini_history = []
        for m in req.history:
            role = "user" if m.role == "user" else "model"
            gemini_history.append({"role": role, "parts": [m.content]})

        chat = model.start_chat(history=gemini_history)
        
        # 5. BUILD PERSONALIZED CONTEXT
        personalization_context = ""
        if chart_details:
            if chart_details.get('moon_sign'):
                personalization_context += f"\n- Moon Sign: {chart_details['moon_sign']} (emotional nature, inner self)"
            if chart_details.get('ascendant'):
                personalization_context += f"\n- Ascendant: {chart_details['ascendant']} (personality, how others see you)"
            if chart_details.get('chart_data'):
                # Extract dominant planets
                planets = [p.get('planet', '') for p in chart_details['chart_data'] if isinstance(p, dict)]
                if planets:
                    personalization_context += f"\n- Key Planets: {', '.join(planets[:5])}"
        
        conversation_context = ""
        if is_first_message:
            conversation_context = "\n[CONVERSATION STATE: This is the FIRST message. You MUST greet the user warmly and personally, referencing their chart elements.]"
        else:
            conversation_context = f"\n[CONVERSATION STATE: This is message #{len(req.history) + 1}. Continue the conversation naturally, building on previous context.]"
        
        # 5. INJECT FULL CONTEXT WITH PERSONALIZATION
        full_prompt = f"""
        [CONVERSATION CONTEXT]
        {conversation_context}
        
        [PERSONALIZATION DATA]
        Chart Summary: {chart_context}
        {personalization_context}
        
        [CURRENT ASTROLOGICAL INFLUENCES]
        Transits: {current_transits}
        
        [KNOWLEDGE BASE]
        Scripture Reference: {scripture_context if scripture_context else "None - use your knowledge"}
        
        [USER PREFERENCES]
        Target Language: {req.target_language}
        
        [USER'S MESSAGE]
        {req.current_message}
        
        [YOUR TASK]
        Respond as a warm, empathetic, psychologically-aware astrologer. Personalize everything. Show you understand them as a person, not just a chart. Make them feel seen, heard, and guided.
        """

        # 6. GENERATE
        response = chat.send_message(full_prompt)
        
        # 7. PARSE JSON AND ENHANCE RESPONSE
        data = parse_ai_json(
            response.text,
            fallback={"reply": response.text, "action_cards": []}
        )
        
        reply = data.get("reply", response.text)
        action_cards = data.get("action_cards", [])
        
        # Post-process: Ensure first message has greeting if missing
        if is_first_message and reply:
            reply_lower = reply.lower()
            greeting_indicators = ["namaste", "hello", "hi", "greetings", "welcome", "hey", "namaskar"]
            has_greeting = any(indicator in reply_lower[:150] for indicator in greeting_indicators)
            
            if not has_greeting and chart_details.get('moon_sign'):
                # Prepend a warm, personalized greeting
                moon_sign = chart_details['moon_sign']
                ascendant = chart_details.get('ascendant', '')
                
                if ascendant:
                    greeting = f"Namaste! 🌙 I see your Moon in {moon_sign} and your Ascendant in {ascendant} - you're someone with a unique blend of emotional depth and outward expression. "
                else:
                    greeting = f"Namaste! 🌙 I see your Moon in {moon_sign} - you're someone who feels deeply and intuitively. "
                
                reply = greeting + reply
        
        return {
            "status": "success", 
            "reply": reply, 
            "cards": action_cards,
            "is_first_message": is_first_message
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat Error: {e}", exc_info=True)
        return format_error_response(
            e,
            user_message="I am aligning the stars... please try again."
        )