import os
import logging
import json
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
import google.generativeai as genai
from supabase import create_client, Client
from utils import parse_ai_json, safe_db_operation, format_error_response

router = APIRouter()
logger = logging.getLogger(__name__)

# Config
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

genai.configure(api_key=GEMINI_KEY)
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

@router.post("/ask-audio")
async def chat_with_audio(
    file: UploadFile = File(...),
    user_id: str = Form(...)
):
    try:
        # 1. Fetch Chart Context from DB
        chart_context = ""
        db_res = safe_db_operation(
            lambda: supabase.table("user_kundli_data")
                .select("ai_summary_text")
                .eq("user_id", user_id)
                .execute(),
            "Failed to fetch user chart"
        )
        if db_res and db_res.data and len(db_res.data) > 0:
            chart_context = db_res.data[0]['ai_summary_text']
        else:
            return {"status": "error", "reply": "Please complete your profile first."}

        # 2. Read Audio Bytes
        audio_bytes = await file.read()

        # 3. Prepare Model
        model = genai.GenerativeModel("gemini-2.0-flash")
        
        # 4. Construct Prompt
        prompt = f"""
        Act as 'Jyotish AI'. The user is speaking to you directly.
        
        USER'S CHART CONTEXT:
        {chart_context}
        
        INSTRUCTIONS:
        1. Listen to the user's audio question.
        2. Answer based on their chart.
        3. Keep the tone conversational, empathetic, and concise (like a voice note reply).
        4. Output JSON: {{ "reply": "..." }}
        """

        # 5. Send Audio + Prompt to Gemini
        response = model.generate_content([
            prompt,
            {
                "mime_type": file.content_type or "audio/webm",
                "data": audio_bytes
            }
        ])

        # 6. Parse Response
        data = parse_ai_json(response.text, fallback={"reply": response.text})
        return {"status": "success", "reply": data.get("reply", response.text)}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Voice Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to process audio. Please try again.")