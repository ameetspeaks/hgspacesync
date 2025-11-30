import os
import logging
import json
from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel
from typing import Optional
import google.generativeai as genai
from supabase import create_client, Client

# Import Engines
from calc import calculate_birth_chart, calculate_kootas, get_synastry_data, SIGNS
from routers.pdf_generator import generate_match_pdf
from utils import parse_ai_json, validate_chart_result, safe_db_operation

router = APIRouter()
logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

genai.configure(api_key=GEMINI_KEY)
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# --- MODELS ---
class PartnerProfile(BaseModel):
    name: str
    dob: str
    time: str
    lat: float
    lon: float
    tz: float

class MatchRequest(BaseModel):
    user_id: Optional[str] = None 
    you: PartnerProfile
    partner: PartnerProfile
    language: str = "English"

# --- HELPER: GET MOON DEGREE ---
def get_moon_abs_degree(chart_data):
    """
    Extracts the Moon's absolute longitude (0-360) from the chart JSON.
    """
    # Find Moon in the list of planets
    moon_obj = next((p for p in chart_data['raw_json'] if p['planet'] == 'Moon'), None)
    if not moon_obj: return 0.0
    
    # Convert Sign + Degree to Absolute Degree
    # Aries(0) to Pisces(11) * 30 + relative degree
    try:
        sign_idx = SIGNS.index(moon_obj['sign'])
        return (sign_idx * 30.0) + moon_obj['degree']
    except:
        return 0.0

# --- ENDPOINTS ---

@router.post("/analyze")
def match_kundli(req: MatchRequest):
    try:
        # 1. Calculate Charts (Astronomy)
        p1 = calculate_birth_chart(req.you.dob, req.you.time, req.you.lat, req.you.lon, req.you.tz)
        p2 = calculate_birth_chart(req.partner.dob, req.partner.time, req.partner.lat, req.partner.lon, req.partner.tz)
        
        if not validate_chart_result(p1) or not validate_chart_result(p2):
            raise HTTPException(status_code=400, detail="Invalid Birth Data provided.")

        # 2. Calculate Gunas (Math)
        deg1 = get_moon_abs_degree(p1)
        deg2 = get_moon_abs_degree(p2)
        
        # Get the detailed scoring breakdown
        koota_result = calculate_kootas(deg1, deg2)
        score_str = f"{koota_result['total']}/36"

        # 3. Prepare Context for AI
        synastry_txt = get_synastry_data(p1, p2)
        
        # 4. AI Prompt (STRICTER REMEDY INSTRUCTION)
        prompt = f"""
        Act as an Expert Vedic Relationship Counselor.
        Target Language: {req.language}
        
        DATA:
        - Boy: {req.you.name} (Moon Nakshatra: {koota_result['boy_nak']['name']})
        - Girl: {req.partner.name} (Moon Nakshatra: {koota_result['girl_nak']['name']})
        - Calculated Guna Score: {score_str}
        - Nadi Score: {koota_result['scores']['nadi']['score']}/8
        - Bhakoot Score: {koota_result['scores']['bhakoot']['score']}/7
        
        CHART CONTEXT:
        {synastry_txt}
        
        TASK:
        Provide a Premium Compatibility Analysis JSON.
        
        CRITICAL REQUIREMENT:
        You MUST generate exactly 3 specific remedies in the 'remedies' array. Do not leave it empty.
        - If the match is good, suggest remedies for bonding (e.g., "Regular Date Nights").
        - If the match is bad, suggest corrections (e.g., "Kumbh Vivah", "Mars Mantra").
        
        OUTPUT JSON STRUCTURE:
        {{
            "title": "One line verdict (e.g., 'Karmic Bond with High Growth')",
            "mangal_dosha": "Check Mars in 1,4,7,8,12. Return 'High', 'Low', or 'Absent'.",
            "psychological_analysis": "2 paragraphs on emotional and mental sync.",
            "sexual_chemistry": "Analysis of Venus/Mars connection.",
            "finance": "Financial outlook for the couple.",
            "progeny": "Brief outlook on family/progeny.",
            "destiny": "Overall shared luck.",
            "remedies": [
                {{"problem": "Nadi Dosh/Anger", "fix_modern": "Cooling breathwork", "fix_vedic": "Mahamrityunjaya Mantra"}},
                {{"problem": "Communication", "fix_modern": "Active listening", "fix_vedic": "Mercury Mantra"}},
                {{"problem": "Financial Stress", "fix_modern": "Joint budget planning", "fix_vedic": "Lakshmi Puja"}}
            ]
        }}
        """
        
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(prompt)
        
        ai_data = parse_ai_json(
            response.text,
            fallback={
                "title": "Analysis Complete", 
                "mangal_dosha": "Check Chart", 
                "psychological_analysis": response.text, 
                "remedies": []
            }
        )
        
        # Inject the Math-Calculated Guna Table into the AI response so Frontend can render it
        # (We rely on Python math for numbers, AI for text)
        ai_data['guna_table'] = [
            {"area": v['name'], "desc": "", "score": str(v['score']), "max": str(v['total'])}
            for k, v in koota_result['scores'].items()
        ]

        # 5. Save to Database
        if req.user_id:
            safe_db_operation(
                lambda: supabase.table("match_reports").insert({
                    "user_id": req.user_id,
                    "profile_a_name": req.you.name,
                    "profile_b_name": req.partner.name,
                    "compatibility_score": score_str,
                    "total_guna_score": float(koota_result['total']),
                    "ai_analysis": ai_data,
                    "report_type": "premium"
                }).execute(),
                "Failed to save match report to DB"
            )

        return {
            "status": "success",
            "data": {
                "score": score_str,
                "analysis": ai_data,
                "charts": {"p1": p1['raw_json'], "p2": p2['raw_json']}
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Match Logic Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to analyze match. Please try again.")


@router.post("/generate-report")
def generate_premium_report(req: MatchRequest):
    """
    Generates the Downloadable PDF (Premium Feature).
    """
    try:
        # 1. Calculate Charts & Math (Re-run to ensure freshness)
        p1 = calculate_birth_chart(req.you.dob, req.you.time, req.you.lat, req.you.lon, req.you.tz)
        p2 = calculate_birth_chart(req.partner.dob, req.partner.time, req.partner.lat, req.partner.lon, req.partner.tz)
        
        deg1 = get_moon_abs_degree(p1)
        deg2 = get_moon_abs_degree(p2)
        koota_result = calculate_kootas(deg1, deg2)
        
        # 2. Generate Content via AI (Dedicated PDF Prompt)
        prompt = f"""
        Act as a Vedic Astrologer. Write content for a PDF Report.
        Boy: {req.you.name}, Girl: {req.partner.name}.
        Score: {koota_result['total']}/36.
        
        CRITICAL: Provide 3 distinct remedies (Modern + Vedic) in the 'remedies' array.
        
        Output JSON:
        {{
            "title": "Verdict Title",
            "mangal_dosha": "Status",
            "psychological_analysis": "Deep text...",
            "sexual_chemistry": "Deep text...",
            "finance": "Text...",
            "progeny": "Text...",
            "destiny": "Text...",
            "remedies": [
                {{"problem": "Issue 1", "fix_modern": "Modern Fix 1", "fix_vedic": "Vedic Fix 1"}},
                {{"problem": "Issue 2", "fix_modern": "Modern Fix 2", "fix_vedic": "Vedic Fix 2"}},
                {{"problem": "Issue 3", "fix_modern": "Modern Fix 3", "fix_vedic": "Vedic Fix 3"}}
            ]
        }}
        """
        model = genai.GenerativeModel("gemini-2.0-flash")
        res = model.generate_content(prompt)
        ai_data = json.loads(res.text.replace("```json", "").replace("```", "").strip())
        
        # Add Guna Table to Data
        ai_data['guna_table'] = koota_result['scores'] # Pass the dict directly for PDF gen

        # 3. Structure Data for PDF Generator
        pdf_input = {
            "p1_name": req.you.name, 
            "p2_name": req.partner.name,
            "p1_chart": p1['raw_json'], 
            "p2_chart": p2['raw_json'],
            "score": koota_result, # Pass full koota object with scores dict
            "ai_analysis": ai_data
        }
        
        # 4. Generate PDF
        pdf_buffer = generate_match_pdf(pdf_input)
        
        return Response(
            content=pdf_buffer.getvalue(), 
            media_type="application/pdf", 
            headers={"Content-Disposition": "attachment; filename=Match_Report.pdf"}
        )

    except Exception as e:
        logger.error(f"PDF Gen Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))