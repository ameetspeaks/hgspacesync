import os
import io
import logging
import json
from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel, field_validator
from typing import Optional, Dict, Any
import google.generativeai as genai
from supabase import create_client, Client

# Import Logic
from calc import calculate_birth_chart, calculate_kootas, SIGNS
from routers.pdf_generator import generate_match_pdf, generate_kundli_pdf
from utils import parse_ai_json, validate_chart_result, format_error_response, safe_db_operation

router = APIRouter()
logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

genai.configure(api_key=GEMINI_KEY)
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# --- ROBUST MODELS (Fixes 422 Errors) ---
class PartnerProfile(BaseModel):
    name: str
    dob: str
    time: str
    # Accept Float OR String, then auto-convert
    lat: float | str 
    lon: float | str
    tz: float | str

    @field_validator('lat', 'lon', 'tz', mode='before')
    @classmethod
    def parse_floats(cls, v):
        if v == "" or v is None: return 0.0
        try:
            return float(v)
        except:
            return 0.0

class MatchPDFRequest(BaseModel):
    you: PartnerProfile
    partner: PartnerProfile
    ai_analysis: Optional[Dict[str, Any]] = None 

class ReportRequest(BaseModel):
    user_id: Optional[str] = None
    name: str
    dob: str
    time: str
    lat: float | str
    lon: float | str
    tz: float | str
    payment_verified: bool = False 

    @field_validator('lat', 'lon', 'tz', mode='before')
    @classmethod
    def parse_floats(cls, v):
        if v == "" or v is None: return 0.0
        try:
            return float(v)
        except:
            return 0.0

# --- HELPERS ---
def get_moon_abs_degree(chart_data):
    """Extracts Moon degree for Guna Calc"""
    moon_obj = next((p for p in chart_data['raw_json'] if p['planet'] == 'Moon'), None)
    if not moon_obj: return 0.0
    try:
        sign_idx = SIGNS.index(moon_obj['sign'])
        return (sign_idx * 30.0) + moon_obj['degree']
    except: return 0.0

# validate_ai_json moved to utils.py

# --- ENDPOINTS ---

@router.post("/match-pdf")
def generate_match_report_pdf(req: MatchPDFRequest):
    """
    Generates the Premium Visual Match Report.
    """
    try:
        # 1. Calculate Charts (Fast & Free)
        p1 = calculate_birth_chart(req.you.dob, req.you.time, req.you.lat, req.you.lon, req.you.tz)
        p2 = calculate_birth_chart(req.partner.dob, req.partner.time, req.partner.lat, req.partner.lon, req.partner.tz)
        
        if not validate_chart_result(p1) or not validate_chart_result(p2):
            raise HTTPException(status_code=400, detail="Chart Calculation Error")

        # 2. Calculate Gunas (The Math)
        deg1 = get_moon_abs_degree(p1)
        deg2 = get_moon_abs_degree(p2)
        koota_result = calculate_kootas(deg1, deg2)
        
        # 3. Generate AI Content (If not provided in request)
        ai_data = req.ai_analysis
        if not ai_data:
            prompt = f"""
            Act as an Expert Vedic Relationship Astrologer.
            Boy: {req.you.name}, Girl: {req.partner.name}.
            Score: {koota_result['total']}/36.
            
            TASK: Write a Premium Compatibility Report.
            
            OUTPUT JSON:
            {{
                "title": "Verdict Title",
                "mangal_dosha": "High/Low/Absent",
                "psychological_analysis": "Detailed analysis...",
                "sexual_chemistry": "Detailed analysis...",
                "finance": "Financial outlook...",
                "progeny": "Family outlook...",
                "remedies": [
                    {{"problem": "Main Issue", "fix_modern": "Psychological fix", "fix_vedic": "Mantra/Ritual"}},
                    {{"problem": "Secondary Issue", "fix_modern": "Habit change", "fix_vedic": "Donation"}}
                ]
            }}
            """
            model = genai.GenerativeModel("gemini-2.0-flash")
            res = model.generate_content(prompt)
            ai_data = parse_ai_json(res.text, fallback={})

        # 4. Inject Math Table into AI Data
        if 'scores' in koota_result:
             ai_data['score'] = koota_result # Ensure score object is passed correctly

        # 5. Prepare Data Package
        pdf_data = {
            "p1_name": req.you.name, 
            "p2_name": req.partner.name,
            "p1_chart": p1['raw_json'], 
            "p2_chart": p2['raw_json'],
            "score": koota_result, 
            "ai_analysis": ai_data
        }
        
        # 6. Generate PDF
        pdf_buffer = generate_match_pdf(pdf_data)
        
        return Response(
            content=pdf_buffer.getvalue(), 
            media_type="application/pdf", 
            headers={"Content-Disposition": "attachment; filename=Match_Report.pdf"}
        )
        
    except Exception as e:
        logger.error(f"Match PDF Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate-kundli-pdf")
def generate_kundli_report(req: ReportRequest):
    """
    Generates Detailed 20+ Page Life Report.
    """
    try:
        # 1. Calculate All Math
        chart_data = calculate_birth_chart(req.dob, req.time, req.lat, req.lon, req.tz)
        
        if not validate_chart_result(chart_data):
            raise HTTPException(status_code=400, detail="Math Error")

        # 2. AI Analysis
        prompt = f"""
        Act as a Senior Vedic Astrologer. Write a 'Grand Life Report' for {req.name}.
        DATA: Chart Summary: {chart_data['ai_summary']}
        
        OUTPUT JSON STRUCTURE:
        {{
            "personality": "400 words on character...",
            "career": "400 words on profession...",
            "marriage": "300 words on love life...",
            "health": "200 words on health...",
            "past_life": "200 words on karma...",
            "varshphal": "Forecast for next 2 years...",
            "remedies": [
                {{"type": "Gemstone", "desc": "Gem details"}},
                {{"type": "Mantra", "desc": "Mantra details"}}
            ]
        }}
        """
        
        model = genai.GenerativeModel("gemini-2.0-flash")
        res = model.generate_content(prompt)
        ai_text = parse_ai_json(
            res.text,
            fallback={"personality": "Analysis pending...", "remedies": []}
        )

        # 3. Save to DB
        if req.user_id:
            safe_db_operation(
                lambda: supabase.table("life_reports").insert({
                    "user_id": req.user_id,
                    "target_name": req.name,
                    "chapters_data": ai_text,
                    "report_type": "grand_kundli"
                }).execute(),
                "Failed to save life report to DB"
            )

        # 4. Generate PDF
        pdf_buffer = generate_kundli_pdf({
            "user": req.dict(),
            "chart": chart_data,
            "analysis": ai_text
        })
        
        return Response(
            content=pdf_buffer.getvalue(), 
            media_type="application/pdf", 
            headers={"Content-Disposition": f"attachment; filename=Kundli_{req.name}.pdf"}
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Report Gen Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to generate report. Please try again.")