import os
import json
import logging
from datetime import datetime, timezone, timedelta
from fastapi import APIRouter, HTTPException, Header
from pydantic import BaseModel, field_validator
from supabase import create_client, Client
import google.generativeai as genai
from calc import get_daily_transits, get_personalized_forecast_data, get_remedy_library

router = APIRouter()
logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
ADMIN_SECRET = os.getenv("ADMIN_SECRET")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
genai.configure(api_key=GEMINI_KEY)

ZODIAC_SIGNS = [
    "aries", "taurus", "gemini", "cancer", "leo", "virgo",
    "libra", "scorpio", "sagittarius", "capricorn", "aquarius", "pisces"
]

model_engine = None

# --- HELPERS ---
def get_ist_date():
    """Returns current date in IST (UTC+5:30)"""
    utc_now = datetime.now(timezone.utc)
    ist_offset = timedelta(hours=5, minutes=30)
    ist_now = utc_now + ist_offset
    return ist_now.date().isoformat()

def get_working_model():
    """Finds a working Gemini model for this environment"""
    candidates = ["gemini-2.0-flash", "gemini-2.0-flash-lite", "gemini-flash-latest", "gemini-1.5-flash", "gemini-pro"]
    for name in candidates:
        try:
            model = genai.GenerativeModel(name)
            model.generate_content("Hi")
            logger.info(f"✅ LOCKED ON: {name}")
            return model
        except: continue
    raise Exception("No working Gemini model found.")

# --- MODELS ---
class PersonalHoroscopeRequest(BaseModel):
    user_id: str = "guest"
    dob: str
    time: str
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

# --- ENDPOINTS ---

@router.post("/api/horoscope/personal")
def get_personal_horoscope(req: PersonalHoroscopeRequest):
    """
    Generates a Personalized Daily Vibe Check.
    """
    try:
        today_str = get_ist_date()
        is_guest = req.user_id == "guest" or not req.user_id

        # 1. CHECK DB CACHE
        if not is_guest:
            try:
                db_check = supabase.table("daily_predictions_log").select("full_prediction_json").eq("user_id", req.user_id).eq("date", today_str).execute()
                if db_check.data and len(db_check.data) > 0:
                    logger.info("✅ Returning Cached Horoscope")
                    transit_data = get_personalized_forecast_data(req.dob, req.time, req.lat, req.lon, req.tz)
                    return {"status": "success", "data": db_check.data[0]['full_prediction_json'], "transits": transit_data}
            except Exception as e:
                logger.warning(f"Cache check failed: {e}")

        # 2. CALCULATE MATH
        transit_data = get_personalized_forecast_data(req.dob, req.time, req.lat, req.lon, req.tz)
        remedy_lib = get_remedy_library()
        transit_context = "\n".join([t['description'] for t in transit_data]) if transit_data else "Planetary energy is stable today."

        # 3. GENERATE AI INSIGHTS
        prompt = f"""
        Act as a Modern Vedic Life Coach.
        USER CONTEXT: Date: {today_str}. Transits: {transit_context}. Remedy Menu: {json.dumps(remedy_lib)}
        
        TASK: Generate a "Dual-Mode" daily guide.
        
        OUTPUT JSON (Strict):
        {{
           "headline": "Punchy 1-line summary",
           "challenge": "One sentence on the main obstacle.",
           "modern_solution": {{ "title": "Physiological Fix", "action": "Specific task from Remedy Menu", "why": "Scientific reason" }},
           "traditional_solution": {{ "title": "Vedic Remedy", "action": "Specific task from Remedy Menu", "why": "Karmic reason" }},
           "lucky_color": "Color Name",
           "power_score": 75,
           "dominant_transit": "Planet Name"
        }}
        """
        
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(prompt)
        
        try:
            clean_text = response.text.replace("```json", "").replace("```", "").strip()
            data = json.loads(clean_text)
        except:
            data = {
                "headline": "Aligning with the Cosmos",
                "challenge": "Stay balanced today.",
                "modern_solution": {"title": "Breathe", "action": "Deep Breathing", "why": "Focus"},
                "traditional_solution": {"title": "Pray", "action": "Chant Om", "why": "Peace"},
                "power_score": 50
            }
        
        if not is_guest:
            try:
                supabase.table("daily_predictions_log").insert({
                    "user_id": req.user_id, "date": today_str, "full_prediction_json": data,
                    "dominant_transit": data.get("dominant_transit", "General"), "power_score": data.get("power_score", 50)
                }).execute()
            except: pass
        
        return {"status": "success", "data": data, "transits": transit_data}

    except Exception as e:
        logger.error(f"Forecast Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/api/horoscope/{sign}")
def get_horoscope(sign: str):
    """Public Generic Daily Horoscope (Returns Specific JSON Format)"""
    if sign.lower() not in ZODIAC_SIGNS:
        raise HTTPException(status_code=400, detail="Invalid Zodiac Sign")
    
    today = get_ist_date()
    res = supabase.table("daily_horoscopes").select("content").eq("date", today).eq("sign", sign.lower()).execute()
    
    if res.data: return res.data[0]['content']
    
    # Fallback matching your specific format
    return {
        "personal": "The stars are aligning. Please check back shortly for your detailed forecast.",
        "career": "Patience is key today. Focus on routine tasks.",
        "health": "Stay hydrated and rest well.",
        "lucky_color": "White",
        "lucky_number": "1"
    }

@router.post("/api/cron/generate")
def generate_daily_batch(x_admin_key: str = Header(None)):
    """
    Cron for Generic Horoscopes (Runs at Midnight).
    Generates data in the specific 'career, health, personal, lucky_color, lucky_number' format.
    """
    global model_engine
    if x_admin_key != ADMIN_SECRET: raise HTTPException(status_code=403, detail="Unauthorized")

    if not model_engine: model_engine = get_working_model()
    today = get_ist_date()
    
    try: transits = get_daily_transits()
    except: transits = "Sun in Scorpio"

    generated_count = 0
    for sign in ZODIAC_SIGNS:
        check = supabase.table("daily_horoscopes").select("id").eq("date", today).eq("sign", sign).execute()
        if not check.data:
            # STRICT SCHEMA FOR YOUR SPECIFIC FORMAT
            prompt = f"""
            Act as an Expert Vedic Astrologer.
            Date: {today}. Sign: {sign}. Transits: {transits}.
            
            Task: Write a daily horoscope.
            
            OUTPUT JSON ONLY (Strict Keys):
            {{
                "career": "2 sentences on work, ambition, and finance/investments.",
                "health": "1 sentence on energy and wellness.",
                "personal": "2 sentences on emotions, relationships, and inner state.",
                "lucky_color": "Color Name",
                "lucky_number": "Number string"
            }}
            """
            try:
                res = model_engine.generate_content(prompt)
                content = json.loads(res.text.replace("```json", "").replace("```", "").strip())
                
                # Save row-by-row for each sign
                supabase.table("daily_horoscopes").insert({
                    "date": today, 
                    "sign": sign, 
                    "content": content
                }).execute()
                generated_count += 1
            except Exception as e:
                logger.error(f"Failed {sign}: {e}")

    return {"status": "success", "generated": generated_count}