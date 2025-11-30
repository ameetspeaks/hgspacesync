import os  # <--- THIS WAS MISSING
import logging
import json
from datetime import datetime, timezone, timedelta
from fastapi import APIRouter, HTTPException, Header
from pydantic import BaseModel
import google.generativeai as genai
from supabase import create_client, Client
from calc import calculate_birth_chart, get_planet_habits_library
from utils import parse_ai_json, validate_chart_result, safe_db_operation

router = APIRouter()
logger = logging.getLogger(__name__)

# Config
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

genai.configure(api_key=GEMINI_KEY)
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# --- MODELS ---
class ResolutionGenRequest(BaseModel):
    user_id: str
    dob: str
    time: str
    lat: float
    lon: float
    tz: float

class CommitRequest(BaseModel):
    user_id: str
    planet: str
    habit_title: str
    reasoning: str

class CheckInRequest(BaseModel):
    user_id: str
    resolution_id: str
    note: str = ""

# --- ENDPOINTS ---

@router.post("/generate")
def generate_resolution_options(req: ResolutionGenRequest):
    try:
        # 1. Calculate Chart
        chart = calculate_birth_chart(req.dob, req.time, req.lat, req.lon, req.tz)
        
        if not validate_chart_result(chart):
            raise HTTPException(status_code=400, detail="Invalid birth data provided")
        
        chart_txt = chart.get('ai_summary', str(chart)) if isinstance(chart, dict) else str(chart)
        
        # 2. Get Habit Library
        habit_lib = get_planet_habits_library()

        # 3. Ask AI
        prompt = f"""
        Role: Vedic Life Coach.
        Task: Analyze this chart for the year 2026.
        CHART: {chart_txt}
        HABIT LIBRARY: {json.dumps(habit_lib)}
        
        INSTRUCTIONS:
        1. Identify the 3 most critical planets that need strengthening or balancing in 2026.
        2. Select a specific habit from the library (or suggest a better one).
        3. Provide a "Cosmic Why" (Reasoning).
        
        OUTPUT JSON:
        [
          {{
            "planet": "Mars",
            "habit": "20 Morning Pushups",
            "reason": "Your Mars is debilitated in Cancer; you need physical fire to fuel your career."
          }},
          ... (2 more)
        ]
        """
        
        model = genai.GenerativeModel("gemini-2.0-flash")
        res = model.generate_content(prompt)
        
        # Clean JSON
        data = parse_ai_json(res.text, fallback=[])
        
        if not isinstance(data, list):
            data = []
        
        return {"status": "success", "options": data}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Resolution Gen Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to generate resolutions. Please try again.")

@router.post("/commit")
def commit_to_sankalpa(req: CommitRequest):
    try:
        # Prepare data
        data = {
            "user_id": req.user_id,
            "year": 2026,
            "planet": req.planet,
            "habit_title": req.habit_title,
            "reasoning": req.reasoning,
            "current_streak": 0,
            "total_completions": 0
        }
        
        # Save to DB
        res = safe_db_operation(
            lambda: supabase.table("user_resolutions").insert(data).select().execute(),
            "Failed to save resolution to DB"
        )
        
        # Handle Supabase response format safely
        if res and res.data and len(res.data) > 0:
            return {"status": "success", "data": res.data[0]}
        else:
            raise HTTPException(status_code=500, detail="Failed to save resolution")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Commit Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to commit resolution. Please try again.")

@router.post("/check-in")
def daily_check_in(req: CheckInRequest):
    today = datetime.now(timezone.utc).date()
    
    try:
        # 1. Log the entry
        log_data = {
            "resolution_id": req.resolution_id,
            "user_id": req.user_id,
            "checkin_date": today.isoformat(),
            "note": req.note
        }
        safe_db_operation(
            lambda: supabase.table("resolution_logs").insert(log_data).execute(),
            "Failed to log check-in"
        )
        
        # 2. Update Streak Logic
        res_data = safe_db_operation(
            lambda: supabase.table("user_resolutions").select("*").eq("id", req.resolution_id).single().execute(),
            "Failed to fetch resolution"
        )
        
        if not res_data or not res_data.data:
            return {"status": "error", "message": "Resolution not found"}
        
        curr = res_data.data
        
        new_streak = curr['current_streak']
        
        if curr['last_checkin']:
            last_date = datetime.fromisoformat(curr['last_checkin']).date()
            if last_date == today - timedelta(days=1):
                new_streak += 1 # Continued streak
            elif last_date == today:
                pass # Already checked in
            else:
                new_streak = 1 # Broken streak
        else:
            new_streak = 1 # First check-in
            
        # Update Resolution Table
        safe_db_operation(
            lambda: supabase.table("user_resolutions").update({
                "current_streak": new_streak,
                "total_completions": curr['total_completions'] + 1,
                "last_checkin": today.isoformat()
            }).eq("id", req.resolution_id).execute(),
            "Failed to update resolution streak"
        )
        
        return {
            "status": "success", 
            "new_streak": new_streak, 
            "message": f"Sankalpa Completed! Streak: {new_streak} Days"
        }

    except Exception as e:
        # Likely a unique constraint violation (already checked in today)
        logger.warning(f"Check-in error (likely duplicate): {e}")
        return {"status": "error", "message": "Already checked in today!"}