import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from datetime import datetime, timezone
from supabase import create_client, Client
from calc import get_detailed_panchang, get_special_event, get_rashi_plant, calculate_birth_chart, get_daily_mantra, get_donation_item

router = APIRouter()
logger = logging.getLogger(__name__)

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

class DailyRequest(BaseModel):
    user_id: str
    dob: str
    time: str
    lat: float
    lon: float
    tz: float
    current_date: str = None 

@router.post("/feed")
def get_daily_feed(req: DailyRequest):
    try:
        dt = datetime.fromisoformat(req.current_date) if req.current_date else datetime.now(timezone.utc)
        panchang = get_detailed_panchang(dt, req.lat, req.lon, req.tz)
        chart = calculate_birth_chart(req.dob, req.time, req.lat, req.lon, req.tz)
        
        moon_sign = chart['key_points']['moon_sign']
        evt = get_special_event(panchang['meta']['tithi'], panchang['meta']['weekday'])
        mantra = get_daily_mantra(panchang['meta']['weekday'], panchang['meta']['tithi'])
        donation = get_donation_item(panchang['meta']['weekday'])
        plant = get_rashi_plant(moon_sign)

        curr_h = dt.hour + (dt.minute/60.0) + req.tz
        rahu_start = panchang['timing']['rahu_start']
        rahu_end = panchang['timing']['rahu_end']
        time_msg = "✅ Good time" if not (rahu_start <= curr_h < rahu_end) else "⚠️ Rahu Kaal"

        stats = {"karma": 0, "streak": 0, "done": []}
        try:
            s = supabase.table("user_gamification_stats").select("*").eq("user_id", req.user_id).execute()
            if s.data: stats["karma"], stats["streak"] = s.data[0]['total_karma'], s.data[0]['current_streak']
            l = supabase.table("user_sadhana_logs").select("action_type").eq("user_id", req.user_id).eq("date", dt.date().isoformat()).execute()
            if l.data: stats["done"] = [r['action_type'] for r in l.data]
        except: pass

        return {
            "status": "success",
            "date": panchang['date'],
            "meta": panchang['meta'],
            "special_event": evt,
            "tasks": [
                {"id": "mantra", "title": "Chant", "text": mantra['text'], "pts": 10, "is_done": "mantra" in stats["done"]},
                {"id": "charity", "title": "Donate", "text": donation, "pts": 20, "is_done": "charity" in stats["done"]},
                {"id": "time_mastery", "title": "Time", "text": time_msg, "pts": 5, "is_done": "time_mastery" in stats["done"]},
                {"id": "plant", "title": "Nature", "text": f"Care for {plant}", "pts": 15, "is_done": "plant" in stats["done"]}
            ],
            "user_stats": stats
        }
    except Exception as e:
        logger.error(f"Feed Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/complete-task")
def complete_task(user_id: str, action_type: str, points: int):
    today = datetime.now(timezone.utc).date().isoformat()
    try:
        supabase.table("user_sadhana_logs").insert({"user_id": user_id, "date": today, "action_type": action_type, "points": points}).execute()
        curr = supabase.table("user_gamification_stats").select("*").eq("user_id", user_id).execute()
        if not curr.data:
            supabase.table("user_gamification_stats").insert({"user_id": user_id, "total_karma": points, "current_streak": 1, "last_checkin": today}).execute()
        else:
            old = curr.data[0]
            new_pts = old['total_karma'] + points
            streak = old['current_streak'] + (1 if old['last_checkin'] != today else 0)
            supabase.table("user_gamification_stats").update({"total_karma": new_pts, "current_streak": streak, "last_checkin": today}).eq("user_id", user_id).execute()
        return {"status": "success"}
    except: return {"status": "error"}