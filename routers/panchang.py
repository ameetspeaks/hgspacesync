import os
import logging
from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException, Header
from pydantic import BaseModel
from supabase import create_client, Client
from calc import get_detailed_panchang

router = APIRouter()
logger = logging.getLogger(__name__)

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
ADMIN_SECRET = os.getenv("ADMIN_SECRET")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

class PanchangRequest(BaseModel):
    date: str = None
    lat: float = 28.61
    lon: float = 77.20
    city: str = "New Delhi"
    tz: float = 5.5

@router.post("/calculate")
def get_panchang(req: PanchangRequest):
    try:
        target_date = datetime.fromisoformat(req.date) if req.date else datetime.now(timezone.utc)
        date_str = target_date.date().isoformat()
        r_lat = round(req.lat, 2)
        r_lon = round(req.lon, 2)

        # DB Check
        try:
            res = supabase.table("daily_panchang_log").select("full_data").eq("date", date_str).eq("latitude", r_lat).eq("longitude", r_lon).execute()
            if res.data: return {"status": "success", "data": res.data[0]['full_data']}
        except: pass

        # Calc
        data = get_detailed_panchang(target_date, req.lat, req.lon, req.tz)
        data['location']['name'] = req.city

        # Save
        try:
            supabase.table("daily_panchang_log").upsert({
                "date": date_str, "latitude": r_lat, "longitude": r_lon,
                "location_name": req.city, "tithi_name": data['meta']['tithi'],
                "nakshatra_name": data['meta']['nakshatra'], "full_data": data
            }, on_conflict="date,latitude,longitude").execute()
        except: pass

        return {"status": "success", "data": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/cron/generate")
def cron_generate(x_admin_key: str = Header(None)):
    if x_admin_key != ADMIN_SECRET: raise HTTPException(status_code=403, detail="Unauthorized")
    cities = [{"lat": 28.61, "lon": 77.20, "tz": 5.5}, {"lat": 19.07, "lon": 72.87, "tz": 5.5}]
    for c in cities: get_detailed_panchang(datetime.now(timezone.utc), c['lat'], c['lon'], c['tz'])
    return {"status": "success"}