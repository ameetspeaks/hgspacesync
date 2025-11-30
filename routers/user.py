import os
import logging
from fastapi import APIRouter, HTTPException, Header
from pydantic import BaseModel
from supabase import create_client, Client
from calc import calculate_birth_chart

router = APIRouter()
logger = logging.getLogger(__name__)

# Config
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
ADMIN_SECRET = os.getenv("ADMIN_SECRET")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

class UserProfileRequest(BaseModel):
    user_id: str # The UUID from Supabase Auth
    dob: str
    time: str
    lat: float
    lon: float
    tz: float

@router.post("/process-kundli")
def process_user_kundli(req: UserProfileRequest, x_admin_key: str = Header(None)):
    """
    Generates Kundli data and saves it PERMANENTLY to the database.
    """
    # 1. Security Check (Prevent random people from overwriting data)
    # In a real app, you'd verify the JWT, but Admin Key works for internal calls
    if x_admin_key != ADMIN_SECRET:
        raise HTTPException(status_code=403, detail="Unauthorized")

    try:
        # 2. Perform Heavy Calculation
        result = calculate_birth_chart(req.dob, req.time, req.lat, req.lon, req.tz)
        
        if isinstance(result, str): # Handle error cases from calc.py
            raise HTTPException(status_code=400, detail=result)

        # 3. Save to Supabase
        data_payload = {
            "user_id": req.user_id,
            "ai_summary_text": result["ai_summary"],
            "chart_data": result["raw_json"],
            "ascendant_sign": result["key_points"]["ascendant"],
            "moon_sign": result["key_points"]["moon_sign"],
            "updated_at": "now()"
        }

        # Upsert (Insert or Update)
        res = supabase.table("user_kundli_data").upsert(data_payload).execute()
        
        return {"status": "success", "data": "Kundli Processed and Saved"}

    except Exception as e:
        logger.error(f"Processing Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))