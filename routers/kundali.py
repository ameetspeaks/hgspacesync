"""
Kundali Storage & PDF Router
Adds: POST /api/kundali/calculate-and-save
      POST /api/kundali/{kundali_id}/pdf
      GET  /api/kundali/my-kundalis

Aligned with the existing hgspacesync architecture:
- Uses calc.py (Skyfield engine) — same as calculator.py and report.py
- Uses pdf_generator.py (ReportLab) — same as report.py
- Uses Supabase client — same pattern as report.py
- Does NOT break any existing routes
"""

import os
import io
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import APIRouter, HTTPException, Header, Response
from pydantic import BaseModel, Field, field_validator

from supabase import create_client, Client

# Re-use existing calc engine (same as calculator.py and report.py)
from calc import calculate_birth_chart, SIGNS
# Re-use existing PDF engine (same as report.py)
from routers.pdf_generator import generate_kundli_pdf
from utils import validate_chart_result, safe_db_operation, parse_ai_json

router = APIRouter()
logger = logging.getLogger(__name__)

# --- CONFIG (same pattern as report.py) ---
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")          # service role key
SUPABASE_ANON = os.getenv("SUPABASE_ANON_KEY", SUPABASE_KEY)

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


# --- MODELS (aligned with BirthChartRequest in calculator.py) ---
class KundaliSaveRequest(BaseModel):
    name: str = Field(..., min_length=1)
    date_of_birth: str = Field(..., description="YYYY-MM-DD")
    time_of_birth: str = Field(..., description="HH:MM")
    place_of_birth: str = Field(..., min_length=1)
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    timezone: float = Field(default=5.5)
    ayanamsa: str = Field(default="lahiri")
    tob_unknown: bool = Field(default=False)

    @field_validator('date_of_birth')
    @classmethod
    def validate_date(cls, v):
        import re
        if not re.match(r'^\d{4}-\d{2}-\d{2}$', v):
            raise ValueError('Date must be YYYY-MM-DD')
        return v

    @field_validator('time_of_birth')
    @classmethod
    def validate_time(cls, v):
        import re
        if not re.match(r'^\d{2}:\d{2}$', v):
            raise ValueError('Time must be HH:MM')
        return v


# --- HELPERS ---
def _get_user_id_from_token(authorization: Optional[str]) -> str:
    """Validates the Bearer token and returns user_id from Supabase auth."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    token = authorization.split(" ", 1)[1]
    try:
        # Use anon-key client to validate user JWT
        user_client: Client = create_client(SUPABASE_URL, SUPABASE_ANON)
        user_resp = user_client.auth.get_user(token)
        return user_resp.user.id
    except Exception as e:
        logger.error(f"Auth token validation failed: {e}")
        raise HTTPException(status_code=401, detail="Invalid or expired token")


def _build_chart_data(req: KundaliSaveRequest) -> dict:
    """Runs the existing Skyfield engine (same as calculator.py)."""
    tob = "12:00" if req.tob_unknown else req.time_of_birth
    result = calculate_birth_chart(
        dob=req.date_of_birth,
        time=tob,
        lat=req.latitude,
        lon=req.longitude,
        tz=req.timezone
    )
    if not validate_chart_result(result):
        raise HTTPException(status_code=400, detail="Failed to calculate birth chart. Check your birth details.")
    # Enrich with panchanga-style fields for completeness of JSON stored in DB
    result["meta"] = {
        "ayanamsa": req.ayanamsa,
        "tob_unknown": req.tob_unknown,
        "place": req.place_of_birth,
        "latitude": req.latitude,
        "longitude": req.longitude,
        "timezone": req.timezone,
    }
    return result


# =====================================================================
# ENDPOINT 1: Calculate chart + save to Supabase kundalis table
# POST /api/kundali/calculate-and-save
# =====================================================================
@router.post("/calculate-and-save")
def calculate_and_save_kundali(req: KundaliSaveRequest, authorization: Optional[str] = Header(None)):
    """
    1. Runs the Skyfield birth-chart engine (same as /api/birth-chart/generate)
    2. Saves the full JSON to the `kundalis` Supabase table
    3. Returns the chart data immediately (no extra DB read needed by Flutter)
    """
    user_id = _get_user_id_from_token(authorization)

    # --- Calc (existing engine) ---
    chart_data = _build_chart_data(req)

    # --- Determine if primary ---
    try:
        existing = supabase.table("kundalis").select("id").eq("user_id", user_id).limit(1).execute()
        is_primary = len(existing.data) == 0
    except Exception:
        is_primary = False

    # --- Save to DB (fire-and-forget style — we still return chart_data) ---
    kundali_id = None
    try:
        insert_resp = supabase.table("kundalis").insert({
            "user_id": user_id,
            "profile_name": req.name,
            "dob": req.date_of_birth,
            "tob": req.time_of_birth + ":00",
            "tob_unknown": req.tob_unknown,
            "pob_name": req.place_of_birth,
            "pob_lat": req.latitude,
            "pob_lng": req.longitude,
            "timezone": f"Etc/GMT{int(-req.timezone):+d}",
            "ayanamsa": req.ayanamsa,
            "chart_data": chart_data,
            "is_primary": is_primary,
        }).select("id").execute()
        if insert_resp.data:
            kundali_id = insert_resp.data[0]["id"]
    except Exception as e:
        # Non-fatal: chart data still returned
        logger.error(f"Kundali save error (non-fatal): {e}")

    return {
        "status": "success",
        "kundali_id": kundali_id,
        "is_primary": is_primary,
        "chart": chart_data,
    }


# =====================================================================
# ENDPOINT 2: Generate & cache Kundali PDF
# POST /api/kundali/{kundali_id}/pdf
# =====================================================================
@router.post("/{kundali_id}/pdf")
def generate_kundali_pdf_endpoint(kundali_id: str, authorization: Optional[str] = Header(None)):
    """
    Checks cache first. If stale/missing, generates a fresh ReportLab PDF
    (same engine as /api/report/generate-kundli-pdf) and uploads to
    Supabase Storage bucket 'kundali-pdfs'.
    Returns a signed URL (1-hour validity).
    """
    user_id = _get_user_id_from_token(authorization)

    # --- 1. Fetch kundali row ---
    try:
        k_resp = supabase.table("kundalis").select("*").eq("id", kundali_id).eq("user_id", user_id).execute()
        if not k_resp.data:
            raise HTTPException(status_code=404, detail="Kundali not found or access denied")
        k = k_resp.data[0]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # --- 2. Check PDF cache ---
    try:
        cache_resp = supabase.table("kundali_pdf_cache") \
            .select("*") \
            .eq("kundali_id", kundali_id) \
            .execute()
        now_utc = datetime.now(timezone.utc)
        valid_cache = [
            c for c in (cache_resp.data or [])
            if datetime.fromisoformat(c["expires_at"].replace("Z", "+00:00")) > now_utc
        ]
        if valid_cache:
            storage_path = valid_cache[0]["storage_path"]
            signed = supabase.storage.from_("kundali-pdfs").create_signed_url(storage_path, 3600)
            return {"pdf_url": signed.get("signedURL") or signed.get("signedUrl"), "cached": True}
    except Exception as e:
        logger.warning(f"Cache check failed (proceeding to generate): {e}")

    # --- 3. Generate fresh PDF using existing ReportLab engine ---
    chart_data = k.get("chart_data", {})
    pdf_buffer = generate_kundli_pdf({
        "user": {
            "name": k["profile_name"],
            "dob": k["dob"],
            "time": k["tob"],
            "place": k["pob_name"],
        },
        "chart": chart_data,
        "analysis": {
            "personality": chart_data.get("ai_summary", "Analysis based on birth chart."),
            "remedies": [],
        }
    })

    # --- 4. Upload to Supabase Storage ---
    storage_path = f"{user_id}/{kundali_id}.pdf"
    try:
        supabase.storage.from_("kundali-pdfs").upload(
            path=storage_path,
            file=pdf_buffer.getvalue(),
            file_options={"content-type": "application/pdf", "upsert": "true"}
        )
    except Exception as e:
        # Fallback: stream bytes directly if storage upload fails
        logger.error(f"Storage upload failed, streaming directly: {e}")
        return Response(
            content=pdf_buffer.getvalue(),
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename=Kundali_{k['profile_name']}.pdf"}
        )

    # --- 5. Insert / update cache row ---
    expires_at = (datetime.now(timezone.utc) + timedelta(hours=72)).isoformat()
    try:
        # Delete stale rows first
        supabase.table("kundali_pdf_cache").delete().eq("kundali_id", kundali_id).execute()
        supabase.table("kundali_pdf_cache").insert({
            "kundali_id": kundali_id,
            "storage_path": storage_path,
            "expires_at": expires_at,
        }).execute()
    except Exception as e:
        logger.warning(f"Cache insert failed (non-fatal): {e}")

    # --- 6. Return signed URL ---
    try:
        signed = supabase.storage.from_("kundali-pdfs").create_signed_url(storage_path, 3600)
        return {"pdf_url": signed.get("signedURL") or signed.get("signedUrl"), "cached": False}
    except Exception as e:
        # Ultimate fallback: stream the PDF
        pdf_buffer.seek(0)
        return Response(
            content=pdf_buffer.getvalue(),
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename=Kundali_{k['profile_name']}.pdf"}
        )


# =====================================================================
# ENDPOINT 3: List saved kundalis for the authenticated user
# GET /api/kundali/my-kundalis
# =====================================================================
@router.get("/my-kundalis")
def list_my_kundalis(authorization: Optional[str] = Header(None)):
    """Returns saved kundalis for the user, primary first then by created_at desc."""
    user_id = _get_user_id_from_token(authorization)
    try:
        resp = supabase.table("kundalis") \
            .select("id, profile_name, dob, tob, is_primary, created_at, pob_name") \
            .eq("user_id", user_id) \
            .order("is_primary", desc=True) \
            .order("created_at", desc=True) \
            .execute()
        return {"status": "success", "kundalis": resp.data or []}
    except Exception as e:
        logger.error(f"List kundalis error: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch kundalis")
