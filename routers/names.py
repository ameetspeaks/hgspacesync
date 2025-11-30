import os
import logging
import json
from fastapi import APIRouter, HTTPException, Header
from pydantic import BaseModel
from typing import Optional, List
import google.generativeai as genai
from calc import get_naming_details

router = APIRouter()
logger = logging.getLogger(__name__)

GEMINI_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_KEY)

class NamingRequest(BaseModel):
    dob: str
    time: str
    lat: float
    lon: float
    tz: float
    gender: str = "U" 
    parent_names: Optional[List[str]] = None 
    mode: str = "vedic" 

@router.post("/generate")
def generate_baby_names(req: NamingRequest):
    try:
        # 1. Math (From updated calc.py)
        vedic_info = get_naming_details(req.dob, req.time, req.lat, req.lon, req.tz)
        if "error" in vedic_info: 
            raise HTTPException(status_code=400, detail=vedic_info['error'])
        
        sounds = vedic_info['all_sounds']
        primary = vedic_info['primary_sound']
        
        # 2. AI
        prompt = f"""
        Act as a Creative Vedic Naming Expert.
        
        CONTEXT:
        - Gender: {req.gender}
        - Nakshatra: {vedic_info['nakshatra']} (Pada {vedic_info['pada']})
        - Sacred Sounds: {', '.join(sounds)}
        - Parents: {', '.join(req.parent_names) if req.parent_names else 'None'}
        - Mode: {req.mode}
        
        TASK:
        Generate 10 unique names.
        
        RULES:
        1. Names MUST start with one of: {', '.join(sounds)}. Priority to '{primary}'.
        2. If Mode='Fusion', blend parent names but keep the starting sound if possible.
        3. Provide deep Sanskrit meaning.
        
        OUTPUT JSON:
        {{
            "vedic_analysis": "Short explanation of why these sounds were chosen.",
            "names": [
                {{ "name": "...", "meaning": "...", "origin": "Sanskrit", "highlight": "Starts with {primary}" }}
            ]
        }}
        """
        
        model = genai.GenerativeModel("gemini-2.0-flash")
        res = model.generate_content(prompt)
        data = json.loads(res.text.replace("```json", "").replace("```", "").strip())
        
        return {
            "status": "success",
            "meta": vedic_info,
            "results": data
        }

    except Exception as e:
        logger.error(f"Naming Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))