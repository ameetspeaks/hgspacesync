import os
import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator
from typing import Optional
from calc import calculate_birth_chart
from utils import validate_chart_result, format_error_response

router = APIRouter()
logger = logging.getLogger(__name__)

# --- DATA MODELS ---
class BirthChartRequest(BaseModel):
    name: str = Field(..., min_length=1, description="Name of the person")
    date_of_birth: str = Field(..., description="Date of birth in YYYY-MM-DD format")
    time_of_birth: str = Field(..., description="Time of birth in HH:MM format")
    place_of_birth: str = Field(..., description="Place of birth")
    latitude: float = Field(..., ge=-90, le=90, description="Latitude of birth place")
    longitude: float = Field(..., ge=-180, le=180, description="Longitude of birth place")
    timezone: float = Field(default=0, description="Timezone offset in hours")
    
    @field_validator('date_of_birth')
    @classmethod
    def validate_date_format(cls, v):
        """Validate date format YYYY-MM-DD"""
        import re
        if not re.match(r'^\d{4}-\d{2}-\d{2}$', v):
            raise ValueError('Date must be in YYYY-MM-DD format')
        try:
            from datetime import datetime
            datetime.strptime(v, '%Y-%m-%d')
        except ValueError:
            raise ValueError('Invalid date')
        return v
    
    @field_validator('time_of_birth')
    @classmethod
    def validate_time_format(cls, v):
        """Validate time format HH:MM"""
        import re
        if not re.match(r'^\d{2}:\d{2}$', v):
            raise ValueError('Time must be in HH:MM format')
        parts = v.split(':')
        hour, minute = int(parts[0]), int(parts[1])
        if not (0 <= hour <= 23) or not (0 <= minute <= 59):
            raise ValueError('Invalid time range')
        return v

class MoonSignRequest(BaseModel):
    date_of_birth: str
    time_of_birth: str
    latitude: float
    longitude: float
    timezone: float = 0

class DashaRequest(BaseModel):
    date_of_birth: str
    time_of_birth: str
    latitude: float
    longitude: float
    timezone: float = 0

# --- ENDPOINT ---
@router.post("/generate")
def generate_birth_chart(req: BirthChartRequest):
    """
    Generate birth chart with planetary positions, houses, and key astrological data.
    """
    try:
        # Calculate birth chart
        result = calculate_birth_chart(
            dob=req.date_of_birth,
            time=req.time_of_birth,
            lat=req.latitude,
            lon=req.longitude,
            tz=req.timezone
        )
        
        # Validate result
        if not validate_chart_result(result):
            error_msg = result.get('ai_summary', 'Invalid birth data provided') if isinstance(result, dict) else 'Invalid birth data provided'
            logger.error(f"Invalid chart result: {error_msg}")
            raise HTTPException(
                status_code=400,
                detail=error_msg if "Error:" not in str(error_msg) else "Invalid birth data. Please check your date, time, and location."
            )
        
        # Extract key information
        key_points = result.get('key_points', {})
        chart_data = result.get('raw_json', [])
        dasha_timeline = result.get('dasha_timeline', [])
        
        # Build response
        response = {
            "status": "success",
            "name": req.name,
            "birth_details": {
                "date_of_birth": req.date_of_birth,
                "time_of_birth": req.time_of_birth,
                "place_of_birth": req.place_of_birth,
                "latitude": req.latitude,
                "longitude": req.longitude,
                "timezone": req.timezone
            },
            "chart": {
                "ascendant": key_points.get('ascendant', ''),
                "moon_sign": key_points.get('moon_sign', ''),
                "planets": chart_data
            },
            "dasha_timeline": dasha_timeline,
            "summary": result.get('ai_summary', '')
        }
        
        return response
        
    except HTTPException:
        raise
    except ValueError as ve:
        # Invalid input data
        logger.error(f"Validation error: {ve}", exc_info=True)
        raise HTTPException(
            status_code=400,
            detail=str(ve)
        )
    except Exception as e:
        # Unexpected error
        logger.error(f"Birth chart generation error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Failed to generate birth chart. Please verify your birth details and try again."
        )

@router.post("/moon-sign")
def calculate_moon_sign(req: MoonSignRequest):
    """
    Calculate moon sign and generate detailed analysis.
    """
    try:
        import google.generativeai as genai
        # 1. Calc math
        result = calculate_birth_chart(
            dob=req.date_of_birth,
            time=req.time_of_birth,
            lat=req.latitude,
            lon=req.longitude,
            tz=req.timezone
        )
        
        moon_sign = result.get('key_points', {}).get('moon_sign', 'Aries')
        
        # 2. Generate content using Gemini
        prompt = f"""
        Act as an expert Vedic Astrologer. Provide a detailed Moon Sign (Rashi) report for someone with {moon_sign} Moon Sign.
        
        OUTPUT JSON FORMAT:
        {{
            "sign": "{moon_sign}",
            "sanskritName": "Sanskrit Name",
            "dates": "Date range (approx)",
            "tagline": "A short catchy tagline",
            "introduction": "A meaningful introduction about this moon sign",
            "sections": [
                {{
                    "heading": "Core Personality",
                    "content": "Description of core personality traits...",
                    "subsections": [
                        {{
                            "title": "Strengths",
                            "points": ["Point 1", "Point 2"]
                        }},
                        {{
                            "title": "Challenges",
                            "points": ["Point 1", "Point 2"]
                        }}
                    ]
                }},
                {{
                    "heading": "Emotional Landscape",
                    "content": "How this person handles emotions...",
                    "subsections": [
                        {{
                            "title": "Inner World",
                            "points": ["Point 1", "Point 2"]
                        }}
                    ]
                }}
            ]
        }}
        """
        model = genai.GenerativeModel("gemini-1.5-flash") # Use 1.5 flash for speed/cost
        res = model.generate_content(prompt)
        ai_data = parse_ai_json(res.text, fallback={})
        
        if not ai_data:
            raise ValueError("Failed to generate AI content for moon sign")
            
        return ai_data
        
    except Exception as e:
        logger.error(f"Moon sign calculation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/dasha")
def calculate_dasha(req: DashaRequest):
    """
    Calculate Vimshottari Dasha periods.
    """
    try:
        # 1. Calc math
        result = calculate_birth_chart(
            dob=req.date_of_birth,
            time=req.time_of_birth,
            lat=req.latitude,
            lon=req.longitude,
            tz=req.timezone
        )
        
        dasha_timeline = result.get('dasha_timeline', [])
        
        if not dasha_timeline:
            raise ValueError("Failed to calculate dasha timeline")
            
        # Find current dasha
        from datetime import datetime
        now = datetime.now().strftime("%Y-%m-%d")
        current = None
        upcoming = []
        
        for p in dasha_timeline:
            if p['start'] <= now <= p['end']:
                current = p
            elif p['start'] > now:
                upcoming.append(p)
                
        # Format response for frontend
        response = {
            "current_dasha": {
                "planet": current['planet'] if current else dasha_timeline[0]['planet'],
                "start_date": current['start'] if current else dasha_timeline[0]['start'],
                "end_date": current['end'] if current else dasha_timeline[0]['end'],
                "description": f"You are currently in {current['planet'] if current else dasha_timeline[0]['planet']} Mahadasha."
            },
            "dasha_periods": [
                {
                    "planet": p['planet'],
                    "start_date": p['start'],
                    "end_date": p['end'],
                    "years": (datetime.strptime(p['end'], "%Y-%m-%d") - datetime.strptime(p['start'], "%Y-%m-%d")).days // 365,
                    "description": f"{p['planet']} Mahadasha period."
                } for p in dasha_timeline
            ]
        }
        
        return response
        
    except Exception as e:
        logger.error(f"Dasha calculation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

