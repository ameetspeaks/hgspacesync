import os
import json
import math
from datetime import datetime, timezone, timedelta
from skyfield.api import load, Topos, wgs84
from skyfield.almanac import find_discrete, sunrise_sunset
from skyfield.framelib import ecliptic_frame

# --- 1. SETUP ENGINE (SKYFIELD) ---
TS = load.timescale()
PLANETS = load('de421.bsp')
EARTH = PLANETS['earth']

# --- 2. CONSTANTS ---
NAKSHATRAS = ["Ashwini", "Bharani", "Krittika", "Rohini", "Mrigashirsha", "Ardra", "Punarvasu", "Pushya", "Ashlesha", "Magha", "Purva Phalguni", "Uttara Phalguni", "Hasta", "Chitra", "Swati", "Vishakha", "Anuradha", "Jyeshtha", "Mula", "Purva Ashadha", "Uttara Ashadha", "Shravana", "Dhanishta", "Shatabhisha", "Purva Bhadrapada", "Uttara Bhadrapada", "Revati"]
TITHIS = ["Pratipada", "Dwitiya", "Tritiya", "Chaturthi", "Panchami", "Shashthi", "Saptami", "Ashtami", "Navami", "Dashami", "Ekadashi", "Dwadashi", "Trayodashi", "Chaturdashi", "Purnima", "Pratipada", "Dwitiya", "Tritiya", "Chaturthi", "Panchami", "Shashthi", "Saptami", "Ashtami", "Navami", "Dashami", "Ekadashi", "Dwadashi", "Trayodashi", "Chaturdashi", "Amavasya"]
YOGAS = ["Vishkumbha", "Priti", "Ayushman", "Saubhagya", "Sobhana", "Atiganda", "Sukarma", "Dhriti", "Shula", "Ganda", "Vriddhi", "Dhruva", "Vyaghata", "Harshana", "Vajra", "Siddhi", "Vyatipata", "Variyan", "Parigha", "Shiva", "Siddha", "Sadhya", "Shubha", "Shukla", "Brahma", "Indra", "Vaidhriti"]
SIGNS = ["Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo", "Libra", "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces"]

# Dasha System Data
DASHA_LORDS = ["Ketu", "Venus", "Sun", "Moon", "Mars", "Rahu", "Jupiter", "Saturn", "Mercury"]
DASHA_YEARS = [7, 20, 6, 10, 7, 18, 16, 19, 17] # Total 120 years

# Naming Syllables
NAKSHATRA_SYLLABLES = [
    ["Chu", "Che", "Cho", "La"], ["Li", "Lu", "Le", "Lo"], ["A", "I", "U", "E"], ["O", "Va", "Vi", "Vu"],
    ["Ve", "Vo", "Ka", "Ki"], ["Ku", "Gha", "Ng", "Chha"], ["Ke", "Ko", "Ha", "Hi"], ["Hu", "He", "Ho", "Da"],
    ["Di", "Du", "De", "Do"], ["Ma", "Mi", "Mu", "Me"], ["Mo", "Ta", "Ti", "Tu"], ["Te", "To", "Pa", "Pi"],
    ["Pu", "Sha", "Na", "Tha"], ["Pe", "Po", "Ra", "Ri"], ["Ru", "Re", "Ro", "Ta"], ["Ti", "Tu", "Te", "To"],
    ["Na", "Ni", "Nu", "Ne"], ["No", "Ya", "Yi", "Yu"], ["Ye", "Yo", "Ba", "Bi"], ["Bu", "Dha", "Bha", "Dha"],
    ["Bhe", "Bho", "Ja", "Ji"], ["Ju", "Je", "Jo", "Gha"], ["Ga", "Gi", "Gu", "Ge"], ["Go", "Sa", "Si", "Su"],
    ["Se", "So", "Da", "Di"], ["Du", "Th", "Jha", "Da"], ["De", "Do", "Cha", "Chi"]
]

# Matching Data
NAK_GANA = [0, 1, 2, 1, 0, 1, 0, 0, 2, 2, 1, 1, 0, 2, 0, 2, 0, 2, 2, 1, 1, 0, 2, 2, 1, 1, 0]
GANA_POINTS = [[6, 6, 1], [6, 6, 0], [1, 0, 6]]
GANA_GROUP = {"Deva": [0, 4, 6, 7, 12, 14, 16, 21, 26], "Manushya": [1, 3, 5, 10, 11, 19, 20, 24, 25], "Rakshasa": [2, 8, 9, 13, 15, 17, 18, 22, 23]}
NAK_NADI = [0, 1, 2, 2, 1, 0, 1, 1, 2, 2, 1, 0, 0, 1, 2, 2, 1, 0, 0, 1, 2, 0, 1, 0, 0, 1, 2]
NAK_YONI = [0, 1, 2, 3, 3, 5, 6, 7, 6, 9, 9, 11, 12, 13, 12, 13, 8, 8, 5, 10, 10, 10, 4, 0, 4, 11, 1]
YONI_MATRIX = [[4, 2, 2, 3, 2, 2, 2, 1, 0, 1, 2, 3, 2, 1], [2, 4, 3, 3, 2, 2, 2, 2, 1, 2, 2, 3, 2, 0], [2, 3, 4, 2, 1, 2, 1, 3, 3, 1, 2, 0, 3, 1], [3, 3, 2, 4, 2, 1, 1, 1, 1, 2, 2, 2, 0, 2], [2, 2, 1, 2, 4, 2, 1, 2, 2, 1, 0, 2, 2, 1], [2, 2, 2, 1, 2, 4, 2, 1, 2, 2, 1, 2, 2, 1], [2, 2, 1, 1, 1, 2, 4, 2, 1, 0, 3, 2, 2, 2], [1, 2, 3, 1, 2, 1, 2, 4, 3, 2, 2, 2, 1, 0], [0, 1, 3, 1, 2, 2, 1, 3, 4, 2, 2, 2, 1, 2], [1, 2, 1, 2, 1, 2, 0, 2, 2, 4, 2, 2, 2, 2], [2, 2, 2, 2, 0, 1, 3, 2, 2, 2, 4, 2, 2, 1], [3, 3, 0, 2, 2, 2, 2, 2, 2, 2, 2, 4, 3, 0], [2, 2, 3, 0, 2, 2, 2, 1, 1, 2, 2, 3, 4, 3], [1, 0, 1, 2, 1, 1, 2, 0, 2, 2, 1, 0, 3, 4]]
SIGN_LORDS = [0, 1, 2, 3, 4, 2, 1, 0, 5, 6, 6, 5]
MAITRI_TABLE = [[5, 3, 0.5, 5, 5, 5, 0.5], [3, 5, 5, 0.5, 0.5, 4, 5], [0.5, 5, 5, 1, 4, 3, 5], [5, 0.5, 1, 5, 5, 4, 1], [5, 0.5, 4, 5, 5, 5, 0.5], [5, 4, 3, 4, 5, 5, 3], [0.5, 5, 5, 1, 0.5, 3, 5]]

# --- 3. MATH HELPERS ---

def setup_ephemeris(): pass

def get_ayanamsa(date_obj):
    if date_obj.tzinfo is None: date_obj = date_obj.replace(tzinfo=timezone.utc)
    days = (date_obj - datetime(2000, 1, 1, tzinfo=timezone.utc)).days
    return 23.85 + ((days / 365.25) * 0.0139)

def normalize_degree(deg): return deg % 360

def decimal_to_time(decimal_hour):
    if decimal_hour is None: return "00:00"
    try:
        hours = int(decimal_hour)
        minutes = int((decimal_hour - hours) * 60)
        return f"{hours:02d}:{minutes:02d}"
    except: return "00:00"

def get_planet_long(planet_name, t):
    target_map = {"Sun": "sun", "Moon": "moon", "Mercury": "mercury", "Venus": "venus", "Mars": "mars", "Jupiter": "jupiter barycenter", "Saturn": "saturn barycenter"}
    if planet_name in ["Rahu", "Ketu"]: return 0.0
    target = PLANETS[target_map.get(planet_name, "sun")]
    astrometric = EARTH.at(t).observe(target)
    lat, lon, distance = astrometric.frame_latlon(ecliptic_frame)
    return normalize_degree(lon.degrees - get_ayanamsa(t.utc_datetime()))

# --- DIVISIONAL CHARTS ---
def get_d9_sign(planet_lon):
    d1 = int(planet_lon / 30)
    deg = planet_lon % 30
    nav = int(deg / (30/9))
    elem = (d1 + 1) % 4
    start = 0 if elem == 1 else 9 if elem == 2 else 6 if elem == 3 else 3
    return (start + nav) % 12

def get_d16_sign(planet_lon):
    """Calculates Shodashamsha (D16) Sign."""
    d1_sign = int(planet_lon / 30)
    deg_in_sign = planet_lon % 30
    part = int(deg_in_sign / (30/16)) # 16 parts
    
    # Moveable(1,4,7,10)->Aries, Fixed(2,5,8,11)->Leo, Dual(3,6,9,12)->Sag
    nature = (d1_sign + 1) % 3
    if nature == 1: start = 0 
    elif nature == 2: start = 4 
    else: start = 8 
    
    return (start + part) % 12

# --- DASHA SYSTEM ---
def calculate_vimshottari(moon_lon, birth_date):
    """Calculates Vimshottari Dasha timeline."""
    nak_len = 360/27
    nak_idx = int(moon_lon / nak_len)
    deg_passed = moon_lon % nak_len
    
    # Find Lord (Ashwini=Ketu=0)
    lord_idx = nak_idx % 9
    
    balance_deg = nak_len - deg_passed
    balance_fraction = balance_deg / nak_len
    balance_years = DASHA_YEARS[lord_idx] * balance_fraction
    
    timeline = []
    current_date = birth_date
    
    # 1. Balance Dasha
    end_date = current_date + timedelta(days=balance_years*365.25)
    timeline.append({"planet": DASHA_LORDS[lord_idx], "start": current_date.strftime("%Y-%m-%d"), "end": end_date.strftime("%Y-%m-%d")})
    current_date = end_date
    
    # 2. Subsequent Dashas
    for i in range(1, 9):
        next_idx = (lord_idx + i) % 9
        years = DASHA_YEARS[next_idx]
        end_date = current_date + timedelta(days=years*365.25)
        timeline.append({"planet": DASHA_LORDS[next_idx], "start": current_date.strftime("%Y-%m-%d"), "end": end_date.strftime("%Y-%m-%d")})
        current_date = end_date
        
    return timeline

def get_rahu_kaal(weekday, sunrise_h, sunset_h):
    duration = sunset_h - sunrise_h
    segment = duration / 8.0
    segments = {"Sunday": 8, "Monday": 2, "Tuesday": 7, "Wednesday": 5, "Thursday": 6, "Friday": 4, "Saturday": 3}
    seg_idx = segments.get(weekday, 8)
    start = sunrise_h + ((seg_idx - 1) * segment)
    return {"display": f"{decimal_to_time(start)} - {decimal_to_time(start + segment)}", "start": start, "end": start + segment}

# --- CONTENT LIBRARIES ---
def get_planet_habits_library():
    return {
        "Sun": ["Wake up 6 AM", "Surya Namaskar", "Drink Copper Water"],
        "Moon": ["Drink 3L Water", "Gratitude Journaling", "Call Mother"],
        "Mars": ["20 Pushups", "High-Intensity Cardio", "Finish a hard task"],
        "Mercury": ["Read 5 Pages", "Write 100 words", "Learn 1 new concept"],
        "Jupiter": ["Read Spiritual Text", "Donate $1", "Teach someone"],
        "Venus": ["Skin care routine", "Create Art", "Clean Room"],
        "Saturn": ["Deep Work (1 Hr)", "Declutter Desk", "Walk barefoot"],
        "Rahu": ["Digital Detox (1 Hr)", "Brainstorm Ideas", "Try something new"],
        "Ketu": ["Silence (15 mins)", "Yoga", "Delete old files"]
    }

def get_remedy_library():
    return {
        "Sun": {"Modern": ["Sunlight"], "Traditional": ["Arghya"]},
        "Moon": {"Modern": ["Hydrate"], "Traditional": ["Respect Mother"]},
        "Mars": {"Modern": ["Exercise"], "Traditional": ["Hanuman Chalisa"]},
        "Mercury": {"Modern": ["Read"], "Traditional": ["Feed Cows"]},
        "Jupiter": {"Modern": ["Teach"], "Traditional": ["Visit Temple"]},
        "Venus": {"Modern": ["Grooming"], "Traditional": ["Donate Curd"]},
        "Saturn": {"Modern": ["Clean"], "Traditional": ["Feed Dogs"]},
        "Rahu": {"Modern": ["Declutter"], "Traditional": ["Feed Birds"]},
        "Ketu": {"Modern": ["Silence"], "Traditional": ["Feed Dogs"]}
    }

def get_daily_mantra(weekday, tithi_name):
    if "Ekadashi" in tithi_name: return {"text": "Om Namo Bhagavate Vasudevaya", "deity": "Vishnu"}
    if "Chaturthi" in tithi_name: return {"text": "Om Gam Ganapataye Namaha", "deity": "Ganesha"}
    mantras = {"Monday": "Om Namah Shivaya", "Tuesday": "Om Hanumate Namaha", "Wednesday": "Om Budhaya Namaha", "Thursday": "Om Brihaspataye Namaha", "Friday": "Om Shukraya Namaha", "Saturday": "Om Shanicharuya Namaha", "Sunday": "Om Suryaya Namaha"}
    return {"text": mantras.get(weekday, "Om Shanti"), "deity": "Peace"}

def get_donation_item(weekday):
    items = {"Monday": "Rice", "Tuesday": "Lentils", "Wednesday": "Green Gram", "Thursday": "Chana Dal", "Friday": "Curd", "Saturday": "Oil", "Sunday": "Wheat"}
    return items.get(weekday, "Food")

def get_rashi_plant(sign_name):
    return "Tulsi"

def get_special_event(tithi_name, weekday):
    event = {"is_special": False, "name": f"{weekday} Vibes", "description": "Daily Routine", "ritual": "Sadhana"}
    if "Ekadashi" in tithi_name: event.update({"is_special": True, "name": "Ekadashi", "description": "Fasting Day", "ritual": "Avoid grains"})
    elif "Amavasya" in tithi_name: event.update({"is_special": True, "name": "Amavasya", "description": "Ancestor Day", "ritual": "Donate food"})
    return event

# --- MAIN FUNCTIONS ---
def get_detailed_panchang(date_obj, lat=28.61, lon=77.20, tz=5.5):
    if date_obj.tzinfo is None: date_obj = date_obj.replace(tzinfo=timezone.utc)
    t = TS.from_datetime(date_obj)
    sun_deg = get_planet_long("Sun", t)
    moon_deg = get_planet_long("Moon", t)
    tithi_idx = int(normalize_degree(moon_deg - sun_deg) / 12)
    nak_idx = int(moon_deg / (360.0/27.0))
    yoga_idx = int(normalize_degree(moon_deg + sun_deg) / (360.0/27.0))
    
    location = wgs84.latlon(lat, lon)
    t0, t1 = t, TS.from_datetime(date_obj + timedelta(days=1))
    times, events = find_discrete(t0, t1, sunrise_sunset(PLANETS, location))
    sunrise_h, sunset_h = 6.0, 18.0
    for ti, event in zip(times, events):
        dt = ti.astimezone(timezone(timedelta(hours=tz)))
        h = dt.hour + (dt.minute/60.0)
        if event == 1: sunrise_h = h
        else: sunset_h = h
    
    weekday = date_obj.strftime("%A")
    rahu = get_rahu_kaal(weekday, sunrise_h, sunset_h)
    midday = sunrise_h + ((sunset_h - sunrise_h) / 2)

    return {
        "date": date_obj.date().isoformat(),
        "location": {"lat": lat, "lon": lon, "name": "User Loc"},
        "meta": {"weekday": weekday, "tithi": TITHIS[tithi_idx], "nakshatra": NAKSHATRAS[nak_idx], "yoga": YOGAS[yoga_idx], "paksha": "Shukla" if tithi_idx < 15 else "Krishna"},
        "timing": {
            "sunrise": decimal_to_time(sunrise_h), "sunset": decimal_to_time(sunset_h),
            "rahu_kaal": rahu["display"], "rahu_start": rahu["start"], "rahu_end": rahu["end"],
            "abhijit_start": decimal_to_time(midday - 0.4), "abhijit_end": decimal_to_time(midday + 0.4)
        },
        "full_data": {"sun_lon": sun_deg, "moon_lon": moon_deg}
    }

def get_daily_transits():
    t = TS.now()
    transits = []
    for p in ["Sun", "Moon", "Jupiter", "Saturn"]:
        deg = get_planet_long(p, t)
        transits.append(f"{p} in {SIGNS[int(deg/30)]}")
    return ", ".join(transits)

def get_personalized_forecast_data(dob, time, lat, lon, tz):
    try:
        y,m,d = map(int, dob.split('-'))
        h,mn = map(int, time.split(':'))
        birth_dt = datetime(y, m, d, h, mn, tzinfo=timezone(timedelta(hours=float(tz))))
        t_birth, t_now = TS.from_datetime(birth_dt), TS.now()
        aspects = []
        for p in ["Sun", "Moon", "Mars", "Jupiter", "Saturn"]:
            n = get_planet_long(p, t_birth)
            t = get_planet_long(p, t_now)
            diff = abs(t - n)
            if diff > 180: diff = 360 - diff
            if abs(diff - 90) < 4: aspects.append({"transit_planet": p, "aspect": "Square", "description": f"{p} Square {p}"})
            if abs(diff - 120) < 4: aspects.append({"transit_planet": p, "aspect": "Trine", "description": f"{p} Trine {p}"})
        return aspects
    except: return []

def calculate_birth_chart(dob, time, lat, lon, tz):
    try:
        # Validate inputs
        if not dob or not time:
            raise ValueError("Date of birth and time are required")
        
        # Parse date
        try:
            y, m, d = map(int, dob.split('-'))
            if not (1900 <= y <= 2100) or not (1 <= m <= 12) or not (1 <= d <= 31):
                raise ValueError("Invalid date range")
        except (ValueError, AttributeError) as e:
            raise ValueError(f"Invalid date format. Use YYYY-MM-DD: {e}")
        
        # Parse time
        try:
            time_parts = time.split(':')
            h = int(time_parts[0])
            mn = int(time_parts[1]) if len(time_parts) > 1 else 0
            if not (0 <= h <= 23) or not (0 <= mn <= 59):
                raise ValueError("Invalid time range")
        except (ValueError, AttributeError, IndexError) as e:
            raise ValueError(f"Invalid time format. Use HH:MM: {e}")
        
        # Validate coordinates
        try:
            lat_f = float(lat)
            lon_f = float(lon)
            tz_f = float(tz)
            if not (-90 <= lat_f <= 90) or not (-180 <= lon_f <= 180):
                raise ValueError("Invalid coordinates")
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid location data: {e}")
        
        # Create birth datetime
        birth_dt = datetime(y, m, d, h, mn, tzinfo=timezone(timedelta(hours=tz_f)))
        t = TS.from_datetime(birth_dt)
        
        chart = []
        sun_deg = get_planet_long("Sun", t)
        asc_sign_id = int(sun_deg / 30) # Approx Ascendant
        
        for p in ["Sun", "Moon", "Mars", "Mercury", "Jupiter", "Venus", "Saturn"]:
            try:
                deg = get_planet_long(p, t)
                sign_id = int(deg / 30)
                if sign_id < 0 or sign_id >= len(SIGNS):
                    sign_id = 0  # Fallback to Aries
                
                house = (sign_id - asc_sign_id) + 1
                if house <= 0: house += 12
                if house > 12: house -= 12
                
                # Calculate Divisionals
                d9 = get_d9_sign(deg)
                d16 = get_d16_sign(deg)
                
                chart.append({
                    "planet": p, 
                    "sign": SIGNS[sign_id], 
                    "d9_sign": SIGNS[d9] if 0 <= d9 < len(SIGNS) else SIGNS[0],
                    "d16_sign": SIGNS[d16] if 0 <= d16 < len(SIGNS) else SIGNS[0],
                    "house": house, 
                    "degree": round(deg % 30, 2), 
                    "abs_degree": deg
                })
            except Exception as planet_error:
                logger.warning(f"Error calculating {p}: {planet_error}")
                continue
        
        if not chart:
            raise ValueError("Failed to calculate any planetary positions")
        
        # Vimshottari Dasha
        moon_obj = next((c for c in chart if c['planet']=='Moon'), None)
        dasha = []
        if moon_obj:
            try:
                dasha = calculate_vimshottari(moon_obj['abs_degree'], birth_dt)
            except Exception as dasha_error:
                logger.warning(f"Dasha calculation failed: {dasha_error}")

        moon_sign = next((c['sign'] for c in chart if c['planet']=='Moon'), SIGNS[0])
        ascendant = SIGNS[asc_sign_id] if 0 <= asc_sign_id < len(SIGNS) else SIGNS[0]

        return {
            "ai_summary": ". ".join([f"{c['planet']} in {c['sign']} ({c['house']}H)" for c in chart]),
            "raw_json": chart,
            "dasha_timeline": dasha,
            "key_points": {
                "ascendant": ascendant,
                "moon_sign": moon_sign
            }
        }
    except ValueError as ve:
        # Re-raise ValueError with clear message
        raise ve
    except Exception as e:
        logger.error(f"Chart calculation error: {e}", exc_info=True)
        raise ValueError(f"Failed to calculate birth chart: {str(e)}")

def get_naming_details(dob, time, lat, lon, tz):
    try:
        y, m, d = map(int, dob.split('-'))
        h, mn = map(int, time.split(':'))
        birth_dt = datetime(y, m, d, h, mn, tzinfo=timezone(timedelta(hours=float(tz))))
        t = TS.from_datetime(birth_dt)
        moon_deg = get_planet_long("Moon", t)
        nak_idx = int(moon_deg / (360.0/27.0))
        deg_in_nak = moon_deg % (360.0/27.0)
        pada = int(deg_in_nak / ((360.0/27.0)/4)) + 1
        if pada > 4: pada = 4
        return {
            "nakshatra": NAKSHATRAS[nak_idx], "pada": pada,
            "rashi": SIGNS[int(moon_deg / 30)],
            "primary_sound": NAKSHATRA_SYLLABLES[nak_idx][pada-1],
            "all_sounds": NAKSHATRA_SYLLABLES[nak_idx]
        }
    except Exception as e: return {"error": str(e)}

def get_nakshatra_details(moon_deg):
    nak_idx = int(moon_deg / (360/27))
    return {"index": nak_idx, "name": NAKSHATRAS[nak_idx], "sign_index": int(moon_deg/30)}

def calculate_kootas(boy_deg, girl_deg):
    b = get_nakshatra_details(boy_deg)
    g = get_nakshatra_details(girl_deg)
    scores = {}
    scores['varna'] = {"score": 1, "total": 1, "name": "Varna"}
    v_score = 2 if (b['sign_index'] % 4) == (g['sign_index'] % 4) else 1
    scores['vashya'] = {"score": v_score, "total": 2, "name": "Vashya"}
    t_score = 3 if ((b['index'] - g['index']) % 9) in [0,2,4,6,8] else 1.5
    scores['tara'] = {"score": t_score, "total": 3, "name": "Tara"}
    b_yoni, g_yoni = NAK_YONI[b['index']], NAK_YONI[g['index']]
    scores['yoni'] = {"score": YONI_MATRIX[b_yoni][g_yoni], "total": 4, "name": "Yoni"}
    b_lord, g_lord = SIGN_LORDS[b['sign_index']], SIGN_LORDS[g['sign_index']]
    scores['maitri'] = {"score": MAITRI_TABLE[b_lord][g_lord], "total": 5, "name": "Maitri"}
    scores['gana'] = {"score": GANA_POINTS[NAK_GANA[b['index']]][NAK_GANA[g['index']]], "total": 6, "name": "Gana"}
    dist = (g['sign_index'] - b['sign_index']) % 12 + 1
    b_score = 0 if dist in [2, 5, 6, 9, 12] else 7
    scores['bhakoot'] = {"score": b_score, "total": 7, "name": "Bhakoot"}
    n_score = 0 if (b['index'] % 3) == (g['index'] % 3) else 8
    scores['nadi'] = {"score": n_score, "total": 8, "name": "Nadi"}
    total = sum(s['score'] for s in scores.values())
    return {"scores": scores, "total": total, "boy_nak": b, "girl_nak": g}

def get_synastry_data(p1, p2):
    return f"Person A: {p1['ai_summary']} \nPerson B: {p2['ai_summary']}"