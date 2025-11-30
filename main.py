import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# --- IMPORT ROUTERS ---
from routers.horoscope import router as horoscope_router
from routers.panchang import router as panchang_router
from routers.chat import router as chat_router
from routers.user import router as user_router
from routers.match import router as match_router   
from routers.report import router as report_router 
from routers.voice import router as voice_router
from routers.seo import router as seo_router # <--- NEW IMPORT
from routers.resolutions import router as res_router # <--- NEW
from routers.names import router as names_router # <--- NEW
from routers.calculator import router as calculator_router # <--- NEW

# --- LOGGING SETUP ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- APP INITIALIZATION ---
app = FastAPI()

# --- CORS MIDDLEWARE ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- INCLUDE ROUTERS (After 'app' is defined) ---

# 1. Horoscope & Cron
app.include_router(horoscope_router, prefix="", tags=["Horoscope"])

# 2. Panchang
app.include_router(panchang_router, prefix="/api/panchang", tags=["Panchang"])

# 3. Chat
app.include_router(chat_router, prefix="/api/chat", tags=["AI Chat"])

# 4. User Data
app.include_router(user_router, prefix="/api/user", tags=["User Data"])

# 5. Matchmaking
app.include_router(match_router, prefix="/api/match", tags=["Kundli Matching"])

# 6. PDF Reports
app.include_router(report_router, prefix="/api/report", tags=["PDF Reports"])

# 7. Voice Chat
app.include_router(voice_router, prefix="/api/voice", tags=["Voice Chat"])

# 8. SEO Automation
app.include_router(seo_router, prefix="/api/seo", tags=["SEO Automation"])

# 9.Resolutions
app.include_router(res_router, prefix="/api/resolutions", tags=["2026 Goals"])

app.include_router(names_router, prefix="/api/names", tags=["Baby Names"])

# 10. Calculators
app.include_router(calculator_router, prefix="/api/birth-chart", tags=["Calculators"])

@app.get("/")
def health():
    return {
        "status": "Alive", 
        "service": "Astrology App Backend",
        "modules": ["Horoscope", "Panchang", "Chat", "User", "Match", "Report", "Voice", "SEO"]
    }