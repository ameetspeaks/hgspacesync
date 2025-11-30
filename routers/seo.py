import os
import logging
import copy
import time
import yake
from fastapi import APIRouter, BackgroundTasks, HTTPException, Header
from pydantic import BaseModel
from supabase import create_client, Client
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted, NotFound
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

router = APIRouter()
logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
ADMIN_SECRET = os.getenv("ADMIN_SECRET")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
genai.configure(api_key=GEMINI_KEY)

kw_extractor = yake.KeywordExtractor(lan="en", n=2, dedupLim=0.9, top=5, features=None)

class RewriteRequest(BaseModel):
    batch_size: int = 5 

class StatusRequest(BaseModel):
    ids: list[int]

# --- MODEL HUNTER ---
def get_working_model():
    candidates = ["gemini-2.0-flash", "gemini-2.0-flash-lite", "gemini-1.5-flash", "gemini-pro"]
    for name in candidates:
        try:
            model = genai.GenerativeModel(name)
            model.generate_content("Test")
            return model
        except Exception as e:
            logger.debug(f"Model {name} failed: {e}")
            continue
    raise Exception("No working Gemini model found.")

# --- HELPER FUNCTIONS ---
def extract_keywords(title, content, existing):
    if existing and len(existing) > 3: return existing
    try:
        text_sample = f"{title} {str(content)[:1000]}"
        keywords = kw_extractor.extract_keywords(text_sample)
        return ", ".join([k[0] for k in keywords[:3]])
    except Exception as e:
        logger.warning(f"Keyword extraction failed: {e}")
        return "general"

def count_words(text):
    return len(str(text).split()) if text else 0

@retry(retry=retry_if_exception_type(ResourceExhausted), wait=wait_exponential(multiplier=2, min=20, max=60), stop=stop_after_attempt(4))
def call_ai_rewrite(text, keywords, length):
    prompt = f"Act as an Expert SEO Copywriter. Rewrite this text. Keywords: [{keywords}]. Length: {length} words. Keep formatting. Original: {text}"
    try:
        model = get_working_model()
        return model.generate_content(prompt).text.strip()
    except NotFound:
        return genai.GenerativeModel("gemini-pro").generate_content(prompt).text.strip()

def process_json(data, keywords):
    if isinstance(data, dict):
        for k, v in data.items():
            if k in ['text', 'content', 'value', 'html'] and isinstance(v, str) and len(v) > 50:
                try:
                    data[k] = call_ai_rewrite(v, keywords, count_words(v))
                    time.sleep(2)
                except Exception as e:
                    logger.warning(f"Failed to rewrite field {k}: {e}")
                    pass
            else: process_json(v, keywords)
    elif isinstance(data, list):
        for item in data: process_json(item, keywords)
    return data

# --- WORKER (Takes Specific IDs) ---
def run_seo_batch_worker(row_ids):
    logger.info(f"🚀 Background Worker processing IDs: {row_ids}")
    
    # Fetch the data for these specific IDs
    response = supabase.table("blog_posts").select("*").in_("id", row_ids).execute()
    rows = response.data

    for row in rows:
        try:
            logger.info(f"Processing ID {row['id']}...")
            keywords = extract_keywords(row['title'], row['content'], row['primary_keyword'])
            original = row['content']
            new_content = None
            
            if isinstance(original, (dict, list)):
                new_content = process_json(copy.deepcopy(original), keywords)
            elif isinstance(original, str):
                if len(original.split()) < 50:
                    supabase.table("blog_posts").update({"rewrite_status": "skipped"}).eq("id", row['id']).execute()
                    continue
                new_content = call_ai_rewrite(original, keywords, count_words(original))
            
            if new_content:
                supabase.table("blog_posts").update({
                    "content": new_content,
                    "primary_keyword": keywords,
                    "seo_keywords": keywords,
                    "rewrite_status": "completed",
                    "updated_at": "now()"
                }).eq("id", row['id']).execute()
                logger.info(f"✅ Finished ID {row['id']}")
            else:
                supabase.table("blog_posts").update({"rewrite_status": "skipped"}).eq("id", row['id']).execute()
            
            logger.info("⏳ Cooling down (4s)...")
            time.sleep(4) 

        except Exception as e:
            logger.error(f"❌ Failed ID {row['id']}: {e}")
            supabase.table("blog_posts").update({"rewrite_status": "failed"}).eq("id", row['id']).execute()

# --- ENDPOINTS ---

@router.post("/rewrite-batch")
def trigger_seo_rewrite(req: RewriteRequest, background_tasks: BackgroundTasks, x_admin_key: str = Header(None)):
    if x_admin_key != ADMIN_SECRET: raise HTTPException(status_code=403, detail="Unauthorized")
    
    # 1. Synchronous: Find Pending Rows
    response = supabase.table("blog_posts").select("id").eq("rewrite_status", "pending").order("id").limit(req.batch_size).execute()
    rows = response.data
    
    if not rows:
        return {"status": "Done", "message": "No pending articles.", "ids": []}

    row_ids = [r['id'] for r in rows]

    # 2. Synchronous: Lock them as 'processing' immediately
    supabase.table("blog_posts").update({"rewrite_status": "processing"}).in_("id", row_ids).execute()

    # 3. Asynchronous: Start work
    background_tasks.add_task(run_seo_batch_worker, row_ids)
    
    return {"status": "Job Started", "ids": row_ids, "message": f"Processing IDs: {row_ids}"}

@router.post("/check-batch")
def check_batch_status(req: StatusRequest, x_admin_key: str = Header(None)):
    if x_admin_key != ADMIN_SECRET: raise HTTPException(status_code=403, detail="Unauthorized")
    
    if not req.ids: return {"pending_count": 0, "failed_count": 0}

    # Check how many of these IDs are still 'processing'
    res = supabase.table("blog_posts").select("rewrite_status").in_("id", req.ids).execute()
    
    processing = sum(1 for r in res.data if r['rewrite_status'] == 'processing')
    failed = sum(1 for r in res.data if r['rewrite_status'] == 'failed')
    completed = sum(1 for r in res.data if r['rewrite_status'] == 'completed')
    
    return {
        "total": len(req.ids),
        "processing": processing,
        "failed": failed,
        "completed": completed,
        "is_done": processing == 0
    }