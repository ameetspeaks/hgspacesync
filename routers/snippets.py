import os
import logging
import json
import time
from fastapi import APIRouter, BackgroundTasks, HTTPException, Header
from pydantic import BaseModel

class BatchStatusRequest(BaseModel):
    ids: list[int]
from supabase import create_client, Client
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted
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
@retry(
    retry=retry_if_exception_type(ResourceExhausted),
    wait=wait_exponential(multiplier=2, min=20, max=60),
    stop=stop_after_attempt(4)
)
def extract_snippets_with_ai(content, keywords, title):
    """Use Gemini to extract snippets from blog content"""
    prompt = f"""Extract 3-5 short, viral-worthy Q&A snippets from this astrology blog post.

Format: Q&A style (Hinglish/English mix)
Example: Q: "Ekadashi ko chawal kyun nahi?" A: "Scientific reason hai water retention..."

Each snippet should be:
- Max 40 words for answer
- Catchy, shareable
- Knowledgeable and accurate

Output JSON:
{{
  "snippets": [
    {{
      "title": "Catchy Header",
      "question": "Direct Question",
      "answer": "Short Answer (Max 40 words)",
      "theme": "purple/gold/red/blue/green",
      "icon": "✨"
    }}
  ]
}}

Keywords: {keywords}
Title: {title}
Content: {content[:3000]}
"""
    
    try:
        model = get_working_model()
        response = model.generate_content(prompt)
        
        # Parse JSON response
        json_text = response.text.replace('```json', '').replace('```', '').strip()
        data = json.loads(json_text)
        
        return data.get('snippets', [])
    except json.JSONDecodeError as e:
        logger.error(f"JSON parse error: {e}")
        logger.error(f"Response text: {response.text[:500]}")
        return []
    except Exception as e:
        logger.error(f"AI extraction failed: {e}")
        return []

def process_blog_snippets(blog_data):
    """Process a single blog to extract and save snippets"""
    log_id = blog_data.get('log_id')
    blog_id = blog_data.get('blog_id')
    content = blog_data.get('content', '')
    title = blog_data.get('title', '')
    primary_keyword = blog_data.get('primary_keyword', '')
    
    if not log_id or not blog_id:
        logger.error(f"Missing log_id or blog_id: {blog_data}")
        return
    
    start_time = time.time()
    
    try:
        # Update status to processing
        supabase.table('snippet_generation_logs').update({
            'status': 'processing',
            'updated_at': 'now()'
        }).eq('id', log_id).execute()
        
        # Extract snippets using AI
        keywords = primary_keyword or title
        snippets = extract_snippets_with_ai(content, keywords, title)
        
        # Insert snippets
        if snippets:
            snippet_payload = []
            for s in snippets:
                snippet_payload.append({
                    'blog_id': int(blog_id),  # Ensure it's an integer
                    'title': str(s.get('title', 'Astrology Tip')).strip(),
                    'question': str(s.get('question', '')).strip(),
                    'answer': str(s.get('answer', '')).strip(),
                    'theme_color': str(s.get('theme', 'purple')).strip(),
                    'icon': str(s.get('icon', '✨')).strip()
                })
            
            if snippet_payload:
                result = supabase.table('content_snippets').insert(snippet_payload).execute()
                logger.info(f"✅ Created {len(snippet_payload)} snippets for blog {blog_id}")
                logger.debug(f"Snippet IDs created: {[r.get('id') for r in (result.data if hasattr(result, 'data') else [])]}")
            else:
                logger.warning(f"⚠️ No valid snippets to insert for blog {blog_id}")
        else:
            logger.warning(f"⚠️ No snippets extracted for blog {blog_id}")
        
        # Calculate processing time
        processing_time_ms = int((time.time() - start_time) * 1000)
        
        # Update log as completed
        supabase.table('snippet_generation_logs').update({
            'status': 'completed',
            'snippets_generated': len(snippets),
            'processing_time_ms': processing_time_ms,
            'metadata': {'snippets': snippets, 'processing_time_ms': processing_time_ms},
            'completed_at': 'now()'
        }).eq('id', log_id).execute()
        
        # Update blog status
        supabase.table('blog_posts').update({
            'snippet_generation_status': 'completed'
        }).eq('id', blog_id).execute()
        
        logger.info(f"✅ Completed blog {blog_id} in {processing_time_ms}ms")
        
    except Exception as e:
        logger.error(f"❌ Failed blog {blog_id}: {e}")
        # Mark as failed
        try:
            supabase.rpc('update_snippet_log_status', {
                'p_log_id': log_id,
                'p_status': 'failed',
                'p_error': str(e)
            }).execute()
            supabase.table('blog_posts').update({
                'snippet_generation_status': 'failed'
            }).eq('id', blog_id).execute()
        except Exception as update_error:
            logger.error(f"Failed to update error status: {update_error}")

def run_snippet_batch_worker(limit=10):
    """Worker function to process pending snippet generation jobs"""
    logger.info(f"🚀 Snippet Worker processing up to {limit} blogs")
    
    try:
        # Get pending blogs
        response = supabase.rpc('get_blogs_pending_snippets', {'p_limit': limit}).execute()
        blogs = response.data if hasattr(response, 'data') else []
        
        if not blogs:
            logger.info("No blogs pending snippet generation")
            return {"status": "no_pending", "processed": 0}
        
        logger.info(f"Found {len(blogs)} pending blogs")
        
        # Process each blog
        for blog in blogs:
            process_blog_snippets(blog)
            time.sleep(2)  # Rate limiting between blogs
        
        return {"status": "completed", "processed": len(blogs)}
        
    except Exception as e:
        logger.error(f"❌ Batch worker failed: {e}")
        return {"status": "error", "error": str(e)}

# --- ENDPOINTS ---

@router.post("/process-pending")
def process_pending_snippets(
    limit: int = 10,
    x_admin_key: str = Header(None)
):
    """Process pending snippet generation jobs"""
    if x_admin_key != ADMIN_SECRET:
        raise HTTPException(status_code=403, detail="Unauthorized")
    
    result = run_snippet_batch_worker(limit)
    return result

@router.post("/process-pending-background")
def process_pending_snippets_background(
    limit: int = 10,
    background_tasks: BackgroundTasks = None,
    x_admin_key: str = Header(None)
):
    """Process pending snippet generation jobs in background"""
    if x_admin_key != ADMIN_SECRET:
        raise HTTPException(status_code=403, detail="Unauthorized")
    
    background_tasks.add_task(run_snippet_batch_worker, limit)
    return {"status": "started", "message": f"Processing up to {limit} blogs in background"}

@router.post("/process-batch")
def process_snippet_batch(
    batch_size: int = 10,
    background_tasks: BackgroundTasks = None,
    x_admin_key: str = Header(None)
):
    """Process a batch of pending blogs (similar to SEO rewrite-batch)"""
    if x_admin_key != ADMIN_SECRET:
        raise HTTPException(status_code=403, detail="Unauthorized")
    
    # Get pending blogs
    try:
        response = supabase.rpc('get_blogs_pending_snippets', {'p_limit': batch_size}).execute()
        blogs = response.data if hasattr(response, 'data') else []
        
        if not blogs:
            return {"status": "Done", "message": "No pending blogs.", "ids": []}
        
        blog_ids = [b.get('blog_id') or b.get('id') for b in blogs if b.get('blog_id') or b.get('id')]
        
        # Lock them as processing
        supabase.table('blog_posts').update({
            'snippet_generation_status': 'processing'
        }).in_('id', blog_ids).execute()
        
        # Start background processing
        background_tasks.add_task(run_snippet_batch_worker, batch_size)
        
        return {
            "status": "Job Started",
            "ids": blog_ids,
            "message": f"Processing {len(blog_ids)} blog(s)",
            "count": len(blog_ids)
        }
    except Exception as e:
        logger.error(f"Error starting batch: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/check-batch")
def check_batch_status(
    req: BatchStatusRequest,
    x_admin_key: str = Header(None)
):
    """Check status of a batch of blogs (similar to SEO check-batch)"""
    if x_admin_key != ADMIN_SECRET:
        raise HTTPException(status_code=403, detail="Unauthorized")
    
    # Validate and log the request
    if not req.ids:
        return {"total": 0, "processing": 0, "completed": 0, "failed": 0, "pending": 0, "is_done": True}
    
    # Ensure all IDs are integers
    try:
        ids = [int(id) for id in req.ids]
        logger.info(f"Checking batch status for {len(ids)} blogs: {ids[:5]}...")
    except (ValueError, TypeError) as e:
        logger.error(f"Invalid IDs format: {req.ids}, error: {e}")
        raise HTTPException(status_code=422, detail=f"Invalid IDs format: {e}")
    
    try:
        # Check blog statuses
        res = supabase.table('blog_posts').select('id, snippet_generation_status').in_('id', ids).execute()
        
        processing = sum(1 for r in res.data if r.get('snippet_generation_status') == 'processing')
        completed = sum(1 for r in res.data if r.get('snippet_generation_status') == 'completed')
        failed = sum(1 for r in res.data if r.get('snippet_generation_status') == 'failed')
        pending = sum(1 for r in res.data if r.get('snippet_generation_status') == 'pending')
        
        return {
            "total": len(ids),
            "processing": processing,
            "completed": completed,
            "failed": failed,
            "pending": pending,
            "is_done": processing == 0 and pending == 0
        }
    except Exception as e:
        logger.error(f"Error checking batch status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

