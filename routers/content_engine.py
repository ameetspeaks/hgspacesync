import os
import logging
import json
import time
import re
from datetime import datetime, timezone
from typing import Dict, Optional
from fastapi import APIRouter, BackgroundTasks, HTTPException, Header
from pydantic import BaseModel
from supabase import create_client, Client
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

router = APIRouter()
logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
ADMIN_SECRET = os.getenv("ADMIN_SECRET")

if not GEMINI_KEY:
    logger.warning("⚠️ GEMINI_API_KEY not set")
if not SUPABASE_URL or not SUPABASE_KEY:
    logger.warning("⚠️ Supabase credentials not set")

if GEMINI_KEY:
    try:
        genai.configure(api_key=GEMINI_KEY)
    except Exception as e:
        logger.error(f"Failed to configure Gemini: {e}")

supabase: Client = None
if SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        logger.info("✅ Supabase client initialized")
    except Exception as e:
        logger.error(f"Failed to initialize Supabase client: {e}")
else:
    logger.warning("⚠️ Supabase credentials not configured")

# --- MODELS ---
class BatchRequest(BaseModel):
    batch_size: int = 3  # Safe default

# --- HELPERS ---
def create_slug(title):
    """Create URL-friendly slug from title"""
    slug = title.lower().strip()
    # Remove special characters, keep only alphanumeric, spaces, and hyphens
    slug = re.sub(r'[^a-z0-9\s-]', '', slug)
    # Replace multiple spaces/hyphens with single hyphen
    slug = re.sub(r'[\s-]+', '-', slug)
    # Remove leading/trailing hyphens
    slug = slug.strip('-')
    return slug

def get_working_model():
    """Get working Gemini model with fallback options"""
    if not GEMINI_KEY:
        raise Exception("GEMINI_API_KEY environment variable is not set")
    
    models_to_try = [
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite",
        "gemini-flash-latest",
        "gemini-1.5-flash",
        "gemini-1.5-pro",
        "gemini-pro"
    ]
    
    last_error = None
    for model_name in models_to_try:
        try:
            logger.info(f"🔍 Trying model: {model_name}")
            model = genai.GenerativeModel(model_name)
            logger.info(f"✅ Model {model_name} initialized successfully")
            return model
        except Exception as e:
            last_error = str(e)
            logger.warning(f"⚠️ Model {model_name} initialization failed: {e}")
            continue
    
    raise Exception(f"No working Gemini model found. Last error: {last_error}")

# --- AI WRITER (HINGLISH SPECIALIST) ---
@retry(
    retry=retry_if_exception_type(ResourceExhausted),
    wait=wait_exponential(multiplier=2, min=30, max=90),
    stop=stop_after_attempt(5)
)
def write_hinglish_article(keyword):
    """
    Generates a structured, SEO-optimized Hinglish blog post.
    Returns dict with: title, slug_base, seo_description, content_markdown, tags, category
    """
    model = get_working_model()
    
    prompt = f"""
Act as a Senior Vedic Astrologer and Content Writer for an Indian Audience.

TOPIC: "{keyword}"
LANGUAGE: Hinglish (Natural mix of Hindi in Roman script and English).
TARGET AUDIENCE: Gen Z and Millennials interested in Astrology.

INSTRUCTIONS:
1. **Tone:** Conversational, relatable, yet authoritative. Use phrases like "Kya aap jante hain?", "Iska matlab hai...", "Dhyan rahe", "Aaiye samajhte hain".

2. **Structure:**
   - **Catchy Title:** English + Hindi mix (e.g., "Gajkesari Yog: Kya Hai Iske Fayde? Complete Guide").
   - **Introduction:** Hook the reader emotionally with a relatable question or statement.
   - **Deep Dive:** Explain the concept astrologically but simply. Use examples and analogies.
   - **Remedies/Tips:** Practical actionable advice that readers can implement.
   - **Conclusion:** Positive closing with encouragement.

3. **SEO Requirements:**
   - Use H2 (##) and H3 (###) tags for headings in markdown format.
   - Include bullet points (-) and numbered lists.
   - Keep keyword density around 1.5% (use the keyword naturally 3-4 times in 800-1200 words).
   - Write 800-1200 words of high-quality content.
   - Use internal linking suggestions in markdown format: [link text](/relevant-page)

4. **Format:** Return ONLY valid JSON. No markdown code blocks, no explanations, just pure JSON.

5. **Content Style:**
   - Mix Hindi (in Roman script) and English naturally
   - Example: "Gajkesari Yog ek bahut hi powerful yog hai jo aapki kundli mein ban sakta hai. Iska matlab hai ki..."
   - Use common Hindi phrases: "Kya hai", "Kaise banta hai", "Kya karna chahiye", "Fayde", "Nuksan"
   - Keep it engaging and easy to read

JSON OUTPUT FORMAT:
{{
    "title": "The SEO Title in Hinglish",
    "slug_base": "english-keyword-slug",
    "seo_description": "150 chars meta description in Hinglish",
    "content_markdown": "The full blog article in markdown format with proper headings (##, ###), bullet points, and paragraphs. NO HTML tags, only markdown.",
    "tags": ["tag1", "tag2", "tag3"],
    "category": "Astrology/Yoga/Remedies"
}}

IMPORTANT: 
- Return ONLY the JSON object, no markdown code fences, no explanations
- content_markdown should be pure markdown (not HTML)
- Make sure the content is well-structured with proper headings
- Include practical tips and remedies
- Keep the tone conversational and engaging
"""

    try:
        response = model.generate_content(prompt)
        response_text = response.text.strip()
        
        # Clean up response (remove markdown code blocks if present)
        clean_text = response_text.replace("```json", "").replace("```", "").strip()
        
        # Try to parse JSON
        try:
            data = json.loads(clean_text)
        except json.JSONDecodeError:
            # If JSON parsing fails, try to extract JSON from the response
            json_match = re.search(r'\{.*\}', clean_text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
            else:
                raise Exception(f"Failed to parse JSON from response: {clean_text[:200]}")
        
        # Validate required fields
        required_fields = ['title', 'slug_base', 'seo_description', 'content_markdown', 'tags', 'category']
        for field in required_fields:
            if field not in data:
                raise Exception(f"Missing required field: {field}")
        
        return data
        
    except Exception as e:
        logger.error(f"Error generating content for '{keyword}': {str(e)}")
        raise

# --- WORKER ---
def run_content_batch(batch_size):
    logger.info(f"🚀 Content Engine Started: Batch {batch_size}")
    
    if not supabase:
        logger.error("❌ Supabase client not initialized")
        return
    
    try:
        # 1. Fetch Pending Keywords
        response = supabase.table("content_queue")\
            .select("*")\
            .eq("status", "pending")\
            .limit(batch_size)\
            .execute()
            
        rows = response.data
        if not rows:
            logger.info("✅ Queue empty. No work.")
            return
        
        logger.info(f"📋 Found {len(rows)} pending keywords")
        
        # 2. Lock Rows
        ids = [r['id'] for r in rows]
        supabase.table("content_queue").update({"status": "processing"}).in_("id", ids).execute()
        logger.info(f"🔒 Locked {len(ids)} keywords for processing")
        
        # 3. Process Loop
        for idx, row in enumerate(rows, 1):
            keyword = row['keyword']
            queue_id = row['id']
            
            try:
                logger.info(f"✍️ [{idx}/{len(rows)}] Writing: {keyword}")
                
                # A. Generate Content
                data = write_hinglish_article(keyword)
                
                # B. Unique Slug Logic
                slug = create_slug(data['slug_base'])
                
                # Check if slug exists
                check = supabase.table("blog_posts").select("id").eq("slug", slug).execute()
                if check.data:
                    # Append timestamp to make it unique
                    slug = f"{slug}-{int(time.time())}"
                    logger.info(f"⚠️ Slug conflict, using: {slug}")
                
                # C. Prepare post payload
                post_payload = {
                    "title": data['title'],
                    "slug": slug,
                    "content": data['content_markdown'],  # Markdown format
                    "short_snippet": data['seo_description'],
                    "seo_title": data['title'],
                    "seo_description": data['seo_description'],
                    "seo_keywords": ", ".join(data['tags']),
                    "primary_keyword": keyword,
                    "is_published": True,  # Publish immediately
                    "published_at": datetime.now(timezone.utc).isoformat(),
                    "author_name": "Jyotish AI"
                }
                
                # D. Save to Blog Table
                insert_response = supabase.table("blog_posts").insert(post_payload).execute()
                
                if insert_response.data:
                    logger.info(f"✅ Published: {slug}")
                    
                    # E. Mark Queue Complete
                    supabase.table("content_queue").update({
                        "status": "completed",
                        "updated_at": datetime.now(timezone.utc).isoformat()
                    }).eq("id", queue_id).execute()
                else:
                    raise Exception("Failed to insert blog post")
                
                # F. THROTTLE (Crucial for Free Tier)
                # 15 RPM limit = 4s per request minimum. We use 15s to be safe and human-like.
                if idx < len(rows):  # Don't sleep after last item
                    logger.info(f"⏳ Waiting 15 seconds before next article...")
                    time.sleep(15)
                    
            except ResourceExhausted as e:
                logger.error(f"❌ Rate limit hit for '{keyword}': {e}")
                supabase.table("content_queue").update({
                    "status": "failed",
                    "updated_at": datetime.now(timezone.utc).isoformat()
                }).eq("id", queue_id).execute()
                # Wait longer on rate limit
                time.sleep(60)
                
            except Exception as e:
                logger.error(f"❌ Failed '{keyword}': {e}")
                supabase.table("content_queue").update({
                    "status": "failed",
                    "updated_at": datetime.now(timezone.utc).isoformat()
                }).eq("id", queue_id).execute()
                time.sleep(10)  # Cool down on error
        
        logger.info(f"✅ Batch processing completed")
        
    except Exception as e:
        logger.error(f"❌ Batch Critical Error: {e}")
        import traceback
        logger.error(traceback.format_exc())

# --- ENDPOINTS ---
@router.get("/health")
def health_check():
    """Health check endpoint to verify router is loaded"""
    return {
        "status": "ok",
        "router": "content_engine",
        "supabase_configured": supabase is not None,
        "gemini_configured": GEMINI_KEY is not None
    }

@router.post("/run-batch")
def trigger_content_generation(
    req: BatchRequest,
    background_tasks: BackgroundTasks,
    x_admin_key: str = Header(None)
):
    """Trigger content generation for pending keywords"""
    
    if x_admin_key != ADMIN_SECRET:
        raise HTTPException(status_code=403, detail="Unauthorized")
    
    if not supabase:
        raise HTTPException(status_code=500, detail="Supabase not configured")
    
    if not GEMINI_KEY:
        raise HTTPException(status_code=500, detail="Gemini API key not configured")
    
    # Validate batch size
    if req.batch_size < 1 or req.batch_size > 10:
        raise HTTPException(status_code=400, detail="Batch size must be between 1 and 10")
    
    # Run in background
    background_tasks.add_task(run_content_batch, req.batch_size)
    
    return {
        "status": "Queued",
        "message": f"Generating content for next {req.batch_size} keywords.",
        "batch_size": req.batch_size
    }

@router.get("/queue-status")
def get_queue_status(x_admin_key: str = Header(None)):
    """Get status of content queue"""
    
    if x_admin_key != ADMIN_SECRET:
        raise HTTPException(status_code=403, detail="Unauthorized")
    
    if not supabase:
        raise HTTPException(status_code=500, detail="Supabase not configured")
    
    try:
        # Get counts by status
        pending = supabase.table("content_queue").select("*", count="exact").eq("status", "pending").execute()
        processing = supabase.table("content_queue").select("*", count="exact").eq("status", "processing").execute()
        completed = supabase.table("content_queue").select("*", count="exact").eq("status", "completed").execute()
        failed = supabase.table("content_queue").select("*", count="exact").eq("status", "failed").execute()
        
        return {
            "pending": pending.count if hasattr(pending, 'count') else len(pending.data) if pending.data else 0,
            "processing": processing.count if hasattr(processing, 'count') else len(processing.data) if processing.data else 0,
            "completed": completed.count if hasattr(completed, 'count') else len(completed.data) if completed.data else 0,
            "failed": failed.count if hasattr(failed, 'count') else len(failed.data) if failed.data else 0
        }
    except Exception as e:
        logger.error(f"Error getting queue status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

