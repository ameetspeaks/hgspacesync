import os
import logging
import copy
import time
import uuid
from datetime import datetime
from typing import Dict, Optional
import yake
from fastapi import APIRouter, BackgroundTasks, HTTPException, Header
from pydantic import BaseModel
from supabase import create_client, Client
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted, NotFound
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

router = APIRouter()
logger = logging.getLogger(__name__)

# --- BATCH TRACKING ---
# In-memory batch tracking (could be moved to DB for persistence)
batch_tracker: Dict[str, dict] = {}

# --- RATE LIMITING CONFIG ---
GEMINI_RATE_LIMIT_DELAY = 15  # Seconds between Gemini API calls
GEMINI_BATCH_DELAY = 30  # Seconds between batches (if processing multiple)
MAX_RETRIES = 3

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

class OptimizationRequest(BaseModel):
    batch_size: int = 5
    target_status: str = "completed"  # Only optimize articles that are already rewritten/completed
    run_sync: bool = False  # If True, runs synchronously (for testing). WARNING: Can timeout on large batches
    run_sync: bool = False  # If True, runs synchronously (for testing). WARNING: Can timeout on large batches

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

def extract_html_from_content(content):
    """
    Extracts HTML content from various formats (JSONB, string, etc.)
    Returns the HTML string ready for optimization.
    """
    if isinstance(content, str):
        # If it's already a string, check if it's HTML
        if "<" in content and ">" in content:
            return content
        # If it's plain text, wrap it in a basic HTML structure
        return f"<div>{content}</div>"
    elif isinstance(content, dict):
        # Try to find HTML in common fields
        html_fields = ['html', 'content', 'text', 'body']
        for field in html_fields:
            if field in content and isinstance(content[field], str):
                return content[field]
        # If no HTML field found, convert dict to HTML
        return f"<div>{str(content)}</div>"
    elif isinstance(content, list):
        # If it's a list, try to extract HTML from items
        html_parts = []
        for item in content:
            if isinstance(item, dict):
                for field in ['html', 'content', 'text', 'body']:
                    if field in item and isinstance(item[field], str):
                        html_parts.append(item[field])
                        break
            elif isinstance(item, str):
                html_parts.append(item)
        return "".join(html_parts) if html_parts else f"<div>{str(content)}</div>"
    else:
        return f"<div>{str(content)}</div>"

def get_sitemap_context():
    """
    Fetches ALL 800+ titles and slugs to build a 'Link Map' for the AI.
    Returns a formatted string with all blog posts for context.
    """
    try:
        # Fetch minimal data to save bandwidth
        res = supabase.table("blog_posts").select("title, slug").execute()
        
        if not res.data:
            logger.warning("No blog posts found for sitemap context")
            return ""
        
        # Format as a compact list for the prompt
        # Format: "Title of Post" -> /blog/slug-of-post
        sitemap = [f"- {r.get('title', r.get('Title', 'Untitled'))} (URL: /blog/{r.get('slug', '')})" for r in res.data]
        return "\n".join(sitemap)
    except Exception as e:
        logger.error(f"Sitemap Fetch Error: {e}")
        return ""

@retry(
    retry=retry_if_exception_type(ResourceExhausted), 
    wait=wait_exponential(multiplier=2, min=30, max=120),  # Increased max wait to 2 minutes
    stop=stop_after_attempt(MAX_RETRIES)
)
def optimize_content_with_ai(content_html, sitemap_context, current_slug, title):
    """
    Injects internal links, fixes alt text, and adds schema using Gemini.
    Returns optimized HTML content.
    """
    try:
        model = get_working_model()
        
        prompt = f"""Act as a Technical SEO Expert.

TASK: Optimize the following HTML Blog Content.

INPUT DATA:
1. **Content:** Provided below.
2. **Internal Link Map:** A list of all other pages on my website.
3. **Current Page:** {current_slug} (DO NOT link to this page itself)

INSTRUCTIONS:
1. **Internal Linking:** Scan the content. If you see text that is semantically relevant to a page in the 'Link Map', wrap it in an <a href="/blog/slug"> tag.
   - Add 3-8 internal links max. 
   - Do NOT link to the current page ({current_slug}).
   - Ensure anchor text is natural and flows with the content.
   - Only link when there's genuine semantic relevance.

2. **Image Optimization:** If you find <img> tags without 'alt' attributes, add descriptive, keyword-rich alt text that describes the image content.

3. **Schema:** Append a valid <script type="application/ld+json"> block at the end with 'Article' schema. Include:
   - @context: "https://schema.org"
   - @type: "Article"
   - headline: "{title}"
   - datePublished: (use current date if not available)
   - author: {{"@type": "Person", "name": "AstrologyApp Team"}}
   - publisher: {{"@type": "Organization", "name": "AstrologyApp"}}

LINK MAP (Reference Only - Use these for internal linking):
{sitemap_context[:50000]}

CONTENT TO OPTIMIZE:
{content_html}

OUTPUT: Return ONLY the updated HTML content. No markdown code blocks, no explanations. Just the HTML."""
        
        response = model.generate_content(prompt)
        optimized_html = response.text.strip()
        
        # Clean up markdown code blocks if present
        optimized_html = optimized_html.replace("```html", "").replace("```", "").strip()
        
        # Remove any leading/trailing markdown formatting
        if optimized_html.startswith("```"):
            optimized_html = optimized_html.split("```")[-1]
        if optimized_html.endswith("```"):
            optimized_html = optimized_html.rsplit("```", 1)[0]
        
        return optimized_html.strip()
        
    except Exception as e:
        logger.error(f"AI Optimization Error: {e}")
        raise

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

# --- SEO OPTIMIZATION WORKER ---
def run_optimization_batch(batch_id: str, batch_size: int, target_status: str = "completed"):
    """
    Background worker that optimizes blog posts with internal links, alt text, and schema.
    Processes articles that have been rewritten (status='completed') but not yet optimized.
    
    Args:
        batch_id: Unique batch identifier for tracking
        batch_size: Number of articles to process
        target_status: Only optimize articles with this rewrite_status
    """
    # Initialize batch tracking
    batch_tracker[batch_id] = {
        "status": "running",
        "started_at": datetime.now().isoformat(),
        "total": 0,
        "processed": 0,
        "failed": 0,
        "skipped": 0,
        "article_ids": [],
        "errors": [],
        "completed_at": None
    }
    
    logger.info(f"🚀 Starting SEO Optimization Batch {batch_id} (batch_size={batch_size}, target_status={target_status})...")
    
    try:
        # 1. Get the Sitemap (Context) - Fetch once per batch
        logger.info("📋 Fetching sitemap context...")
        sitemap_ctx = get_sitemap_context()
        
        if not sitemap_ctx:
            logger.warning("⚠️ No sitemap context available. Continuing with empty context.")
        
        # 2. Fetch Candidates - Articles that need optimization
        # First, check how many articles match our criteria
        logger.info(f"🔍 Searching for articles with rewrite_status='{target_status}'...")
        
        # Try to check if seo_optimized column exists by querying one article
        test_query = supabase.table("blog_posts").select("id, seo_optimized").limit(1).execute()
        has_seo_column = test_query.data and len(test_query.data) > 0 and 'seo_optimized' in test_query.data[0]
        
        logger.info(f"📊 seo_optimized column exists: {has_seo_column}")
        
        # Build query based on whether column exists
        if has_seo_column:
            # Query with seo_optimized filter
            logger.info("🔍 Querying with seo_optimized filter...")
            response = supabase.table("blog_posts")\
                .select("*")\
                .eq("rewrite_status", target_status)\
                .or_("seo_optimized.is.null,seo_optimized.eq.false")\
                .limit(batch_size)\
                .execute()
        else:
            # If column doesn't exist, use rewrite_status only
            logger.info("⚠️ seo_optimized column not found. Using rewrite_status filter only.")
            response = supabase.table("blog_posts")\
                .select("*")\
                .eq("rewrite_status", target_status)\
                .limit(batch_size)\
                .execute()
        
        rows = response.data if response else []
        
        logger.info(f"📝 Query returned {len(rows)} articles")
        
        # Update batch tracker with total
        batch_tracker[batch_id]["total"] = len(rows)
        batch_tracker[batch_id]["article_ids"] = [r['id'] for r in rows]
        
        if not rows:
            # Log why no articles were found
            total_completed = supabase.table("blog_posts")\
                .select("id", count="exact")\
                .eq("rewrite_status", target_status)\
                .execute()
            total_count = total_completed.count if hasattr(total_completed, 'count') else len(total_completed.data) if total_completed.data else 0
            
            if has_seo_column:
                optimized_count = supabase.table("blog_posts")\
                    .select("id", count="exact")\
                    .eq("rewrite_status", target_status)\
                    .eq("seo_optimized", True)\
                    .execute()
                opt_count = optimized_count.count if hasattr(optimized_count, 'count') else len(optimized_count.data) if optimized_count.data else 0
                logger.info(f"ℹ️ Found {total_count} articles with status '{target_status}', {opt_count} already optimized")
            else:
                logger.info(f"ℹ️ Found {total_count} articles with status '{target_status}'")
            
            logger.info("✅ No unoptimized articles found.")
            batch_tracker[batch_id].update({
                "status": "completed",
                "completed_at": datetime.now().isoformat(),
                "message": "No articles to optimize"
            })
            return {"status": "done", "message": "No articles to optimize", "processed": 0, "total_available": total_count}
        
        logger.info(f"📝 Found {len(rows)} articles to optimize")
        
        processed = 0
        failed = 0
        skipped = 0
        
        for idx, row in enumerate(rows, 1):
            try:
                article_title = row.get('title') or row.get('Title', 'Untitled')
                article_slug = row.get('slug', '')
                article_id = row['id']
                
                logger.info(f"✨ [{idx}/{len(rows)}] Optimizing: {article_slug} (ID: {article_id})")
                
                # Update batch progress
                batch_tracker[batch_id]["current_article"] = {
                    "id": article_id,
                    "slug": article_slug,
                    "title": article_title,
                    "status": "processing"
                }
                
                # Extract HTML from content
                original_content = row.get('content', '')
                html_content = extract_html_from_content(original_content)
                
                if not html_content or len(html_content.strip()) < 50:
                    logger.warning(f"⚠️ Skipping ID {article_id}: Content too short or empty")
                    skipped += 1
                    batch_tracker[batch_id]["skipped"] = skipped
                    continue
                
                # Call AI to optimize with rate limiting
                try:
                    optimized_html = optimize_content_with_ai(
                        html_content, 
                        sitemap_ctx, 
                        article_slug,
                        article_title
                    )
                except ResourceExhausted as rate_error:
                    logger.warning(f"⚠️ Rate limit hit for ID {article_id}, waiting longer...")
                    # Wait longer on rate limit
                    time.sleep(60)  # Wait 1 minute on rate limit
                    # Retry once
                    try:
                        optimized_html = optimize_content_with_ai(
                            html_content, 
                            sitemap_ctx, 
                            article_slug,
                            article_title
                        )
                    except Exception as retry_error:
                        raise retry_error
                
                # Update DB - try to set seo_optimized, but handle if column doesn't exist
                update_data = {
                    "content": optimized_html,
                    "updated_at": "now()"
                }
                
                # Try to set seo_optimized flag if column exists
                try:
                    update_data["seo_optimized"] = True
                    supabase.table("blog_posts").update(update_data).eq("id", article_id).execute()
                except Exception as col_error:
                    # If column doesn't exist, just update content
                    logger.debug(f"seo_optimized column update failed (may not exist): {col_error}")
                    supabase.table("blog_posts").update({
                        "content": optimized_html,
                        "updated_at": "now()"
                    }).eq("id", article_id).execute()
                
                processed += 1
                batch_tracker[batch_id]["processed"] = processed
                logger.info(f"✅ [{idx}/{len(rows)}] Done: {article_slug} (ID: {article_id})")
                
                # Rate limiting: Wait between API calls (except for last article)
                if idx < len(rows):
                    logger.info(f"⏳ Rate limiting: Waiting {GEMINI_RATE_LIMIT_DELAY}s before next article...")
                    time.sleep(GEMINI_RATE_LIMIT_DELAY)
                
            except Exception as e:
                failed += 1
                error_msg = str(e)
                logger.error(f"❌ Failed ID {row['id']}: {error_msg}", exc_info=True)
                batch_tracker[batch_id]["failed"] = failed
                batch_tracker[batch_id]["errors"].append({
                    "article_id": row['id'],
                    "slug": row.get('slug', ''),
                    "error": error_msg
                })
                # Shorter wait on error
                time.sleep(5)
        
        logger.info(f"🎉 Batch {batch_id} complete! Processed: {processed}, Failed: {failed}, Skipped: {skipped}")
        
        # Update batch tracker
        batch_tracker[batch_id].update({
            "status": "completed",
            "processed": processed,
            "failed": failed,
            "skipped": skipped,
            "completed_at": datetime.now().isoformat(),
            "current_article": None
        })
        
        return {"status": "completed", "processed": processed, "failed": failed, "skipped": skipped}
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"❌ Batch {batch_id} Error: {error_msg}", exc_info=True)
        
        # Update batch tracker with error
        batch_tracker[batch_id].update({
            "status": "error",
            "completed_at": datetime.now().isoformat(),
            "error": error_msg
        })
        
        return {"status": "error", "message": error_msg}

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

@router.post("/optimize-batch")
def trigger_seo_polish(
    req: OptimizationRequest, 
    background_tasks: BackgroundTasks, 
    x_admin_key: str = Header(None)
):
    """
    Triggers SEO optimization batch job.
    Adds internal links, image alt text, and JSON-LD schema to blog articles.
    
    Returns a batch_id that can be used to track progress.
    """
    try:
        if x_admin_key != ADMIN_SECRET:
            raise HTTPException(status_code=403, detail="Unauthorized")
        
        # Generate unique batch ID
        batch_id = str(uuid.uuid4())
        
        # Get run_sync flag (defaults to False if not provided)
        run_sync = getattr(req, 'run_sync', False)
        
        if run_sync:
            # Run synchronously for immediate feedback (testing only)
            logger.info(f"🔄 Running optimization synchronously (testing mode) - Batch ID: {batch_id}")
            result = run_optimization_batch(batch_id, req.batch_size, req.target_status)
            return {
                "status": "completed",
                "batch_id": batch_id,
                "message": f"SEO optimization completed synchronously.",
                "result": result
            }
        else:
            # Run asynchronously (production)
            background_tasks.add_task(run_optimization_batch, batch_id, req.batch_size, req.target_status)
            return {
                "status": "queued", 
                "batch_id": batch_id,
                "message": f"SEO optimization queued. Processing {req.batch_size} articles with status '{req.target_status}'.", 
                "batch_size": req.batch_size,
                "check_status_url": f"/api/seo/batch-status/{batch_id}",
                "note": "Use the batch_id to check status. Task is running in background."
            }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in trigger_seo_polish: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to trigger optimization: {str(e)}")

@router.get("/batch-status/{batch_id}")
def get_batch_status(batch_id: str, x_admin_key: str = Header(None)):
    """
    Get the status of a running or completed optimization batch.
    
    Returns:
        - status: "running", "completed", "error"
        - progress: processed/total
        - details: article IDs, errors, etc.
    """
    if x_admin_key != ADMIN_SECRET:
        raise HTTPException(status_code=403, detail="Unauthorized")
    
    if batch_id not in batch_tracker:
        raise HTTPException(status_code=404, detail=f"Batch {batch_id} not found. It may have expired or never existed.")
    
    batch_info = batch_tracker[batch_id].copy()
    
    # Calculate progress percentage
    if batch_info["total"] > 0:
        batch_info["progress_percentage"] = round(
            (batch_info["processed"] + batch_info["failed"] + batch_info["skipped"]) / batch_info["total"] * 100, 
            2
        )
    else:
        batch_info["progress_percentage"] = 0
    
    # Calculate elapsed time
    if batch_info.get("started_at"):
        started = datetime.fromisoformat(batch_info["started_at"])
        if batch_info.get("completed_at"):
            completed = datetime.fromisoformat(batch_info["completed_at"])
            elapsed = (completed - started).total_seconds()
        else:
            elapsed = (datetime.now() - started).total_seconds()
        batch_info["elapsed_seconds"] = round(elapsed, 2)
    
    return batch_info

@router.get("/optimize-status")
def get_seo_optimization_status(x_admin_key: str = Header(None)):
    """
    Get overall SEO optimization status across all articles.
    Returns counts of optimized vs unoptimized articles.
    """
    if x_admin_key != ADMIN_SECRET:
        raise HTTPException(status_code=403, detail="Unauthorized")
    
    try:
        # Try to get counts with seo_optimized column
        try:
            # Total articles
            total_res = supabase.table("blog_posts").select("id", count="exact").execute()
            total_count = total_res.count if hasattr(total_res, 'count') else len(total_res.data) if total_res.data else 0
            
            # Optimized articles
            optimized_res = supabase.table("blog_posts").select("id", count="exact").eq("seo_optimized", True).execute()
            optimized_count = optimized_res.count if hasattr(optimized_res, 'count') else len(optimized_res.data) if optimized_res.data else 0
            
            # Unoptimized articles (completed but not optimized)
            unoptimized_res = supabase.table("blog_posts").select("id", count="exact").eq("rewrite_status", "completed").is_("seo_optimized", "null").execute()
            unoptimized_count = unoptimized_res.count if hasattr(unoptimized_res, 'count') else len(unoptimized_res.data) if unoptimized_res.data else 0
            
            # Fallback: if seo_optimized column doesn't exist, count by rewrite_status
            if optimized_count == 0 and unoptimized_count == 0:
                completed_res = supabase.table("blog_posts").select("id", count="exact").eq("rewrite_status", "completed").execute()
                completed_count = completed_res.count if hasattr(completed_res, 'count') else len(completed_res.data) if completed_res.data else 0
                unoptimized_count = completed_count
        except Exception as e:
            logger.debug(f"Error querying seo_optimized column: {e}")
            # Fallback to rewrite_status only
            total_res = supabase.table("blog_posts").select("id", count="exact").execute()
            total_count = total_res.count if hasattr(total_res, 'count') else len(total_res.data) if total_res.data else 0
            
            completed_res = supabase.table("blog_posts").select("id", count="exact").eq("rewrite_status", "completed").execute()
            completed_count = completed_res.count if hasattr(completed_res, 'count') else len(completed_res.data) if completed_res.data else 0
            
            optimized_count = 0
            unoptimized_count = completed_count
        
        return {
            "total_articles": total_count,
            "optimized": optimized_count,
            "unoptimized": unoptimized_count,
            "optimization_percentage": round((optimized_count / total_count * 100), 2) if total_count > 0 else 0,
            "ready_for_optimization": unoptimized_count,
            "note": "seo_optimized column exists" if optimized_count > 0 or unoptimized_count > 0 else "Using rewrite_status only (seo_optimized column may not exist)"
        }
    except Exception as e:
        logger.error(f"Error getting SEO optimization status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get status: {str(e)}")

@router.get("/optimize-status/{article_id}")
def get_article_seo_status(article_id: int, x_admin_key: str = Header(None)):
    """
    Get SEO optimization status for a specific article by ID.
    """
    if x_admin_key != ADMIN_SECRET:
        raise HTTPException(status_code=403, detail="Unauthorized")
    
    try:
        res = supabase.table("blog_posts").select("id, slug, title, seo_optimized, rewrite_status, updated_at").eq("id", article_id).single().execute()
        
        if not res.data:
            raise HTTPException(status_code=404, detail="Article not found")
        
        article = res.data
        return {
            "id": article.get("id"),
            "slug": article.get("slug"),
            "title": article.get("title") or article.get("Title", "Untitled"),
            "seo_optimized": article.get("seo_optimized", False),
            "rewrite_status": article.get("rewrite_status"),
            "last_updated": article.get("updated_at"),
            "is_ready": article.get("rewrite_status") == "completed" and not article.get("seo_optimized", False)
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting article SEO status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get article status: {str(e)}")

@router.get("/optimize-diagnostic")
def diagnostic_seo_optimization(x_admin_key: str = Header(None)):
    """
    Diagnostic endpoint to check why optimization might not be working.
    Returns detailed information about articles and database state.
    """
    if x_admin_key != ADMIN_SECRET:
        raise HTTPException(status_code=403, detail="Unauthorized")
    
    try:
        # Check if seo_optimized column exists
        test_query = supabase.table("blog_posts").select("id, seo_optimized").limit(1).execute()
        has_seo_column = test_query.data and len(test_query.data) > 0 and 'seo_optimized' in test_query.data[0]
        
        # Get counts by rewrite_status
        all_statuses = supabase.table("blog_posts").select("rewrite_status").execute()
        status_counts = {}
        for row in all_statuses.data:
            status = row.get('rewrite_status', 'unknown')
            status_counts[status] = status_counts.get(status, 0) + 1
        
        # Get completed articles
        completed_res = supabase.table("blog_posts")\
            .select("id, slug, title, seo_optimized")\
            .eq("rewrite_status", "completed")\
            .limit(10)\
            .execute()
        
        completed_articles = completed_res.data if completed_res else []
        
        # Count optimized if column exists
        optimized_count = 0
        if has_seo_column:
            opt_res = supabase.table("blog_posts")\
                .select("id", count="exact")\
                .eq("seo_optimized", True)\
                .execute()
            optimized_count = opt_res.count if hasattr(opt_res, 'count') else len(opt_res.data) if opt_res.data else 0
        
        return {
            "database_state": {
                "seo_optimized_column_exists": has_seo_column,
                "total_articles": len(all_statuses.data) if all_statuses.data else 0
            },
            "status_breakdown": status_counts,
            "completed_articles": {
                "total": len(completed_articles),
                "sample": completed_articles[:5],  # Show first 5
                "optimized_count": optimized_count
            },
            "recommendations": [
                "Run migration to add seo_optimized column" if not has_seo_column else "Column exists ✓",
                f"Found {status_counts.get('completed', 0)} articles with rewrite_status='completed'",
                f"Found {optimized_count} articles already optimized" if has_seo_column else "Cannot check optimized count (column missing)",
                "Use /api/seo/optimize-batch with run_sync=true for testing" if status_counts.get('completed', 0) > 0 else "No articles ready for optimization"
            ]
        }
    except Exception as e:
        logger.error(f"Error in diagnostic: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Diagnostic failed: {str(e)}")