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

class OptimizationRequest(BaseModel):
    batch_size: int = 5
    target_status: str = "completed"  # Only optimize articles that are already rewritten/completed

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
    wait=wait_exponential(multiplier=2, min=30, max=90),
    stop=stop_after_attempt(3)
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
def run_optimization_batch(batch_size, target_status="completed"):
    """
    Background worker that optimizes blog posts with internal links, alt text, and schema.
    Processes articles that have been rewritten (status='completed') but not yet optimized.
    """
    logger.info(f"🚀 Starting SEO Optimization Batch (batch_size={batch_size}, target_status={target_status})...")
    
    try:
        # 1. Get the Sitemap (Context) - Fetch once per batch
        logger.info("📋 Fetching sitemap context...")
        sitemap_ctx = get_sitemap_context()
        
        if not sitemap_ctx:
            logger.warning("⚠️ No sitemap context available. Continuing with empty context.")
        
        # 2. Fetch Candidates - Articles that need optimization
        # Check if seo_optimized column exists by trying to query it
        # If it doesn't exist, we'll use rewrite_status as a fallback
        try:
            # Try to query with seo_optimized filter
            response = supabase.table("blog_posts")\
                .select("*")\
                .eq("rewrite_status", target_status)\
                .is_("seo_optimized", "null")\
                .limit(batch_size)\
                .execute()
        except Exception as e:
            # If seo_optimized column doesn't exist, use rewrite_status only
            logger.info("⚠️ seo_optimized column not found. Using rewrite_status filter only.")
            response = supabase.table("blog_posts")\
                .select("*")\
                .eq("rewrite_status", target_status)\
                .limit(batch_size)\
                .execute()
        
        rows = response.data if response else []
        
        if not rows:
            logger.info("✅ No unoptimized articles found.")
            return {"status": "done", "message": "No articles to optimize", "processed": 0}
        
        logger.info(f"📝 Found {len(rows)} articles to optimize")
        
        processed = 0
        failed = 0
        
        for row in rows:
            try:
                article_title = row.get('title') or row.get('Title', 'Untitled')
                article_slug = row.get('slug', '')
                logger.info(f"✨ Optimizing: {article_slug} (ID: {row['id']})")
                
                # Extract HTML from content
                original_content = row.get('content', '')
                html_content = extract_html_from_content(original_content)
                
                if not html_content or len(html_content.strip()) < 50:
                    logger.warning(f"⚠️ Skipping ID {row['id']}: Content too short or empty")
                    continue
                
                # Call AI to optimize
                optimized_html = optimize_content_with_ai(
                    html_content, 
                    sitemap_ctx, 
                    article_slug,
                    article_title
                )
                
                # Update DB - try to set seo_optimized, but handle if column doesn't exist
                update_data = {
                    "content": optimized_html,
                    "updated_at": "now()"
                }
                
                # Try to set seo_optimized flag if column exists
                try:
                    update_data["seo_optimized"] = True
                    supabase.table("blog_posts").update(update_data).eq("id", row['id']).execute()
                except Exception as col_error:
                    # If column doesn't exist, just update content
                    logger.debug(f"seo_optimized column update failed (may not exist): {col_error}")
                    supabase.table("blog_posts").update({
                        "content": optimized_html,
                        "updated_at": "now()"
                    }).eq("id", row['id']).execute()
                
                processed += 1
                logger.info(f"✅ Done: {article_slug} (ID: {row['id']})")
                
                # Throttle to avoid rate limits
                time.sleep(10)
                
            except Exception as e:
                failed += 1
                logger.error(f"❌ Failed ID {row['id']}: {e}", exc_info=True)
                time.sleep(5)
        
        logger.info(f"🎉 Batch complete! Processed: {processed}, Failed: {failed}")
        return {"status": "completed", "processed": processed, "failed": failed}
        
    except Exception as e:
        logger.error(f"❌ Batch Error: {e}", exc_info=True)
        return {"status": "error", "message": str(e)}

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
    """
    if x_admin_key != ADMIN_SECRET:
        raise HTTPException(status_code=403, detail="Unauthorized")
    
    background_tasks.add_task(run_optimization_batch, req.batch_size, req.target_status)
    return {
        "status": "queued", 
        "message": f"SEO optimization started. Processing {req.batch_size} articles with status '{req.target_status}'.",
        "batch_size": req.batch_size
    }