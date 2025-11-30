import os
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from fastapi import APIRouter, HTTPException, Query, Header
from pydantic import BaseModel, Field
from supabase import create_client, Client
from utils import safe_db_operation, format_error_response

router = APIRouter()
logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
ADMIN_SECRET = os.getenv("ADMIN_SECRET")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# --- DATA MODELS ---
class SEOTrackingRequest(BaseModel):
    page_url: str = Field(..., description="URL of the page being tracked")
    page_type: str = Field(..., description="Type of page: blog, calculator, horoscope, etc.")
    page_id: Optional[str] = None
    title: Optional[str] = None
    meta_description: Optional[str] = None
    primary_keyword: Optional[str] = None
    word_count: Optional[int] = None
    internal_links_count: Optional[int] = None
    external_links_count: Optional[int] = None
    images_count: Optional[int] = None
    has_structured_data: Optional[bool] = None
    seo_score: Optional[float] = None  # 0-100

class SEOStatusUpdate(BaseModel):
    page_url: str
    google_indexed: Optional[bool] = None
    google_rank: Optional[int] = None
    impressions: Optional[int] = None
    clicks: Optional[int] = None
    ctr: Optional[float] = None
    avg_position: Optional[float] = None
    last_crawled: Optional[str] = None

class SEORankingRequest(BaseModel):
    keyword: str
    page_url: str
    rank: Optional[int] = None
    search_volume: Optional[int] = None
    difficulty: Optional[float] = None

# --- HELPER FUNCTIONS ---
def calculate_seo_score(page_data: Dict[str, Any]) -> float:
    """Calculate SEO score (0-100) based on various factors"""
    score = 0.0
    max_score = 100.0
    
    # Title optimization (15 points)
    if page_data.get('title'):
        title_len = len(page_data['title'])
        if 50 <= title_len <= 60:
            score += 15
        elif 40 <= title_len < 50 or 60 < title_len <= 70:
            score += 10
        else:
            score += 5
    
    # Meta description (15 points)
    if page_data.get('meta_description'):
        desc_len = len(page_data['meta_description'])
        if 120 <= desc_len <= 160:
            score += 15
        elif 100 <= desc_len < 120 or 160 < desc_len <= 180:
            score += 10
        else:
            score += 5
    
    # Content length (20 points)
    word_count = page_data.get('word_count', 0)
    if word_count >= 1000:
        score += 20
    elif word_count >= 500:
        score += 15
    elif word_count >= 300:
        score += 10
    elif word_count >= 100:
        score += 5
    
    # Internal links (15 points)
    internal_links = page_data.get('internal_links_count', 0)
    if internal_links >= 5:
        score += 15
    elif internal_links >= 3:
        score += 10
    elif internal_links >= 1:
        score += 5
    
    # External links (10 points)
    external_links = page_data.get('external_links_count', 0)
    if external_links >= 2:
        score += 10
    elif external_links >= 1:
        score += 5
    
    # Images (10 points)
    images = page_data.get('images_count', 0)
    if images >= 3:
        score += 10
    elif images >= 1:
        score += 5
    
    # Structured data (15 points)
    if page_data.get('has_structured_data'):
        score += 15
    
    return min(score, max_score)

# --- ENDPOINTS ---

@router.post("/track")
def track_seo_page(req: SEOTrackingRequest, x_admin_key: str = Header(None)):
    """
    Track SEO metrics for a page.
    Creates or updates SEO tracking record.
    """
    try:
        # Calculate SEO score
        page_data = {
            'title': req.title,
            'meta_description': req.meta_description,
            'word_count': req.word_count or 0,
            'internal_links_count': req.internal_links_count or 0,
            'external_links_count': req.external_links_count or 0,
            'images_count': req.images_count or 0,
            'has_structured_data': req.has_structured_data or False,
        }
        
        seo_score = req.seo_score or calculate_seo_score(page_data)
        
        # Check if record exists
        existing = safe_db_operation(
            lambda: supabase.table("seo_tracking")
                .select("id")
                .eq("page_url", req.page_url)
                .execute(),
            "Failed to check existing SEO tracking"
        )
        
        tracking_data = {
            "page_url": req.page_url,
            "page_type": req.page_type,
            "page_id": req.page_id,
            "title": req.title,
            "meta_description": req.meta_description,
            "primary_keyword": req.primary_keyword,
            "word_count": req.word_count or 0,
            "internal_links_count": req.internal_links_count or 0,
            "external_links_count": req.external_links_count or 0,
            "images_count": req.images_count or 0,
            "has_structured_data": req.has_structured_data or False,
            "seo_score": seo_score,
            "last_updated": datetime.utcnow().isoformat(),
        }
        
        if existing and existing.data:
            # Update existing record
            result = safe_db_operation(
                lambda: supabase.table("seo_tracking")
                    .update(tracking_data)
                    .eq("page_url", req.page_url)
                    .execute(),
                "Failed to update SEO tracking"
            )
        else:
            # Create new record
            tracking_data["created_at"] = datetime.utcnow().isoformat()
            result = safe_db_operation(
                lambda: supabase.table("seo_tracking")
                    .insert(tracking_data)
                    .execute(),
                "Failed to create SEO tracking"
            )
        
        return {
            "status": "success",
            "message": "SEO tracking updated",
            "data": result.data[0] if result.data else tracking_data,
            "seo_score": seo_score
        }
        
    except Exception as e:
        logger.error(f"Error tracking SEO: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to track SEO: {str(e)}"
        )

@router.post("/status")
def update_seo_status(req: SEOStatusUpdate, x_admin_key: str = Header(None)):
    """
    Update SEO status from Google Search Console or other sources.
    """
    try:
        update_data = {
            "last_updated": datetime.utcnow().isoformat(),
        }
        
        if req.google_indexed is not None:
            update_data["google_indexed"] = req.google_indexed
        if req.google_rank is not None:
            update_data["google_rank"] = req.google_rank
        if req.impressions is not None:
            update_data["impressions"] = req.impressions
        if req.clicks is not None:
            update_data["clicks"] = req.clicks
        if req.ctr is not None:
            update_data["ctr"] = req.ctr
        if req.avg_position is not None:
            update_data["avg_position"] = req.avg_position
        if req.last_crawled:
            update_data["last_crawled"] = req.last_crawled
        
        result = safe_db_operation(
            lambda: supabase.table("seo_tracking")
                .update(update_data)
                .eq("page_url", req.page_url)
                .execute(),
            "Failed to update SEO status"
        )
        
        return {
            "status": "success",
            "message": "SEO status updated",
            "data": result.data[0] if result.data else None
        }
        
    except Exception as e:
        logger.error(f"Error updating SEO status: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update SEO status: {str(e)}"
        )

@router.post("/ranking")
def track_keyword_ranking(req: SEORankingRequest, x_admin_key: str = Header(None)):
    """
    Track keyword rankings for a page.
    """
    try:
        ranking_data = {
            "keyword": req.keyword,
            "page_url": req.page_url,
            "rank": req.rank,
            "search_volume": req.search_volume,
            "difficulty": req.difficulty,
            "tracked_at": datetime.utcnow().isoformat(),
        }
        
        result = safe_db_operation(
            lambda: supabase.table("seo_keyword_rankings")
                .insert(ranking_data)
                .execute(),
            "Failed to track keyword ranking"
        )
        
        return {
            "status": "success",
            "message": "Keyword ranking tracked",
            "data": result.data[0] if result.data else ranking_data
        }
        
    except Exception as e:
        logger.error(f"Error tracking keyword ranking: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to track keyword ranking: {str(e)}"
        )

@router.get("/dashboard")
def get_seo_dashboard(
    page_type: Optional[str] = Query(None, description="Filter by page type"),
    limit: int = Query(50, description="Number of results"),
    x_admin_key: str = Header(None)
):
    """
    Get SEO dashboard data with aggregated metrics.
    """
    try:
        query = supabase.table("seo_tracking").select("*")
        
        if page_type:
            query = query.eq("page_type", page_type)
        
        query = query.order("seo_score", desc=True).limit(limit)
        
        result = safe_db_operation(
            lambda: query.execute(),
            "Failed to fetch SEO dashboard data"
        )
        
        pages = result.data if result.data else []
        
        # Calculate aggregate metrics
        total_pages = len(pages)
        avg_seo_score = sum(p.get('seo_score', 0) for p in pages) / total_pages if total_pages > 0 else 0
        indexed_count = sum(1 for p in pages if p.get('google_indexed'))
        pages_with_structured_data = sum(1 for p in pages if p.get('has_structured_data'))
        
        # Group by page type
        by_type = {}
        for page in pages:
            ptype = page.get('page_type', 'unknown')
            if ptype not in by_type:
                by_type[ptype] = {'count': 0, 'avg_score': 0, 'total_score': 0}
            by_type[ptype]['count'] += 1
            by_type[ptype]['total_score'] += page.get('seo_score', 0)
        
        for ptype in by_type:
            by_type[ptype]['avg_score'] = by_type[ptype]['total_score'] / by_type[ptype]['count']
        
        return {
            "status": "success",
            "summary": {
                "total_pages": total_pages,
                "average_seo_score": round(avg_seo_score, 2),
                "indexed_pages": indexed_count,
                "indexed_percentage": round((indexed_count / total_pages * 100) if total_pages > 0 else 0, 2),
                "pages_with_structured_data": pages_with_structured_data,
                "structured_data_percentage": round((pages_with_structured_data / total_pages * 100) if total_pages > 0 else 0, 2),
            },
            "by_page_type": by_type,
            "pages": pages
        }
        
    except Exception as e:
        logger.error(f"Error fetching SEO dashboard: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch SEO dashboard: {str(e)}"
        )

@router.get("/page/{page_url:path}")
def get_page_seo_status(page_url: str):
    """
    Get SEO status for a specific page.
    """
    try:
        # Decode URL
        from urllib.parse import unquote
        decoded_url = unquote(page_url)
        
        result = safe_db_operation(
            lambda: supabase.table("seo_tracking")
                .select("*")
                .eq("page_url", decoded_url)
                .single()
                .execute(),
            "Failed to fetch page SEO status"
        )
        
        if not result.data:
            raise HTTPException(status_code=404, detail="Page not found in SEO tracking")
        
        # Get keyword rankings for this page
        rankings_result = safe_db_operation(
            lambda: supabase.table("seo_keyword_rankings")
                .select("*")
                .eq("page_url", decoded_url)
                .order("tracked_at", desc=True)
                .limit(10)
                .execute(),
            "Failed to fetch keyword rankings"
        )
        
        page_data = result.data
        page_data['keyword_rankings'] = rankings_result.data if rankings_result.data else []
        
        return {
            "status": "success",
            "data": page_data
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching page SEO status: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch page SEO status: {str(e)}"
        )

@router.get("/rankings")
def get_keyword_rankings(
    page_url: Optional[str] = Query(None),
    keyword: Optional[str] = Query(None),
    limit: int = Query(50)
):
    """
    Get keyword ranking data.
    """
    try:
        query = supabase.table("seo_keyword_rankings").select("*")
        
        if page_url:
            query = query.eq("page_url", page_url)
        if keyword:
            query = query.ilike("keyword", f"%{keyword}%")
        
        query = query.order("tracked_at", desc=True).limit(limit)
        
        result = safe_db_operation(
            lambda: query.execute(),
            "Failed to fetch keyword rankings"
        )
        
        return {
            "status": "success",
            "data": result.data if result.data else []
        }
        
    except Exception as e:
        logger.error(f"Error fetching keyword rankings: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch keyword rankings: {str(e)}"
        )

@router.get("/health")
def tracking_health():
    """Health check for SEO tracking service"""
    return {
        "status": "healthy",
        "service": "SEO Tracking & Analytics",
        "features": [
            "Page SEO Tracking",
            "Keyword Ranking Tracking",
            "SEO Score Calculation",
            "Dashboard Analytics"
        ]
    }

