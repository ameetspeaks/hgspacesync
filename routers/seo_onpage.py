import os
import logging
import json
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
import google.generativeai as genai
from supabase import create_client, Client
from utils import safe_db_operation, format_error_response

router = APIRouter()
logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
BASE_URL = os.getenv("BASE_URL", "https://astrologyapp.in")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
genai.configure(api_key=GEMINI_KEY)

# --- DATA MODELS ---
class SEORequest(BaseModel):
    page_type: str = Field(..., description="Type of page: blog, calculator, horoscope, compatibility, etc.")
    page_url: str = Field(..., description="Full URL of the page")
    page_title: Optional[str] = None
    page_content: Optional[str] = None
    page_id: Optional[str] = None  # For blog posts, calculator pages, etc.
    primary_keyword: Optional[str] = None
    secondary_keywords: Optional[List[str]] = None
    image_url: Optional[str] = None
    author: Optional[str] = None
    publish_date: Optional[str] = None

class SEOResponse(BaseModel):
    meta_tags: Dict[str, Any]
    structured_data: Dict[str, Any]
    internal_links: List[Dict[str, str]]
    external_links: List[Dict[str, str]]
    breadcrumbs: List[Dict[str, str]]
    recommendations: List[str]

# --- HELPER FUNCTIONS ---
def get_working_model():
    """Get a working Gemini model"""
    candidates = ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-pro"]
    for name in candidates:
        try:
            model = genai.GenerativeModel(name)
            model.generate_content("Test")
            return model
        except Exception as e:
            logger.debug(f"Model {name} failed: {e}")
            continue
    raise Exception("No working Gemini model found")

def generate_meta_tags(page_type: str, title: str, content: str, keyword: str, url: str, image: Optional[str] = None) -> Dict[str, Any]:
    """Generate optimized meta tags using AI"""
    try:
        model = get_working_model()
        
        prompt = f"""You are an expert SEO specialist. Generate optimized meta tags for this page.

Page Type: {page_type}
Title: {title}
Primary Keyword: {keyword}
URL: {url}
Content Preview: {content[:500] if content else "N/A"}

Generate:
1. Meta Title (50-60 characters, include keyword naturally)
2. Meta Description (150-160 characters, compelling and keyword-rich)
3. Meta Keywords (5-10 relevant keywords, comma-separated)
4. Focus Keyword (primary keyword)

Output JSON format:
{{
    "title": "Optimized title here",
    "description": "Compelling description here",
    "keywords": "keyword1, keyword2, keyword3",
    "focus_keyword": "primary keyword"
}}"""

        response = model.generate_content(prompt)
        text = response.text.strip()
        
        # Clean JSON response
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()
        
        meta_data = json.loads(text)
        
        # Ensure title includes brand
        if "AstrologyApp" not in meta_data.get("title", ""):
            meta_data["title"] = f"{meta_data.get('title', title)} | AstrologyApp"
        
        return {
            "title": meta_data.get("title", title),
            "description": meta_data.get("description", content[:160] if content else ""),
            "keywords": meta_data.get("keywords", keyword),
            "focus_keyword": meta_data.get("focus_keyword", keyword),
            "og_title": meta_data.get("title", title),
            "og_description": meta_data.get("description", content[:160] if content else ""),
            "og_image": image or f"{BASE_URL}/images/og-default.jpg",
            "og_url": url,
            "og_type": "article" if page_type == "blog" else "website",
            "twitter_card": "summary_large_image",
            "twitter_title": meta_data.get("title", title),
            "twitter_description": meta_data.get("description", content[:160] if content else ""),
            "twitter_image": image or f"{BASE_URL}/images/og-default.jpg",
            "canonical": url,
            "robots": "index, follow"
        }
    except Exception as e:
        logger.error(f"Error generating meta tags: {e}")
        # Fallback to basic meta tags
        return {
            "title": f"{title} | AstrologyApp" if "AstrologyApp" not in title else title,
            "description": content[:160] if content else f"Explore {title} on AstrologyApp",
            "keywords": keyword,
            "focus_keyword": keyword,
            "og_title": title,
            "og_description": content[:160] if content else f"Explore {title} on AstrologyApp",
            "og_image": image or f"{BASE_URL}/images/og-default.jpg",
            "og_url": url,
            "og_type": "article" if page_type == "blog" else "website",
            "twitter_card": "summary_large_image",
            "twitter_title": title,
            "twitter_description": content[:160] if content else f"Explore {title} on AstrologyApp",
            "twitter_image": image or f"{BASE_URL}/images/og-default.jpg",
            "canonical": url,
            "robots": "index, follow"
        }

def generate_structured_data(page_type: str, title: str, content: str, url: str, 
                            author: Optional[str] = None, publish_date: Optional[str] = None,
                            image: Optional[str] = None) -> Dict[str, Any]:
    """Generate JSON-LD structured data based on page type"""
    
    base_schema = {
        "@context": "https://schema.org",
        "@type": "WebPage",
        "name": title,
        "url": url,
        "description": content[:200] if content else f"Explore {title} on AstrologyApp",
        "publisher": {
            "@type": "Organization",
            "name": "AstrologyApp",
            "url": BASE_URL,
            "logo": {
                "@type": "ImageObject",
                "url": f"{BASE_URL}/logo.png"
            }
        }
    }
    
    if page_type == "blog":
        schema = {
            "@context": "https://schema.org",
            "@type": "BlogPosting",
            "headline": title,
            "url": url,
            "description": content[:200] if content else "",
            "image": image or f"{BASE_URL}/images/og-default.jpg",
            "datePublished": publish_date or "",
            "dateModified": publish_date or "",
            "author": {
                "@type": "Person",
                "name": author or "AstrologyApp Team"
            },
            "publisher": {
                "@type": "Organization",
                "name": "AstrologyApp",
                "logo": {
                    "@type": "ImageObject",
                    "url": f"{BASE_URL}/logo.png"
                }
            },
            "mainEntityOfPage": {
                "@type": "WebPage",
                "@id": url
            }
        }
    elif page_type == "calculator":
        schema = {
            "@context": "https://schema.org",
            "@type": "WebApplication",
            "name": title,
            "url": url,
            "description": content[:200] if content else f"Use our {title} calculator",
            "applicationCategory": "AstrologyCalculator",
            "operatingSystem": "Web",
            "offers": {
                "@type": "Offer",
                "price": "0",
                "priceCurrency": "USD"
            }
        }
    elif page_type == "horoscope":
        schema = {
            "@context": "https://schema.org",
            "@type": "Article",
            "headline": title,
            "url": url,
            "description": content[:200] if content else f"Read your {title}",
            "publisher": {
                "@type": "Organization",
                "name": "AstrologyApp",
                "logo": {
                    "@type": "ImageObject",
                    "url": f"{BASE_URL}/logo.png"
                }
            }
        }
    else:
        schema = base_schema
    
    return schema

def get_internal_links(page_type: str, keyword: str, page_id: Optional[str] = None) -> List[Dict[str, str]]:
    """Get relevant internal links based on page type and keyword"""
    try:
        internal_links = []
        
        # Common internal links for all pages
        common_links = [
            {"url": f"{BASE_URL}/", "text": "Home", "anchor": "AstrologyApp Home"},
            {"url": f"{BASE_URL}/astrology-calculators-tools", "text": "Calculators", "anchor": "Astrology Calculators"},
            {"url": f"{BASE_URL}/horoscope/today-horoscope", "text": "Horoscope", "anchor": "Daily Horoscope"},
            {"url": f"{BASE_URL}/panchanga", "text": "Panchanga", "anchor": "Daily Panchanga"},
            {"url": f"{BASE_URL}/blog", "text": "Blog", "anchor": "Astrology Blog"},
        ]
        
        # Page-specific internal links
        if page_type == "blog":
            # Get related blog posts
            try:
                related = safe_db_operation(
                    lambda: supabase.table("blog_posts")
                        .select("slug, Title, primary_keyword")
                        .neq("id", page_id if page_id else "")
                        .limit(5)
                        .execute(),
                    "Failed to fetch related posts"
                )
                if related and related.data:
                    for post in related.data[:3]:
                        internal_links.append({
                            "url": f"{BASE_URL}/blog/{post.get('slug', '')}",
                            "text": post.get('Title', ''),
                            "anchor": post.get('Title', '')
                        })
            except Exception as e:
                logger.warning(f"Could not fetch related posts: {e}")
        
        elif page_type == "calculator":
            calculator_links = [
                {"url": f"{BASE_URL}/calculators/birth-chart", "text": "Birth Chart Calculator", "anchor": "Birth Chart"},
                {"url": f"{BASE_URL}/calculators/love-compatibility", "text": "Love Compatibility", "anchor": "Love Compatibility Calculator"},
                {"url": f"{BASE_URL}/calculators/marriage-compatibility", "text": "Marriage Compatibility", "anchor": "Marriage Compatibility Calculator"},
            ]
            internal_links.extend(calculator_links)
        
        elif page_type == "horoscope":
            horoscope_links = [
                {"url": f"{BASE_URL}/horoscope/today-horoscope", "text": "Today's Horoscope", "anchor": "Daily Horoscope"},
                {"url": f"{BASE_URL}/horoscope/weekly-horoscope", "text": "Weekly Horoscope", "anchor": "Weekly Predictions"},
                {"url": f"{BASE_URL}/horoscope/monthly-horoscope", "text": "Monthly Horoscope", "anchor": "Monthly Predictions"},
            ]
            internal_links.extend(horoscope_links)
        
        # Add common links
        internal_links.extend(common_links[:3])  # Limit to 3 common links
        
        return internal_links[:8]  # Max 8 internal links
        
    except Exception as e:
        logger.error(f"Error getting internal links: {e}")
        return []

def get_external_links(keyword: str, page_type: str) -> List[Dict[str, str]]:
    """Get relevant external links (authoritative sources)"""
    try:
        # Authoritative astrology and Vedic knowledge sources
        external_sources = {
            "vedic": [
                {"url": "https://www.sanatan.org", "text": "Sanatan.org", "anchor": "Vedic Knowledge"},
                {"url": "https://www.hinduwebsite.com", "text": "Hindu Website", "anchor": "Hindu Philosophy"},
            ],
            "astrology": [
                {"url": "https://www.astrology.com", "text": "Astrology.com", "anchor": "Astrology Resources"},
            ],
            "general": [
                {"url": "https://en.wikipedia.org/wiki/Vedic_astrology", "text": "Wikipedia - Vedic Astrology", "anchor": "Vedic Astrology"},
            ]
        }
        
        external_links = []
        
        # Add relevant external links based on keyword
        keyword_lower = keyword.lower()
        if any(term in keyword_lower for term in ["vedic", "hindu", "sanatan", "jyotish"]):
            external_links.extend(external_sources["vedic"][:2])
        if any(term in keyword_lower for term in ["astrology", "horoscope", "zodiac"]):
            external_links.extend(external_sources["astrology"][:1])
        
        # Always add at least one general reference
        if len(external_links) < 2:
            external_links.extend(external_sources["general"][:1])
        
        return external_links[:3]  # Max 3 external links
        
    except Exception as e:
        logger.error(f"Error getting external links: {e}")
        return []

def generate_breadcrumbs(page_type: str, url: str, title: str) -> List[Dict[str, str]]:
    """Generate breadcrumb navigation"""
    breadcrumbs = [
        {"name": "Home", "url": f"{BASE_URL}/"}
    ]
    
    url_parts = url.replace(BASE_URL, "").strip("/").split("/")
    
    if page_type == "blog":
        breadcrumbs.append({"name": "Blog", "url": f"{BASE_URL}/blog"})
        if len(url_parts) > 1:
            breadcrumbs.append({"name": title, "url": url})
    elif page_type == "calculator":
        breadcrumbs.append({"name": "Calculators", "url": f"{BASE_URL}/astrology-calculators-tools"})
        breadcrumbs.append({"name": title, "url": url})
    elif page_type == "horoscope":
        breadcrumbs.append({"name": "Horoscope", "url": f"{BASE_URL}/horoscope"})
        if len(url_parts) > 1:
            breadcrumbs.append({"name": title, "url": url})
    else:
        if len(url_parts) > 0:
            breadcrumbs.append({"name": title, "url": url})
    
    return breadcrumbs

def generate_breadcrumb_schema(breadcrumbs: List[Dict[str, str]]) -> Dict[str, Any]:
    """Generate BreadcrumbList structured data"""
    return {
        "@context": "https://schema.org",
        "@type": "BreadcrumbList",
        "itemListElement": [
            {
                "@type": "ListItem",
                "position": i + 1,
                "name": crumb["name"],
                "item": crumb["url"]
            }
            for i, crumb in enumerate(breadcrumbs)
        ]
    }

def get_seo_recommendations(page_type: str, content: Optional[str], meta_tags: Dict) -> List[str]:
    """Generate SEO recommendations"""
    recommendations = []
    
    if not content or len(content) < 300:
        recommendations.append("Add more content (aim for at least 300 words)")
    
    if not meta_tags.get("description") or len(meta_tags.get("description", "")) < 120:
        recommendations.append("Meta description should be 120-160 characters")
    
    if page_type == "blog" and not content:
        recommendations.append("Blog posts should have at least 800-1000 words for better SEO")
    
    if not meta_tags.get("og_image"):
        recommendations.append("Add an Open Graph image (1200x630px) for better social sharing")
    
    recommendations.append("Ensure images have descriptive alt text")
    recommendations.append("Use H1 tag for main title, H2-H3 for sections")
    recommendations.append("Include primary keyword in first 100 words of content")
    
    return recommendations

# --- ENDPOINTS ---

@router.post("/generate", response_model=SEOResponse)
def generate_seo_data(req: SEORequest):
    """
    Generate comprehensive on-page SEO data for any page.
    Returns meta tags, structured data, internal/external links, breadcrumbs, and recommendations.
    """
    try:
        # Generate meta tags
        meta_tags = generate_meta_tags(
            page_type=req.page_type,
            title=req.page_title or "AstrologyApp",
            content=req.page_content or "",
            keyword=req.primary_keyword or req.page_type,
            url=req.page_url,
            image=req.image_url
        )
        
        # Generate structured data
        structured_data = generate_structured_data(
            page_type=req.page_type,
            title=req.page_title or "AstrologyApp",
            content=req.page_content or "",
            url=req.page_url,
            author=req.author,
            publish_date=req.publish_date,
            image=req.image_url
        )
        
        # Get internal links
        internal_links = get_internal_links(
            page_type=req.page_type,
            keyword=req.primary_keyword or req.page_type,
            page_id=req.page_id
        )
        
        # Get external links
        external_links = get_external_links(
            keyword=req.primary_keyword or req.page_type,
            page_type=req.page_type
        )
        
        # Generate breadcrumbs
        breadcrumbs = generate_breadcrumbs(
            page_type=req.page_type,
            url=req.page_url,
            title=req.page_title or "AstrologyApp"
        )
        
        # Add breadcrumb structured data
        structured_data["breadcrumb"] = generate_breadcrumb_schema(breadcrumbs)
        
        # Get SEO recommendations
        recommendations = get_seo_recommendations(
            page_type=req.page_type,
            content=req.page_content,
            meta_tags=meta_tags
        )
        
        return SEOResponse(
            meta_tags=meta_tags,
            structured_data=structured_data,
            internal_links=internal_links,
            external_links=external_links,
            breadcrumbs=breadcrumbs,
            recommendations=recommendations
        )
        
    except Exception as e:
        logger.error(f"Error generating SEO data: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate SEO data: {str(e)}"
        )

@router.get("/page/{page_type}")
def get_seo_for_page(
    page_type: str,
    url: str = Query(..., description="Page URL"),
    page_id: Optional[str] = Query(None, description="Page ID (for blog posts, etc.)")
):
    """
    Get SEO data for a specific page by fetching from database.
    """
    try:
        page_data = None
        
        if page_type == "blog" and page_id:
            # Fetch blog post
            result = safe_db_operation(
                lambda: supabase.table("blog_posts")
                    .select("*")
                    .eq("id", page_id)
                    .single()
                    .execute(),
                "Failed to fetch blog post"
            )
            if result and result.data:
                page_data = {
                    "title": result.data.get("Title", ""),
                    "content": str(result.data.get("content", ""))[:1000],
                    "primary_keyword": result.data.get("primary_keyword", ""),
                    "image_url": result.data.get("coverImage", {}).get("url", "") if isinstance(result.data.get("coverImage"), dict) else "",
                    "author": result.data.get("author", {}).get("name", "") if isinstance(result.data.get("author"), dict) else "",
                    "publish_date": result.data.get("publishedAt", "")
                }
        
        if not page_data:
            raise HTTPException(status_code=404, detail="Page not found")
        
        # Generate SEO data
        req = SEORequest(
            page_type=page_type,
            page_url=url,
            page_title=page_data.get("title"),
            page_content=page_data.get("content"),
            page_id=page_id,
            primary_keyword=page_data.get("primary_keyword"),
            image_url=page_data.get("image_url"),
            author=page_data.get("author"),
            publish_date=page_data.get("publish_date")
        )
        
        return generate_seo_data(req)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting SEO for page: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get SEO data: {str(e)}"
        )

@router.get("/health")
def seo_health():
    """Health check for SEO service"""
    return {
        "status": "healthy",
        "service": "On-Page SEO Generator",
        "features": [
            "Meta Tags Generation",
            "Structured Data (JSON-LD)",
            "Internal Links",
            "External Links",
            "Breadcrumbs",
            "SEO Recommendations"
        ]
    }

