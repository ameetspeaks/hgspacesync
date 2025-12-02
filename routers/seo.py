import os
import logging
import copy
import time
import uuid
import re
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

class RestructureRequest(BaseModel):
    batch_size: int = 10
    article_ids: Optional[list[int]] = None  # If provided, only restructure these articles
    dry_run: bool = False  # If True, only shows what would be changed without updating
    seo_optimize: bool = True  # If True, also optimizes for SEO (internal links, headings, etc.)

# --- MODEL HUNTER ---
def get_working_model():
    """Try different models in order of preference."""
    # First, check if API key is configured
    if not GEMINI_KEY:
        error_msg = "GEMINI_API_KEY environment variable is not set. Please configure it in your environment."
        logger.error(f"❌ {error_msg}")
        raise Exception(error_msg)
    
    # Ensure genai is configured
    try:
        genai.configure(api_key=GEMINI_KEY)
    except Exception as e:
        error_msg = f"Failed to configure Gemini API: {str(e)}"
        logger.error(f"❌ {error_msg}")
        raise Exception(error_msg)
    
    models_to_try = [
        "gemini-1.5-flash",
        "gemini-1.5-pro",
        "gemini-pro"
    ]
    
    last_error = None
    for model_name in models_to_try:
        try:
            logger.info(f"🔍 Trying model: {model_name}")
            model = genai.GenerativeModel(model_name)
            # Don't test - just return the model. Testing can fail due to rate limits or network issues
            # but the model might still work for actual requests
            logger.info(f"✅ Model {model_name} initialized successfully")
            return model
        except Exception as e:
            last_error = str(e)
            logger.warning(f"⚠️ Model {model_name} initialization failed: {e}")
            continue
    
    # If all models failed to initialize
    error_msg = f"No working Gemini model found. Tried: {', '.join(models_to_try)}. Last error: {last_error}. Please check: 1) GEMINI_API_KEY is set correctly, 2) API key is valid, 3) API quota is available, 4) Network connectivity."
    logger.error(f"❌ {error_msg}")
    raise Exception(error_msg)

def extract_keywords(title, content, existing):
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
    Converts structured content blocks to HTML for optimization.
    Returns the HTML string ready for optimization.
    """
    import json
    
    if isinstance(content, str):
        # Check if it's JSON string
        try:
            parsed = json.loads(content)
            if isinstance(parsed, (dict, list)):
                return convert_content_blocks_to_html(parsed)
        except (json.JSONDecodeError, ValueError):
            pass
        
        # If it's already a string, check if it's HTML
        if "<" in content and ">" in content:
            return content
        # If it's plain text, wrap it in a basic HTML structure
        return f"<div>{content}</div>"
    elif isinstance(content, dict):
        # Check if it's structured content blocks
        if 'contentBlocks' in content or '__component' in str(content):
            return convert_content_blocks_to_html(content)
        
        # Try to find HTML in common fields
        html_fields = ['html', 'content', 'text', 'body']
        for field in html_fields:
            if field in content and isinstance(content[field], str):
                return content[field]
        # If no HTML field found, convert dict to HTML
        return f"<div>{str(content)}</div>"
    elif isinstance(content, list):
        # Check if it's a list of content blocks
        if content and isinstance(content[0], dict) and ('__component' in content[0] or 'body' in content[0]):
            return convert_content_blocks_to_html(content)
        
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

def convert_content_blocks_to_html(content):
    """
    Converts structured content blocks (Strapi/Contentful format) to HTML.
    Handles rich-text blocks, paragraphs, headings, lists, etc.
    """
    html_parts = []
    
    # Handle list of blocks
    if isinstance(content, list):
        blocks = content
    elif isinstance(content, dict):
        # Check if contentBlocks field exists
        if 'contentBlocks' in content:
            blocks = content['contentBlocks']
        elif 'body' in content:
            blocks = content['body'] if isinstance(content['body'], list) else [content]
        else:
            blocks = [content]
    else:
        return f"<div>{str(content)}</div>"
    
    for block in blocks:
        if not isinstance(block, dict):
            continue
            
        component = block.get('__component', '')
        body = block.get('body', [])
        
        # Handle rich-text blocks
        if 'rich-text' in component or 'body' in block:
            if isinstance(body, list):
                for para in body:
                    if isinstance(para, dict):
                        html_parts.append(convert_paragraph_to_html(para))
            elif isinstance(body, str):
                html_parts.append(f"<p>{body}</p>")
        # Handle other block types
        elif isinstance(block, str):
            html_parts.append(f"<p>{block}</p>")
    
    return "".join(html_parts) if html_parts else f"<div>{str(content)}</div>"

def convert_paragraph_to_html(para):
    """Convert a paragraph/heading/list block to HTML."""
    import re
    from html import escape
    
    para_type = para.get('type', 'paragraph')
    
    if para_type == 'heading':
        level = para.get('level', 2)
        children = para.get('children', [])
        text = convert_children_to_text(children)
        return f"<h{level}>{text}</h{level}>"
    elif para_type == 'paragraph':
        children = para.get('children', [])
        text = convert_children_to_html(children)
        return f"<p>{text}</p>"
    elif para_type == 'list':
        format_type = para.get('format', 'unordered')
        tag = 'ol' if format_type == 'ordered' else 'ul'
        items = para.get('children', [])
        items_html = []
        for item in items:
            item_children = item.get('children', [])
            item_text = convert_children_to_html(item_children)
            items_html.append(f"<li>{item_text}</li>")
        return f"<{tag}>{''.join(items_html)}</{tag}>"
    else:
        children = para.get('children', [])
        text = convert_children_to_text(children)
        return f"<p>{text}</p>"

def convert_children_to_html(children):
    """Convert children array to HTML string with links and formatting."""
    import re
    from html import escape
    
    if not children:
        return ""
    
    html_parts = []
    for child in children:
        if child.get('type') == 'text':
            text = escape(child.get('text', ''))
            if child.get('bold'):
                text = f"<strong>{text}</strong>"
            if child.get('italic'):
                text = f"<em>{text}</em>"
            html_parts.append(text)
        elif child.get('type') == 'link':
            url = child.get('url', '#')
            link_children = child.get('children', [])
            link_text = convert_children_to_text(link_children)
            html_parts.append(f'<a href="{escape(url)}">{escape(link_text)}</a>')
        else:
            text = escape(str(child.get('text', '')))
            html_parts.append(text)
    
    return ''.join(html_parts)

def convert_children_to_text(children):
    """Convert children array to plain text."""
    if not children:
        return ""
    
    text_parts = []
    for child in children:
        if child.get('type') == 'text':
            text_parts.append(child.get('text', ''))
        elif child.get('type') == 'link':
            link_children = child.get('children', [])
            link_text = convert_children_to_text(link_children)
            text_parts.append(link_text)
        else:
            text_parts.append(str(child.get('text', '')))
    
    return ''.join(text_parts)

def convert_html_to_content_blocks(html_content, original_content):
    """
    Converts optimized HTML back to structured content blocks format.
    Uses regex-based parsing for reliability.
    """
    import re
    from html import unescape
    
    # Remove script tags (schema) - we'll add it separately if needed
    html_content = re.sub(r'<script[^>]*>.*?</script>', '', html_content, flags=re.DOTALL | re.IGNORECASE)
    
    # Split by block-level tags
    blocks = []
    
    # Pattern to match block elements
    block_pattern = r'<(h[1-6]|p|div|ul|ol|li)[^>]*>(.*?)</\1>'
    
    # Split content by block tags
    parts = re.split(r'(<(?:h[1-6]|p|div|ul|ol)[^>]*>.*?</(?:h[1-6]|p|div|ul|ol)>)', html_content, flags=re.DOTALL | re.IGNORECASE)
    
    for part in parts:
        if not part.strip():
            continue
            
        part = part.strip()
        
        # Heading
        heading_match = re.match(r'<h([1-6])[^>]*>(.*?)</h\1>', part, re.DOTALL | re.IGNORECASE)
        if heading_match:
            level = int(heading_match.group(1))
            content = heading_match.group(2)
            children = parse_inline_content(content)
            blocks.append({
                'type': 'heading',
                'level': level,
                'children': children
            })
            continue
        
        # Paragraph
        para_match = re.match(r'<p[^>]*>(.*?)</p>', part, re.DOTALL | re.IGNORECASE)
        if para_match:
            content = para_match.group(1)
            children = parse_inline_content(content)
            if children:
                blocks.append({
                    'type': 'paragraph',
                    'children': children
                })
            continue
        
        # List
        list_match = re.match(r'<(ul|ol)[^>]*>(.*?)</\1>', part, re.DOTALL | re.IGNORECASE)
        if list_match:
            list_type = 'unordered' if list_match.group(1).lower() == 'ul' else 'ordered'
            list_content = list_match.group(2)
            items = re.findall(r'<li[^>]*>(.*?)</li>', list_content, re.DOTALL | re.IGNORECASE)
            list_children = []
            for item in items:
                item_children = parse_inline_content(item)
                if item_children:
                    list_children.append({'children': item_children})
            if list_children:
                blocks.append({
                    'type': 'list',
                    'format': list_type,
                    'children': list_children
                })
            continue
        
        # Div or other content - treat as paragraph
        div_match = re.match(r'<div[^>]*>(.*?)</div>', part, re.DOTALL | re.IGNORECASE)
        if div_match:
            content = div_match.group(1)
            children = parse_inline_content(content)
            if children:
                blocks.append({
                    'type': 'paragraph',
                    'children': children
                })
            continue
        
        # Plain text - create paragraph
        text = re.sub(r'<[^>]+>', '', part)  # Remove any remaining tags
        text = unescape(text).strip()
        if text:
            blocks.append({
                'type': 'paragraph',
                'children': [{'type': 'text', 'text': text}]
            })
    
    # If we got blocks, wrap in contentBlocks structure
    if blocks:
        return [{
            '__component': 'blog-components.rich-text',
            'body': blocks
        }]
    
    # Fallback: return original if conversion failed
    logger.warning("Failed to convert HTML to content blocks, returning original structure")
    return original_content

def parse_inline_content(html):
    """
    Parses inline HTML content (text, links, bold, italic) into structured children.
    """
    import re
    from html import unescape
    
    if not html:
        return []
    
    children = []
    # Pattern to match links
    link_pattern = r'<a[^>]*href=["\']([^"\']*)["\'][^>]*>(.*?)</a>'
    
    last_pos = 0
    for match in re.finditer(link_pattern, html, re.DOTALL | re.IGNORECASE):
        # Text before link
        before = html[last_pos:match.start()]
        if before:
            before_text = parse_text_formatting(before)
            children.extend(before_text)
        
        # Link
        href = match.group(1)
        link_text = match.group(2)
        link_children = parse_text_formatting(link_text)
        children.append({
            'type': 'link',
            'url': href,
            'children': link_children if link_children else [{'type': 'text', 'text': link_text}]
        })
        
        last_pos = match.end()
    
    # Text after last link
    after = html[last_pos:]
    if after:
        after_text = parse_text_formatting(after)
        children.extend(after_text)
    
    return children if children else [{'type': 'text', 'text': unescape(re.sub(r'<[^>]+>', '', html))}]

def parse_text_formatting(text):
    """
    Parses text with bold/italic formatting.
    """
    import re
    from html import unescape
    
    if not text:
        return []
    
    children = []
    # Pattern for bold
    bold_pattern = r'<(strong|b)[^>]*>(.*?)</\1>'
    # Pattern for italic
    italic_pattern = r'<(em|i)[^>]*>(.*?)</\1>'
    
    # Find all formatting tags
    positions = []
    for match in re.finditer(bold_pattern, text, re.DOTALL | re.IGNORECASE):
        positions.append((match.start(), match.end(), 'bold', match.group(2)))
    for match in re.finditer(italic_pattern, text, re.DOTALL | re.IGNORECASE):
        positions.append((match.start(), match.end(), 'italic', match.group(2)))
    
    if not positions:
        # No formatting, return plain text
        clean_text = unescape(re.sub(r'<[^>]+>', '', text)).strip()
        if clean_text:
            return [{'type': 'text', 'text': clean_text}]
        return []
    
    # Sort by position
    positions.sort(key=lambda x: x[0])
    
    last_pos = 0
    for start, end, format_type, content in positions:
        # Text before formatting
        if start > last_pos:
            before = text[last_pos:start]
            clean_before = unescape(re.sub(r'<[^>]+>', '', before)).strip()
            if clean_before:
                children.append({'type': 'text', 'text': clean_before})
        
        # Formatted text
        clean_content = unescape(re.sub(r'<[^>]+>', '', content)).strip()
        if clean_content:
            child = {'type': 'text', 'text': clean_content}
            if format_type == 'bold':
                child['bold'] = True
            elif format_type == 'italic':
                child['italic'] = True
            children.append(child)
        
        last_pos = end
    
    # Text after last formatting
    if last_pos < len(text):
        after = text[last_pos:]
        clean_after = unescape(re.sub(r'<[^>]+>', '', after)).strip()
        if clean_after:
            children.append({'type': 'text', 'text': clean_after})
    
    return children if children else [{'type': 'text', 'text': unescape(re.sub(r'<[^>]+>', '', text)).strip()}]

def get_sitemap_context():
    """
    Fetches ALL blog post titles and slugs to build a 'Link Map' for internal linking.
    """
    try:
        # Fetch minimal data to save bandwidth
        res = supabase.table("blog_posts").select("title, slug").execute()
        # Format as a compact list for the prompt
        sitemap = [f"- {r['title']} (URL: /blog/{r['slug']})" for r in res.data]
        return "\n".join(sitemap)
    except Exception as e:
        logger.error(f"Sitemap Fetch Error: {e}")
        return ""

def optimize_and_restructure_content(content, sitemap_context, current_slug, title):
    """
    Restructures content AND optimizes it for SEO with internal links, proper headings, etc.
    This is the enhanced version that does both restructuring and SEO optimization.
    """
    import json
    import re
    from html import unescape
    
    if not content:
        return []
    
    # Step 1: Convert to HTML first (for AI processing)
    content_html = extract_html_from_content(content)
    
    # Step 2: Optimize with AI (add internal links, ensure proper structure)
    try:
        optimized_html = optimize_content_with_ai_for_restructure(
            content_html, 
            sitemap_context, 
            current_slug,
            title
        )
    except Exception as e:
        logger.warning(f"AI optimization failed, using basic restructuring: {e}")
        optimized_html = content_html
    
    # Step 3: Convert optimized HTML to structured blocks
    structured_blocks = convert_html_to_content_blocks(optimized_html, [])
    
    # Step 4: Ensure proper heading hierarchy and add footer if needed
    structured_blocks = enhance_seo_structure(structured_blocks, title)
    
    return structured_blocks

@retry(
    retry=retry_if_exception_type(ResourceExhausted),
    wait=wait_exponential(multiplier=2, min=20, max=60),
    stop=stop_after_attempt(3)
)
def optimize_content_with_ai_for_restructure(content_html, sitemap_context, current_slug, title):
    """
    Uses AI to optimize content with internal links, proper headings, and SEO elements.
    """
    try:
        model = get_working_model()
        
        prompt = f"""Act as a Technical SEO Expert and Content Structuring Specialist.

TASK: Optimize and structure the following content for SEO.

INPUT DATA:
1. **Content:** Provided below (may be HTML, markdown, or mixed format).
2. **Internal Link Map:** A list of all other pages on my website.
3. **Current Page:** {current_slug} (DO NOT link to this page itself)
4. **Page Title:** {title}

INSTRUCTIONS:
1. **Content Structure:**
   - Ensure proper heading hierarchy: Use <h1> for main title (if not present), <h2> for main sections, <h3> for subsections
   - Convert all content to clean HTML format
   - Ensure paragraphs are properly wrapped in <p> tags
   - Convert lists to proper <ul> or <ol> format

2. **Internal Linking:**
   - Scan the content for text that is semantically relevant to pages in the 'Link Map'
   - Wrap relevant text in <a href="/blog/slug"> tags
   - Add 5-10 internal links naturally throughout the content
   - DO NOT link to the current page ({current_slug})
   - Ensure anchor text is natural and flows with the content
   - Only link when there's genuine semantic relevance

3. **SEO Optimization:**
   - Ensure first paragraph is strong and keyword-rich
   - Add relevant keywords naturally throughout
   - Ensure proper heading structure (H1 → H2 → H3 hierarchy)
   - Make sure content is well-organized and readable

4. **Image Optimization:**
   - If you find <img> tags without 'alt' attributes, add descriptive, keyword-rich alt text

LINK MAP (Reference Only - Use these for internal linking):
{sitemap_context[:50000]}

CONTENT TO OPTIMIZE:
{content_html[:50000]}

OUTPUT: Return ONLY the optimized HTML content. No markdown code blocks, no explanations. Just clean HTML with proper structure, internal links, and SEO elements."""

        response = model.generate_content(prompt)
        optimized_html = response.text.strip()
        
        # Clean up markdown code blocks if present
        optimized_html = optimized_html.replace("```html", "").replace("```", "").strip()
        
        # Remove any leading/trailing markdown formatting
        if optimized_html.startswith("```"):
            optimized_html = optimized_html.split("```")[-1]
        if optimized_html.endswith("```"):
            optimized_html = optimized_html.rsplit("```", 1)[0]
        
        # Convert markdown-style links [text](url) to HTML <a> tags if present
        optimized_html = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2">\1</a>', optimized_html)
        
        # Remove standalone brackets [text] that aren't links
        optimized_html = re.sub(r'\[([^\]]+)\](?!\()', r'\1', optimized_html)
        
        return optimized_html.strip()
        
                except Exception as e:
        logger.error(f"AI Optimization Error: {e}")
        raise

def enhance_seo_structure(blocks, title):
    """
    Enhances structured blocks with proper SEO elements:
    - Ensures proper heading hierarchy
    - Adds footer/CTA if needed
    - Validates structure
    """
    if not blocks:
        return blocks
    
    enhanced_blocks = []
    has_h1 = False
    has_h2 = False
    
    # First pass: Check heading structure and fix hierarchy
    for block in blocks:
        if block.get('__component') == 'blog-components.rich-text':
            body = block.get('body', [])
            enhanced_body = []
            
            for item in body:
                if item.get('type') == 'heading':
                    level = item.get('level', 2)
                    
                    # Ensure we have at least one H2 if no H1
                    if level == 1:
                        has_h1 = True
                    elif level == 2:
                        has_h2 = True
                    
                    # If first heading is H3 or below, promote to H2
                    if not has_h1 and not has_h2 and level > 2:
                        item['level'] = 2
                        has_h2 = True
                    
                    enhanced_body.append(item)
                else:
                    enhanced_body.append(item)
            
            block['body'] = enhanced_body
        
        enhanced_blocks.append(block)
    
    # Add footer/CTA block if content is substantial
    total_blocks = sum(
        len(b.get('body', [])) if b.get('__component') == 'blog-components.rich-text' else 1
        for b in enhanced_blocks
    )
    
    # Add CTA/footer if we have substantial content (5+ paragraphs)
    if total_blocks >= 5:
        # Check if footer already exists
        has_footer = any(
            b.get('__component') == 'blog-components.cta-block' 
            for b in enhanced_blocks
        )
        
        if not has_footer:
            # Add a simple CTA block
            enhanced_blocks.append({
                '__component': 'blog-components.cta-block',
                'heading': 'Get Personalized Guidance',
                'text': 'Connect with our expert astrologers for personalized insights and remedies tailored to your birth chart.',
                'buttonText': 'Book Consultation',
                'buttonLink': '/consultation'
            })
    
    return enhanced_blocks

def restructure_content_to_blocks(content, use_seo_optimization=False, sitemap_context=None, current_slug=None, title=None):
    """
    Converts any content format (HTML, markdown, plain text) to structured content blocks.
    If use_seo_optimization is True, also optimizes for SEO with internal links.
    """
    import json
    import re
    from html import unescape
    
    if not content:
        return []
    
    # If SEO optimization is requested, use the enhanced version
    if use_seo_optimization and sitemap_context and current_slug and title:
        return optimize_and_restructure_content(content, sitemap_context, current_slug, title)
    
    # If already structured, return as-is
    if isinstance(content, list):
        # Check if it's already in the correct format
        if content and isinstance(content[0], dict) and content[0].get('__component'):
            return content
        # Otherwise, try to convert
    elif isinstance(content, dict):
        # Check if it's already structured
        if content.get('__component') or content.get('contentBlocks'):
            return [content] if content.get('__component') else content.get('contentBlocks', [])
    
    # Convert to string for processing
    content_str = str(content)
    
    # Check if it's JSON string
    if isinstance(content, str):
        try:
            parsed = json.loads(content)
            if isinstance(parsed, (dict, list)):
                # Recursively process parsed JSON
                return restructure_content_to_blocks(parsed, use_seo_optimization, sitemap_context, current_slug, title)
        except (json.JSONDecodeError, ValueError):
            pass
    
    # Check if content has HTML tags
    has_html = re.search(r'<[a-z][\s\S]*>', content_str, re.IGNORECASE)
    
    if has_html:
        # Parse HTML
        return convert_html_to_content_blocks(content_str, [])
    
    # Check if content has markdown (headings, lists, etc.)
    has_markdown = re.search(r'^#{1,6}\s+|^[-*+]\s+|^\d+\.\s+', content_str, re.MULTILINE)
    
    if has_markdown:
        # Parse markdown
        return parse_markdown_to_blocks(content_str)
    
    # Plain text - convert to paragraphs
    return parse_plain_text_to_blocks(content_str)

def parse_markdown_to_blocks(markdown):
    """Parse markdown to structured blocks."""
    import re
    
    blocks = []
    lines = markdown.split('\n')
    current_paragraph = []
    current_list = []
    list_type = None
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        if not line:
            # Empty line - end current paragraph/list
            if current_paragraph:
                blocks.append({
                    'type': 'paragraph',
                    'children': [{'type': 'text', 'text': ' '.join(current_paragraph)}]
                })
                current_paragraph = []
            if current_list:
                blocks.append({
                    'type': 'list',
                    'format': list_type,
                    'children': [{'children': [{'type': 'text', 'text': item}]} for item in current_list]
                })
                current_list = []
                list_type = None
            i += 1
            continue
        
        # Heading
        heading_match = re.match(r'^(#{1,6})\s+(.+)$', line)
        if heading_match:
            # Save any pending content
            if current_paragraph:
                blocks.append({
                    'type': 'paragraph',
                    'children': [{'type': 'text', 'text': ' '.join(current_paragraph)}]
                })
                current_paragraph = []
            if current_list:
                blocks.append({
                    'type': 'list',
                    'format': list_type,
                    'children': [{'children': [{'type': 'text', 'text': item}]} for item in current_list]
                })
                current_list = []
                list_type = None
            
            level = len(heading_match.group(1))
            text = heading_match.group(2).strip()
            # Parse links in heading
            children = parse_markdown_links(text)
            blocks.append({
                'type': 'heading',
                'level': level,
                'children': children
            })
            i += 1
            continue
        
        # Unordered list
        ul_match = re.match(r'^[-*+]\s+(.+)$', line)
        if ul_match:
            if current_paragraph:
                blocks.append({
                    'type': 'paragraph',
                    'children': [{'type': 'text', 'text': ' '.join(current_paragraph)}]
                })
                current_paragraph = []
            if list_type != 'unordered':
                if current_list:
                    blocks.append({
                        'type': 'list',
                        'format': list_type,
                        'children': [{'children': [{'type': 'text', 'text': item}]} for item in current_list]
                    })
                current_list = []
            list_type = 'unordered'
            text = ul_match.group(1).strip()
            children = parse_markdown_links(text)
            current_list.append(children[0]['text'] if children and children[0].get('type') == 'text' else text)
            i += 1
            continue
        
        # Ordered list
        ol_match = re.match(r'^\d+\.\s+(.+)$', line)
        if ol_match:
            if current_paragraph:
                blocks.append({
                    'type': 'paragraph',
                    'children': [{'type': 'text', 'text': ' '.join(current_paragraph)}]
                })
                current_paragraph = []
            if list_type != 'ordered':
                if current_list:
                    blocks.append({
                        'type': 'list',
                        'format': list_type,
                        'children': [{'children': [{'type': 'text', 'text': item}]} for item in current_list]
                    })
                current_list = []
            list_type = 'ordered'
            text = ol_match.group(1).strip()
            children = parse_markdown_links(text)
            current_list.append(children[0]['text'] if children and children[0].get('type') == 'text' else text)
            i += 1
                    continue
        
        # Regular text
        if current_list:
            # End list, start paragraph
            blocks.append({
                'type': 'list',
                'format': list_type,
                'children': [{'children': [{'type': 'text', 'text': item}]} for item in current_list]
            })
            current_list = []
            list_type = None
        
        # Parse links in line
        children = parse_markdown_links(line)
        if children:
            # If has links, create paragraph with links
            current_paragraph.append(' '.join([c.get('text', '') if c.get('type') == 'text' else c.get('children', [{}])[0].get('text', '') for c in children]))
        else:
            current_paragraph.append(line)
        
        i += 1
    
    # Add remaining content
    if current_paragraph:
        children = parse_markdown_links(' '.join(current_paragraph))
        blocks.append({
            'type': 'paragraph',
            'children': children if children else [{'type': 'text', 'text': ' '.join(current_paragraph)}]
        })
    if current_list:
        blocks.append({
            'type': 'list',
            'format': list_type,
            'children': [{'children': [{'type': 'text', 'text': item}]} for item in current_list]
        })
    
    # Wrap in contentBlocks structure
    if blocks:
        return [{
            '__component': 'blog-components.rich-text',
            'body': blocks
        }]
    
    return []

def parse_markdown_links(text):
    """Parse markdown-style links [text](url) to structured children."""
    import re
    
    children = []
    # Pattern for markdown links: [text](url)
    link_pattern = r'\[([^\]]+)\]\(([^)]+)\)'
    
    last_pos = 0
    for match in re.finditer(link_pattern, text):
        # Text before link
        before = text[last_pos:match.start()]
        if before:
            children.append({'type': 'text', 'text': before})
        
        # Link
        link_text = match.group(1)
        link_url = match.group(2)
        children.append({
            'type': 'link',
            'url': link_url,
            'children': [{'type': 'text', 'text': link_text}]
        })
        
        last_pos = match.end()
    
    # Text after last link
    after = text[last_pos:]
    if after:
        children.append({'type': 'text', 'text': after})
    
    return children if children else [{'type': 'text', 'text': text}]

def parse_plain_text_to_blocks(text):
    """Parse plain text to structured blocks."""
    if not text:
        return []
    
    # Split by double newlines (paragraphs)
    paragraphs = re.split(r'\n{2,}', text.strip())
    
    blocks = []
    for para in paragraphs:
        if para.strip():
            # Check for headings (all caps or specific patterns)
            lines = para.split('\n')
            first_line = lines[0].strip()
            
            # Simple heuristic: if first line is short and ends without period, might be heading
            if len(first_line) < 100 and not first_line.endswith('.') and len(lines) > 1:
                blocks.append({
                    'type': 'heading',
                    'level': 2,
                    'children': [{'type': 'text', 'text': first_line}]
                })
                # Rest as paragraph
                rest = '\n'.join(lines[1:]).strip()
                if rest:
                    blocks.append({
                        'type': 'paragraph',
                        'children': [{'type': 'text', 'text': rest}]
                    })
            else:
                blocks.append({
                    'type': 'paragraph',
                    'children': [{'type': 'text', 'text': para.strip()}]
                })
    
    if blocks:
        return [{
            '__component': 'blog-components.rich-text',
            'body': blocks
        }]
    
    return []

# --- RESTRUCTURE ENDPOINT ---
@router.post("/restructure-content")
def restructure_content_batch(
    req: RestructureRequest,
    background_tasks: BackgroundTasks,
    x_admin_key: str = Header(None)
):
    """
    Restructures content column from HTML/markdown to proper structured content blocks.
    This fixes rendering issues where content shows as raw markdown/HTML.
    """
    if x_admin_key != ADMIN_SECRET:
        raise HTTPException(status_code=403, detail="Unauthorized")
    
    try:
        # Build query
        query = supabase.table("blog_posts").select("*")
        
        if req.article_ids:
            # Restructure specific articles
            query = query.in_("id", req.article_ids)
        else:
            # Get articles that need restructuring (have content but might be malformed)
            query = query.limit(req.batch_size)
        
        response = query.execute()
        articles = response.data if response else []
        
        if not articles:
            return {
                "status": "no_articles",
                "message": "No articles found to restructure.",
                "count": 0
            }
        
        # Get sitemap context for SEO optimization (if enabled)
        sitemap_ctx = ""
        if req.seo_optimize:
            logger.info("📋 Fetching sitemap context for SEO optimization...")
            sitemap_ctx = get_sitemap_context()
            if not sitemap_ctx:
                logger.warning("⚠️ No sitemap context available. SEO optimization may be limited.")
        
        if req.dry_run:
            # Dry run - just show what would be changed
            results = []
            for article in articles:
                original = article.get('content', '')
                slug = article.get('slug', '')
                title = article.get('title', '')
                
                restructured = restructure_content_to_blocks(
                    original,
                    use_seo_optimization=req.seo_optimize,
                    sitemap_context=sitemap_ctx if req.seo_optimize else None,
                    current_slug=slug,
                    title=title
                )
                
                # Count internal links in restructured content
                internal_links_count = 0
                if restructured:
                    for block in restructured:
                        if block.get('__component') == 'blog-components.rich-text':
                            body = block.get('body', [])
                            for item in body:
                                if item.get('type') == 'paragraph' or item.get('type') == 'heading':
                                    children = item.get('children', [])
                                    for child in children:
                                        if child.get('type') == 'link' and child.get('url', '').startswith('/blog/'):
                                            internal_links_count += 1
                
                results.append({
                    "id": article.get('id'),
                    "slug": slug,
                    "title": title,
                    "original_type": type(original).__name__,
                    "restructured_blocks": len(restructured),
                    "internal_links_added": internal_links_count if req.seo_optimize else 0,
                    "preview": restructured[:1] if restructured else []
                })
            
            return {
                "status": "dry_run",
                "message": f"Would restructure {len(articles)} articles (dry run)" + (" with SEO optimization" if req.seo_optimize else ""),
                "count": len(articles),
                "seo_optimized": req.seo_optimize,
                "articles": results
            }
        
        # Actually restructure
        restructured_count = 0
        errors = []
        
        # Rate limiting for AI calls
        for idx, article in enumerate(articles):
            try:
                article_id = article.get('id')
                original_content = article.get('content', '')
                slug = article.get('slug', '')
                title = article.get('title', '')
                
                logger.info(f"🔄 Restructuring article ID {article_id}: {slug}")
                
                # Restructure content (with SEO optimization if enabled)
                restructured_blocks = restructure_content_to_blocks(
                    original_content,
                    use_seo_optimization=req.seo_optimize,
                    sitemap_context=sitemap_ctx if req.seo_optimize else None,
                    current_slug=slug,
                    title=title
                )
                
                if restructured_blocks:
                    # Update database
                    supabase.table("blog_posts").update({
                        "content": restructured_blocks,
                        "updated_at": "now()"
                    }).eq("id", article_id).execute()
                    
                    restructured_count += 1
                    logger.info(f"✅ Restructured article ID {article_id}: {slug}")
                    
                    # Rate limiting for AI calls
                    if req.seo_optimize and idx < len(articles) - 1:
                        time.sleep(GEMINI_RATE_LIMIT_DELAY)
                else:
                    logger.warning(f"⚠️ No blocks generated for article ID {article_id}")
                    errors.append({
                        "id": article_id,
                        "error": "No blocks generated"
                    })
                    
            except Exception as e:
                logger.error(f"❌ Failed to restructure article ID {article.get('id')}: {e}")
                errors.append({
                    "id": article.get('id'),
                    "error": str(e)
                })
                # Continue with next article even if one fails
        
        return {
            "status": "completed",
            "message": f"Restructured {restructured_count} out of {len(articles)} articles",
            "restructured": restructured_count,
            "total": len(articles),
            "errors": errors if errors else None
        }
        
    except Exception as e:
        logger.error(f"Error in restructure_content_batch: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to restructure content: {str(e)}")

# --- SEO OPTIMIZATION ENDPOINT ---
def optimize_content_with_ai(content_html, sitemap_context, current_slug, title):
    """
    Uses AI to optimize content with internal links, alt text, and schema.
    This is the original SEO optimization function.
    """
    try:
        model = get_working_model()
        
        prompt = f"""Act as a Technical SEO Expert.

TASK: Optimize the following HTML Blog Content.

INPUT DATA:
1. **Content:** Provided below.
2. **Internal Link Map:** A list of all other pages on my website.
3. **Current Page:** {current_slug} (DO NOT link to this page itself)
4. **Page Title:** {title}

INSTRUCTIONS:
1. **Internal Linking:** Scan the content. If you see text that is semantically relevant to a page in the 'Link Map', wrap it in an <a href="/blog/slug"> tag.
   - Add 3-8 internal links max.
   - Do NOT link to the current page ({current_slug}).
   - Ensure anchor text is natural.

2. **Image Optimization:** If you find <img> tags without 'alt' attributes, add descriptive, keyword-rich alt text.

3. **Schema:** Append a valid <script type="application/ld+json"> block at the end with 'Article' schema.

LINK MAP (Reference Only):
{sitemap_context[:50000]}

CONTENT TO OPTIMIZE:
{content_html[:50000]}

OUTPUT: Return ONLY the updated HTML content. No markdown ticks, no explanations."""

        response = model.generate_content(prompt)
        optimized_html = response.text.strip()
        
        # Clean up markdown code blocks if present
        optimized_html = optimized_html.replace("```html", "").replace("```", "").strip()
        
        # Convert markdown-style links [text](url) to HTML <a> tags if present
        optimized_html = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2">\1</a>', optimized_html)
        
        return optimized_html.strip()
        
    except Exception as e:
        logger.error(f"AI Optimization Error: {e}")
        raise

def run_optimization_batch(batch_id, batch_size, target_status):
    """
    Background worker for SEO optimization batch processing.
    """
    try:
        logger.info(f"🚀 Starting SEO Optimization Batch {batch_id}...")
        
        # Update batch status
        batch_tracker[batch_id] = {
            "status": "running",
            "started_at": datetime.now().isoformat(),
            "processed": 0,
            "total": 0,
            "errors": []
        }
        
        # 1. Get the Sitemap (Context)
        sitemap_ctx = get_sitemap_context()
        if not sitemap_ctx:
            logger.warning("⚠️ No sitemap context available. Internal linking may be limited.")
        
        # 2. Check if seo_optimized column exists
        try:
            # Try to fetch one row to check column existence
            test_res = supabase.table("blog_posts").select("id, seo_optimized").limit(1).execute()
        except Exception as e:
            logger.warning(f"⚠️ seo_optimized column may not exist: {e}")
            # Continue anyway - we'll handle it
        
        # 3. Fetch Candidates
        query = supabase.table("blog_posts").select("*")
        
        # Filter by target_status if provided
        if target_status:
            query = query.eq("rewrite_status", target_status)
        
        # Filter out already optimized articles (if column exists)
        try:
            query = query.eq("seo_optimized", False)
        except:
            # Column might not exist, continue without filter
            pass
        
        query = query.limit(batch_size)
        
        response = query.execute()
        rows = response.data if response else []
    
    if not rows:
            batch_tracker[batch_id]["status"] = "completed"
            batch_tracker[batch_id]["message"] = "No unoptimized articles found."
            logger.info("✅ No unoptimized articles found.")
            return
        
        batch_tracker[batch_id]["total"] = len(rows)
        
        # 4. Process each article
        for idx, row in enumerate(rows):
            try:
                article_id = row.get('id')
                slug = row.get('slug', '')
                title = row.get('title', '')
                content = row.get('content', '')
                
                logger.info(f"✨ Optimizing: {slug} (ID: {article_id})")
                
                # Extract HTML from content
                content_html = extract_html_from_content(content)
                
                # Optimize with AI
                optimized_html = optimize_content_with_ai(content_html, sitemap_ctx, slug, title)
                
                # Convert optimized HTML back to structured blocks if original was structured
                # Check if original content was structured
                is_structured = isinstance(content, (dict, list)) or (
                    isinstance(content, str) and 
                    (content.strip().startswith('[') or content.strip().startswith('{'))
                )
                
                if is_structured:
                    # Convert back to structured blocks
                    optimized_blocks = convert_html_to_content_blocks(optimized_html, content)
                    final_content = optimized_blocks
                else:
                    # Keep as HTML string
                    final_content = optimized_html
                
                # Update database
                update_data = {
                    "content": final_content,
                    "updated_at": "now()"
                }
                
                # Try to set seo_optimized flag if column exists
                try:
                    update_data["seo_optimized"] = True
                except:
                    pass
                
                supabase.table("blog_posts").update(update_data).eq("id", article_id).execute()
                
                batch_tracker[batch_id]["processed"] += 1
                logger.info(f"✅ Done: {slug}")
                
                # Rate limiting
                if idx < len(rows) - 1:
                    time.sleep(GEMINI_RATE_LIMIT_DELAY)
                    
            except Exception as e:
                error_msg = f"Failed ID {row.get('id')}: {str(e)}"
                logger.error(f"❌ {error_msg}")
                batch_tracker[batch_id]["errors"].append({
                    "id": row.get('id'),
                    "slug": row.get('slug'),
                    "error": str(e)
                })
                time.sleep(5)  # Shorter delay on error
        
        # Mark batch as completed
        batch_tracker[batch_id]["status"] = "completed"
        batch_tracker[batch_id]["completed_at"] = datetime.now().isoformat()
        batch_tracker[batch_id]["message"] = f"Processed {batch_tracker[batch_id]['processed']} out of {batch_tracker[batch_id]['total']} articles"
        
        logger.info(f"✅ Batch {batch_id} completed!")
        
    except Exception as e:
        logger.error(f"❌ Batch {batch_id} Error: {e}", exc_info=True)
        batch_tracker[batch_id]["status"] = "failed"
        batch_tracker[batch_id]["error"] = str(e)
        batch_tracker[batch_id]["completed_at"] = datetime.now().isoformat()

@router.post("/optimize-batch")
def trigger_seo_optimization(
    req: OptimizationRequest,
    background_tasks: BackgroundTasks,
    x_admin_key: str = Header(None)
):
    """
    Triggers SEO optimization batch (internal links, alt text, schema).
    """
    if x_admin_key != ADMIN_SECRET:
        raise HTTPException(status_code=403, detail="Unauthorized")
    
    try:
        # Generate unique batch ID
        batch_id = str(uuid.uuid4())
        
        if req.run_sync:
            # Synchronous mode (for testing, can timeout on large batches)
            logger.warning("⚠️ Running in SYNC mode - this may timeout on large batches!")
            run_optimization_batch(batch_id, req.batch_size, req.target_status)
            return {
                "status": "completed",
                "batch_id": batch_id,
                "message": "SEO optimization completed synchronously.",
                "batch_size": req.batch_size
            }
        else:
            # Asynchronous mode (recommended)
            background_tasks.add_task(run_optimization_batch, batch_id, req.batch_size, req.target_status)
            
            return {
                "status": "queued",
                "batch_id": batch_id,
                "message": f"SEO optimization started. Processing {req.batch_size} articles with status '{req.target_status}'.",
                "batch_size": req.batch_size
            }
            
    except Exception as e:
        logger.error(f"Error triggering SEO optimization: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to trigger SEO optimization: {str(e)}")

@router.get("/batch-status/{batch_id}")
def get_batch_status(batch_id: str, x_admin_key: str = Header(None)):
    """
    Get the status of a specific batch.
    """
    if x_admin_key != ADMIN_SECRET:
        raise HTTPException(status_code=403, detail="Unauthorized")
    
    if batch_id not in batch_tracker:
        raise HTTPException(status_code=404, detail=f"Batch {batch_id} not found")
    
    return batch_tracker[batch_id]

@router.get("/optimize-diagnostic")
def optimize_diagnostic(x_admin_key: str = Header(None)):
    """
    Diagnostic endpoint to check configuration and API connectivity.
    """
    if x_admin_key != ADMIN_SECRET:
        raise HTTPException(status_code=403, detail="Unauthorized")
    
    diagnostic = {
        "timestamp": datetime.now().isoformat(),
        "configuration": {},
        "api_test": {},
        "recommendations": []
    }
    
    # Check environment variables
    diagnostic["configuration"]["GEMINI_API_KEY"] = "✅ Set" if GEMINI_KEY else "❌ Missing"
    diagnostic["configuration"]["SUPABASE_URL"] = "✅ Set" if SUPABASE_URL else "❌ Missing"
    diagnostic["configuration"]["SUPABASE_KEY"] = "✅ Set" if SUPABASE_KEY else "❌ Missing"
    diagnostic["configuration"]["ADMIN_SECRET"] = "✅ Set" if ADMIN_SECRET else "❌ Missing"
    
    # Test Gemini API
    if GEMINI_KEY:
        try:
            model = get_working_model()
            diagnostic["api_test"]["gemini"] = "✅ Working"
            diagnostic["api_test"]["model_name"] = "Initialized successfully"
        except Exception as e:
            diagnostic["api_test"]["gemini"] = f"❌ Failed: {str(e)}"
            diagnostic["recommendations"].append(f"Gemini API issue: {str(e)}")
    else:
        diagnostic["api_test"]["gemini"] = "❌ Skipped (no API key)"
        diagnostic["recommendations"].append("Set GEMINI_API_KEY environment variable")
    
    # Test Supabase connection
    if SUPABASE_URL and SUPABASE_KEY:
        try:
            test_res = supabase.table("blog_posts").select("id").limit(1).execute()
            diagnostic["api_test"]["supabase"] = "✅ Connected"
            diagnostic["api_test"]["article_count"] = "Can query database"
        except Exception as e:
            diagnostic["api_test"]["supabase"] = f"❌ Failed: {str(e)}"
            diagnostic["recommendations"].append(f"Supabase connection issue: {str(e)}")
    else:
        diagnostic["api_test"]["supabase"] = "❌ Skipped (missing credentials)"
    
    # Check for seo_optimized column
    try:
        test_res = supabase.table("blog_posts").select("id, seo_optimized").limit(1).execute()
        diagnostic["configuration"]["seo_optimized_column"] = "✅ Exists"
    except Exception as e:
        if "seo_optimized" in str(e).lower():
            diagnostic["configuration"]["seo_optimized_column"] = "❌ Missing - run migration"
            diagnostic["recommendations"].append("Run migration to add seo_optimized column")
        else:
            diagnostic["configuration"]["seo_optimized_column"] = f"⚠️ Unknown: {str(e)}"
    
    return diagnostic
