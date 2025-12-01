-- Migration: Add seo_optimized column to blog_posts table
-- Purpose: Track which articles have been optimized with internal links, alt text, and schema
-- Date: 2025-01-XX

-- Add a flag to track internal linking and SEO optimization status
ALTER TABLE public.blog_posts 
ADD COLUMN IF NOT EXISTS seo_optimized BOOLEAN DEFAULT FALSE;

-- Create index for faster queries on unoptimized articles
CREATE INDEX IF NOT EXISTS idx_blog_posts_seo_optimized ON public.blog_posts(seo_optimized) 
WHERE seo_optimized = FALSE;

-- Optional: Add a comment to document the column
COMMENT ON COLUMN public.blog_posts.seo_optimized IS 
'Flag indicating if the article has been optimized with internal links, image alt text, and JSON-LD schema';

