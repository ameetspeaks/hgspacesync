# SEO Optimization System Setup

This document explains how to use the **Smart SEO Polisher** system that adds internal links, image alt text, and JSON-LD schema to your 800+ blog articles.

## Quick Start - Check Status

**Check overall optimization progress:**
```bash
curl -X GET "https://ameetspeaks-astrologyapp.hf.space/api/seo/optimize-status" \
     -H "x-admin-key: YOUR_SECRET"
```

**Check specific article:**
```bash
curl -X GET "https://ameetspeaks-astrologyapp.hf.space/api/seo/optimize-status/123" \
     -H "x-admin-key: YOUR_SECRET"
```

## Overview

The SEO optimization system uses **Gemini AI** with its massive context window to:
1. **Add Internal Links**: Automatically links relevant text to other articles on your site
2. **Add Image Alt Text**: Ensures all images have descriptive, keyword-rich alt attributes
3. **Add JSON-LD Schema**: Injects Article schema for better Google rich snippets

## Database Setup

### Step 1: Run the Migration

Execute the SQL migration to add the `seo_optimized` column:

```sql
-- Run this in your Supabase SQL Editor
ALTER TABLE public.blog_posts 
ADD COLUMN IF NOT EXISTS seo_optimized BOOLEAN DEFAULT FALSE;

CREATE INDEX IF NOT EXISTS idx_blog_posts_seo_optimized 
ON public.blog_posts(seo_optimized) 
WHERE seo_optimized = FALSE;
```

Or use the migration file:
```bash
# In Supabase Dashboard > SQL Editor, paste the contents of:
migrations/add_seo_optimized_column.sql
```

## API Usage

### Endpoint

**POST** `/api/seo/optimize-batch`

### Authentication

Include the admin secret in the header:
```
x-admin-key: YOUR_ADMIN_SECRET
```

### Request Body

```json
{
  "batch_size": 5,
  "target_status": "completed",
  "run_sync": false
}
```

**Parameters:**
- `batch_size` (int, default: 5): Number of articles to process per batch
- `target_status` (string, default: "completed"): Only optimize articles with this rewrite_status
- `run_sync` (bool, default: false): If true, runs synchronously and returns immediate results. **Use only for testing with small batches (1-3 articles)** as it can timeout on larger batches.

### Response

```json
{
  "status": "queued",
  "message": "SEO optimization started. Processing 5 articles with status 'completed'.",
  "batch_size": 5
}
```

## How It Works

1. **Sitemap Context**: The system fetches ALL blog post titles and slugs to build a comprehensive link map
2. **AI Processing**: Gemini AI analyzes each article and:
   - Identifies semantically relevant text that should link to other articles
   - Finds images without alt text and adds descriptive alt attributes
   - Generates and appends JSON-LD Article schema
3. **Batch Processing**: Processes articles in batches to avoid rate limits
4. **Status Tracking**: Uses `seo_optimized` flag to track which articles have been processed

## Example Usage

### Using cURL

```bash
curl -X POST "https://ameetspeaks-astrologyapp.hf.space/api/seo/optimize-batch" \
     -H "Content-Type: application/json" \
     -H "x-admin-key: YOUR_SECRET" \
     -d '{"batch_size": 5}'
```

### Using Python

```python
import requests

url = "https://ameetspeaks-astrologyapp.hf.space/api/seo/optimize-batch"
headers = {
    "Content-Type": "application/json",
    "x-admin-key": "YOUR_SECRET"
}
data = {
    "batch_size": 5,
    "target_status": "completed"
}

response = requests.post(url, json=data, headers=headers)
print(response.json())
```

### Using JavaScript/Node.js

```javascript
const response = await fetch('https://ameetspeaks-astrologyapp.hf.space/api/seo/optimize-batch', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'x-admin-key': 'YOUR_SECRET'
  },
  body: JSON.stringify({
    batch_size: 5,
    target_status: 'completed'
  })
});

const result = await response.json();
console.log(result);
```

## Automation

### GitHub Actions

A GitHub Actions workflow is already set up at `.github/workflows/seo_optimization.yml`.

#### Setup Instructions:

1. **Add Repository Secrets:**
   - Go to: Settings → Secrets and variables → Actions → Secrets
   - Add `ADMIN_SECRET` with your admin secret key
   - (Optional) Add `HF_SPACE_URL` with your full Space URL (e.g., `https://ameetspeaks-astrologyapp.hf.space`)

2. **Add Repository Variables (if not using HF_SPACE_URL):**
   - Go to: Settings → Secrets and variables → Actions → Variables
   - Add `HF_USERNAME` (e.g., `ameetspeaks`)
   - Add `SPACE_NAME` (e.g., `astrologyapp`)
   - The workflow will construct the URL automatically

3. **The workflow will:**
   - Run automatically every 6 hours (configurable in the workflow file)
   - Can be triggered manually from Actions tab
   - Process 10 articles per batch (configurable when manually triggered)
   - Log response status and body for monitoring

#### Manual Trigger:

1. Go to: Actions → SEO Optimization Batch → Run workflow
2. Optionally set a custom `batch_size`
3. Click "Run workflow"

#### Customize Schedule:

Edit `.github/workflows/seo_optimization.yml` and change the cron schedule:
```yaml
schedule:
  - cron: '0 */6 * * *'  # Every 6 hours
  # Examples:
  # '0 */3 * * *'  - Every 3 hours
  # '0 0 * * *'    - Daily at midnight
  # '0 9,21 * * *' - Twice daily at 9 AM and 9 PM
```

### Manual Batch Processing

For 800+ articles, process in batches:

```bash
# Process 10 articles at a time, run multiple times
for i in {1..80}; do
  echo "Batch $i"
  curl -X POST "https://ameetspeaks-astrologyapp.hf.space/api/seo/optimize-batch" \
       -H "Content-Type: application/json" \
       -H "x-admin-key: YOUR_SECRET" \
       -d '{"batch_size": 10}'
  sleep 300  # Wait 5 minutes between batches
done
```

## Monitoring Progress

### Method 1: API Endpoints (Recommended)

#### Get Overall Status

Check the overall optimization progress across all articles:

```bash
curl -X GET "https://ameetspeaks-astrologyapp.hf.space/api/seo/optimize-status" \
     -H "x-admin-key: YOUR_SECRET"
```

**Response:**
```json
{
  "total_articles": 800,
  "optimized": 245,
  "unoptimized": 555,
  "optimization_percentage": 30.63,
  "ready_for_optimization": 555,
  "note": "seo_optimized column exists"
}
```

#### Get Status for Specific Article

Check if a specific article has been optimized:

```bash
curl -X GET "https://ameetspeaks-astrologyapp.hf.space/api/seo/optimize-status/123" \
     -H "x-admin-key: YOUR_SECRET"
```

**Response:**
```json
{
  "id": 123,
  "slug": "article-slug",
  "title": "Article Title",
  "seo_optimized": true,
  "rewrite_status": "completed",
  "last_updated": "2025-01-15T10:30:00Z",
  "is_ready": false
}
```

#### Python Example

```python
import requests

url = "https://ameetspeaks-astrologyapp.hf.space/api/seo/optimize-status"
headers = {"x-admin-key": "YOUR_SECRET"}

response = requests.get(url, headers=headers)
status = response.json()

print(f"Total Articles: {status['total_articles']}")
print(f"Optimized: {status['optimized']}")
print(f"Unoptimized: {status['unoptimized']}")
print(f"Progress: {status['optimization_percentage']}%")
```

#### JavaScript Example

```javascript
const response = await fetch('https://ameetspeaks-astrologyapp.hf.space/api/seo/optimize-status', {
  headers: {
    'x-admin-key': 'YOUR_SECRET'
  }
});

const status = await response.json();
console.log(`Progress: ${status.optimization_percentage}% (${status.optimized}/${status.total_articles})`);
```

### Method 2: SQL Queries (Direct Database)

#### Check Overall Status

```sql
-- Total articles
SELECT COUNT(*) as total_articles FROM blog_posts;

-- Optimized articles
SELECT COUNT(*) as optimized 
FROM blog_posts 
WHERE seo_optimized = TRUE;

-- Unoptimized articles (ready for optimization)
SELECT COUNT(*) as unoptimized 
FROM blog_posts 
WHERE seo_optimized = FALSE 
AND rewrite_status = 'completed';

-- Optimization progress percentage
SELECT 
  COUNT(*) FILTER (WHERE seo_optimized = TRUE) as optimized,
  COUNT(*) FILTER (WHERE seo_optimized = FALSE AND rewrite_status = 'completed') as unoptimized,
  COUNT(*) as total,
  ROUND(
    COUNT(*) FILTER (WHERE seo_optimized = TRUE)::numeric / 
    NULLIF(COUNT(*), 0) * 100, 
    2
  ) as optimization_percentage
FROM blog_posts;
```

#### View Recent Optimizations

```sql
SELECT id, slug, title, seo_optimized, updated_at
FROM blog_posts
WHERE seo_optimized = TRUE
ORDER BY updated_at DESC
LIMIT 10;
```

#### Find Articles Ready for Optimization

```sql
SELECT id, slug, title, rewrite_status
FROM blog_posts
WHERE rewrite_status = 'completed'
AND (seo_optimized = FALSE OR seo_optimized IS NULL)
ORDER BY id
LIMIT 20;
```

#### Check Specific Article

```sql
SELECT 
  id, 
  slug, 
  title, 
  seo_optimized, 
  rewrite_status, 
  updated_at
FROM blog_posts
WHERE id = 123;  -- Replace with your article ID
```

## Features

### 1. Smart Internal Linking
- Uses entire sitemap (800+ articles) as context
- AI identifies semantic relevance automatically
- Adds 3-8 natural internal links per article
- Never links to the current article itself

### 2. Image Alt Text
- Automatically detects images without alt attributes
- Adds descriptive, keyword-rich alt text
- Improves accessibility and SEO

### 3. JSON-LD Schema
- Generates Article schema automatically
- Includes publisher, author, and metadata
- Improves Google rich snippet eligibility

## Rate Limiting

The system includes:
- **10-second delay** between articles
- **Retry logic** for API rate limits
- **Exponential backoff** for ResourceExhausted errors
- **Batch processing** to avoid overwhelming the API

## Troubleshooting

### Task Queued But No Articles Optimized

If the task is queued successfully but no articles are being optimized:

1. **Run Diagnostic Endpoint:**
   ```bash
   curl -X GET "https://ameetspeaks-astrologyapp.hf.space/api/seo/optimize-diagnostic" \
        -H "x-admin-key: YOUR_SECRET"
   ```
   This shows database state, article counts, and recommendations.

2. **Test Synchronously (for debugging):**
   ```bash
   curl -X POST "https://ameetspeaks-astrologyapp.hf.space/api/seo/optimize-batch" \
        -H "Content-Type: application/json" \
        -H "x-admin-key: YOUR_SECRET" \
        -d '{"batch_size": 1, "run_sync": true}'
   ```
   This runs immediately and shows errors. Use small batch_size (1-3) for testing.

3. **Check Common Issues:**
   - Articles must have `rewrite_status = 'completed'` first
   - Run `/api/seo/rewrite-batch` if articles aren't rewritten yet
   - Check Hugging Face Space logs for detailed error messages
   - Verify `seo_optimized` column exists (run migration if needed)

See `TROUBLESHOOTING.md` for detailed troubleshooting guide.

### Column Doesn't Exist Error

If you see errors about `seo_optimized` column:
1. Run the migration SQL (see Database Setup above)
2. The system will gracefully fall back to using `rewrite_status` only

### No Articles Found

If no articles are returned:
- Check that articles have `rewrite_status = 'completed'`
- Verify the `target_status` parameter matches your data
- Check if all articles are already optimized (`seo_optimized = TRUE`)
- Run diagnostic endpoint to see detailed breakdown

### API Rate Limits

If you hit rate limits:
- Reduce `batch_size` (try 3-5 instead of 10)
- Increase delay between batches
- The system automatically retries with exponential backoff

## Best Practices

1. **Start Small**: Begin with `batch_size: 5` to test
2. **Monitor Results**: Check a few optimized articles manually
3. **Gradual Rollout**: Process in batches over time
4. **Review Sample**: Always review a sample of optimized content
5. **Backup First**: Consider backing up your database before bulk processing

## Technical Details

- **Model**: Uses `get_working_model()` which tries:
  - gemini-2.0-flash
  - gemini-2.0-flash-lite
  - gemini-1.5-flash
  - gemini-pro (fallback)
- **Context Window**: Leverages Gemini's large context (1M+ tokens) for sitemap
- **Content Handling**: Supports JSONB, string, and nested content formats
- **Error Handling**: Comprehensive error handling with logging

## Support

For issues or questions:
1. Check logs in Hugging Face Space logs
2. Verify database schema matches expected structure
3. Ensure environment variables are set correctly:
   - `SUPABASE_URL`
   - `SUPABASE_KEY`
   - `GEMINI_API_KEY`
   - `ADMIN_SECRET`

