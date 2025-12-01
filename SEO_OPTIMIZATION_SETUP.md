# SEO Optimization System Setup

This document explains how to use the **Smart SEO Polisher** system that adds internal links, image alt text, and JSON-LD schema to your 800+ blog articles.

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
  "target_status": "completed"
}
```

**Parameters:**
- `batch_size` (int, default: 5): Number of articles to process per batch
- `target_status` (string, default: "completed"): Only optimize articles with this rewrite_status

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

### Check Unoptimized Articles

```sql
SELECT COUNT(*) 
FROM blog_posts 
WHERE seo_optimized = FALSE 
AND rewrite_status = 'completed';
```

### Check Optimized Articles

```sql
SELECT COUNT(*) 
FROM blog_posts 
WHERE seo_optimized = TRUE;
```

### View Recent Optimizations

```sql
SELECT id, slug, title, seo_optimized, updated_at
FROM blog_posts
WHERE seo_optimized = TRUE
ORDER BY updated_at DESC
LIMIT 10;
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

### Column Doesn't Exist Error

If you see errors about `seo_optimized` column:
1. Run the migration SQL (see Database Setup above)
2. The system will gracefully fall back to using `rewrite_status` only

### No Articles Found

If no articles are returned:
- Check that articles have `rewrite_status = 'completed'`
- Verify the `target_status` parameter matches your data
- Check if all articles are already optimized (`seo_optimized = TRUE`)

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

