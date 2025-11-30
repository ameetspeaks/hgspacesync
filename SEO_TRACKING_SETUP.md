# SEO Tracking & Analytics System

## Overview

The SEO Tracking system automatically monitors and tracks SEO performance for all pages on your site. It provides:

- ✅ **SEO Score Calculation** (0-100 based on multiple factors)
- ✅ **Page Tracking** (tracks all pages automatically)
- ✅ **Keyword Ranking Tracking**
- ✅ **Google Index Status**
- ✅ **Performance Metrics** (impressions, clicks, CTR, position)
- ✅ **Dashboard** for viewing all metrics

## Database Setup

You need to create two tables in Supabase:

### 1. `seo_tracking` Table

```sql
CREATE TABLE IF NOT EXISTS seo_tracking (
  id BIGSERIAL PRIMARY KEY,
  page_url TEXT UNIQUE NOT NULL,
  page_type TEXT NOT NULL,
  page_id TEXT,
  title TEXT,
  meta_description TEXT,
  primary_keyword TEXT,
  word_count INTEGER DEFAULT 0,
  internal_links_count INTEGER DEFAULT 0,
  external_links_count INTEGER DEFAULT 0,
  images_count INTEGER DEFAULT 0,
  has_structured_data BOOLEAN DEFAULT false,
  seo_score FLOAT DEFAULT 0,
  google_indexed BOOLEAN,
  google_rank INTEGER,
  impressions INTEGER DEFAULT 0,
  clicks INTEGER DEFAULT 0,
  ctr FLOAT,
  avg_position FLOAT,
  last_crawled TIMESTAMP,
  created_at TIMESTAMP DEFAULT NOW(),
  last_updated TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_seo_tracking_page_url ON seo_tracking(page_url);
CREATE INDEX IF NOT EXISTS idx_seo_tracking_page_type ON seo_tracking(page_type);
CREATE INDEX IF NOT EXISTS idx_seo_tracking_seo_score ON seo_tracking(seo_score DESC);
```

### 2. `seo_keyword_rankings` Table

```sql
CREATE TABLE IF NOT EXISTS seo_keyword_rankings (
  id BIGSERIAL PRIMARY KEY,
  keyword TEXT NOT NULL,
  page_url TEXT NOT NULL,
  rank INTEGER,
  search_volume INTEGER,
  difficulty FLOAT,
  tracked_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_seo_keyword_rankings_page_url ON seo_keyword_rankings(page_url);
CREATE INDEX IF NOT EXISTS idx_seo_keyword_rankings_keyword ON seo_keyword_rankings(keyword);
CREATE INDEX IF NOT EXISTS idx_seo_keyword_rankings_tracked_at ON seo_keyword_rankings(tracked_at DESC);
```

## How It Works

### Automatic Tracking

When you use the `OnPageSEO` component, it automatically tracks SEO metrics:

```javascript
import OnPageSEO from '../components/SEO/OnPageSEO';

// This automatically tracks the page
<OnPageSEO
  pageType="blog"
  pageTitle="My Blog Post"
  pageContent="..."
  primaryKeyword="vedic astrology"
/>
```

### Manual Tracking

You can also track pages manually:

```javascript
import { trackSEOPage } from '../lib/hooks/useSEOTracking';

await trackSEOPage({
  page_url: 'https://astrologyapp.in/blog/my-post',
  page_type: 'blog',
  page_title: 'My Blog Post',
  page_content: '...',
  primary_keyword: 'vedic astrology',
  word_count: 800,
  internal_links_count: 5,
  external_links_count: 2,
  images_count: 3,
  has_structured_data: true,
});
```

## SEO Score Calculation

The SEO score (0-100) is calculated based on:

- **Title Optimization** (15 points) - 50-60 characters optimal
- **Meta Description** (15 points) - 120-160 characters optimal
- **Content Length** (20 points) - 1000+ words = full points
- **Internal Links** (15 points) - 5+ links = full points
- **External Links** (10 points) - 2+ links = full points
- **Images** (10 points) - 3+ images = full points
- **Structured Data** (15 points) - Present = full points

## Dashboard Access

Visit: `/admin/seo-dashboard`

The dashboard shows:
- Total pages tracked
- Average SEO score
- Indexed pages percentage
- Pages with structured data
- Performance by page type
- Detailed page list with scores

## API Endpoints

### Track a Page
```
POST /api/seo/tracking?endpoint=track
Body: {
  page_url: "...",
  page_type: "blog",
  ...
}
```

### Get Dashboard Data
```
GET /api/seo/tracking?endpoint=dashboard&page_type=blog
```

### Get Page Status
```
GET /api/seo/tracking?endpoint=page/{encoded_url}
```

### Update SEO Status (from Google Search Console)
```
POST /api/seo/tracking?endpoint=status
Body: {
  page_url: "...",
  google_indexed: true,
  google_rank: 5,
  impressions: 1000,
  clicks: 50,
  ctr: 5.0,
  avg_position: 3.5
}
```

### Track Keyword Ranking
```
POST /api/seo/tracking?endpoint=ranking
Body: {
  keyword: "vedic astrology",
  page_url: "...",
  rank: 3,
  search_volume: 10000,
  difficulty: 45.5
}
```

## Integration with Google Search Console

You can integrate with Google Search Console API to automatically update status:

```javascript
// Example: Fetch from Google Search Console and update
const gscData = await fetchFromGoogleSearchConsole(pageUrl);

await updateSEOStatus({
  page_url: pageUrl,
  google_indexed: gscData.indexed,
  impressions: gscData.impressions,
  clicks: gscData.clicks,
  ctr: gscData.ctr,
  avg_position: gscData.avgPosition,
});
```

## Monitoring & Alerts

You can set up monitoring to:
- Alert when SEO score drops below threshold
- Track ranking changes
- Monitor indexing status
- Track keyword position changes

## Best Practices

1. **Track all pages** - Use OnPageSEO component on all pages
2. **Update regularly** - Sync with Google Search Console weekly
3. **Monitor trends** - Check dashboard regularly
4. **Fix low scores** - Address pages with SEO score < 60
5. **Track keywords** - Monitor important keyword rankings

## Next Steps

1. **Create database tables** (run SQL above in Supabase)
2. **Add OnPageSEO to pages** - Start with blog posts
3. **Access dashboard** - Visit `/admin/seo-dashboard`
4. **Set up Google Search Console sync** (optional)
5. **Monitor and improve** - Use insights to optimize pages

