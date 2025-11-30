# On-Page SEO Backend System

A comprehensive backend system for generating high-quality on-page SEO data including meta tags, structured data, internal/external links, breadcrumbs, and SEO recommendations.

## Features

✅ **Meta Tags Generation**
- Optimized title tags (50-60 characters)
- Compelling meta descriptions (150-160 characters)
- Keyword-rich meta keywords
- Open Graph tags for social sharing
- Twitter Card tags
- Canonical URLs
- Robots meta tags

✅ **Structured Data (JSON-LD)**
- BlogPosting schema for blog posts
- WebApplication schema for calculators
- Article schema for horoscopes
- BreadcrumbList schema
- Organization schema
- Customizable based on page type

✅ **Internal Links**
- Context-aware internal linking
- Related blog posts
- Calculator cross-links
- Horoscope navigation
- Common site-wide links

✅ **External Links**
- Authoritative source recommendations
- Vedic knowledge sources
- Astrology references
- Wikipedia links
- Properly formatted with rel="noopener noreferrer"

✅ **Breadcrumbs**
- Automatic breadcrumb generation
- BreadcrumbList structured data
- Context-aware navigation

✅ **SEO Recommendations**
- Content length suggestions
- Meta tag optimization tips
- Image optimization recommendations
- Keyword placement advice

## API Endpoints

### 1. Generate SEO Data (POST)

**Endpoint:** `/api/seo/onpage/generate`

**Request Body:**
```json
{
  "page_type": "blog",
  "page_url": "https://astrologyapp.in/blog/vedic-astrology-guide",
  "page_title": "Complete Guide to Vedic Astrology",
  "page_content": "Vedic astrology is an ancient system...",
  "page_id": "123",
  "primary_keyword": "vedic astrology",
  "secondary_keywords": ["jyotish", "hindu astrology"],
  "image_url": "https://astrologyapp.in/images/vedic-astrology.jpg",
  "author": "AstrologyApp Team",
  "publish_date": "2024-01-15"
}
```

**Response:**
```json
{
  "meta_tags": {
    "title": "Complete Guide to Vedic Astrology | AstrologyApp",
    "description": "Learn about Vedic astrology...",
    "keywords": "vedic astrology, jyotish, hindu astrology",
    "og_title": "...",
    "og_description": "...",
    "og_image": "...",
    "canonical": "..."
  },
  "structured_data": {
    "@context": "https://schema.org",
    "@type": "BlogPosting",
    ...
  },
  "internal_links": [
    {
      "url": "https://astrologyapp.in/calculators/birth-chart",
      "text": "Birth Chart Calculator",
      "anchor": "Birth Chart"
    }
  ],
  "external_links": [
    {
      "url": "https://www.sanatan.org",
      "text": "Sanatan.org",
      "anchor": "Vedic Knowledge"
    }
  ],
  "breadcrumbs": [
    {"name": "Home", "url": "https://astrologyapp.in/"},
    {"name": "Blog", "url": "https://astrologyapp.in/blog"},
    {"name": "Complete Guide to Vedic Astrology", "url": "..."}
  ],
  "recommendations": [
    "Add more content (aim for at least 300 words)",
    "Meta description should be 120-160 characters"
  ]
}
```

### 2. Get SEO for Page (GET)

**Endpoint:** `/api/seo/onpage/page/{page_type}?url={url}&page_id={page_id}`

Fetches page data from database and generates SEO automatically.

**Example:**
```
GET /api/seo/onpage/page/blog?url=https://astrologyapp.in/blog/my-post&page_id=123
```

### 3. Health Check

**Endpoint:** `/api/seo/onpage/health`

Returns service status and available features.

## Frontend Usage

### Using the React Hook

```javascript
import { useSEO } from '../lib/hooks/useSEO';

function BlogPost({ post }) {
  const { seoData, loading, error } = useSEO({
    pageType: 'blog',
    pageId: post.id,
    pageTitle: post.title,
    pageContent: post.content,
    primaryKeyword: post.primaryKeyword,
    imageUrl: post.coverImage?.url,
    author: post.author?.name,
    publishDate: post.publishedAt,
    autoFetch: true,
  });

  if (loading) return <div>Loading SEO data...</div>;
  if (error) return <div>Error: {error}</div>;

  return (
    <div>
      {/* Your content */}
      {seoData?.internal_links && (
        <InternalLinks links={seoData.internal_links} />
      )}
    </div>
  );
}
```

### Using the SEO Component

```javascript
import OnPageSEO, { InternalLinks, ExternalLinks } from '../components/SEO/OnPageSEO';

function BlogPost({ post }) {
  return (
    <>
      <OnPageSEO
        pageType="blog"
        pageId={post.id}
        pageTitle={post.title}
        pageContent={post.content}
        primaryKeyword={post.primaryKeyword}
        imageUrl={post.coverImage?.url}
        author={post.author?.name}
        publishDate={post.publishedAt}
      />
      
      <article>
        {/* Your blog content */}
      </article>

      <InternalLinks links={seoData?.internal_links} />
      <ExternalLinks links={seoData?.external_links} />
    </>
  );
}
```

### Manual API Call

```javascript
const response = await fetch('/api/seo/onpage', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    page_type: 'calculator',
    page_url: 'https://astrologyapp.in/calculators/birth-chart',
    page_title: 'Birth Chart Calculator',
    page_content: 'Calculate your birth chart...',
    primary_keyword: 'birth chart calculator',
  }),
});

const { data } = await response.json();
```

## Page Types Supported

- **blog** - Blog posts with BlogPosting schema
- **calculator** - Calculator pages with WebApplication schema
- **horoscope** - Horoscope pages with Article schema
- **compatibility** - Compatibility pages
- **general** - General web pages

## Environment Variables

```bash
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key
GEMINI_API_KEY=your_gemini_key
BASE_URL=https://astrologyapp.in
```

## Best Practices

1. **Always provide page_type** - This determines the structured data schema
2. **Include primary_keyword** - Helps generate better meta tags
3. **Provide page_content** - AI uses this to generate better descriptions
4. **Use page_id for blogs** - Enables automatic related post linking
5. **Include image_url** - Better social sharing with OG images
6. **Review recommendations** - Follow SEO recommendations for better rankings

## Integration Examples

### Blog Post Page
```javascript
<OnPageSEO
  pageType="blog"
  pageId={post.id}
  pageTitle={post.title}
  pageContent={post.content}
  primaryKeyword={post.primaryKeyword}
  imageUrl={post.coverImage?.url}
  author={post.author?.name}
  publishDate={post.publishedAt}
/>
```

### Calculator Page
```javascript
<OnPageSEO
  pageType="calculator"
  pageTitle="Birth Chart Calculator"
  pageContent="Calculate your Vedic birth chart..."
  primaryKeyword="birth chart calculator"
  pageUrl="https://astrologyapp.in/calculators/birth-chart"
/>
```

### Horoscope Page
```javascript
<OnPageSEO
  pageType="horoscope"
  pageTitle="Today's Horoscope for Aries"
  pageContent={horoscopeContent}
  primaryKeyword="aries horoscope today"
  pageUrl="https://astrologyapp.in/horoscope/today-horoscope/aries"
/>
```

## Notes

- The system uses AI (Gemini) to generate optimized meta tags
- Internal links are automatically fetched from database for blog posts
- External links are curated authoritative sources
- Breadcrumbs are automatically generated based on URL structure
- All structured data follows Schema.org standards
- Meta tags are optimized for search engines and social media

