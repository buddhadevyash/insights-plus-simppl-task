from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional, Any, Union
from datetime import datetime, date, timedelta
import json, logging, os
from pathlib import Path
from collections import defaultdict, Counter
import re
from functools import lru_cache
import asyncio
from concurrent.futures import ThreadPoolExecutor
from groq import Groq

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Social Media Insights API",
    description="Optimized API for social media analytics with 6 visualization endpoints",
    version="3.1.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load .env file from your path
load_dotenv("/home/y21tbh/Documents/insights-plus/insights-plus-simppl-task/backend/app/.env")

# Read credentials

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

DATA_DIR = Path(__file__).parent.parent / "json_data"

# Initialize Groq client
try:
    client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
except Exception as e:
    logger.error(f"Failed to initialize Groq client: {e}")
    client = None

# Thread pool for async operations
executor = ThreadPoolExecutor(max_workers=4)

# ------------------ MODELS ------------------
class DateRange(BaseModel):
    start_date: date
    end_date: date
    
    @validator('end_date')
    def end_date_must_be_after_start_date(cls, v, values):
        if 'start_date' in values and v < values['start_date']:
            raise ValueError('end_date must be after start_date')
        return v

class PlatformRequest(BaseModel):
    platform: str = Field(..., pattern="^(youtube|reddit|both)$")
    date_range: DateRange

# Response Models for Each Visualization
class PlatformDistribution(BaseModel):
    platform: str
    posts_count: int
    percentage: float

class EngagementTrend(BaseModel):
    month: str
    youtube_engagement: int
    reddit_engagement: int
    total_engagement: int

class TopPost(BaseModel):
    rank: int
    title: str
    content: str
    engagement_score: int
    platform: str
    url: str

class InfluentialUser(BaseModel):
    rank: int
    username: str
    total_engagement: int
    posts_count: int
    avg_engagement: float
    platform: str

class TrendingKeyword(BaseModel):
    keyword: str
    frequency: int
    platforms: List[str]

class PostsHistogram(BaseModel):
    time_period: str  # Week or Month
    youtube_posts: int
    reddit_posts: int
    total_posts: int

# Main Dashboard Response
class DashboardResponse(BaseModel):
    platform_distribution: List[PlatformDistribution]
    engagement_trends: List[EngagementTrend]
    top_posts: List[TopPost]
    influential_users: List[InfluentialUser]
    trending_keywords: List[TrendingKeyword]
    posts_histogram: List[PostsHistogram]

# ------------------ DATA LOADING AND CACHING ------------------
class DataCache:
    def __init__(self):
        self._cache = {}
        self._timestamps = {}
    
    def get(self, filename: str):
        filepath = DATA_DIR / filename
        if not filepath.exists():
            logger.warning(f"File not found: {filepath}")
            return []
        
        current_mtime = filepath.stat().st_mtime
        
        # Check if cached version is still valid
        if filename in self._cache and self._timestamps.get(filename) == current_mtime:
            return self._cache[filename]
        
        # Load fresh data
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self._cache[filename] = data
                self._timestamps[filename] = current_mtime
                logger.info(f"Loaded {len(data)} items from {filename}")
                return data
        except Exception as e:
            logger.error(f"Error loading {filename}: {e}")
            return []

# Global cache instance
data_cache = DataCache()

# ------------------ UTILITY FUNCTIONS ------------------
def parse_date_safely(date_str: Any) -> Optional[date]:
    """Parse date string safely with multiple format support"""
    if not date_str:
        return None
    
    try:
        date_str = str(date_str).strip()
        
        # Handle Unix timestamps
        if date_str.isdigit() and len(date_str) >= 10:
            timestamp = int(date_str)
            if timestamp > 1000000000:  # Valid Unix timestamp
                return datetime.fromtimestamp(timestamp).date()
        
        # Handle ISO format dates
        if "-" in date_str:
            date_part = date_str.split(" ")[0].split("T")[0]  # Remove time part
            if len(date_part) >= 8:
                return datetime.strptime(date_part[:10], "%Y-%m-%d").date()
        
        return None
    except Exception as e:
        logger.debug(f"Date parsing error for '{date_str}': {e}")
        return None

def determine_platform(item: Dict) -> str:
    """Determine platform from item data"""
    # Check for Reddit-specific fields
    reddit_indicators = ['subreddit', 'permalink', 'subreddit_id', 'subreddit_name_prefixed']
    if any(field in item for field in reddit_indicators):
        return 'Reddit'
    
    # Check for YouTube-specific fields
    youtube_indicators = ['video_id', 'channel_id', 'channel_title', 'video_url']
    if any(field in item for field in youtube_indicators):
        return 'YouTube'
    
    # Check URL patterns
    url = item.get('url', '') or item.get('permalink', '')
    if 'reddit.com' in str(url) or '/r/' in str(url):
        return 'Reddit'
    elif 'youtube.com' in str(url) or 'youtu.be' in str(url):
        return 'YouTube'
    
    # Fallback to filename-based detection would happen at data loading level
    return 'Unknown'

def get_engagement_score(item: Dict) -> int:
    """Calculate comprehensive engagement score with improved logic"""
    score = 0
    
    # YouTube reactions
    reactions = item.get('reactions')
    if reactions:
        try:
            if isinstance(reactions, str):
                # Handle both single and double quotes
                reactions = reactions.replace("'", '"')
                reactions = json.loads(reactions)
            if isinstance(reactions, dict):
                likes = int(reactions.get('likes', 0))
                dislikes = int(reactions.get('dislikes', 0))
                score += max(0, likes - dislikes)
        except Exception as e:
            logger.debug(f"Reactions parsing error: {e}")
    
    # Direct likes/dislikes fields
    likes = item.get('likes') or item.get('like_count') or 0
    dislikes = item.get('dislikes') or item.get('dislike_count') or 0
    if likes or dislikes:
        score += max(0, int(likes) - int(dislikes))
    
    # Reddit votes
    upvotes = int(item.get('upvotes', 0) or item.get('ups', 0) or 0)
    downvotes = int(item.get('downvotes', 0) or item.get('downs', 0) or 0)
    score += max(0, upvotes - downvotes)
    
    # Reddit score field
    reddit_score = item.get('score')
    if reddit_score is not None:
        score += max(0, int(reddit_score))
    
    # Comments boost engagement
    comment_count = (item.get('comment_count') or item.get('num_comments') or 
                    item.get('reply_count') or item.get('comments', 0))
    if comment_count:
        score += int(comment_count) * 2
    
    # Views (scaled down)
    views = item.get('view_count') or item.get('views') or item.get('view_count_text')
    if views:
        try:
            # Handle view count text like "1.2M views"
            if isinstance(views, str):
                views = views.replace(',', '').replace(' views', '').replace(' view', '')
                if 'K' in views:
                    views = float(views.replace('K', '')) * 1000
                elif 'M' in views:
                    views = float(views.replace('M', '')) * 1000000
                else:
                    views = float(views)
            score += int(views) // 100  # Scale down views
        except:
            pass
    
    return max(0, score)

def filter_by_date_range(data: List[Dict], start_date: date, end_date: date, source_platform: str = None) -> List[Dict]:
    """Filter data by date range and add platform info"""
    date_fields = ['date_of_post', 'published_date', 'created_utc', 'date_of_comment', 'published_at', 'timestamp']
    filtered = []
    
    for item in data:
        # Add platform info based on source if not determinable from content
        if source_platform and determine_platform(item) == 'Unknown':
            item['_source_platform'] = source_platform
        
        item_date = None
        for field in date_fields:
            if field in item and item[field]:
                item_date = parse_date_safely(item[field])
                if item_date:
                    break
        
        # If no date found, include item (might be recent data without proper dates)
        if item_date is None or (start_date <= item_date <= end_date):
            filtered.append(item)
    
    logger.info(f"Filtered {len(filtered)} items from {len(data)} for platform {source_platform}")
    return filtered

def get_platform_from_item(item: Dict) -> str:
    """Get platform with fallback to source platform"""
    platform = determine_platform(item)
    if platform == 'Unknown':
        platform = item.get('_source_platform', 'Unknown')
    return platform

# ------------------ VISUALIZATION DATA GENERATORS ------------------
async def generate_platform_distribution(yt_posts: List[Dict], rd_posts: List[Dict]) -> List[PlatformDistribution]:
    """Generate platform distribution pie chart data"""
    youtube_count = len(yt_posts)
    reddit_count = len(rd_posts)
    total = youtube_count + reddit_count
    
    if total == 0:
        return [
            PlatformDistribution(platform="YouTube", posts_count=0, percentage=0.0),
            PlatformDistribution(platform="Reddit", posts_count=0, percentage=0.0)
        ]
    
    return [
        PlatformDistribution(
            platform="YouTube",
            posts_count=youtube_count,
            percentage=round((youtube_count / total) * 100, 2)
        ),
        PlatformDistribution(
            platform="Reddit", 
            posts_count=reddit_count,
            percentage=round((reddit_count / total) * 100, 2)
        )
    ]

async def generate_engagement_trends(yt_posts: List[Dict], yt_comments: List[Dict], 
                                   rd_posts: List[Dict], rd_comments: List[Dict]) -> List[EngagementTrend]:
    """Generate monthly engagement trends"""
    monthly_data = defaultdict(lambda: {'youtube': 0, 'reddit': 0})
    
    # Process YouTube data
    for item in yt_posts + yt_comments:
        date_obj = None
        for field in ['date_of_post', 'published_date', 'created_utc', 'date_of_comment', 'published_at', 'timestamp']:
            if field in item and item[field]:
                date_obj = parse_date_safely(item[field])
                if date_obj:
                    break
        
        if date_obj:
            month_key = date_obj.strftime("%Y-%m")
            monthly_data[month_key]['youtube'] += get_engagement_score(item)
    
    # Process Reddit data
    for item in rd_posts + rd_comments:
        date_obj = None
        for field in ['date_of_post', 'created_utc', 'date_of_comment', 'timestamp']:
            if field in item and item[field]:
                date_obj = parse_date_safely(item[field])
                if date_obj:
                    break
        
        if date_obj:
            month_key = date_obj.strftime("%Y-%m")
            monthly_data[month_key]['reddit'] += get_engagement_score(item)
    
    # Convert to sorted list
    trends = []
    for month in sorted(monthly_data.keys()):
        data = monthly_data[month]
        trends.append(EngagementTrend(
            month=month,
            youtube_engagement=data['youtube'],
            reddit_engagement=data['reddit'],
            total_engagement=data['youtube'] + data['reddit']
        ))
    
    # If no trends found, create a default entry
    if not trends:
        current_month = datetime.now().strftime("%Y-%m")
        trends.append(EngagementTrend(
            month=current_month,
            youtube_engagement=0,
            reddit_engagement=0,
            total_engagement=0
        ))
    
    return trends

async def generate_top_posts(posts: List[Dict]) -> List[TopPost]:
    """Generate top 5 engaged posts"""
    if not posts:
        return []
    
    # Calculate engagement scores and add platform info
    processed_posts = []
    for post in posts:
        engagement_score = get_engagement_score(post)
        if engagement_score > 0:  # Only include posts with some engagement
            post_copy = post.copy()
            post_copy['_engagement_score'] = engagement_score
            post_copy['_platform'] = get_platform_from_item(post)
            processed_posts.append(post_copy)
    
    # Sort by engagement
    sorted_posts = sorted(processed_posts, key=lambda x: x.get('_engagement_score', 0), reverse=True)[:10]
    
    top_posts = []
    for i, post in enumerate(sorted_posts[:5], 1):
        platform = post.get('_platform', 'Unknown')
        
        # Get content with fallback options
        content_options = ['content', 'selftext', 'description', 'text', 'body']
        content = None
        for field in content_options:
            if field in post and post[field]:
                content = str(post[field])[:200]
                break
        
        if not content:
            content = "No content available"
        
        # Get title with fallback
        title = str(post.get('title') or post.get('name') or 'Untitled')[:100]
        
        # Get URL with fallback
        url = str(post.get('url') or post.get('permalink') or post.get('link') or '#')
        
        top_posts.append(TopPost(
            rank=i,
            title=title,
            content=content,
            engagement_score=post.get('_engagement_score', 0),
            platform=platform,
            url=url
        ))
    
    return top_posts

async def generate_influential_users(posts: List[Dict], comments: List[Dict]) -> List[InfluentialUser]:
    """Generate top 5 influential users with improved platform detection"""
    user_stats = defaultdict(lambda: {'engagement': 0, 'posts': 0, 'platforms': set()})
    
    # Process posts
    for post in posts:
        username = str(post.get('username') or post.get('author') or post.get('user') or 'Unknown')
        if username not in ['Unknown', 'None', '[deleted]', '[removed]', '']:
            engagement = get_engagement_score(post)
            platform = get_platform_from_item(post)
            
            user_stats[username]['engagement'] += engagement
            user_stats[username]['posts'] += 1
            user_stats[username]['platforms'].add(platform)
    
    # Process comments
    for comment in comments:
        username = str(comment.get('username') or comment.get('author') or comment.get('user') or 'Unknown')
        if username not in ['Unknown', 'None', '[deleted]', '[removed]', '']:
            engagement = get_engagement_score(comment)
            platform = get_platform_from_item(comment)
            
            user_stats[username]['engagement'] += engagement
            user_stats[username]['platforms'].add(platform)
    
    # Filter out users with no engagement
    filtered_users = {k: v for k, v in user_stats.items() if v['engagement'] > 0}
    
    # Sort by total engagement
    sorted_users = sorted(filtered_users.items(), key=lambda x: x[1]['engagement'], reverse=True)[:5]
    
    influential_users = []
    for i, (username, stats) in enumerate(sorted_users, 1):
        avg_engagement = stats['engagement'] / max(stats['posts'], 1)
        
        # Determine primary platform
        platforms = list(stats['platforms'])
        primary_platform = platforms[0] if platforms else 'Unknown'
        
        influential_users.append(InfluentialUser(
            rank=i,
            username=username,
            total_engagement=stats['engagement'],
            posts_count=stats['posts'],
            avg_engagement=round(avg_engagement, 2),
            platform=primary_platform
        ))
    
    return influential_users

async def generate_trending_keywords(posts: List[Dict], comments: List[Dict]) -> List[TrendingKeyword]:
    """Generate trending keywords using AI with improved fallback"""
    # Collect all text and track platforms
    all_text = []
    platform_tracking = defaultdict(set)
    
    # Process posts
    for post in posts:
        platform = get_platform_from_item(post)
        text_fields = ['title', 'content', 'selftext', 'text', 'description']
        
        for field in text_fields:
            text = post.get(field)
            if text and len(str(text)) > 10:  # Only meaningful text
                text_str = str(text)
                all_text.append(text_str)
                
                # Extract and track keywords
                words = re.findall(r'\b[a-zA-Z]{3,}\b', text_str.lower())
                for word in words:
                    if len(word) >= 4:  # Only longer words
                        platform_tracking[word].add(platform)
    
    # Process comments
    for comment in comments:
        platform = get_platform_from_item(comment)
        text_fields = ['text', 'body', 'content']
        
        for field in text_fields:
            text = comment.get(field)
            if text and len(str(text)) > 10:
                text_str = str(text)
                all_text.append(text_str)
                
                words = re.findall(r'\b[a-zA-Z]{3,}\b', text_str.lower())
                for word in words:
                    if len(word) >= 4:
                        platform_tracking[word].add(platform)
    
    if not all_text:
        return []
    
    combined_text = ' '.join(all_text[:50])  # Limit text for processing
    
    # Try Groq API first
    if client and len(combined_text) > 100:
        try:
            text_sample = combined_text[:3000]  # Limit for API
            prompt = f"""Analyze this social media content and extract exactly 10 trending keywords or topics. 
Focus on:
- Specific topics, brands, technologies, or subjects
- Avoid common words like 'people', 'time', 'good', etc.
- Include relevant hashtag topics (without #)
- Focus on nouns and proper nouns

Return ONLY the keywords separated by commas, no other text.

Content sample: {text_sample}"""
            
            response = await asyncio.get_event_loop().run_in_executor(
                executor,
                lambda: client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model="llama-3.1-8b-instant",
                    max_tokens=150,
                    temperature=0.3
                )
            )
            
            keywords_text = response.choices[0].message.content.strip()
            keywords = [kw.strip().lower() for kw in keywords_text.split(',') if kw.strip() and len(kw.strip()) > 2]
            
            # Count frequencies and get platforms
            trending = []
            for keyword in keywords[:15]:
                frequency = combined_text.lower().count(keyword)
                if frequency >= 1:  # Lower threshold
                    platforms = list(platform_tracking.get(keyword, ['Unknown']))
                    platforms = [p for p in platforms if p != 'Unknown'] or ['Mixed']
                    
                    trending.append(TrendingKeyword(
                        keyword=keyword.title(),
                        frequency=frequency,
                        platforms=platforms
                    ))
            
            if trending:
                return trending[:10]
            
        except Exception as e:
            logger.error(f"Groq API error: {e}")
    
    # Enhanced fallback method
    words = re.findall(r'\b[a-zA-Z]{4,}\b', combined_text.lower())
    word_counts = Counter(words)
    
    # Enhanced stop words
    stop_words = {
        'this', 'that', 'with', 'from', 'have', 'were', 'your', 'they', 'their',
        'what', 'which', 'when', 'where', 'about', 'just', 'like', 'some', 'would',
        'could', 'should', 'them', 'then', 'than', 'been', 'much', 'such', 'very',
        'more', 'also', 'will', 'make', 'most', 'only', 'other', 'said', 'each',
        'time', 'people', 'good', 'know', 'think', 'really', 'well', 'right'
    }
    
    trending = []
    for word, count in word_counts.most_common(30):
        if word not in stop_words and count >= 2 and len(word) >= 4:
            platforms = list(platform_tracking.get(word, ['Unknown']))
            platforms = [p for p in platforms if p != 'Unknown'] or ['Mixed']
            
            trending.append(TrendingKeyword(
                keyword=word.title(),
                frequency=count,
                platforms=platforms
            ))
            
            if len(trending) >= 10:
                break
    
    return trending

async def generate_posts_histogram(yt_posts: List[Dict], rd_posts: List[Dict]) -> List[PostsHistogram]:
    """Generate weekly posts histogram with improved date handling"""
    weekly_data = defaultdict(lambda: {'youtube': 0, 'reddit': 0})
    
    # Process YouTube posts
    for post in yt_posts:
        date_obj = None
        for field in ['date_of_post', 'published_date', 'created_utc', 'timestamp']:
            if field in post and post[field]:
                date_obj = parse_date_safely(post[field])
                if date_obj:
                    break
        
        if date_obj:
            # Get Monday of the week
            monday = date_obj - timedelta(days=date_obj.weekday())
            week_key = monday.strftime("%Y-%m-%d")
            weekly_data[week_key]['youtube'] += 1
    
    # Process Reddit posts
    for post in rd_posts:
        date_obj = None
        for field in ['date_of_post', 'created_utc', 'timestamp']:
            if field in post and post[field]:
                date_obj = parse_date_safely(post[field])
                if date_obj:
                    break
        
        if date_obj:
            monday = date_obj - timedelta(days=date_obj.weekday())
            week_key = monday.strftime("%Y-%m-%d")
            weekly_data[week_key]['reddit'] += 1
    
    # Convert to sorted list
    histogram = []
    for week in sorted(weekly_data.keys()):
        data = weekly_data[week]
        histogram.append(PostsHistogram(
            time_period=week,
            youtube_posts=data['youtube'],
            reddit_posts=data['reddit'],
            total_posts=data['youtube'] + data['reddit']
        ))
    
    # If no data, create a default entry
    if not histogram:
        current_week = datetime.now().date()
        monday = current_week - timedelta(days=current_week.weekday())
        histogram.append(PostsHistogram(
            time_period=monday.strftime("%Y-%m-%d"),
            youtube_posts=0,
            reddit_posts=0,
            total_posts=0
        ))
    
    return histogram

# ------------------ API ROUTES ------------------
@app.get("/")
async def root():
    return {
        "message": "Social Media Insights API v3.1 - FIXED", 
        "status": "active",
        "improvements": [
            "Fixed empty response issues",
            "Improved platform detection",
            "Better date parsing",
            "Enhanced engagement scoring",
            "More robust error handling"
        ],
        "visualizations": [
            "platform-distribution",
            "engagement-trends", 
            "top-posts",
            "influential-users",
            "trending-keywords",
            "posts-histogram"
        ]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint with data validation"""
    # Check data files
    data_status = {}
    required_files = ["posts_youtube.json", "comments_youtube.json", "posts_reddit.json", "comments_reddit.json"]
    
    for filename in required_files:
        data = data_cache.get(filename)
        data_status[filename] = {
            "exists": len(data) > 0,
            "count": len(data),
            "sample_fields": list(data[0].keys()) if data else []
        }
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "groq_available": client is not None,
        "data_dir_exists": DATA_DIR.exists(),
        "data_files": data_status
    }

@app.post("/dashboard", response_model=DashboardResponse)
async def get_complete_dashboard(request: PlatformRequest):
    """Get all dashboard visualizations in one response"""
    try:
        start_time = datetime.now()
        platform = request.platform.lower()
        start_date = request.date_range.start_date
        end_date = request.date_range.end_date
        
        logger.info(f"Generating dashboard for {platform} from {start_date} to {end_date}")
        
        # Load all data with platform tagging
        yt_posts = filter_by_date_range(data_cache.get("posts_youtube.json"), start_date, end_date, "YouTube")
        yt_comments = filter_by_date_range(data_cache.get("comments_youtube.json"), start_date, end_date, "YouTube")
        rd_posts = filter_by_date_range(data_cache.get("posts_reddit.json"), start_date, end_date, "Reddit")
        rd_comments = filter_by_date_range(data_cache.get("comments_reddit.json"), start_date, end_date, "Reddit")
        
        logger.info(f"Data loaded - YT Posts: {len(yt_posts)}, YT Comments: {len(yt_comments)}, "
                   f"RD Posts: {len(rd_posts)}, RD Comments: {len(rd_comments)}")
        
        # Select data based on platform filter
        if platform == "both":
            posts = yt_posts + rd_posts
            comments = yt_comments + rd_comments
        elif platform == "youtube":
            posts = yt_posts
            comments = yt_comments
        else:  # reddit
            posts = rd_posts
            comments = rd_comments
        
        logger.info(f"Filtered data - Posts: {len(posts)}, Comments: {len(comments)}")
        
        # Generate all visualizations concurrently
        tasks = [
            generate_platform_distribution(yt_posts, rd_posts),
            generate_engagement_trends(yt_posts, yt_comments, rd_posts, rd_comments),
            generate_top_posts(posts),
            generate_influential_users(posts, comments),
            generate_trending_keywords(posts, comments),
            generate_posts_histogram(yt_posts, rd_posts)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions in results
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Error in visualization {i}: {result}")
                final_results.append([])  # Return empty list for failed visualizations
            else:
                final_results.append(result)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Dashboard generated in {processing_time:.2f} seconds")
        
        response = DashboardResponse(
            platform_distribution=final_results[0],
            engagement_trends=final_results[1],
            top_posts=final_results[2],
            influential_users=final_results[3],
            trending_keywords=final_results[4],
            posts_histogram=final_results[5]
        )
        
        # Log response summary
        logger.info(f"Response summary - Platform dist: {len(response.platform_distribution)}, "
                   f"Trends: {len(response.engagement_trends)}, Top posts: {len(response.top_posts)}, "
                   f"Users: {len(response.influential_users)}, Keywords: {len(response.trending_keywords)}, "
                   f"Histogram: {len(response.posts_histogram)}")
        
        return response
        
    except Exception as e:
        logger.error(f"Error generating dashboard: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# Individual endpoints for specific visualizations
@app.post("/platform-distribution")
async def get_platform_distribution(request: PlatformRequest):
    """Get platform distribution pie chart data"""
    try:
        yt_posts = filter_by_date_range(data_cache.get("posts_youtube.json"), 
                                       request.date_range.start_date, request.date_range.end_date, "YouTube")
        rd_posts = filter_by_date_range(data_cache.get("posts_reddit.json"), 
                                       request.date_range.start_date, request.date_range.end_date, "Reddit")
        return await generate_platform_distribution(yt_posts, rd_posts)
    except Exception as e:
        logger.error(f"Error in platform distribution: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/engagement-trends")
async def get_engagement_trends(request: PlatformRequest):
    """Get monthly engagement trends"""
    try:
        yt_posts = filter_by_date_range(data_cache.get("posts_youtube.json"), 
                                       request.date_range.start_date, request.date_range.end_date, "YouTube")
        yt_comments = filter_by_date_range(data_cache.get("comments_youtube.json"), 
                                          request.date_range.start_date, request.date_range.end_date, "YouTube")
        rd_posts = filter_by_date_range(data_cache.get("posts_reddit.json"), 
                                       request.date_range.start_date, request.date_range.end_date, "Reddit")
        rd_comments = filter_by_date_range(data_cache.get("comments_reddit.json"), 
                                          request.date_range.start_date, request.date_range.end_date, "Reddit")
        return await generate_engagement_trends(yt_posts, yt_comments, rd_posts, rd_comments)
    except Exception as e:
        logger.error(f"Error in engagement trends: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/top-posts")
async def get_top_posts(request: PlatformRequest):
    """Get top 5 engaged posts"""
    try:
        posts = []
        if request.platform in ["youtube", "both"]:
            posts.extend(filter_by_date_range(data_cache.get("posts_youtube.json"), 
                                             request.date_range.start_date, request.date_range.end_date, "YouTube"))
        if request.platform in ["reddit", "both"]:
            posts.extend(filter_by_date_range(data_cache.get("posts_reddit.json"), 
                                             request.date_range.start_date, request.date_range.end_date, "Reddit"))
        return await generate_top_posts(posts)
    except Exception as e:
        logger.error(f"Error in top posts: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/influential-users")
async def get_influential_users(request: PlatformRequest):
    """Get top 5 influential users"""
    try:
        posts, comments = [], []
        if request.platform in ["youtube", "both"]:
            posts.extend(filter_by_date_range(data_cache.get("posts_youtube.json"), 
                                             request.date_range.start_date, request.date_range.end_date, "YouTube"))
            comments.extend(filter_by_date_range(data_cache.get("comments_youtube.json"), 
                                                request.date_range.start_date, request.date_range.end_date, "YouTube"))
        if request.platform in ["reddit", "both"]:
            posts.extend(filter_by_date_range(data_cache.get("posts_reddit.json"), 
                                             request.date_range.start_date, request.date_range.end_date, "Reddit"))
            comments.extend(filter_by_date_range(data_cache.get("comments_reddit.json"), 
                                                request.date_range.start_date, request.date_range.end_date, "Reddit"))
        return await generate_influential_users(posts, comments)
    except Exception as e:
        logger.error(f"Error in influential users: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/trending-keywords")
async def get_trending_keywords(request: PlatformRequest):
    """Get trending keywords"""
    try:
        posts, comments = [], []
        if request.platform in ["youtube", "both"]:
            posts.extend(filter_by_date_range(data_cache.get("posts_youtube.json"), 
                                             request.date_range.start_date, request.date_range.end_date, "YouTube"))
            comments.extend(filter_by_date_range(data_cache.get("comments_youtube.json"), 
                                                request.date_range.start_date, request.date_range.end_date, "YouTube"))
        if request.platform in ["reddit", "both"]:
            posts.extend(filter_by_date_range(data_cache.get("posts_reddit.json"), 
                                             request.date_range.start_date, request.date_range.end_date, "Reddit"))
            comments.extend(filter_by_date_range(data_cache.get("comments_reddit.json"), 
                                                request.date_range.start_date, request.date_range.end_date, "Reddit"))
        return await generate_trending_keywords(posts, comments)
    except Exception as e:
        logger.error(f"Error in trending keywords: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/posts-histogram")
async def get_posts_histogram(request: PlatformRequest):
    """Get posts histogram data"""
    try:
        yt_posts = filter_by_date_range(data_cache.get("posts_youtube.json"), 
                                       request.date_range.start_date, request.date_range.end_date, "YouTube")
        rd_posts = filter_by_date_range(data_cache.get("posts_reddit.json"), 
                                       request.date_range.start_date, request.date_range.end_date, "Reddit")
        return await generate_posts_histogram(yt_posts, rd_posts)
    except Exception as e:
        logger.error(f"Error in posts histogram: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/available-dates")
async def get_available_dates():
    """Get available date range from data with improved parsing"""
    try:
        dates = []
        data_files = [
            ("posts_youtube.json", "YouTube"),
            ("comments_youtube.json", "YouTube"), 
            ("posts_reddit.json", "Reddit"),
            ("comments_reddit.json", "Reddit")
        ]
        date_fields = ['date_of_post', 'published_date', 'created_utc', 'date_of_comment', 'published_at', 'timestamp']
        
        file_stats = {}
        
        for filename, platform in data_files:
            data_list = data_cache.get(filename)
            file_dates = []
            
            for item in data_list:
                for field in date_fields:
                    if field in item and item[field]:
                        parsed_date = parse_date_safely(item[field])
                        if parsed_date:
                            dates.append(parsed_date)
                            file_dates.append(parsed_date)
                            break
            
            file_stats[filename] = {
                "platform": platform,
                "total_items": len(data_list),
                "items_with_dates": len(file_dates),
                "date_range": {
                    "min": min(file_dates).isoformat() if file_dates else None,
                    "max": max(file_dates).isoformat() if file_dates else None
                }
            }
        
        if not dates:
            return {
                "min_date": None, 
                "max_date": None,
                "total_dates": 0,
                "files": file_stats,
                "message": "No valid dates found in data"
            }
        
        return {
            "min_date": min(dates).isoformat(),
            "max_date": max(dates).isoformat(),
            "total_dates": len(set(dates)),
            "unique_dates": len(set(dates)),
            "files": file_stats
        }
        
    except Exception as e:
        logger.error(f"Error getting available dates: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving date range")

@app.get("/data-stats")
async def get_data_statistics():
    """Get detailed data statistics for debugging"""
    try:
        stats = {}
        data_files = [
            ("posts_youtube.json", "YouTube Posts"),
            ("comments_youtube.json", "YouTube Comments"), 
            ("posts_reddit.json", "Reddit Posts"),
            ("comments_reddit.json", "Reddit Comments")
        ]
        
        for filename, description in data_files:
            data_list = data_cache.get(filename)
            
            if data_list:
                sample_item = data_list[0] if data_list else {}
                
                # Check for engagement indicators
                engagement_fields = []
                for field in ['likes', 'upvotes', 'score', 'reactions', 'view_count']:
                    if field in sample_item:
                        engagement_fields.append(field)
                
                # Check for date fields
                date_fields = []
                for field in ['date_of_post', 'created_utc', 'published_date', 'timestamp']:
                    if field in sample_item:
                        date_fields.append(field)
                
                # Check for text fields
                text_fields = []
                for field in ['title', 'content', 'text', 'body', 'selftext']:
                    if field in sample_item:
                        text_fields.append(field)
                
                stats[filename] = {
                    "description": description,
                    "total_items": len(data_list),
                    "sample_fields": list(sample_item.keys()),
                    "engagement_fields": engagement_fields,
                    "date_fields": date_fields,
                    "text_fields": text_fields,
                    "sample_item": {k: str(v)[:100] + "..." if len(str(v)) > 100 else v 
                                  for k, v in list(sample_item.items())[:5]}
                }
            else:
                stats[filename] = {
                    "description": description,
                    "total_items": 0,
                    "error": "No data found"
                }
        
        return {
            "timestamp": datetime.now().isoformat(),
            "data_directory": str(DATA_DIR),
            "files": stats
        }
        
    except Exception as e:
        logger.error(f"Error getting data stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("startup")
async def startup_event():
    """Application startup with enhanced logging"""
    logger.info("Starting Social Media Insights API v3.1 - FIXED VERSION")
    
    required_files = [
        ("comments_reddit.json", "Reddit Comments"),
        ("posts_reddit.json", "Reddit Posts"), 
        ("comments_youtube.json", "YouTube Comments"),
        ("posts_youtube.json", "YouTube Posts")
    ]
    
    total_items = 0
    for filename, description in required_files:
        data = data_cache.get(filename)
        logger.info(f"✓ {description} ({filename}): {len(data)} items loaded")
        total_items += len(data)
        
        # Log sample data structure
        if data:
            sample_keys = list(data[0].keys())[:10]  # First 10 fields
            logger.info(f"  Sample fields: {sample_keys}")
    
    logger.info(f"Total data items loaded: {total_items}")
    logger.info(f"Groq client available: {client is not None}")
    logger.info("API ready with 6 visualization endpoints + debugging endpoints")

@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown"""
    logger.info("Shutting down Social Media Insights API")
    executor.shutdown(wait=True)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")