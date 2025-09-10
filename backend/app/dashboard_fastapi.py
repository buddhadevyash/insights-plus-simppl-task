from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional, Any, Union
from datetime import datetime, date, timedelta
import json
import logging
import os
from pathlib import Path
from collections import defaultdict, Counter
import re
from functools import lru_cache
import asyncio
from concurrent.futures import ThreadPoolExecutor
from groq import Groq
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Social Media Insights API",
    description="Optimized API for social media analytics with 8 visualization endpoints",
    version="4.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load .env file
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

# Response Models for Visualizations
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
    author: str

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
    time_period: str
    youtube_posts: int
    reddit_posts: int
    total_posts: int

# New Models for additional visualizations
class TrendingComment(BaseModel):
    rank: int
    author: str
    content: str
    engagement_score: int
    platform: str
    url: str
    post_title: str

class SentimentPoint(BaseModel):
    month: str
    positive_score: float
    negative_score: float
    neutral_score: float

class SentimentTrend(BaseModel):
    youtube: List[SentimentPoint]
    reddit: List[SentimentPoint]

# Main Dashboard Response - UPDATED
class DashboardResponse(BaseModel):
    platform_distribution: List[PlatformDistribution]
    engagement_trends: List[EngagementTrend]
    top_posts: List[TopPost]
    influential_users: List[InfluentialUser]
    trending_keywords: List[TrendingKeyword]
    posts_histogram: List[PostsHistogram]
    trending_comments: List[TrendingComment]
    sentiment_trends: SentimentTrend


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
        
        if filename in self._cache and self._timestamps.get(filename) == current_mtime:
            return self._cache[filename]
        
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

data_cache = DataCache()


# ------------------ UTILITY FUNCTIONS ------------------

def parse_date_safely(date_str: Any) -> Optional[date]:
    if not date_str:
        return None
    try:
        date_str = str(date_str).strip()
        if date_str.isdigit() and len(date_str) >= 10:
            return datetime.fromtimestamp(int(date_str)).date()
        if "-" in date_str:
            date_part = date_str.split(" ")[0].split("T")[0]
            if len(date_part) >= 8:
                return datetime.strptime(date_part[:10], "%Y-%m-%d").date()
        return None
    except Exception as e:
        logger.debug(f"Date parsing error for '{date_str}': {e}")
        return None

def determine_platform(item: Dict) -> str:
    reddit_indicators = ['subreddit', 'permalink', 'subreddit_id', 'subreddit_name_prefixed', 'comment_id']
    if any(field in item for field in reddit_indicators) or ('t1_' in str(item.get('comment_id', ''))):
        return 'Reddit'
    
    youtube_indicators = ['video_id', 'channel_id', 'channel_title', 'video_url']
    if any(field in item for field in youtube_indicators):
        return 'YouTube'
    
    url = item.get('url', '') or item.get('permalink', '')
    if 'reddit.com' in str(url) or '/r/' in str(url):
        return 'Reddit'
    elif 'youtube.com' in str(url) or 'youtu.be' in str(url):
        return 'YouTube'
    
    return 'Unknown'

def get_engagement_score(item: Dict) -> int:
    score = 0
    
    if item.get('reactions'):
        try:
            reactions_data = item['reactions']
            if isinstance(reactions_data, str):
                reactions = json.loads(reactions_data.replace("'", '"'))
            else:
                reactions = reactions_data
            if isinstance(reactions, dict):
                likes = int(reactions.get('likes', 0) or 0)
                dislikes = int(reactions.get('dislikes', 0) or 0)
                score += max(0, likes - dislikes)
        except Exception as e:
            logger.debug(f"Reactions parsing error: {e}")

    likes = int(item.get('likes') or item.get('like_count') or 0)
    dislikes = int(item.get('dislikes') or item.get('dislike_count') or 0)
    score += max(0, likes - dislikes)
    
    upvotes = int(item.get('upvotes', 0) or item.get('ups', 0) or 0)
    downvotes = int(item.get('downvotes', 0) or item.get('downs', 0) or 0)
    score += max(0, upvotes - downvotes)
    
    if item.get('score') is not None:
        score += max(0, int(item['score']))

    comment_count = int(item.get('comment_count') or item.get('num_comments') or 0)
    score += comment_count * 2
    
    return max(0, score)

def filter_by_date_range(data: List[Dict], start_date: date, end_date: date, source_platform: str = None) -> List[Dict]:
    date_fields = ['date_of_post', 'published_date', 'created_utc', 'date_of_comment', 'published_at', 'timestamp']
    filtered = []
    for item in data:
        if source_platform and determine_platform(item) == 'Unknown':
            item['_source_platform'] = source_platform
        
        item_date = next((parse_date_safely(item[field]) for field in date_fields if item.get(field)), None)
        
        if item_date is None or (start_date <= item_date <= end_date):
            filtered.append(item)
            
    logger.info(f"Filtered {len(filtered)} items from {len(data)} for platform {source_platform}")
    return filtered

def get_platform_from_item(item: Dict) -> str:
    platform = determine_platform(item)
    return platform if platform != 'Unknown' else item.get('_source_platform', 'Unknown')


# ------------------ VISUALIZATION DATA GENERATORS ------------------

async def generate_platform_distribution(yt_posts: List[Dict], rd_posts: List[Dict]) -> List[PlatformDistribution]:
    youtube_count, reddit_count = len(yt_posts), len(rd_posts)
    total = youtube_count + reddit_count
    if total == 0:
        return [
            PlatformDistribution(platform="YouTube", posts_count=0, percentage=0.0),
            PlatformDistribution(platform="Reddit", posts_count=0, percentage=0.0)
        ]
    return [
        PlatformDistribution(platform="YouTube", posts_count=youtube_count, percentage=round((youtube_count / total) * 100, 2)),
        PlatformDistribution(platform="Reddit", posts_count=reddit_count, percentage=round((reddit_count / total) * 100, 2))
    ]

async def generate_engagement_trends(yt_posts: List[Dict], yt_comments: List[Dict], rd_posts: List[Dict], rd_comments: List[Dict]) -> List[EngagementTrend]:
    monthly_data = defaultdict(lambda: {'youtube': 0, 'reddit': 0})
    date_fields = ['date_of_post', 'published_date', 'created_utc', 'date_of_comment', 'published_at', 'timestamp']

    for item in yt_posts + yt_comments:
        date_obj = next((parse_date_safely(item[field]) for field in date_fields if item.get(field)), None)
        if date_obj:
            monthly_data[date_obj.strftime("%Y-%m")]['youtube'] += get_engagement_score(item)
    
    for item in rd_posts + rd_comments:
        date_obj = next((parse_date_safely(item[field]) for field in date_fields if item.get(field)), None)
        if date_obj:
            monthly_data[date_obj.strftime("%Y-%m")]['reddit'] += get_engagement_score(item)

    trends = [EngagementTrend(month=m, youtube_engagement=d['youtube'], reddit_engagement=d['reddit'], total_engagement=d['youtube'] + d['reddit']) for m, d in sorted(monthly_data.items())]
    return trends if trends else [EngagementTrend(month=datetime.now().strftime("%Y-%m"), youtube_engagement=0, reddit_engagement=0, total_engagement=0)]

async def generate_top_posts(posts: List[Dict]) -> List[TopPost]:
    processed_posts = []
    for post in posts:
        engagement_score = get_engagement_score(post)
        if engagement_score > 0:
            post_copy = post.copy()
            post_copy['_engagement_score'] = engagement_score
            post_copy['_platform'] = get_platform_from_item(post)
            processed_posts.append(post_copy)

    sorted_posts = sorted(processed_posts, key=lambda x: x.get('_engagement_score', 0), reverse=True)[:5]
    
    top_posts_list = []
    for i, post in enumerate(sorted_posts, 1):
        platform = post.get('_platform', 'Unknown')
        content_options = ['content', 'selftext', 'description', 'text', 'body']
        content = next((str(post[field]) for field in content_options if post.get(field)), "")
        
        title_candidate = post.get('title') or post.get('name')
        title = str(title_candidate) if title_candidate else (content[:70] + "..." if content else f"Post from {platform}")

        url = "#"
        if platform == 'YouTube':
            if post.get('video_id'):
                url = f"https://www.youtube.com/watch?v={post['video_id']}"
        elif platform == 'Reddit':
            permalink = post.get('permalink')
            if permalink and str(permalink).startswith('/r/'):
                url = f"https://www.reddit.com{permalink}"
            else:
                url = str(permalink or '#')

        top_posts_list.append(TopPost(
            rank=i,
            title=title.strip(),
            content=content.strip()[:200],
            engagement_score=post.get('_engagement_score', 0),
            platform=platform,
            url=url,
            author=str(post.get('username') or post.get('author') or 'Unknown')
        ))
    return top_posts_list

async def generate_influential_users(posts: List[Dict], comments: List[Dict]) -> List[InfluentialUser]:
    user_stats = defaultdict(lambda: {'engagement': 0, 'posts': 0, 'platforms': set()})
    
    for item in posts + comments:
        username = str(item.get('username') or item.get('author') or 'Unknown')
        if username not in ['Unknown', 'None', '[deleted]', '[removed]', '']:
            platform = get_platform_from_item(item)
            user_stats[username]['engagement'] += get_engagement_score(item)
            user_stats[username]['platforms'].add(platform)
            if any(k in item for k in ['title', 'video_id', 'selftext']): # Heuristic to count posts vs comments
                 user_stats[username]['posts'] += 1

    filtered_users = {k: v for k, v in user_stats.items() if v['engagement'] > 0}
    sorted_users = sorted(filtered_users.items(), key=lambda x: x[1]['engagement'], reverse=True)[:5]

    return [InfluentialUser(
        rank=i, username=username, total_engagement=stats['engagement'],
        posts_count=stats['posts'], avg_engagement=round(stats['engagement'] / max(stats['posts'], 1), 2),
        platform=list(stats['platforms'])[0] if stats['platforms'] else 'Unknown'
    ) for i, (username, stats) in enumerate(sorted_users, 1)]

async def generate_trending_keywords(posts: List[Dict], comments: List[Dict]) -> List[TrendingKeyword]:
    all_text = []
    platform_tracking = defaultdict(set)
    text_fields = ['title', 'content', 'selftext', 'text', 'description', 'body', 'raw_text']
    
    for item in posts + comments:
        platform = get_platform_from_item(item)
        for field in text_fields:
            if text := item.get(field):
                text_str = str(text)
                all_text.append(text_str)
                words = re.findall(r'\b[a-zA-Z]{4,}\b', text_str.lower())
                for word in words:
                    platform_tracking[word].add(platform)

    if not all_text: return []
    
    combined_text = ' '.join(all_text[:100])
    
    if client and len(combined_text) > 100:
        try:
            prompt = f"Extract exactly 10 trending topics or keywords from this text. Focus on specific nouns, technologies, or subjects. Avoid common words. Return ONLY a comma-separated list.\n\nContent: {combined_text[:3000]}"
            response = await asyncio.to_thread(client.chat.completions.create, messages=[{"role": "user", "content": prompt}], model="llama-3.1-8b-instant", max_tokens=150, temperature=0.3)
            keywords_text = response.choices[0].message.content.strip()
            keywords = [kw.strip().lower() for kw in keywords_text.split(',') if kw.strip() and len(kw.strip()) > 2]
            
            trending = []
            for keyword in keywords[:10]:
                freq = combined_text.lower().count(keyword)
                if freq > 1:
                    platforms = [p for p in platform_tracking.get(keyword, []) if p != 'Unknown'] or ['Mixed']
                    trending.append(TrendingKeyword(keyword=keyword.title(), frequency=freq, platforms=list(set(platforms))))
            if trending: return trending
        except Exception as e:
            logger.error(f"Groq API error: {e}")

    # Fallback method
    stop_words = {'this', 'that', 'with', 'from', 'have', 'were', 'your', 'they', 'their', 'what', 'which', 'about', 'just', 'like', 'would', 'could', 'content', 'https'}
    words = re.findall(r'\b[a-zA-Z]{4,}\b', combined_text.lower())
    word_counts = Counter(w for w in words if w not in stop_words)
    
    trending = []
    for word, count in word_counts.most_common(20):
        if len(trending) >= 10: break
        platforms = [p for p in platform_tracking.get(word, []) if p != 'Unknown'] or ['Mixed']
        trending.append(TrendingKeyword(keyword=word.title(), frequency=count, platforms=list(set(platforms))))
        
    return trending

async def generate_posts_histogram(yt_posts: List[Dict], rd_posts: List[Dict]) -> List[PostsHistogram]:
    weekly_data = defaultdict(lambda: {'youtube': 0, 'reddit': 0})
    date_fields = ['date_of_post', 'published_date', 'created_utc', 'timestamp']

    for post in yt_posts:
        if date_obj := next((parse_date_safely(post[field]) for field in date_fields if post.get(field)), None):
            week_key = (date_obj - timedelta(days=date_obj.weekday())).strftime("%Y-%m-%d")
            weekly_data[week_key]['youtube'] += 1
            
    for post in rd_posts:
        if date_obj := next((parse_date_safely(post[field]) for field in date_fields if post.get(field)), None):
            week_key = (date_obj - timedelta(days=date_obj.weekday())).strftime("%Y-%m-%d")
            weekly_data[week_key]['reddit'] += 1

    histogram = [PostsHistogram(time_period=w, youtube_posts=d['youtube'], reddit_posts=d['reddit'], total_posts=d['youtube'] + d['reddit']) for w, d in sorted(weekly_data.items())]
    return histogram if histogram else [PostsHistogram(time_period=(datetime.now().date() - timedelta(days=datetime.now().weekday())).strftime("%Y-%m-%d"), youtube_posts=0, reddit_posts=0, total_posts=0)]

async def generate_trending_comments(comments: List[Dict], posts: List[Dict]) -> List[TrendingComment]:
    post_map = {str(post.get('id') or post.get('post_id')): post.get('title', 'Original Post') for post in posts}
    
    processed_comments = []
    for comment in comments:
        engagement_score = get_engagement_score(comment)
        if engagement_score > 0:
            comment_copy = comment.copy()
            comment_copy['_engagement_score'] = engagement_score
            comment_copy['_platform'] = get_platform_from_item(comment)
            processed_comments.append(comment_copy)

    sorted_comments = sorted(processed_comments, key=lambda x: x.get('_engagement_score', 0), reverse=True)[:5]

    trending_list = []
    for i, comment in enumerate(sorted_comments, 1):
        platform = comment.get('_platform', 'Unknown')
        content = str(comment.get('text') or comment.get('raw_text') or comment.get('body') or "")
        
        post_id = str(comment.get('post_id', ''))
        post_title = post_map.get(post_id, "Context Unavailable")

        url = '#'
        if platform == 'Reddit' and comment.get('comment_id') and post_id:
            url = f"https://www.reddit.com/comments/{post_id}/_/{comment['comment_id']}"
        elif platform == 'YouTube' and comment.get('video_id') and comment.get('comment_id'):
             url = f"https://www.youtube.com/watch?v={comment['video_id']}&lc={comment['comment_id']}"

        trending_list.append(TrendingComment(
            rank=i,
            author=str(comment.get('username') or comment.get('author') or 'Unknown'),
            content=content.strip()[:200],
            engagement_score=comment.get('_engagement_score', 0),
            platform=platform,
            url=url,
            post_title=str(post_title)
        ))
    return trending_list

async def generate_sentiment_trends(posts: List[Dict], comments: List[Dict]) -> SentimentTrend:
    monthly_data = defaultdict(lambda: defaultdict(lambda: {'pos': 0, 'neg': 0, 'neu': 0, 'count': 0}))
    date_fields = ['date_of_post', 'published_date', 'created_utc', 'date_of_comment', 'published_at', 'timestamp']
    
    for item in posts + comments:
        date_obj = next((parse_date_safely(item[field]) for field in date_fields if item.get(field)), None)
        analysis_str = item.get('text_analysis')
        platform = get_platform_from_item(item)

        if date_obj and analysis_str and platform != 'Unknown':
            try:
                analysis = json.loads(analysis_str)
                if sentiment := analysis.get('Sentiment'):
                    month_key = date_obj.strftime("%Y-%m")
                    stats = monthly_data[platform][month_key]
                    stats['pos'] += sentiment.get('positive', 0)
                    stats['neg'] += sentiment.get('negative', 0)
                    stats['neu'] += sentiment.get('neutral', 0)
                    stats['count'] += 1
            except (json.JSONDecodeError, TypeError):
                continue
    
    yt_trends, rd_trends = [], []
    for platform, data in monthly_data.items():
        for month, stats in sorted(data.items()):
            if stats['count'] > 0:
                point = SentimentPoint(
                    month=month,
                    positive_score=round(stats['pos'] / stats['count'], 3),
                    negative_score=round(stats['neg'] / stats['count'], 3),
                    neutral_score=round(stats['neu'] / stats['count'], 3)
                )
                if platform == "YouTube": yt_trends.append(point)
                elif platform == "Reddit": rd_trends.append(point)

    return SentimentTrend(youtube=yt_trends, reddit=rd_trends)


# ------------------ API ROUTES ------------------

@app.get("/")
async def root():
    return {
        "message": "Social Media Insights API v4.0 - Now with more visualizations!",
        "status": "active",
        "visualizations": [
            "platform-distribution", "engagement-trends", "top-posts",
            "influential-users", "trending-keywords", "posts-histogram",
            "trending-comments", "sentiment-trends"
        ]
    }

@app.get("/health")
async def health_check():
    data_status = {}
    for filename in ["posts_youtube.json", "comments_youtube.json", "posts_reddit.json", "comments_reddit.json"]:
        data = data_cache.get(filename)
        data_status[filename] = {"exists": (data is not None and len(data) > 0), "count": len(data)}
    return {"status": "healthy", "groq_available": client is not None, "data_files": data_status}

@app.post("/dashboard", response_model=DashboardResponse)
async def get_complete_dashboard(request: PlatformRequest):
    try:
        start_time = datetime.now()
        platform, start_date, end_date = request.platform.lower(), request.date_range.start_date, request.date_range.end_date
        logger.info(f"Generating dashboard for {platform} from {start_date} to {end_date}")

        yt_posts = filter_by_date_range(data_cache.get("posts_youtube.json"), start_date, end_date, "YouTube")
        yt_comments = filter_by_date_range(data_cache.get("comments_youtube.json"), start_date, end_date, "YouTube")
        rd_posts = filter_by_date_range(data_cache.get("posts_reddit.json"), start_date, end_date, "Reddit")
        rd_comments = filter_by_date_range(data_cache.get("comments_reddit.json"), start_date, end_date, "Reddit")

        if platform == "both":
            posts, comments = yt_posts + rd_posts, yt_comments + rd_comments
        elif platform == "youtube":
            posts, comments = yt_posts, yt_comments
        else: # reddit
            posts, comments = rd_posts, rd_comments

        tasks = [
            generate_platform_distribution(yt_posts, rd_posts),
            generate_engagement_trends(yt_posts, yt_comments, rd_posts, rd_comments),
            generate_top_posts(posts),
            generate_influential_users(posts, comments),
            generate_trending_keywords(posts, comments),
            generate_posts_histogram(yt_posts, rd_posts),
            generate_trending_comments(comments, yt_posts + rd_posts),
            generate_sentiment_trends(posts, comments)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        final_results = [res if not isinstance(res, Exception) else (logger.error(f"Error in viz {i}: {res}"), []) for i, res in enumerate(results)]

        response = DashboardResponse(
            platform_distribution=final_results[0], engagement_trends=final_results[1],
            top_posts=final_results[2], influential_users=final_results[3],
            trending_keywords=final_results[4], posts_histogram=final_results[5],
            trending_comments=final_results[6], sentiment_trends=final_results[7]
        )
        logger.info(f"Dashboard generated in {(datetime.now() - start_time).total_seconds():.2f}s")
        return response
    except Exception as e:
        logger.error(f"Error generating dashboard: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# --- Individual Endpoints ---
def _get_filtered_data(platform: str, start_date: date, end_date: date):
    yt_posts = filter_by_date_range(data_cache.get("posts_youtube.json"), start_date, end_date, "YouTube") if platform in ["youtube", "both"] else []
    yt_comments = filter_by_date_range(data_cache.get("comments_youtube.json"), start_date, end_date, "YouTube") if platform in ["youtube", "both"] else []
    rd_posts = filter_by_date_range(data_cache.get("posts_reddit.json"), start_date, end_date, "Reddit") if platform in ["reddit", "both"] else []
    rd_comments = filter_by_date_range(data_cache.get("comments_reddit.json"), start_date, end_date, "Reddit") if platform in ["reddit", "both"] else []
    return yt_posts, yt_comments, rd_posts, rd_comments

@app.post("/platform-distribution")
async def get_platform_distribution(request: PlatformRequest):
    yt_p, _, rd_p, _ = _get_filtered_data("both", request.date_range.start_date, request.date_range.end_date)
    return await generate_platform_distribution(yt_p, rd_p)

@app.post("/engagement-trends")
async def get_engagement_trends(request: PlatformRequest):
    yt_p, yt_c, rd_p, rd_c = _get_filtered_data("both", request.date_range.start_date, request.date_range.end_date)
    return await generate_engagement_trends(yt_p, yt_c, rd_p, rd_c)

@app.post("/top-posts")
async def get_top_posts(request: PlatformRequest):
    yt_p, _, rd_p, _ = _get_filtered_data(request.platform, request.date_range.start_date, request.date_range.end_date)
    return await generate_top_posts(yt_p + rd_p)

@app.post("/influential-users")
async def get_influential_users(request: PlatformRequest):
    yt_p, yt_c, rd_p, rd_c = _get_filtered_data(request.platform, request.date_range.start_date, request.date_range.end_date)
    return await generate_influential_users(yt_p + rd_p, yt_c + rd_c)

@app.post("/trending-keywords")
async def get_trending_keywords(request: PlatformRequest):
    yt_p, yt_c, rd_p, rd_c = _get_filtered_data(request.platform, request.date_range.start_date, request.date_range.end_date)
    return await generate_trending_keywords(yt_p + rd_p, yt_c + rd_c)

@app.post("/posts-histogram")
async def get_posts_histogram(request: PlatformRequest):
    yt_p, _, rd_p, _ = _get_filtered_data("both", request.date_range.start_date, request.date_range.end_date)
    return await generate_posts_histogram(yt_p, rd_p)

@app.post("/trending-comments")
async def get_trending_comments(request: PlatformRequest):
    yt_p, yt_c, rd_p, rd_c = _get_filtered_data(request.platform, request.date_range.start_date, request.date_range.end_date)
    all_posts = yt_p + rd_p
    all_comments = yt_c + rd_c
    return await generate_trending_comments(all_comments, all_posts)

@app.post("/sentiment-trends")
async def get_sentiment_trends(request: PlatformRequest):
    yt_p, yt_c, rd_p, rd_c = _get_filtered_data(request.platform, request.date_range.start_date, request.date_range.end_date)
    return await generate_sentiment_trends(yt_p + rd_p, yt_c + rd_c)


@app.get("/available-dates")
async def get_available_dates():
    dates = []
    data_files = ["posts_youtube.json", "comments_youtube.json", "posts_reddit.json", "comments_reddit.json"]
    date_fields = ['date_of_post', 'published_date', 'created_utc', 'date_of_comment', 'published_at', 'timestamp']
    for filename in data_files:
        for item in data_cache.get(filename):
            if date_obj := next((parse_date_safely(item[field]) for field in date_fields if item.get(field)), None):
                dates.append(date_obj)
    return {"min_date": min(dates).isoformat() if dates else None, "max_date": max(dates).isoformat() if dates else None}


@app.on_event("startup")
async def startup_event():
    logger.info("Starting Social Media Insights API v4.0")
    for filename in ["comments_reddit.json", "posts_reddit.json", "comments_youtube.json", "posts_youtube.json"]:
        data_cache.get(filename) # Pre-load cache
    logger.info(f"Groq client available: {client is not None}")
    logger.info("API is ready.")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down API")
    executor.shutdown(wait=True)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")