// src/app/dashboard/page.jsx
'use client';

import React, { useState, useEffect } from 'react';
import { Bar, Line, Pie, Doughnut } from 'react-chartjs-2';
import {
  Chart as ChartJS, CategoryScale, LinearScale, BarElement, LineElement,
  PointElement, ArcElement, Title, Tooltip, Legend
} from 'chart.js';
import { 
  Calendar as CalendarIcon, 
  Youtube, 
  Rss, 
  TrendingUp, 
  Users, 
  MessageSquare, 
  ExternalLink,
  Hash,
  Eye,
  ThumbsUp,
  Share2,
  Crown,
  Activity,
  AlertCircle
} from 'lucide-react';
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { Button } from "@/components/ui/button";
import { Calendar } from "@/components/ui/calendar";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { addDays, format } from 'date-fns';
import { cn } from '@/lib/utils';

ChartJS.register(
  CategoryScale, LinearScale, BarElement, LineElement, PointElement, ArcElement, Title, Tooltip, Legend
);

// Modern ChartCard component
const ChartCard = ({ title, description, children, className = "" }) => (
  <Card className={cn("", className)}>
    <CardHeader className="pb-2">
      <CardTitle className="text-lg font-semibold">{title}</CardTitle>
      {description && <CardDescription>{description}</CardDescription>}
    </CardHeader>
    <CardContent>
      <div className="h-72 relative">
        {children}
      </div>
    </CardContent>
  </Card>
);

// Stats Card component
const StatsCard = ({ title, value, change, icon: Icon, className = "" }) => (
  <Card className={cn("", className)}>
    <CardContent className="p-6">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-sm font-medium text-muted-foreground">{title}</p>
          <p className="text-2xl font-bold">{value}</p>
          {change && (
            <p className="text-xs text-muted-foreground">
              <TrendingUp className="inline h-3 w-3 mr-1" />
              {change}
            </p>
          )}
        </div>
        {Icon && <Icon className="h-8 w-8 text-muted-foreground" />}
      </div>
    </CardContent>
  </Card>
);

// List Card component for posts, users, keywords
const ListCard = ({ title, items, type, className = "" }) => (
  <Card className={cn("", className)}>
    <CardHeader>
      <CardTitle className="text-lg font-semibold flex items-center gap-2">
        {type === 'posts' && <MessageSquare className="h-5 w-5" />}
        {type === 'users' && <Users className="h-5 w-5" />}
        {type === 'keywords' && <Hash className="h-5 w-5" />}
        {title}
      </CardTitle>
    </CardHeader>
    <CardContent className="space-y-3">
      {items.map((item, index) => (
        <div key={index} className="flex items-center justify-between p-3 bg-muted/30 rounded-lg hover:bg-muted/50 transition-colors">
          <div className="flex-1 min-w-0">
            {type === 'posts' && (
              <>
                <div className="flex items-center gap-2 mb-2">
                  <Badge variant="outline" className="text-xs">
                    #{item.rank}
                  </Badge>
                  <Badge variant={item.platform === 'YouTube' ? 'destructive' : 'default'} className="text-xs">
                    {item.platform}
                  </Badge>
                </div>
                <p className="font-medium text-sm truncate">
                  {item.title !== "Untitled" ? item.title : `Post ${item.rank}`}
                </p>
                <div className="flex items-center gap-4 mt-2 text-xs text-muted-foreground">
                  <span className="flex items-center gap-1">
                    <Activity className="h-3 w-3" />
                    {item.engagement_score.toLocaleString()} engagement
                  </span>
                  {item.url && (
                    <a 
                      href={item.url} 
                      target="_blank" 
                      rel="noopener noreferrer"
                      className="flex items-center gap-1 hover:text-foreground transition-colors"
                    >
                      <ExternalLink className="h-3 w-3" />
                      View
                    </a>
                  )}
                </div>
              </>
            )}
            
            {type === 'users' && (
              <>
                <div className="flex items-center gap-2 mb-2">
                  <Badge variant="outline" className="text-xs">
                    #{item.rank}
                  </Badge>
                  <Badge variant={item.platform === 'YouTube' ? 'destructive' : 'default'} className="text-xs">
                    {item.platform}
                  </Badge>
                </div>
                <p className="font-medium text-sm">{item.username}</p>
                <div className="flex items-center gap-4 mt-2 text-xs text-muted-foreground">
                  <span className="flex items-center gap-1">
                    <Activity className="h-3 w-3" />
                    {item.total_engagement.toLocaleString()} total
                  </span>
                  <span className="flex items-center gap-1">
                    <MessageSquare className="h-3 w-3" />
                    {item.posts_count} posts
                  </span>
                  <span className="flex items-center gap-1">
                    <TrendingUp className="h-3 w-3" />
                    {Math.round(item.avg_engagement).toLocaleString()} avg
                  </span>
                </div>
              </>
            )}
            
            {type === 'keywords' && (
              <>
                <div className="flex items-center gap-2 mb-2">
                  <p className="font-medium text-sm">#{item.keyword}</p>
                  <Badge variant="secondary" className="text-xs">
                    {item.frequency} mentions
                  </Badge>
                </div>
                <div className="flex items-center gap-2 mt-2">
                  {item.platforms.map((platform, idx) => (
                    <Badge 
                      key={idx}
                      variant={platform === 'YouTube' ? 'destructive' : platform === 'Reddit' ? 'default' : 'outline'} 
                      className="text-xs"
                    >
                      {platform}
                    </Badge>
                  ))}
                </div>
              </>
            )}
          </div>
        </div>
      ))}
    </CardContent>
  </Card>
);

// Skeleton component
const SkeletonDashboard = () => (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {Array.from({ length: 4 }).map((_, i) => (
            <Card key={i} className="animate-pulse">
                <CardContent className="p-6">
                    <div className="h-5 bg-muted rounded w-1/2 mb-2"></div>
                    <div className="h-8 bg-muted rounded w-3/4"></div>
                </CardContent>
            </Card>
        ))}
        <Card className="lg:col-span-3 animate-pulse">
            <CardHeader>
                <div className="h-5 bg-muted rounded w-1/3"></div>
            </CardHeader>
            <CardContent>
                <div className="h-72 bg-muted rounded-md"></div>
            </CardContent>
        </Card>
        <Card className="lg:col-span-1 animate-pulse">
             <CardHeader>
                <div className="h-5 bg-muted rounded w-1/2"></div>
            </CardHeader>
            <CardContent>
                <div className="h-72 bg-muted rounded-md"></div>
            </CardContent>
        </Card>
    </div>
);


export default function DashboardPage() {
  const [platform, setPlatform] = useState('both');
  const [date, setDate] = useState({
    from: addDays(new Date(), -30),
    to: new Date(),
  });
  const [chartData, setChartData] = useState(null);
  const [rawData, setRawData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchDashboardData = async () => {
      if (!date?.from || !date?.to) return;
      setLoading(true);
      setError(null);
      
      try {
        // Simulating network delay
        await new Promise(resolve => setTimeout(resolve, 500));

        const data = {
          platform_distribution: [ { platform: "YouTube", posts_count: 1321, percentage: 62.87 }, { platform: "Reddit", posts_count: 780, percentage: 37.13 } ],
          engagement_trends: [ { month: "2025-05", youtube_engagement: 0, reddit_engagement: 514622, total_engagement: 514622 }, { month: "2025-07", youtube_engagement: 90, reddit_engagement: 0, total_engagement: 90 }, { month: "2025-08", youtube_engagement: 6640358, reddit_engagement: 0, total_engagement: 6640358 } ],
          top_posts: [ { rank: 1, title: "Untitled", content: "{\"tags\": \"skit, funny, vine, youtube, jaden williams, jaden, jadenw, tiktok, edit, gaming, fyp, short, short film, foryou, 4u, jk, joking, comedy, meme, hd, funny tiktok, funny skit, comedy skit, come", engagement_score: 387795, platform: "YouTube", url: "https://www.youtube.com/watch?v=9eO-LqtoGps" }, { rank: 2, title: "Untitled", content: "{\"tags\": \"\", \"title\": \"unique way to use chat gpt shorts rajshamani motivation india\", \"description\": \"\", \"topicDetails\": \"\"}", engagement_score: 373670, platform: "YouTube", url: "https://www.youtube.com/watch?v=CRe_O0_W3-8" }, { rank: 3, title: "Untitled", content: "{\"tags\": \"\", \"title\": \"i love you chatgpt smiling_face_with_hearts viral funny comedy explorepage\", \"description\": \"\", \"topicDetails\": \"https://en.wikipedia.org/wiki/entertainment\"}", engagement_score: 370006, platform: "YouTube", url: "https://www.youtube.com/watch?v=kBoke4HY8RM" }, { rank: 4, title: "Untitled", content: "{\"tags\": \"facttechz, chatgpt, ai, artificial intelligence, gemini, grok, chat, chatbot, privacy, google chrome, history, cookies, youtube shorts, short video, fact, facts, hindi, short fact video, ama", engagement_score: 272392, platform: "YouTube", url: "https://www.youtube.com/watch?v=UKmv_pY2F0Q" }, { rank: 5, title: "Untitled", content: "{\"tags\": \"chatgpt, gpt, chat gpt 5, chatgpt 5\", \"title\": \"the new chatgpt-5 is crazy\", \"description\": \"hands on with chatgpt 5! i spend a lot of time trying to make my videos as concise, polished and ", engagement_score: 150117, platform: "YouTube", url: "https://www.youtube.com/watch?v=n7cG9twTrms" } ],
          influential_users: [ { rank: 1, username: "Jaden Williams", total_engagement: 402077, posts_count: 2, avg_engagement: 201038.5, platform: "YouTube" }, { rank: 2, username: "learn with shamani", total_engagement: 373670, posts_count: 1, avg_engagement: 373670.0, platform: "YouTube" }, { rank: 3, username: "Theonlyzainnn", total_engagement: 370006, posts_count: 1, avg_engagement: 370006.0, platform: "YouTube" }, { rank: 4, username: "Mrwhosetheboss", total_engagement: 297752, posts_count: 2, avg_engagement: 148876.0, platform: "YouTube" }, { rank: 5, username: "FactTechz", total_engagement: 272392, posts_count: 1, avg_engagement: 272392.0, platform: "YouTube" } ],
          trending_keywords: [ { keyword: "Chatgpt", frequency: 177, platforms: [ "Reddit", "YouTube" ] }, { keyword: "Aiims", frequency: 1, platforms: [ "YouTube" ] }, { keyword: "Ias", frequency: 5, platforms: [ "Mixed" ] }, { keyword: "Upsc", frequency: 1, platforms: [ "YouTube" ] }, { keyword: "Ssc", frequency: 1, platforms: [ "Mixed" ] }, { keyword: "Youtube", frequency: 13, platforms: [ "Reddit", "YouTube" ] }, { keyword: "Bihar", frequency: 1, platforms: [ "YouTube" ] }, { keyword: "India", frequency: 8, platforms: [ "Reddit", "YouTube" ] }, { keyword: "Wikipedia", frequency: 55, platforms: [ "Reddit", "YouTube" ] }, { keyword: "Openai", frequency: 8, platforms: [ "Reddit", "YouTube" ] } ],
          posts_histogram: [ { time_period: "2025-05-12", youtube_posts: 0, reddit_posts: 17, total_posts: 17 }, { time_period: "2025-05-19", youtube_posts: 0, reddit_posts: 503, total_posts: 503 }, { time_period: "2025-05-26", youtube_posts: 0, reddit_posts: 260, total_posts: 260 }, { time_period: "2025-07-28", youtube_posts: 362, reddit_posts: 0, total_posts: 362 }, { time_period: "2025-08-04", youtube_posts: 959, reddit_posts: 0, total_posts: 959 } ]
        };
        
        setRawData(data);
        
        const formattedData = {
          postsHistogram: {
            labels: data.posts_histogram.map(item => format(new Date(item.time_period), 'MMM dd')),
            datasets: [
              { label: "YouTube Posts", data: data.posts_histogram.map(item => item.youtube_posts), backgroundColor: 'rgba(255, 99, 132, 0.8)', borderColor: 'rgb(255, 99, 132)', borderWidth: 1, borderRadius: 4, borderSkipped: false, },
              { label: "Reddit Posts", data: data.posts_histogram.map(item => item.reddit_posts), backgroundColor: 'rgba(54, 162, 235, 0.8)', borderColor: 'rgb(54, 162, 235)', borderWidth: 1, borderRadius: 4, borderSkipped: false, }
            ]
          },
          engagementLineChart: {
            labels: data.engagement_trends.map(item => format(new Date(item.month + '-01'), 'MMM yyyy')),
            datasets: [
              { label: "YouTube Engagement", data: data.engagement_trends.map(item => item.youtube_engagement), borderColor: 'rgb(255, 99, 132)', backgroundColor: 'rgba(255, 99, 132, 0.1)', fill: true, tension: 0.4, pointBackgroundColor: 'rgb(255, 99, 132)', pointBorderColor: '#fff', pointBorderWidth: 2, pointRadius: 5, },
              { label: "Reddit Engagement", data: data.engagement_trends.map(item => item.reddit_engagement), borderColor: 'rgb(54, 162, 235)', backgroundColor: 'rgba(54, 162, 235, 0.1)', fill: true, tension: 0.4, pointBackgroundColor: 'rgb(54, 162, 235)', pointBorderColor: '#fff', pointBorderWidth: 2, pointRadius: 5, }
            ]
          },
          platformDoughnutChart: {
            labels: data.platform_distribution.map(item => item.platform),
            datasets: [{ data: data.platform_distribution.map(item => item.percentage), backgroundColor: [ 'rgba(255, 99, 132, 0.8)', 'rgba(54, 162, 235, 0.8)' ], borderColor: [ 'rgb(255, 99, 132)', 'rgb(54, 162, 235)' ], borderWidth: 2, hoverBorderWidth: 3, cutout: '60%' }]
          }
        };
        
        setChartData(formattedData);
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };
    
    fetchDashboardData();
  }, [date, platform]);

  const getSummaryStats = () => {
    if (!rawData) return null;
    const totalPosts = rawData.platform_distribution.reduce((sum, item) => sum + item.posts_count, 0);
    const totalEngagement = rawData.engagement_trends.reduce((sum, item) => sum + item.total_engagement, 0);
    const totalKeywords = rawData.trending_keywords.length;
    return { totalPosts, totalEngagement, totalKeywords };
  };

  const stats = getSummaryStats();

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { position: 'bottom', labels: { padding: 20, usePointStyle: true, font: { size: 12 } } },
      tooltip: {
        backgroundColor: 'rgba(0, 0, 0, 0.8)',
        titleColor: '#fff',
        bodyColor: '#fff',
        borderColor: 'rgba(255, 255, 255, 0.1)',
        borderWidth: 1,
        cornerRadius: 8,
        displayColors: true,
        callbacks: {
          label: function(context) {
            if (context.dataset.label?.includes('Engagement')) {
              return `${context.dataset.label}: ${context.parsed.y?.toLocaleString() || 0}`;
            }
            return `${context.dataset.label}: ${context.parsed.y || context.parsed || 0}`;
          }
        }
      }
    },
    scales: {
      x: { grid: { display: false }, ticks: { font: { size: 11 } } },
      y: {
        grid: { color: 'rgba(0, 0, 0, 0.05)' },
        ticks: {
          font: { size: 11 },
          callback: function(value) {
            if (value >= 1000000) return (value / 1000000).toFixed(1) + 'M';
            if (value >= 1000) return (value / 1000).toFixed(1) + 'K';
            return value;
          }
        }
      }
    }
  };

  return (
    <div className="space-y-6">
      <header>
        <h1 className="text-2xl font-bold tracking-tight text-foreground">Dashboard</h1>
        <p className="text-muted-foreground">An overview of your content analytics and engagement metrics.</p>
      </header>

      <div className="flex flex-col sm:flex-row gap-4 items-start sm:items-center justify-between">
        <Popover>
          <PopoverTrigger asChild>
            <Button id="date" variant="outline" className={cn("w-full sm:w-[280px] justify-start text-left font-normal", !date && "text-muted-foreground")}>
              <CalendarIcon className="mr-2 h-4 w-4" />
              {date?.from ? ( date.to ? (<>{format(date.from, "LLL dd, y")} - {format(date.to, "LLL dd, y")}</>) : (format(date.from, "LLL dd, y"))) : (<span>Pick a date</span>)}
            </Button>
          </PopoverTrigger>
          <PopoverContent className="w-auto p-0" align="start">
            <Calendar initialFocus mode="range" defaultMonth={date?.from} selected={date} onSelect={setDate} numberOfMonths={2}/>
          </PopoverContent>
        </Popover>
        
        <div className="flex items-center gap-2">
          <Button variant={platform === 'both' ? 'default' : 'outline'} onClick={() => setPlatform('both')} size="sm">All Platforms</Button>
          <Button variant={platform === 'youtube' ? 'destructive' : 'outline'} onClick={() => setPlatform('youtube')} size="sm"><Youtube className="mr-2 h-4 w-4" />YouTube</Button>
          <Button variant={platform === 'reddit' ? 'secondary' : 'outline'} onClick={() => setPlatform('reddit')} size="sm"><Rss className="mr-2 h-4 w-4" />Reddit</Button>
        </div>
      </div>
      
      {loading ? (
        <SkeletonDashboard />
      ) : error ? (
        <Card className="p-8">
          <div className="text-center text-destructive">
            <AlertCircle className="h-12 w-12 mx-auto mb-4" />
            <h3 className="text-lg font-semibold mb-2">Error Loading Data</h3>
            <p>{error}</p>
          </div>
        </Card>
      ) : chartData && rawData && stats ? (
        <>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            <StatsCard title="Total Posts" value={stats.totalPosts.toLocaleString()} icon={MessageSquare}/>
            <StatsCard title="Total Engagement" value={stats.totalEngagement.toLocaleString()} icon={Activity}/>
            <StatsCard title="Active Users" value={rawData.influential_users.length.toLocaleString()} icon={Users}/>
            <StatsCard title="Trending Keywords" value={stats.totalKeywords.toLocaleString()} icon={Hash}/>
          </div>
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <ChartCard title="Posts Over Time" description="Distribution of posts across platforms" className="lg:col-span-2">
              <Bar options={{ ...chartOptions, scales: { ...chartOptions.scales, x: { ...chartOptions.scales.x, stacked: true }, y: { ...chartOptions.scales.y, stacked: true } } }} data={chartData.postsHistogram} />
            </ChartCard>
            <ChartCard title="Platform Distribution" description="Posts by platform">
              <Doughnut options={{ responsive: true, maintainAspectRatio: false, plugins: { legend: { position: 'bottom', labels: { padding: 20, usePointStyle: true } }, tooltip: { backgroundColor: 'rgba(0, 0, 0, 0.8)', callbacks: { label: function(context) { return `${context.label}: ${context.parsed.toFixed(2)}%`; } } } } }} data={chartData.platformDoughnutChart} />
            </ChartCard>
          </div>
          <div className="grid grid-cols-1">
            <ChartCard title="Engagement Trends" description="Monthly engagement by platform">
              <Line options={{ ...chartOptions, interaction: { intersect: false, mode: 'index' } }} data={chartData.engagementLineChart} />
            </ChartCard>
          </div>
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <ListCard title="Most Engaged Posts" items={rawData.top_posts} type="posts"/>
            <ListCard title="Top Users by Engagement" items={rawData.influential_users} type="users"/>
            <ListCard title="Trending Keywords" items={rawData.trending_keywords.sort((a, b) => b.frequency - a.frequency)} type="keywords"/>
          </div>
        </>
      ) : (
        <Card className="p-8">
          <div className="text-center text-muted-foreground">
            <Eye className="h-12 w-12 mx-auto mb-4" />
            <h3 className="text-lg font-semibold mb-2">No Data Available</h3>
            <p>No data available for the selected filters. Try adjusting your date range or platform selection.</p>
          </div>
        </Card>
      )}
    </div>
  );
}

