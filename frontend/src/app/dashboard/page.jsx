// src/app/dashboard/page.jsx

'use client';

import React, { useState, useEffect, useMemo } from 'react';
import { Bar, Line, Doughnut } from 'react-chartjs-2';
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
  Activity,
  AlertCircle,
  MessageCircle // Added for comments list
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

// --- Reusable Color Palettes ---
const COLORS = {
  youtube: {
    solid: 'rgb(236, 72, 153)', // Pink
    light: 'rgba(236, 72, 153, 0.8)',
    faded: 'rgba(236, 72, 153, 0.1)',
  },
  reddit: {
    solid: 'rgb(54, 162, 235)', // Blue
    light: 'rgba(54, 162, 235, 0.8)',
    faded: 'rgba(54, 162, 235, 0.1)',
  }
};

const SENTIMENT_COLORS = {
    positive: 'rgba(75, 192, 192, 0.7)',  // Teal
    neutral: 'rgba(201, 203, 207, 0.7)', // Gray
    negative: 'rgba(255, 99, 132, 0.7)'   // Red
};


// Modern ChartCard component
const ChartCard = ({ title, description, children, className = "" }) => (
  <Card className={cn("", className)}>
    <CardHeader className="pb-2">
      <div>
        <CardTitle className="text-lg font-semibold">{title}</CardTitle>
        {description && <CardDescription>{description}</CardDescription>}
      </div>
    </CardHeader>
    <CardContent>
      <div className="h-72 relative">
        {children}
      </div>
    </CardContent>
  </Card>
);

// Stats Card component
const StatsCard = ({ title, value, icon: Icon, className = "" }) => (
  <Card className={cn("", className)}>
    <CardContent className="p-6">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-sm font-medium text-muted-foreground">{title}</p>
          <p className="text-2xl font-bold">{value}</p>
        </div>
        {Icon && <Icon className="h-8 w-8 text-muted-foreground" />}
      </div>
    </CardContent>
  </Card>
);

// Helper function to parse JSON strings and extract titles
const parseTitleFromJSON = (titleStr) => {
  if (typeof titleStr !== 'string') return titleStr;
  
  try {
    const parsed = JSON.parse(titleStr);
    
    if (typeof parsed === 'object') {
      if (parsed.title) return parsed.title;
      if (parsed.tags && typeof parsed.tags === 'string') {
        return parsed.tags.split(',')[0].trim();
      }
      if (Array.isArray(parsed) && parsed.length > 0) {
        return parsed[0];
      }
    }
    return titleStr;
  } catch (e) {
    if (titleStr.startsWith('{') && titleStr.includes('"title"')) {
      const titleMatch = titleStr.match(/"title":\s*"([^"]*)"/);
      if (titleMatch && titleMatch[1]) {
        return titleMatch[1];
      }
    }
    if (titleStr.includes('tags') && titleStr.includes(',')) {
      const tagsMatch = titleStr.match(/"tags":\s*"([^"]*)"/);
      if (tagsMatch && tagsMatch[1]) {
        return tagsMatch[1].split(',')[0].trim();
      }
    }
    return titleStr.replace(/{|}|"|tags|title|:/g, '').substring(0, 60) + '...';
  }
};

// List Card component for posts, users, keywords
const ListCard = ({ title, items, type, className = "" }) => {
  return (
    <Card className={cn("flex flex-col", className)}>
      <CardHeader>
        <CardTitle className="text-lg font-semibold flex items-center gap-2">
          {type === 'posts' && <MessageSquare className="h-5 w-5" />}
          {type === 'users' && <Users className="h-5 w-5" />}
          {type === 'keywords' && <Hash className="h-5 w-5" />}
          {type === 'comments' && <MessageCircle className="h-5 w-5" />}
          {title}
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-3 flex-1 overflow-y-auto" style={{maxHeight: '400px'}}>
        {items.map((item, index) => (
          <div key={index} className="flex items-start justify-between p-3 bg-muted/30 rounded-lg hover:bg-muted/50 transition-colors">
            <div className="flex-1 min-w-0">
              {type === 'posts' && (
                <>
                  <div className="flex items-center gap-2 mb-2">
                    <Badge variant="outline" className="text-xs">
                      #{item.rank}
                    </Badge>
                    <Badge variant={item.platform === 'YouTube' ? 'destructive' : 'secondary'} className="text-xs">
                      {item.platform}
                    </Badge>
                  </div>
                  <p className="font-medium text-sm mb-1 truncate" title={parseTitleFromJSON(item.title)}>
                    {parseTitleFromJSON(item.title)}
                  </p>
                  {item.author && (
                    <p className="text-xs text-muted-foreground mb-2">by {item.author}</p>
                  )}
                  <div className="flex items-center gap-4 text-xs text-muted-foreground">
                    <span className="flex items-center gap-1">
                      <Activity className="h-3 w-3" />
                      {item.engagement_score.toLocaleString()} engagement
                    </span>
                    {item.url && item.url !== "#" && (
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
                    <Badge variant={item.platform === 'YouTube' ? 'destructive' : 'secondary'} className="text-xs">
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
                        variant={platform === 'YouTube' ? 'destructive' : platform === 'Reddit' ? 'secondary' : 'outline'}
                        className="text-xs"
                      >
                        {platform}
                      </Badge>
                    ))}
                  </div>
                </>
              )}

              {type === 'comments' && (
                  <>
                    <div className="flex items-center gap-2 mb-2">
                        <Badge variant="outline" className="text-xs">#{item.rank}</Badge>
                        <Badge variant={item.platform === 'YouTube' ? 'destructive' : 'secondary'} className="text-xs">{item.platform}</Badge>
                        <p className="text-xs text-muted-foreground truncate">by {item.author}</p>
                    </div>
                    <p className="text-sm text-foreground mb-2 italic">"{item.content}"</p>
                    <div className="flex items-center gap-4 text-xs text-muted-foreground">
                        <span className="flex items-center gap-1">
                            <Activity className="h-3 w-3" />
                            {item.engagement_score.toLocaleString()} engagement
                        </span>
                        {item.url && item.url !== "#" && (
                            <a href={item.url} target="_blank" rel="noopener noreferrer" className="flex items-center gap-1 hover:text-foreground transition-colors">
                                <ExternalLink className="h-3 w-3" />
                                View Comment
                            </a>
                        )}
                    </div>
                  </>
              )}
            </div>
          </div>
        ))}
      </CardContent>
    </Card>
  );
};


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
    <Card className="lg:col-span-4 animate-pulse">
      <CardHeader>
        <div className="h-5 bg-muted rounded w-1/3"></div>
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
    from: addDays(new Date(), -365),
    to: new Date(),
  });
  const [rawData, setRawData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchDashboardData = async () => {
      if (!date?.from || !date?.to) return;
      setLoading(true);
      setError(null);

      try {
        const response = await fetch('http://localhost:8002/dashboard',{
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            platform: platform,
            date_range: {
              start_date: format(date.from, 'yyyy-MM-dd'),
              end_date: format(date.to, 'yyyy-MM-dd'),
            },
          }),
        });

        if (!response.ok) {
          throw new Error('Network response was not ok');
        }
        const data = await response.json();
        setRawData(data);
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

    fetchDashboardData();
  }, [date, platform]);

  const stats = useMemo(() => {
    if (!rawData) return null;

    let totalPosts = 0;
    let totalEngagement = 0;

    if (platform === 'youtube') {
      const youtubeData = rawData.platform_distribution.find(p => p.platform === 'YouTube');
      totalPosts = youtubeData ? youtubeData.posts_count : 0;
    } else if (platform === 'reddit') {
      const redditData = rawData.platform_distribution.find(p => p.platform === 'Reddit');
      totalPosts = redditData ? redditData.posts_count : 0;
    } else { // 'both'
      totalPosts = rawData.platform_distribution.reduce((sum, item) => sum + item.posts_count, 0);
    }

    if (platform === 'youtube') {
      totalEngagement = rawData.engagement_trends.reduce((sum, item) => sum + item.youtube_engagement, 0);
    } else if (platform === 'reddit') {
      totalEngagement = rawData.engagement_trends.reduce((sum, item) => sum + item.reddit_engagement, 0);
    } else { // 'both'
      totalEngagement = rawData.engagement_trends.reduce((sum, item) => sum + item.total_engagement, 0);
    }

    const totalUsers = rawData.influential_users.length;
    const totalKeywords = rawData.trending_keywords.length;

    return { totalPosts, totalEngagement, totalKeywords, totalUsers };
  }, [rawData, platform]);

  const formattedChartData = useMemo(() => {
    if (!rawData) return null;

    // --- Bar Chart Data ---
    const postHistogramData = {
      labels: rawData.posts_histogram.map(item => format(new Date(item.time_period), 'MMM dd')),
      datasets: []
    };

    if (platform === 'youtube' || platform === 'both') {
      postHistogramData.datasets.push({
        label: "YouTube Posts",
        data: rawData.posts_histogram.map(item => item.youtube_posts),
        backgroundColor: COLORS.youtube.light
      });
    }

    if (platform === 'reddit' || platform === 'both') {
      postHistogramData.datasets.push({
        label: "Reddit Posts",
        data: rawData.posts_histogram.map(item => item.reddit_posts),
        backgroundColor: COLORS.reddit.light
      });
    }

    // --- Doughnut Chart Data ---
    let doughnutConfig = { labels: [], data: [], colors: [] };

    if (platform === 'youtube') {
      doughnutConfig = { labels: ['YouTube'], data: [100], colors: [COLORS.youtube.light] };
    } else if (platform === 'reddit') {
      doughnutConfig = { labels: ['Reddit'], data: [100], colors: [COLORS.reddit.light] };
    } else {
      doughnutConfig = {
        labels: rawData.platform_distribution.map(item => item.platform),
        data: rawData.platform_distribution.map(item => item.percentage),
        colors: [COLORS.youtube.light, COLORS.reddit.light]
      };
    }
    
    // --- Sentiment Chart Data ---
    const sentimentYoutube = rawData.sentiment_trends.youtube || [];
    const sentimentReddit = rawData.sentiment_trends.reddit || [];
    const allMonths = [...new Set([...sentimentYoutube.map(s => s.month), ...sentimentReddit.map(s => s.month)])].sort();

    const sentimentData = {
        labels: allMonths.map(month => format(new Date(month), 'MMM yyyy')),
        datasets: [
            { label: 'Positive', data: [], backgroundColor: SENTIMENT_COLORS.positive },
            { label: 'Neutral', data: [], backgroundColor: SENTIMENT_COLORS.neutral },
            { label: 'Negative', data: [], backgroundColor: SENTIMENT_COLORS.negative },
        ]
    };

    allMonths.forEach(month => {
        const yt = sentimentYoutube.find(s => s.month === month);
        const rd = sentimentReddit.find(s => s.month === month);

        let positive = 0, neutral = 0, negative = 0;

        if (platform === 'youtube') {
            if (yt) { positive = yt.positive_score; neutral = yt.neutral_score; negative = yt.negative_score; }
        } else if (platform === 'reddit') {
            if (rd) { positive = rd.positive_score; neutral = rd.neutral_score; negative = rd.negative_score; }
        } else { // 'both'
            let count = 0;
            if (yt) { positive += yt.positive_score; neutral += yt.neutral_score; negative += yt.negative_score; count++; }
            if (rd) { positive += rd.positive_score; neutral += rd.neutral_score; negative += rd.negative_score; count++; }
            if (count > 0) { positive /= count; neutral /= count; negative /= count; }
        }

        sentimentData.datasets[0].data.push(positive * 100);
        sentimentData.datasets[1].data.push(neutral * 100);
        sentimentData.datasets[2].data.push(negative * 100);
    });


    return {
      postsHistogram: postHistogramData,
      engagementLineChart: {
        labels: rawData.engagement_trends.map(item => format(new Date(item.month), 'MMM yyyy')),
        datasets: [
          { label: "YouTube Engagement", data: rawData.engagement_trends.map(item => item.youtube_engagement), borderColor: COLORS.youtube.solid, backgroundColor: COLORS.youtube.faded, fill: true, tension: 0.4 },
          { label: "Reddit Engagement", data: rawData.engagement_trends.map(item => item.reddit_engagement), borderColor: COLORS.reddit.solid, backgroundColor: COLORS.reddit.faded, fill: true, tension: 0.4 }
        ]
      },
      platformDoughnutChart: {
        labels: doughnutConfig.labels,
        datasets: [{ data: doughnutConfig.data, backgroundColor: doughnutConfig.colors, borderWidth: 2, cutout: '60%' }]
      },
      sentimentTrendsChart: sentimentData,
    };
  }, [rawData, platform]);

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { position: 'bottom', labels: { padding: 20, usePointStyle: true, font: { size: 12 } } },
      tooltip: {
        backgroundColor: 'rgba(0, 0, 0, 0.8)',
        callbacks: {
          label: function(context) {
            let label = context.dataset.label || context.label || '';
            if (label) {
              label += ': ';
            }
            let value = context.parsed.y ?? context.parsed;
            if (value !== null) {
              label += new Intl.NumberFormat('en-US').format(value);
            }
            return label;
          }
        }
      }
    },
    scales: {
      x: { grid: { display: false } },
      y: {
        grid: { color: 'rgba(0, 0, 0, 0.05)' },
        ticks: {
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
      ) : formattedChartData && rawData && stats ? (
        <>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            <StatsCard title="Total Posts" value={stats.totalPosts.toLocaleString()} icon={MessageSquare}/>
            <StatsCard title="Total Engagement" value={stats.totalEngagement.toLocaleString()} icon={Activity}/>
            <StatsCard title="Influential Users" value={stats.totalUsers.toLocaleString()} icon={Users}/>
            <StatsCard title="Trending Keywords" value={stats.totalKeywords.toLocaleString()} icon={Hash}/>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <ChartCard title="Posts Over Time" description="Distribution of posts across platforms" className="lg:col-span-2">
              <Bar options={{ ...chartOptions, scales: { ...chartOptions.scales, x: { ...chartOptions.scales.x, stacked: true }, y: { ...chartOptions.scales.y, stacked: true } } }} data={formattedChartData.postsHistogram} />
            </ChartCard>
            <ChartCard title="Platform Distribution" description="Posts by platform">
              <Doughnut options={{ ...chartOptions, scales: {}, plugins: {...chartOptions.plugins, tooltip: { ...chartOptions.plugins.tooltip, callbacks: { label: function(context) { return `${context.label}: ${context.parsed.toFixed(2)}%`; }}}} }} data={formattedChartData.platformDoughnutChart} />
            </ChartCard>
          </div>

          <div className="grid grid-cols-1 gap-6">
            <ChartCard title="Engagement Trends" description="Monthly engagement by platform">
              <Line options={{ ...chartOptions, interaction: { intersect: false, mode: 'index' } }} data={formattedChartData.engagementLineChart} />
            </ChartCard>
          </div>

          <div className="grid grid-cols-1 gap-6">
            <ChartCard title="Sentiment Trends" description="Monthly sentiment analysis by platform">
              <Bar 
                options={{ 
                    ...chartOptions, 
                    scales: { 
                        x: { ...chartOptions.scales.x, stacked: true }, 
                        y: { ...chartOptions.scales.y, stacked: true, max: 100, ticks: { callback: (value) => value + '%' } }
                    },
                    plugins: { ...chartOptions.plugins, tooltip: { ...chartOptions.plugins.tooltip, callbacks: { label: function(context) { return `${context.dataset.label}: ${context.parsed.y.toFixed(2)}%`; } } } }
                }} 
                data={formattedChartData.sentimentTrendsChart} 
              />
            </ChartCard>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <ListCard title="Most Engaged Posts" items={rawData.top_posts} type="posts" />
            <ListCard title="Top Users by Engagement" items={rawData.influential_users} type="users" />
            <ListCard title="Trending Keywords" items={rawData.trending_keywords} type="keywords" />
            <ListCard title="Trending Comments" items={rawData.trending_comments} type="comments" />
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