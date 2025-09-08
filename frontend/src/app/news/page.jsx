// src/app/news/page.jsx
'use client';

import React, { useState, useEffect, useCallback, useRef } from 'react';
import { NewsCard } from '../components/NewsCard'; // Assuming this component exists
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Search } from 'lucide-react';

const staticDomains = [
  { name: 'YouTube', query: 'youtube.com' },
  { name: 'Reddit', query: 'reddit.com' },
];
const categories = ['Business', 'Entertainment', 'Sports', 'Technology'];

const SkeletonGrid = () => (
  <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
    {Array.from({ length: 12 }).map((_, index) => ( // Changed to 12 to match limit
      <div key={index} className="bg-card p-4 rounded-lg border animate-pulse space-y-3">
        <div className="bg-muted h-40 rounded-md"></div>
        <div className="bg-muted h-5 rounded w-3/4"></div>
        <div className="bg-muted h-4 rounded w-full"></div>
        <div className="flex justify-between pt-2">
          <div className="bg-muted h-4 rounded w-1/4"></div>
          <div className="bg-muted h-4 rounded w-1/4"></div>
        </div>
      </div>
    ))}
  </div>
);

// REMOVED: LoadingSpinner is no longer needed without infinite scroll

export default function NewsPage() {
  const [articles, setArticles] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [activeFilter, setActiveFilter] = useState('general');
  
  // REMOVED: State for page and hasMore is no longer needed

  const fetchNews = useCallback(async (filter, search) => {
    setLoading(true);
    setArticles([]); // Clear previous articles on new fetch
    setError(null);
    
    // Always fetch page 1 since there's no pagination
    let url = `/api/news?page=1`; 
    
    const domain = staticDomains.find((d) => d.name.toLowerCase() === filter);
    const currentSearchTerm = search.trim();

    if (currentSearchTerm) {
      url += `&q=${encodeURIComponent(currentSearchTerm)}`;
    } else if (domain) {
      url += `&q=${encodeURIComponent(domain.query)}`;
    } else if (filter !== 'general') {
      url += `&category=${filter}`;
    }

    try {
      const fetchPromise = fetch(url).then((res) => {
        if (!res.ok) throw new Error('Network response was not ok');
        return res.json();
      });
      const delayPromise = new Promise((resolve) => setTimeout(resolve, 500));
      const [data] = await Promise.all([fetchPromise, delayPromise]);
      if (data.error) throw new Error(data.error);
      
      if (data.articles && data.articles.length > 0) {
        setArticles(data.articles);
      } else {
        setArticles([]);
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, []);
  
  // Simplified effect to fetch news when filters or search term change
  useEffect(() => {
    fetchNews(activeFilter, searchTerm);
  }, [activeFilter, searchTerm, fetchNews]);

  // REMOVED: useEffect for infinite scroll is no longer needed

  const handleSearch = (e) => {
    e.preventDefault();
    const newSearchTerm = e.target.elements.search.value;
    setActiveFilter('');
    setSearchTerm(newSearchTerm);
  };

  const handleFilterClick = (newFilter) => {
    const filter = newFilter.toLowerCase();
    setSearchTerm('');
    if (document.querySelector('input[name="search"]')) {
      document.querySelector('input[name="search"]').value = '';
    }
    setActiveFilter(filter);
  };

  const allFilters = ['General', ...staticDomains.map(d => d.name), ...categories];

  return (
    <div className="space-y-6">
      <header>
        <h1 className="text-2xl font-bold tracking-tight text-foreground">News Feed</h1>
        <p className="text-muted-foreground">Your real-time window into global headlines.</p>
      </header>

      <div className="space-y-4">
        <form onSubmit={handleSearch} className="relative">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-5 w-5 text-muted-foreground" />
          <Input name="search" type="text" placeholder="Search for any topic..." className="pl-10"/>
        </form>

        <div className="flex flex-wrap gap-2">
          {allFilters.map((filterName) => (
            <Button
              key={filterName}
              variant={activeFilter === filterName.toLowerCase() ? 'default' : 'outline'}
              onClick={() => handleFilterClick(filterName)}
            >
              {filterName}
            </Button>
          ))}
        </div>
      </div>
      
      {loading && <SkeletonGrid />}
      {error && <p className="text-center text-destructive-foreground bg-destructive/10 py-4 rounded-md font-medium">{error}</p>}
      
      {!loading && articles.length > 0 && (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
          {/* FIX: Use .slice(0, 12) to limit the number of articles displayed */}
          {articles.slice(0, 12).map((article) => (
            <NewsCard key={article.url} article={article} />
          ))}
        </div>
      )}
      
      {!loading && !error && articles.length === 0 && (
        <div className="text-center py-10">
          <p className="text-muted-foreground">No articles found. Try a different search or category.</p>
        </div>
      )}
    </div>
  );
}