// src/app/components/NewsCard.jsx
import React from 'react';

export const NewsCard = ({ article }) => {
  const { title, description, url, urlToImage, source, publishedAt } = article;
  
  const fallbackImage = `https://placehold.co/600x400/1a202c/ffffff?text=${encodeURIComponent(source?.name || 'News')}`;

  return (
    <a 
      href={url} 
      target="_blank" 
      rel="noopener noreferrer" 
      // MODIFIED: Added a black border
      className="group block rounded-lg overflow-hidden bg-card shadow-sm border border-black hover:shadow-lg transition-all duration-300 ease-in-out transform hover:-translate-y-1 h-full flex flex-col"
    >
      <div className="relative overflow-hidden">
        <img 
          src={urlToImage || fallbackImage} 
          alt={title || 'News Article Image'} 
          className="w-full h-48 object-cover group-hover:scale-105 transition-transform duration-300 ease-in-out" 
          onError={(e) => { 
            e.target.onerror = null; 
            e.target.src = fallbackImage; 
          }}
        />
        <div className="absolute inset-0 bg-gradient-to-t from-black/60 to-transparent"></div>
        <span className="absolute top-2 right-2 bg-primary text-primary-foreground text-xs font-semibold px-2 py-1 rounded-full">
          {source?.name}
        </span>
      </div>
      
      <div className="p-4 flex flex-col flex-grow">
        <h3 className="font-bold text-foreground text-lg mb-2 leading-tight">
          {title || "No Title Available"}
        </h3>
        <p className="text-muted-foreground text-sm mb-4 line-clamp-3 flex-grow">
          {description}
        </p>
        
        <div className="text-xs text-muted-foreground pt-3 border-t">
          <span>{publishedAt ? new Date(publishedAt).toLocaleDateString() : 'No Date'}</span>
        </div>

        <div className="mt-4 w-full bg-primary text-primary-foreground text-center text-sm font-semibold py-2 rounded-md transition-colors hover:bg-primary/90">
          Click Here
        </div>
      </div>
    </a>
  );
};