"""
ETL Pipeline module for the Analytics Platform.
This module handles the extraction, transformation, and loading of data.
"""

import os
import sys
import logging
import json
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

# Add the project root to the path so we can import the config
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.config import DATA_PROCESSING, DATA_DIR, LOGS_DIR

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / "etl_pipeline.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("etl_pipeline")


class ETLPipeline:
    """
    Handles the extraction, transformation, and loading of data from various sources.
    """
    
    def __init__(self):
        """Initialize the ETL Pipeline."""
        self.config = DATA_PROCESSING
        self.data_dir = DATA_DIR
        self.processed_dir = DATA_DIR / "processed"
        self.processed_dir.mkdir(exist_ok=True)
        
    def process_all_sources(self) -> Dict[str, pd.DataFrame]:
        """
        Process data from all available sources.
        
        Returns:
            Dictionary mapping source names to processed DataFrames
        """
        logger.info("Starting ETL process for all data sources")
        results = {}
        
        # Process each data source
        for source_dir in self.data_dir.iterdir():
            if source_dir.is_dir() and source_dir.name != "processed":
                try:
                    source_name = source_dir.name
                    logger.info(f"Processing data from {source_name}")
                    
                    # Get the latest data file for this source
                    latest_file = self._get_latest_file(source_dir)
                    if latest_file:
                        # Process the data
                        processed_data = self.process_source(source_name, latest_file)
                        results[source_name] = processed_data
                    else:
                        logger.warning(f"No data files found for {source_name}")
                except Exception as e:
                    logger.error(f"Error processing data from {source_name}: {str(e)}")
        
        logger.info("Completed ETL process for all data sources")
        return results
    
    def _get_latest_file(self, source_dir: Path) -> Optional[Path]:
        """
        Get the latest data file from a source directory.
        
        Args:
            source_dir: Directory containing data files
            
        Returns:
            Path to the latest file, or None if no files exist
        """
        json_files = list(source_dir.glob("*.json"))
        if not json_files:
            return None
        
        # Sort by modification time, newest first
        return sorted(json_files, key=lambda x: x.stat().st_mtime, reverse=True)[0]
    
    def process_source(self, source_name: str, file_path: Path) -> pd.DataFrame:
        """
        Process data from a specific source file.
        
        Args:
            source_name: Name of the data source
            file_path: Path to the data file
            
        Returns:
            Processed DataFrame
        """
        logger.info(f"Processing {source_name} data from {file_path}")
        
        # Load the raw data
        with open(file_path, 'r') as f:
            raw_data = json.load(f)
        
        # Apply source-specific processing
        if "yahoo_finance" in source_name:
            processed_data = self._process_yahoo_finance(raw_data)
        elif "google_trends" in source_name:
            processed_data = self._process_google_trends(raw_data)
        elif "social_media" in source_name:
            processed_data = self._process_social_media(raw_data, source_name)
        elif "e_commerce" in source_name:
            processed_data = self._process_e_commerce(raw_data, source_name)
        else:
            logger.warning(f"Unknown data source type: {source_name}")
            processed_data = pd.DataFrame()
        
        # Save the processed data
        if not processed_data.empty:
            self._save_processed_data(source_name, processed_data)
        
        return processed_data
    
    def _process_yahoo_finance(self, raw_data: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Process Yahoo Finance data.
        
        Args:
            raw_data: Raw data from Yahoo Finance
            
        Returns:
            Processed DataFrame
        """
        # Initialize an empty list to store all stock data
        all_stock_data = []
        
        for stock_item in raw_data:
            symbol = stock_item.get("symbol", "")
            dates = stock_item.get("dates", [])
            price_data = stock_item.get("price_data", [])
            
            # Skip if missing essential data
            if not symbol or not dates or not price_data:
                continue
            
            # Create records for each date
            for i, date in enumerate(dates):
                if i < len(price_data):
                    record = {
                        "symbol": symbol,
                        "date": date,
                        "open": price_data[i].get("open", None),
                        "high": price_data[i].get("high", None),
                        "low": price_data[i].get("low", None),
                        "close": price_data[i].get("close", None),
                        "volume": price_data[i].get("volume", None)
                    }
                    all_stock_data.append(record)
        
        # Convert to DataFrame
        df = pd.DataFrame(all_stock_data)
        
        # Convert date column to datetime
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        
        # Calculate additional metrics
        if not df.empty and "close" in df.columns:
            # Calculate daily returns
            df["daily_return"] = df.groupby("symbol")["close"].pct_change()
            
            # Calculate moving averages
            df["ma_7"] = df.groupby("symbol")["close"].transform(lambda x: x.rolling(window=7).mean())
            df["ma_30"] = df.groupby("symbol")["close"].transform(lambda x: x.rolling(window=30).mean())
            
            # Calculate volatility (standard deviation of returns)
            df["volatility_30d"] = df.groupby("symbol")["daily_return"].transform(lambda x: x.rolling(window=30).std())
        
        logger.info(f"Processed {len(df)} Yahoo Finance records")
        return df
    
    def _process_google_trends(self, raw_data: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Process Google Trends data.
        
        Args:
            raw_data: Raw data from Google Trends
            
        Returns:
            Processed DataFrame
        """
        # Initialize an empty list to store all trend data
        all_trend_data = []
        
        for trend_item in raw_data:
            keyword = trend_item.get("keyword", "")
            dates = trend_item.get("dates", [])
            values = trend_item.get("values", [])
            
            # Skip if missing essential data
            if not keyword or not dates or not values:
                continue
            
            # Create records for each date
            for i, date in enumerate(dates):
                if i < len(values):
                    record = {
                        "keyword": keyword,
                        "date": date,
                        "interest": values[i]
                    }
                    all_trend_data.append(record)
        
        # Convert to DataFrame
        df = pd.DataFrame(all_trend_data)
        
        # Convert date column to datetime
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        
        # Calculate additional metrics
        if not df.empty and "interest" in df.columns:
            # Calculate moving averages
            df["ma_7"] = df.groupby("keyword")["interest"].transform(lambda x: x.rolling(window=7).mean())
            df["ma_30"] = df.groupby("keyword")["interest"].transform(lambda x: x.rolling(window=30).mean())
            
            # Calculate momentum (change over last 7 days)
            df["momentum_7d"] = df.groupby("keyword")["interest"].transform(lambda x: x - x.shift(7))
            
            # Calculate percentile rank (to identify relative popularity)
            df["percentile_rank"] = df.groupby("keyword")["interest"].transform(lambda x: x.rank(pct=True))
        
        logger.info(f"Processed {len(df)} Google Trends records")
        return df
    
    def _process_social_media(self, raw_data: List[Dict[str, Any]], source_name: str) -> pd.DataFrame:
        """
        Process social media data.
        
        Args:
            raw_data: Raw data from social media
            source_name: Name of the social media source
            
        Returns:
            Processed DataFrame
        """
        # Initialize an empty list to store all social media data
        all_social_data = []
        
        for item in raw_data:
            keyword = item.get("keyword", "")
            platform = "twitter" if "twitter" in source_name else "reddit"
            
            # Process Twitter data
            if platform == "twitter" and "tweets" in item:
                for tweet in item.get("tweets", []):
                    record = {
                        "platform": platform,
                        "keyword": keyword,
                        "content_id": tweet.get("id", ""),
                        "text": tweet.get("text", ""),
                        "date": tweet.get("created_at", ""),
                        "user": tweet.get("user", {}).get("username", ""),
                        "followers": tweet.get("user", {}).get("followers_count", 0),
                        "likes": tweet.get("metrics", {}).get("likes", 0),
                        "shares": tweet.get("metrics", {}).get("retweets", 0),
                        "comments": tweet.get("metrics", {}).get("replies", 0)
                    }
                    all_social_data.append(record)
            
            # Process Reddit data
            elif platform == "reddit" and "posts" in item:
                for post in item.get("posts", []):
                    record = {
                        "platform": platform,
                        "keyword": keyword,
                        "content_id": post.get("id", ""),
                        "text": post.get("title", ""),
                        "date": post.get("created_at", ""),
                        "user": post.get("author", ""),
                        "subreddit": post.get("subreddit", ""),
                        "likes": post.get("metrics", {}).get("upvotes", 0),
                        "comments": post.get("metrics", {}).get("comments", 0),
                        "upvote_ratio": post.get("metrics", {}).get("upvote_ratio", 0)
                    }
                    all_social_data.append(record)
        
        # Convert to DataFrame
        df = pd.DataFrame(all_social_data)
        
        # Convert date column to datetime
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        
        # Calculate engagement score
        if not df.empty:
            if "likes" in df.columns and "shares" in df.columns and "comments" in df.columns:
                # For Twitter
                df["engagement_score"] = (df["likes"] * 1 + df["shares"] * 2 + df["comments"] * 3) / 100
            elif "likes" in df.columns and "comments" in df.columns:
                # For Reddit
                df["engagement_score"] = (df["likes"] * 1 + df["comments"] * 3) / 100
        
        logger.info(f"Processed {len(df)} {source_name} records")
        return df
    
    def _process_e_commerce(self, raw_data: List[Dict[str, Any]], source_name: str) -> pd.DataFrame:
        """
        Process e-commerce data.
        
        Args:
            raw_data: Raw data from e-commerce platforms
            source_name: Name of the e-commerce source
            
        Returns:
            Processed DataFrame
        """
        # Initialize an empty list to store all product data
        all_product_data = []
        
        for item in raw_data:
            platform = item.get("platform", "")
            category = item.get("category", "")
            
            for product in item.get("products", []):
                record = {
                    "platform": platform,
                    "category": category,
                    "product_id": product.get("id", ""),
                    "name": product.get("name", ""),
                    "price": product.get("price", 0),
                    "currency": product.get("currency", "USD"),
                    "rating": product.get("rating", 0),
                    "review_count": product.get("review_count", 0),
                    "sales_rank": product.get("sales_rank", 0),
                    "rank_change_30d": product.get("rank_change_30d", 0),
                    "in_stock": product.get("in_stock", True),
                    "url": product.get("url", "")
                }
                all_product_data.append(record)
        
        # Convert to DataFrame
        df = pd.DataFrame(all_product_data)
        
        # Calculate additional metrics
        if not df.empty:
            # Calculate popularity score based on rank, reviews, and rating
            if "sales_rank" in df.columns and "review_count" in df.columns and "rating" in df.columns:
                # Normalize sales rank (lower is better)
                max_rank = df["sales_rank"].max()
                if max_rank > 0:
                    df["norm_rank"] = 1 - (df["sales_rank"] / max_rank)
                else:
                    df["norm_rank"] = 0
                
                # Normalize review count (higher is better)
                max_reviews = df["review_count"].max()
                if max_reviews > 0:
                    df["norm_reviews"] = df["review_count"] / max_reviews
                else:
                    df["norm_reviews"] = 0
                
                # Normalize rating (scale to 0-1)
                df["norm_rating"] = df["rating"] / 5.0
                
                # Calculate popularity score (weighted average)
                df["popularity_score"] = (
                    df["norm_rank"] * 0.5 + 
                    df["norm_reviews"] * 0.3 + 
                    df["norm_rating"] * 0.2
                )
            
            # Calculate trend score based on rank change
            if "rank_change_30d" in df.columns:
                # Positive change means worsening rank, so we invert it
                df["trend_score"] = -df["rank_change_30d"] / 100
                # Clip to range [-1, 1]
                df["trend_score"] = df["trend_score"].clip(-1, 1)
        
        logger.info(f"Processed {len(df)} {source_name} records")
        return df
    
    def _save_processed_data(self, source_name: str, df: pd.DataFrame):
        """
        Save processed data to disk.
        
        Args:
            source_name: Name of the data source
            df: Processed DataFrame
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{source_name}_processed_{timestamp}.csv"
        filepath = self.processed_dir / filename
        
        df.to_csv(filepath, index=False)
        logger.info(f"Saved processed {source_name} data to {filepath}")


def main():
    """Main entry point for ETL pipeline."""
    etl = ETLPipeline()
    results = etl.p
(Content truncated due to size limit. Use line ranges to read in chunks)