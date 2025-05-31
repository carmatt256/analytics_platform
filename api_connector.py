"""
API Connector Framework for the Analytics Platform.
This module provides a standardized way to connect to various APIs and retrieve data.
"""

import requests
import json
import time
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import sys
import os

# Add the project root to the path so we can import the config
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.config import API_KEYS, LOGS_DIR

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / "api_connector.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("api_connector")


class APIConnector(ABC):
    """Base class for all API connectors."""
    
    def __init__(self, base_url: str, api_key: str, rate_limit: int = 60):
        """
        Initialize the API connector.
        
        Args:
            base_url: The base URL for the API
            api_key: The API key for authentication
            rate_limit: Maximum number of requests per minute
        """
        self.base_url = base_url
        self.api_key = api_key
        self.rate_limit = rate_limit
        self.last_request_time = 0
        self.session = requests.Session()
        self.headers = self._get_default_headers()
        
    def _get_default_headers(self) -> Dict[str, str]:
        """Get default headers for API requests."""
        return {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
    
    def _respect_rate_limit(self):
        """Ensure we don't exceed the API rate limit."""
        current_time = time.time()
        elapsed = current_time - self.last_request_time
        min_interval = 60.0 / self.rate_limit
        
        if elapsed < min_interval:
            sleep_time = min_interval - elapsed
            logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f} seconds")
            time.sleep(sleep_time)
            
        self.last_request_time = time.time()
    
    def make_request(self, endpoint: str, method: str = "GET", params: Optional[Dict[str, Any]] = None, 
                    data: Optional[Dict[str, Any]] = None, retry_count: int = 3) -> Dict[str, Any]:
        """
        Make an API request with retry logic.
        
        Args:
            endpoint: The API endpoint to call
            method: HTTP method (GET, POST, etc.)
            params: URL parameters
            data: Request body for POST/PUT requests
            retry_count: Number of retries on failure
            
        Returns:
            The API response as a dictionary
        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        self._respect_rate_limit()
        
        for attempt in range(retry_count + 1):
            try:
                logger.debug(f"Making {method} request to {url}")
                
                if method.upper() == "GET":
                    response = self.session.get(url, headers=self.headers, params=params)
                elif method.upper() == "POST":
                    response = self.session.post(url, headers=self.headers, params=params, json=data)
                elif method.upper() == "PUT":
                    response = self.session.put(url, headers=self.headers, params=params, json=data)
                elif method.upper() == "DELETE":
                    response = self.session.delete(url, headers=self.headers, params=params)
                else:
                    raise ValueError(f"Unsupported HTTP method: {method}")
                
                response.raise_for_status()
                return response.json()
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"Request failed (attempt {attempt+1}/{retry_count+1}): {str(e)}")
                
                if attempt < retry_count:
                    # Exponential backoff
                    wait_time = 2 ** attempt
                    logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Request failed after {retry_count+1} attempts: {str(e)}")
                    raise
    
    @abstractmethod
    def get_data(self, **kwargs) -> List[Dict[str, Any]]:
        """
        Get data from the API. Must be implemented by subclasses.
        
        Returns:
            List of data items retrieved from the API
        """
        pass


class YahooFinanceConnector(APIConnector):
    """Connector for Yahoo Finance API."""
    
    def __init__(self):
        super().__init__(
            base_url="https://yahoo-finance-api.example.com",
            api_key=API_KEYS["yahoo_finance"],
            rate_limit=30
        )
    
    def get_data(self, symbols: List[str], interval: str = "1d", range_period: str = "3mo") -> List[Dict[str, Any]]:
        """
        Get stock data for the specified symbols.
        
        Args:
            symbols: List of stock symbols
            interval: Data interval (1m, 5m, 1h, 1d, 1wk, 1mo)
            range_period: Data range (1d, 5d, 1mo, 3mo, 6mo, 1y, 5y, max)
            
        Returns:
            List of stock data items
        """
        results = []
        
        for symbol in symbols:
            logger.info(f"Fetching Yahoo Finance data for {symbol}")
            
            try:
                # In a real implementation, this would use the actual Yahoo Finance API
                # For this example, we'll simulate the API response
                params = {
                    "symbol": symbol,
                    "interval": interval,
                    "range": range_period
                }
                
                # Simulated API call
                # response = self.make_request("v8/finance/chart", params=params)
                
                # For demonstration, create simulated data
                data = self._generate_simulated_data(symbol, interval, range_period)
                results.append(data)
                
            except Exception as e:
                logger.error(f"Error fetching data for {symbol}: {str(e)}")
        
        return results
    
    def _generate_simulated_data(self, symbol: str, interval: str, range_period: str) -> Dict[str, Any]:
        """Generate simulated stock data for demonstration purposes."""
        import random
        from datetime import datetime, timedelta
        
        # Determine number of data points based on interval and range
        if interval == "1d" and range_period == "3mo":
            days = 90
        else:
            days = 30  # Default
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Generate random price data
        base_price = random.uniform(50, 500)
        price_data = []
        dates = []
        
        current_date = start_date
        while current_date <= end_date:
            if current_date.weekday() < 5:  # Only business days
                price_change = random.uniform(-0.03, 0.03)  # -3% to +3%
                base_price = base_price * (1 + price_change)
                price_data.append({
                    "open": base_price * random.uniform(0.99, 1.0),
                    "high": base_price * random.uniform(1.0, 1.03),
                    "low": base_price * random.uniform(0.97, 1.0),
                    "close": base_price,
                    "volume": int(random.uniform(100000, 10000000))
                })
                dates.append(current_date.strftime("%Y-%m-%d"))
            
            if interval == "1d":
                current_date += timedelta(days=1)
            elif interval == "1wk":
                current_date += timedelta(days=7)
            elif interval == "1mo":
                current_date += timedelta(days=30)
        
        return {
            "symbol": symbol,
            "interval": interval,
            "range": range_period,
            "dates": dates,
            "price_data": price_data,
            "metadata": {
                "currency": "USD",
                "exchange": "NASDAQ",
                "timezone": "America/New_York"
            }
        }


class GoogleTrendsConnector(APIConnector):
    """Connector for Google Trends API."""
    
    def __init__(self):
        super().__init__(
            base_url="https://trends.google.com/trends/api",
            api_key=API_KEYS["google_trends"],
            rate_limit=20
        )
    
    def get_data(self, keywords: List[str], geo: str = "US", timeframe: str = "today 3-m") -> List[Dict[str, Any]]:
        """
        Get trend data for the specified keywords.
        
        Args:
            keywords: List of keywords to get trend data for
            geo: Geographic region (e.g., "US", "GB")
            timeframe: Time range (e.g., "today 3-m", "today 12-m")
            
        Returns:
            List of trend data items
        """
        logger.info(f"Fetching Google Trends data for {keywords}")
        
        try:
            # In a real implementation, this would use the actual Google Trends API
            # For this example, we'll simulate the API response
            
            # For demonstration, create simulated data
            results = []
            for keyword in keywords:
                data = self._generate_simulated_data(keyword, geo, timeframe)
                results.append(data)
            
            return results
            
        except Exception as e:
            logger.error(f"Error fetching Google Trends data: {str(e)}")
            return []
    
    def _generate_simulated_data(self, keyword: str, geo: str, timeframe: str) -> Dict[str, Any]:
        """Generate simulated trend data for demonstration purposes."""
        import random
        from datetime import datetime, timedelta
        
        # Parse timeframe to determine number of data points
        if timeframe == "today 3-m":
            days = 90
        elif timeframe == "today 1-m":
            days = 30
        else:
            days = 90  # Default
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Generate random trend data
        trend_data = []
        dates = []
        
        # Base value and trend direction
        base_value = random.randint(20, 60)
        trend_direction = random.choice([-1, 1])  # Downward or upward trend
        
        current_date = start_date
        while current_date <= end_date:
            # Add some randomness but maintain a general trend
            random_factor = random.uniform(-10, 10)
            day_progress = (current_date - start_date).days / days
            trend_factor = trend_direction * day_progress * 40  # Amplify trend over time
            
            value = max(0, min(100, base_value + random_factor + trend_factor))
            trend_data.append(int(value))
            dates.append(current_date.strftime("%Y-%m-%d"))
            
            current_date += timedelta(days=1)
        
        # Add some spikes for realism
        num_spikes = random.randint(1, 3)
        for _ in range(num_spikes):
            spike_index = random.randint(0, len(trend_data) - 1)
            trend_data[spike_index] = min(100, trend_data[spike_index] + random.randint(20, 40))
        
        return {
            "keyword": keyword,
            "geo": geo,
            "timeframe": timeframe,
            "dates": dates,
            "values": trend_data,
            "related_queries": [
                {"query": f"buy {keyword}", "value": random.randint(50, 100)},
                {"query": f"best {keyword}", "value": random.randint(30, 90)},
                {"query": f"{keyword} review", "value": random.randint(20, 80)},
                {"query": f"new {keyword}", "value": random.randint(10, 70)},
                {"query": f"{keyword} price", "value": random.randint(5, 60)}
            ]
        }


class SocialMediaConnector(APIConnector):
    """Base class for social media API connectors."""
    
    def __init__(self, platform: str):
        self.platform = platform
        super().__init__(
            base_url=self._get_platform_url(),
            api_key=API_KEYS.get(platform, ""),
            rate_limit=self._get_platform_rate_limit()
        )
    
    def _get_platform_url(self) -> str:
        """Get the base URL for the platform API."""
        if self.platform == "twitter":
            return "https://api.twitter.com/2"
        elif self.platform == "reddit":
            return "https://oauth.reddit.com"
        else:
            raise ValueError(f"Unsupported social media platform: {self.platform}")
    
    def _get_platform_rate_limit(self) -> int:
        """Get the rate limit for the platform API."""
        if self.platform == "twitter":
            return 15  # 15 requests per minute
        elif self.platform == "reddit":
            return 60  # 60 requests per minute
        else:
            return 30  # Default
    
    @abstractmethod
    def get_data(self, **kwargs) -> List[Dict[str, Any]]:
        """Get data from the social media API."""
        pass


class TwitterConnector(SocialMediaConnector):
    """Connector for Twitter API."""
    
    def __init__(self):
        super().__init__(platform="twitter")
    
    def get_data(self, keywords: List[str], limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get tweets containing the specified keywords.
        
        Args:
            keywords: List of keywords to search for
            limit: Maximum number of tweets to retrieve
            
        Returns:
            List of tweet data items
        """
        results = []
        
        for keyword in keywords:
            logger.info(f"Fetching Twitter data for keyword: {keyword}")
            
            try:
                # In a real implementation, this would use the actual Twitter API
                # For this example, we'll simulate the API response
                
                # For demonstration, create simulated data
                data = self._generate_simulated_data(keyword, limit)
                results.append(data)
                
            except Exception as e:
                logger.error(f"Error fetching Twitter data for {keyword}: {str(e)}")
        
        return results
    
    def _generate_simulated_data(self, keyword: str, limit: int) -> Dict[str, Any]:
        """Generate simulated Twitter data for demonstration purposes."""
        import random
        from datetime import datetime, timedelta
        
        # Generate random tweets
        tweets = []
        end_date = datetime.now()
        
        for i in range(min(limit, 100)):
            tweet_age = random.randint(0, 90)  # Random age up to 90 days
            tweet_date = end_date - timedelta(days=tweet_age, 
                                             hours=random.randint(0, 23),
                                             minutes=random.randint(0, 59))
            
            # Generate tweet text with the keyword
            tweet_templates = [
                f"Just bought a new {keyword} and it's amazing! #musthave #trending",
                f"Has anyone tried the new {keyword}? Worth the hype? #askingforafriend",
                f"This {keyword} is the best purchase I've made all year! Highly recommend.",
                f"Not impressed with my new {keyword}. Expected better quality. #disappointed",
                f"Looking for recommendations on which {keyword} to buy. Any suggestions?",
                f"The {keyword} is trending for a reason - it's a game changer!",
                f"Just saw an ad for a {keyword} and now I want one
(Content truncated due to size limit. Use line ranges to read in chunks)