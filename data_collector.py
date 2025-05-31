"""
Data Collector module for the Analytics Platform.
This module orchestrates the collection of data from various sources.
"""

import os
import sys
import logging
import json
from datetime import datetime
from typing import Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add the project root to the path so we can import the config
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.config import DATA_COLLECTION, DATA_DIR, LOGS_DIR
from src.data_collection.api_connector import get_connector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / "data_collector.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("data_collector")


class DataCollector:
    """
    Orchestrates the collection of data from various sources.
    """
    
    def __init__(self):
        """Initialize the DataCollector."""
        self.config = DATA_COLLECTION
        self.data_dir = DATA_DIR
        
        # Create data directories if they don't exist
        self._create_data_directories()
    
    def _create_data_directories(self):
        """Create necessary data directories."""
        for source in self.config.keys():
            source_dir = self.data_dir / source
            source_dir.mkdir(exist_ok=True)
    
    def collect_all_data(self, max_workers: int = 4) -> Dict[str, List[Dict[str, Any]]]:
        """
        Collect data from all configured sources.
        
        Args:
            max_workers: Maximum number of concurrent workers
            
        Returns:
            Dictionary mapping source names to collected data
        """
        logger.info("Starting data collection from all sources")
        results = {}
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_source = {
                executor.submit(self.collect_data, source): source
                for source in self.config.keys()
            }
            
            for future in as_completed(future_to_source):
                source = future_to_source[future]
                try:
                    data = future.result()
                    results[source] = data
                    logger.info(f"Completed data collection for {source}")
                except Exception as e:
                    logger.error(f"Error collecting data from {source}: {str(e)}")
        
        logger.info("Completed data collection from all sources")
        return results
    
    def collect_data(self, source: str) -> List[Dict[str, Any]]:
        """
        Collect data from a specific source.
        
        Args:
            source: Name of the data source
            
        Returns:
            List of collected data items
        """
        logger.info(f"Collecting data from {source}")
        
        try:
            if source == "yahoo_finance":
                return self._collect_yahoo_finance_data()
            elif source == "google_trends":
                return self._collect_google_trends_data()
            elif source == "social_media":
                return self._collect_social_media_data()
            elif source == "e_commerce":
                return self._collect_e_commerce_data()
            else:
                logger.warning(f"Unknown data source: {source}")
                return []
        except Exception as e:
            logger.error(f"Error collecting data from {source}: {str(e)}")
            raise
    
    def _collect_yahoo_finance_data(self) -> List[Dict[str, Any]]:
        """Collect data from Yahoo Finance."""
        config = self.config["yahoo_finance"]
        connector = get_connector("yahoo_finance")
        
        data = connector.get_data(
            symbols=config["symbols"],
            interval=config["interval"],
            range_period=config["range"]
        )
        
        # Save the collected data
        self._save_data("yahoo_finance", data)
        
        return data
    
    def _collect_google_trends_data(self) -> List[Dict[str, Any]]:
        """Collect data from Google Trends."""
        config = self.config["google_trends"]
        connector = get_connector("google_trends")
        
        data = connector.get_data(
            keywords=config["keywords"],
            geo=config["geo"],
            timeframe=config["timeframe"]
        )
        
        # Save the collected data
        self._save_data("google_trends", data)
        
        return data
    
    def _collect_social_media_data(self) -> List[Dict[str, Any]]:
        """Collect data from social media platforms."""
        config = self.config["social_media"]
        results = []
        
        for platform in config["platforms"]:
            try:
                connector = get_connector(platform)
                data = connector.get_data(
                    keywords=config["keywords"],
                    limit=config["limit"]
                )
                results.extend(data)
                
                # Save the platform-specific data
                self._save_data(f"social_media_{platform}", data)
            except Exception as e:
                logger.error(f"Error collecting data from {platform}: {str(e)}")
        
        return results
    
    def _collect_e_commerce_data(self) -> List[Dict[str, Any]]:
        """Collect data from e-commerce platforms."""
        config = self.config["e_commerce"]
        results = []
        
        for platform in config["platforms"]:
            try:
                connector = get_connector(platform)
                data = connector.get_data(
                    categories=config["categories"],
                    limit=config["limit"]
                )
                results.extend(data)
                
                # Save the platform-specific data
                self._save_data(f"e_commerce_{platform}", data)
            except Exception as e:
                logger.error(f"Error collecting data from {platform}: {str(e)}")
        
        return results
    
    def _save_data(self, source: str, data: List[Dict[str, Any]]):
        """
        Save collected data to disk.
        
        Args:
            source: Name of the data source
            data: Collected data to save
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{source}_{timestamp}.json"
        filepath = self.data_dir / source / filename
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved {len(data)} items from {source} to {filepath}")


def main():
    """Main entry point for data collection."""
    collector = DataCollector()
    results = collector.collect_all_data()
    
    # Print summary of collected data
    for source, data in results.items():
        if isinstance(data, list):
            logger.info(f"Collected {len(data)} items from {source}")
        else:
            logger.info(f"Collected data from {source}")


if __name__ == "__main__":
    main()
