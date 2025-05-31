"""
Feature Engineering module for the Analytics Platform.
This module creates derived features from processed data to enhance analysis.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
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
        logging.FileHandler(LOGS_DIR / "feature_engineering.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("feature_engineering")


class FeatureEngineer:
    """
    Creates derived features from processed data to enhance analysis.
    """
    
    def __init__(self):
        """Initialize the Feature Engineer."""
        self.config = DATA_PROCESSING["feature_engineering"]
        self.processed_dir = DATA_DIR / "processed"
        self.features_dir = DATA_DIR / "features"
        self.features_dir.mkdir(exist_ok=True)
    
    def engineer_features(self, dataframes: Dict[str, pd.DataFrame] = None) -> Dict[str, pd.DataFrame]:
        """
        Engineer features from processed data.
        
        Args:
            dataframes: Dictionary of processed DataFrames (optional)
            
        Returns:
            Dictionary of DataFrames with engineered features
        """
        logger.info("Starting feature engineering process")
        
        # If no dataframes provided, load the latest processed data
        if dataframes is None:
            dataframes = self._load_latest_processed_data()
        
        results = {}
        
        # Engineer features for each data source
        for source_name, df in dataframes.items():
            try:
                logger.info(f"Engineering features for {source_name}")
                
                if "yahoo_finance" in source_name:
                    enhanced_df = self._engineer_finance_features(df)
                elif "google_trends" in source_name:
                    enhanced_df = self._engineer_trends_features(df)
                elif "social_media" in source_name:
                    enhanced_df = self._engineer_social_features(df)
                elif "e_commerce" in source_name:
                    enhanced_df = self._engineer_ecommerce_features(df)
                else:
                    logger.warning(f"Unknown data source type: {source_name}")
                    enhanced_df = df
                
                results[source_name] = enhanced_df
                
                # Save the engineered features
                self._save_engineered_features(source_name, enhanced_df)
                
            except Exception as e:
                logger.error(f"Error engineering features for {source_name}: {str(e)}")
                results[source_name] = df  # Return original dataframe on error
        
        # Create integrated feature set
        try:
            integrated_features = self._create_integrated_features(results)
            results["integrated"] = integrated_features
            self._save_engineered_features("integrated", integrated_features)
        except Exception as e:
            logger.error(f"Error creating integrated features: {str(e)}")
        
        logger.info("Completed feature engineering process")
        return results
    
    def _load_latest_processed_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load the latest processed data files.
        
        Returns:
            Dictionary mapping source names to DataFrames
        """
        results = {}
        
        # Group files by source
        file_groups = {}
        for file_path in self.processed_dir.glob("*_processed_*.csv"):
            parts = file_path.stem.split("_processed_")
            if len(parts) >= 1:
                source_name = parts[0]
                if source_name not in file_groups:
                    file_groups[source_name] = []
                file_groups[source_name].append(file_path)
        
        # Get the latest file for each source
        for source_name, files in file_groups.items():
            if files:
                # Sort by modification time, newest first
                latest_file = sorted(files, key=lambda x: x.stat().st_mtime, reverse=True)[0]
                try:
                    df = pd.read_csv(latest_file)
                    results[source_name] = df
                    logger.info(f"Loaded {len(df)} records from {latest_file}")
                except Exception as e:
                    logger.error(f"Error loading {latest_file}: {str(e)}")
        
        return results
    
    def _engineer_finance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer features for financial data.
        
        Args:
            df: DataFrame with financial data
            
        Returns:
            DataFrame with engineered features
        """
        # Make a copy to avoid modifying the original
        result = df.copy()
        
        # Skip if DataFrame is empty or missing required columns
        if result.empty or "close" not in result.columns or "symbol" not in result.columns:
            return result
        
        # Calculate time windows from config
        time_windows = self.config.get("time_windows", [7, 14, 30, 90])
        
        # Calculate price momentum for different time windows
        for window in time_windows:
            col_name = f"momentum_{window}d"
            result[col_name] = result.groupby("symbol")["close"].transform(
                lambda x: x.pct_change(periods=window)
            )
        
        # Calculate relative strength index (RSI)
        def calculate_rsi(series, window=14):
            delta = series.diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            avg_gain = gain.rolling(window=window).mean()
            avg_loss = loss.rolling(window=window).mean()
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        
        result["rsi_14"] = result.groupby("symbol")["close"].transform(
            lambda x: calculate_rsi(x, window=14)
        )
        
        # Calculate Bollinger Bands
        result["bollinger_middle"] = result.groupby("symbol")["close"].transform(
            lambda x: x.rolling(window=20).mean()
        )
        result["bollinger_std"] = result.groupby("symbol")["close"].transform(
            lambda x: x.rolling(window=20).std()
        )
        result["bollinger_upper"] = result["bollinger_middle"] + 2 * result["bollinger_std"]
        result["bollinger_lower"] = result["bollinger_middle"] - 2 * result["bollinger_std"]
        
        # Calculate distance from Bollinger Bands (as a percentage)
        result["bollinger_position"] = (result["close"] - result["bollinger_lower"]) / (
            result["bollinger_upper"] - result["bollinger_lower"]
        )
        
        # Calculate trading volume features
        if "volume" in result.columns:
            # Volume moving average
            result["volume_ma_10"] = result.groupby("symbol")["volume"].transform(
                lambda x: x.rolling(window=10).mean()
            )
            
            # Volume ratio (current volume / average volume)
            result["volume_ratio"] = result["volume"] / result["volume_ma_10"]
            
            # On-balance volume (OBV)
            def calculate_obv(data):
                obv = pd.Series(index=data.index)
                obv.iloc[0] = 0
                
                for i in range(1, len(data)):
                    if data["close"].iloc[i] > data["close"].iloc[i-1]:
                        obv.iloc[i] = obv.iloc[i-1] + data["volume"].iloc[i]
                    elif data["close"].iloc[i] < data["close"].iloc[i-1]:
                        obv.iloc[i] = obv.iloc[i-1] - data["volume"].iloc[i]
                    else:
                        obv.iloc[i] = obv.iloc[i-1]
                
                return obv
            
            # Apply OBV calculation for each symbol
            for symbol, group in result.groupby("symbol"):
                result.loc[group.index, "obv"] = calculate_obv(group)
        
        # Normalize features if configured
        if self.config.get("normalize_features", True):
            # Normalize momentum features
            for window in time_windows:
                col_name = f"momentum_{window}d"
                if col_name in result.columns:
                    result[col_name] = result[col_name].clip(-1, 1)  # Clip to [-1, 1]
            
            # RSI is already normalized (0-100)
            
            # Bollinger position is already normalized (typically 0-1)
            
            # Normalize volume ratio
            if "volume_ratio" in result.columns:
                result["volume_ratio"] = result["volume_ratio"].clip(0, 5)  # Clip to [0, 5]
                result["volume_ratio"] = result["volume_ratio"] / 5  # Scale to [0, 1]
        
        logger.info(f"Engineered {len(result.columns) - len(df.columns)} new features for financial data")
        return result
    
    def _engineer_trends_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer features for Google Trends data.
        
        Args:
            df: DataFrame with trends data
            
        Returns:
            DataFrame with engineered features
        """
        # Make a copy to avoid modifying the original
        result = df.copy()
        
        # Skip if DataFrame is empty or missing required columns
        if result.empty or "interest" not in result.columns or "keyword" not in result.columns:
            return result
        
        # Calculate time windows from config
        time_windows = self.config.get("time_windows", [7, 14, 30, 90])
        
        # Calculate trend momentum for different time windows
        for window in time_windows:
            col_name = f"trend_momentum_{window}d"
            result[col_name] = result.groupby("keyword")["interest"].transform(
                lambda x: x.diff(periods=window)
            )
        
        # Calculate acceleration (change in momentum)
        result["trend_acceleration_7d"] = result.groupby("keyword")["momentum_7d"].transform(
            lambda x: x.diff()
        )
        
        # Calculate relative interest (compared to all keywords)
        result["date_str"] = result["date"].dt.strftime("%Y-%m-%d")
        date_totals = result.groupby("date_str")["interest"].sum().reset_index()
        date_totals.columns = ["date_str", "total_interest"]
        
        # Merge the totals back
        result = pd.merge(result, date_totals, on="date_str", how="left")
        
        # Calculate relative interest
        result["relative_interest"] = result["interest"] / result["total_interest"]
        
        # Drop temporary column
        result = result.drop(columns=["date_str"])
        
        # Calculate trend stability (inverse of standard deviation over time)
        result["trend_stability_30d"] = 1 - result.groupby("keyword")["interest"].transform(
            lambda x: x.rolling(window=30).std() / x.rolling(window=30).mean()
        ).clip(0, 1)  # Clip to [0, 1]
        
        # Calculate trend seasonality (correlation with 7-day lagged data)
        def calculate_seasonality(series, lag=7):
            return series.corr(series.shift(lag))
        
        # Apply seasonality calculation for each keyword
        for keyword, group in result.groupby("keyword"):
            if len(group) > 14:  # Need sufficient data points
                result.loc[group.index, "weekly_seasonality"] = calculate_seasonality(group["interest"], lag=7)
        
        # Normalize features if configured
        if self.config.get("normalize_features", True):
            # Normalize momentum features
            for window in time_windows:
                col_name = f"trend_momentum_{window}d"
                if col_name in result.columns:
                    max_abs = result[col_name].abs().max()
                    if max_abs > 0:
                        result[col_name] = result[col_name] / max_abs
                    result[col_name] = result[col_name].clip(-1, 1)  # Clip to [-1, 1]
            
            # Normalize acceleration
            if "trend_acceleration_7d" in result.columns:
                max_abs = result["trend_acceleration_7d"].abs().max()
                if max_abs > 0:
                    result["trend_acceleration_7d"] = result["trend_acceleration_7d"] / max_abs
                result["trend_acceleration_7d"] = result["trend_acceleration_7d"].clip(-1, 1)
            
            # Relative interest is already normalized (sum to 1)
            
            # Weekly seasonality is already normalized (-1 to 1)
        
        logger.info(f"Engineered {len(result.columns) - len(df.columns)} new features for trends data")
        return result
    
    def _engineer_social_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer features for social media data.
        
        Args:
            df: DataFrame with social media data
            
        Returns:
            DataFrame with engineered features
        """
        # Make a copy to avoid modifying the original
        result = df.copy()
        
        # Skip if DataFrame is empty or missing required columns
        if result.empty or "text" not in result.columns or "keyword" not in result.columns:
            return result
        
        # Add date-related features
        if "date" in result.columns:
            result["day_of_week"] = result["date"].dt.dayofweek
            result["hour_of_day"] = result["date"].dt.hour
            result["is_weekend"] = result["day_of_week"].isin([5, 6]).astype(int)
        
        # Calculate text length
        result["text_length"] = result["text"].str.len()
        
        # Calculate hashtag count
        result["hashtag_count"] = result["text"].str.count(r'#\w+')
        
        # Calculate mention count
        result["mention_count"] = result["text"].str.count(r'@\w+')
        
        # Calculate URL count
        result["url_count"] = result["text"].str.count(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        
        # Calculate engagement rate
        if "engagement_score" in result.columns and "followers" in result.columns:
            # Only for Twitter data where we have follower counts
            mask = (result["platform"] == "twitter") & (result["followers"] > 0)
            result.loc[mask, "engagement_rate"] = result.loc[mask, "engagement_score"] / result.loc[mask, "followers"]
        
        # Calculate keyword relevance (how many times the keyword appears in the text)
        result["keyword_count"] = result.apply(
            lambda row: row["text"].lower().count(row["keyword"].lower()),
            axis=1
        )
        
        # Calculate keyword relevance score (normalized by text length)
        mask = result["text_length"] > 0
        result.loc[mask, "keyword_relevance"] = result.loc[mask, "keyword_count"] / result.loc[mask, "text_length"]
        
        # Calculate recency score (newer content gets higher score)
        if "date" in result.columns:
            max_date = result["date"].max()
            result["days_old"] = (max_date - result["date"]).dt.total_seconds() / (24 * 3600)
            result["recency_score"] = 1 / (1 + result["days_old"])
        
        # Calculate virality score (combination of engagement and recency)
        if "engagement_score" in 
(Content truncated due to size limit. Use line ranges to read in chunks)