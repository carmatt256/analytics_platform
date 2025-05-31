"""
AI Analytics module for the Analytics Platform.
This module applies AI models to analyze processed data and identify trends.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import joblib
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from prophet import Prophet

# Add the project root to the path so we can import the config
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.config import AI_ANALYTICS, DATA_DIR, MODELS_DIR, LOGS_DIR

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / "ai_analytics.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ai_analytics")


class TrendAnalyzer:
    """
    Applies AI models to analyze data and identify trends.
    """
    
    def __init__(self):
        """Initialize the Trend Analyzer."""
        self.config = AI_ANALYTICS
        self.features_dir = DATA_DIR / "features"
        self.models_dir = MODELS_DIR
        self.models_dir.mkdir(exist_ok=True)
        self.results_dir = DATA_DIR / "analysis_results"
        self.results_dir.mkdir(exist_ok=True)
    
    def analyze_trends(self, dataframes: Dict[str, pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Analyze trends in the data using AI models.
        
        Args:
            dataframes: Dictionary of DataFrames with engineered features (optional)
            
        Returns:
            Dictionary of analysis results
        """
        logger.info("Starting trend analysis")
        
        # If no dataframes provided, load the latest feature-engineered data
        if dataframes is None:
            dataframes = self._load_latest_features()
        
        results = {}
        
        # Apply time series forecasting
        try:
            time_series_results = self._apply_time_series_forecasting(dataframes)
            results["time_series"] = time_series_results
        except Exception as e:
            logger.error(f"Error in time series forecasting: {str(e)}")
        
        # Apply clustering
        try:
            clustering_results = self._apply_clustering(dataframes)
            results["clustering"] = clustering_results
        except Exception as e:
            logger.error(f"Error in clustering: {str(e)}")
        
        # Apply anomaly detection
        try:
            anomaly_results = self._apply_anomaly_detection(dataframes)
            results["anomalies"] = anomaly_results
        except Exception as e:
            logger.error(f"Error in anomaly detection: {str(e)}")
        
        # Identify top trending products and services
        try:
            trending_items = self._identify_trending_items(dataframes)
            results["trending_items"] = trending_items
        except Exception as e:
            logger.error(f"Error identifying trending items: {str(e)}")
        
        # Save analysis results
        self._save_analysis_results(results)
        
        logger.info("Completed trend analysis")
        return results
    
    def _load_latest_features(self) -> Dict[str, pd.DataFrame]:
        """
        Load the latest feature-engineered data files.
        
        Returns:
            Dictionary mapping source names to DataFrames
        """
        results = {}
        
        # Group files by source
        file_groups = {}
        for file_path in self.features_dir.glob("*_features_*.csv"):
            parts = file_path.stem.split("_features_")
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
    
    def _apply_time_series_forecasting(self, dataframes: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Apply time series forecasting to predict future trends.
        
        Args:
            dataframes: Dictionary of DataFrames with engineered features
            
        Returns:
            Dictionary of forecasting results
        """
        logger.info("Applying time series forecasting")
        forecast_results = {}
        
        # Apply forecasting to Google Trends data
        if "google_trends" in dataframes:
            trends_df = dataframes["google_trends"]
            
            # Skip if DataFrame is empty or missing required columns
            if not trends_df.empty and "date" in trends_df.columns and "keyword" in trends_df.columns and "interest" in trends_df.columns:
                # Convert date to datetime if it's not already
                if not pd.api.types.is_datetime64_any_dtype(trends_df["date"]):
                    trends_df["date"] = pd.to_datetime(trends_df["date"])
                
                # Get unique keywords
                keywords = trends_df["keyword"].unique()
                
                # Forecast for each keyword
                keyword_forecasts = {}
                for keyword in keywords:
                    try:
                        # Filter data for this keyword
                        keyword_data = trends_df[trends_df["keyword"] == keyword].copy()
                        
                        # Skip if insufficient data
                        if len(keyword_data) < 14:  # Need at least 2 weeks of data
                            continue
                        
                        # Prepare data for Prophet
                        prophet_data = keyword_data[["date", "interest"]].rename(
                            columns={"date": "ds", "interest": "y"}
                        )
                        
                        # Create and fit model
                        model = Prophet(
                            daily_seasonality=False,
                            weekly_seasonality=True,
                            yearly_seasonality=False
                        )
                        model.fit(prophet_data)
                        
                        # Create future dataframe for prediction
                        forecast_horizon = self.config["time_series"]["forecast_horizon"]
                        future = model.make_future_dataframe(periods=forecast_horizon)
                        
                        # Make prediction
                        forecast = model.predict(future)
                        
                        # Extract relevant columns
                        forecast_result = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(forecast_horizon)
                        
                        # Convert to dictionary for easier serialization
                        forecast_dict = {
                            "dates": forecast_result["ds"].dt.strftime("%Y-%m-%d").tolist(),
                            "values": forecast_result["yhat"].tolist(),
                            "lower_bounds": forecast_result["yhat_lower"].tolist(),
                            "upper_bounds": forecast_result["yhat_upper"].tolist()
                        }
                        
                        # Calculate trend direction and strength
                        last_value = prophet_data["y"].iloc[-1]
                        forecast_end = forecast_result["yhat"].iloc[-1]
                        trend_change = (forecast_end - last_value) / last_value if last_value > 0 else 0
                        
                        trend_direction = "up" if trend_change > 0.05 else "down" if trend_change < -0.05 else "stable"
                        trend_strength = abs(trend_change)
                        
                        # Add to results
                        keyword_forecasts[keyword] = {
                            "forecast": forecast_dict,
                            "trend_direction": trend_direction,
                            "trend_strength": trend_strength,
                            "current_value": float(last_value),
                            "forecast_value": float(forecast_end)
                        }
                        
                        # Save the model
                        model_path = self.models_dir / f"prophet_{keyword.replace(' ', '_')}.pkl"
                        with open(model_path, 'wb') as f:
                            joblib.dump(model, f)
                        
                    except Exception as e:
                        logger.error(f"Error forecasting for keyword {keyword}: {str(e)}")
                
                forecast_results["google_trends"] = keyword_forecasts
                logger.info(f"Completed forecasting for {len(keyword_forecasts)} keywords")
        
        # Apply forecasting to financial data if available
        if "yahoo_finance" in dataframes:
            finance_df = dataframes["yahoo_finance"]
            
            # Skip if DataFrame is empty or missing required columns
            if not finance_df.empty and "date" in finance_df.columns and "symbol" in finance_df.columns and "close" in finance_df.columns:
                # Convert date to datetime if it's not already
                if not pd.api.types.is_datetime64_any_dtype(finance_df["date"]):
                    finance_df["date"] = pd.to_datetime(finance_df["date"])
                
                # Get unique symbols
                symbols = finance_df["symbol"].unique()
                
                # Forecast for each symbol
                symbol_forecasts = {}
                for symbol in symbols:
                    try:
                        # Filter data for this symbol
                        symbol_data = finance_df[finance_df["symbol"] == symbol].copy()
                        
                        # Skip if insufficient data
                        if len(symbol_data) < 30:  # Need at least 30 days of data
                            continue
                        
                        # Prepare data for Prophet
                        prophet_data = symbol_data[["date", "close"]].rename(
                            columns={"date": "ds", "close": "y"}
                        )
                        
                        # Create and fit model
                        model = Prophet(
                            daily_seasonality=False,
                            weekly_seasonality=True,
                            yearly_seasonality=False,
                            changepoint_prior_scale=0.05
                        )
                        model.fit(prophet_data)
                        
                        # Create future dataframe for prediction
                        forecast_horizon = self.config["time_series"]["forecast_horizon"]
                        future = model.make_future_dataframe(periods=forecast_horizon)
                        
                        # Make prediction
                        forecast = model.predict(future)
                        
                        # Extract relevant columns
                        forecast_result = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(forecast_horizon)
                        
                        # Convert to dictionary for easier serialization
                        forecast_dict = {
                            "dates": forecast_result["ds"].dt.strftime("%Y-%m-%d").tolist(),
                            "values": forecast_result["yhat"].tolist(),
                            "lower_bounds": forecast_result["yhat_lower"].tolist(),
                            "upper_bounds": forecast_result["yhat_upper"].tolist()
                        }
                        
                        # Calculate trend direction and strength
                        last_value = prophet_data["y"].iloc[-1]
                        forecast_end = forecast_result["yhat"].iloc[-1]
                        trend_change = (forecast_end - last_value) / last_value if last_value > 0 else 0
                        
                        trend_direction = "up" if trend_change > 0.05 else "down" if trend_change < -0.05 else "stable"
                        trend_strength = abs(trend_change)
                        
                        # Add to results
                        symbol_forecasts[symbol] = {
                            "forecast": forecast_dict,
                            "trend_direction": trend_direction,
                            "trend_strength": trend_strength,
                            "current_value": float(last_value),
                            "forecast_value": float(forecast_end)
                        }
                        
                    except Exception as e:
                        logger.error(f"Error forecasting for symbol {symbol}: {str(e)}")
                
                forecast_results["yahoo_finance"] = symbol_forecasts
                logger.info(f"Completed forecasting for {len(symbol_forecasts)} symbols")
        
        return forecast_results
    
    def _apply_clustering(self, dataframes: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Apply clustering to identify groups of related items.
        
        Args:
            dataframes: Dictionary of DataFrames with engineered features
            
        Returns:
            Dictionary of clustering results
        """
        logger.info("Applying clustering")
        clustering_results = {}
        
        # Apply clustering to e-commerce data
        if "e_commerce_amazon" in dataframes:
            ecommerce_df = dataframes["e_commerce_amazon"]
            
            # Skip if DataFrame is empty or missing required columns
            if not ecommerce_df.empty and "category" in ecommerce_df.columns:
                # Get unique categories
                categories = ecommerce_df["category"].unique()
                
                category_clusters = {}
                for category in categories:
                    try:
                        # Filter data for this category
                        category_data = ecommerce_df[ecommerce_df["category"] == category].copy()
                        
                        # Skip if insufficient data
                        if len(category_data) < 10:  # Need at least 10 products
                            continue
                        
                        # Select features for clustering
                        feature_cols = [
                            "price", "rating", "review_count", "popularity_score", 
                            "trend_score", "value_score"
                        ]
                        
                        # Only use columns that exist
                        available_features = [col for col in feature_cols if col in category_data.columns]
                        
                        if not available_features:
                            continue
                        
                        # Prepare data for clustering
                        X = category_data[available_f
(Content truncated due to size limit. Use line ranges to read in chunks)