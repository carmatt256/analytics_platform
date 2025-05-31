"""
Configuration settings for the Analytics Platform.
This module contains all configuration parameters for the platform components.
"""

import os
from pathlib import Path

# Base directory structure
BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
LOGS_DIR = BASE_DIR / "logs"

# Create directories if they don't exist
for directory in [DATA_DIR, MODELS_DIR, REPORTS_DIR, LOGS_DIR]:
    directory.mkdir(exist_ok=True)

# Data collection settings
DATA_COLLECTION = {
    "yahoo_finance": {
        "api_base_url": "https://yahoo-finance-api.example.com",
        "symbols": ["AMZN", "AAPL", "MSFT", "GOOGL", "META", "TSLA", "WMT", "TGT", "COST", "EBAY"],
        "interval": "1d",
        "range": "3mo",
    },
    "google_trends": {
        "api_base_url": "https://trends.google.com/trends/api",
        "keywords": ["smartphone", "laptop", "smart home", "wireless earbuds", "fitness tracker"],
        "geo": "US",
        "timeframe": "today 3-m",
    },
    "social_media": {
        "platforms": ["twitter", "reddit"],
        "keywords": ["trending products", "must have gadgets", "best selling", "viral products"],
        "limit": 100,
    },
    "e_commerce": {
        "platforms": ["amazon", "walmart", "bestbuy"],
        "categories": ["electronics", "home", "fashion", "beauty", "sports"],
        "limit": 50,
    },
}

# Data processing settings
DATA_PROCESSING = {
    "etl": {
        "batch_size": 1000,
        "workers": 4,
    },
    "data_quality": {
        "min_completeness": 0.8,
        "outlier_threshold": 3.0,
    },
    "feature_engineering": {
        "time_windows": [7, 14, 30, 90],  # days
        "normalize_features": True,
    },
}

# AI analytics settings
AI_ANALYTICS = {
    "time_series": {
        "models": ["arima", "prophet", "lstm"],
        "forecast_horizon": 30,  # days
        "train_test_split": 0.8,
    },
    "nlp": {
        "models": ["sentiment", "topic", "entity"],
        "min_document_frequency": 0.01,
        "max_document_frequency": 0.95,
    },
    "clustering": {
        "algorithms": ["kmeans", "hierarchical", "dbscan"],
        "max_clusters": 10,
    },
    "anomaly_detection": {
        "algorithms": ["isolation_forest", "one_class_svm"],
        "contamination": 0.05,
    },
}

# Insight generation settings
INSIGHT_GENERATION = {
    "trend_scoring": {
        "growth_weight": 0.4,
        "volume_weight": 0.3,
        "sentiment_weight": 0.2,
        "novelty_weight": 0.1,
    },
    "opportunity_assessment": {
        "market_size_weight": 0.3,
        "competition_weight": 0.3,
        "growth_potential_weight": 0.4,
    },
}

# Presentation settings
PRESENTATION = {
    "reports": {
        "formats": ["pdf", "html", "pptx"],
        "template_dir": BASE_DIR / "templates",
    },
    "dashboards": {
        "refresh_interval": 3600,  # seconds
        "default_timeframe": "90d",
    },
    "alerts": {
        "channels": ["email", "dashboard"],
        "threshold": 0.7,  # minimum score to trigger alert
    },
}

# Logging settings
LOGGING = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": LOGS_DIR / "platform.log",
}

# API keys (in production, these would be stored securely and not in code)
API_KEYS = {
    "yahoo_finance": "dummy_key_for_yahoo_finance",
    "google_trends": "dummy_key_for_google_trends",
    "twitter": "dummy_key_for_twitter",
    "reddit": "dummy_key_for_reddit",
}
