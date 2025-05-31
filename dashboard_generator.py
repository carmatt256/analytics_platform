"""
Dashboard Generator module for the Analytics Platform.
This module creates interactive dashboards for visualizing market trends.
"""

import os
import sys
import logging
import json
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output

# Add the project root to the path so we can import the config
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.config import PRESENTATION, DATA_DIR, REPORTS_DIR, LOGS_DIR

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / "dashboard_generator.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("dashboard_generator")


class DashboardGenerator:
    """
    Creates interactive dashboards for visualizing market trends.
    """
    
    def __init__(self):
        """Initialize the Dashboard Generator."""
        self.config = PRESENTATION
        self.analysis_results_dir = DATA_DIR / "analysis_results"
        self.features_dir = DATA_DIR / "features"
        self.dashboard_dir = REPORTS_DIR / "dashboard"
        self.dashboard_dir.mkdir(exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.dashboard_assets_dir = self.dashboard_dir / "assets"
        self.dashboard_assets_dir.mkdir(exist_ok=True)
        
        # Load data
        self.analysis_results = self._load_latest_analysis_results()
        self.features_data = self._load_latest_features()
        
        # Initialize Dash app
        self.app = dash.Dash(
            __name__,
            external_stylesheets=[dbc.themes.BOOTSTRAP],
            assets_folder=str(self.dashboard_assets_dir)
        )
        self.app.title = "Market Trends Dashboard"
    
    def _load_latest_analysis_results(self) -> dict:
        """
        Load the latest analysis results file.
        
        Returns:
            Dictionary containing analysis results
        """
        json_files = list(self.analysis_results_dir.glob("analysis_results_*.json"))
        if not json_files:
            return {}
        
        # Sort by modification time, newest first
        latest_file = sorted(json_files, key=lambda x: x.stat().st_mtime, reverse=True)[0]
        
        try:
            with open(latest_file, 'r') as f:
                results = json.load(f)
            logger.info(f"Loaded analysis results from {latest_file}")
            return results
        except Exception as e:
            logger.error(f"Error loading analysis results from {latest_file}: {str(e)}")
            return {}
    
    def _load_latest_features(self) -> dict:
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
                    df = pd.read_csv(latest_file, parse_dates=["date"] if "date" in pd.read_csv(latest_file, nrows=1).columns else None)
                    results[source_name] = df
                    logger.info(f"Loaded features data from {latest_file}")
                except Exception as e:
                    logger.error(f"Error loading features from {latest_file}: {str(e)}")
        
        return results
    
    def create_dashboard(self) -> dash.Dash:
        """
        Create the interactive dashboard.
        
        Returns:
            Dash app instance
        """
        logger.info("Creating interactive dashboard")
        
        # Create layout
        self.app.layout = self._create_layout()
        
        # Add callbacks
        self._add_callbacks()
        
        logger.info("Dashboard creation complete")
        return self.app
    
    def _create_layout(self) -> html.Div:
        """
        Create the dashboard layout.
        
        Returns:
            Dash HTML layout
        """
        # Create navbar
        navbar = dbc.Navbar(
            dbc.Container(
                [
                    html.A(
                        dbc.Row(
                            [
                                dbc.Col(html.Img(src="/assets/logo.png", height="30px"), width="auto"),
                                dbc.Col(dbc.NavbarBrand("Market Trends Dashboard", className="ms-2")),
                            ],
                            align="center",
                        ),
                        href="/",
                    ),
                    dbc.NavbarToggler(id="navbar-toggler"),
                    dbc.Collapse(
                        dbc.Nav(
                            [
                                dbc.NavItem(dbc.NavLink("Overview", href="#overview")),
                                dbc.NavItem(dbc.NavLink("Trending Items", href="#trending")),
                                dbc.NavItem(dbc.NavLink("Forecasts", href="#forecasts")),
                                dbc.NavItem(dbc.NavLink("Market Segments", href="#segments")),
                                dbc.NavItem(dbc.NavLink("Anomalies", href="#anomalies")),
                            ],
                            className="ms-auto",
                            navbar=True,
                        ),
                        id="navbar-collapse",
                        navbar=True,
                    ),
                ]
            ),
            color="primary",
            dark=True,
            className="mb-4",
        )
        
        # Create overview section
        overview_section = html.Div(
            [
                html.H2("Market Trends Overview", id="overview"),
                html.P("Interactive dashboard showing the most popular products and services in the US market over the last 3 months."),
                dbc.Row(
                    [
                        dbc.Col(
                            dbc.Card(
                                dbc.CardBody(
                                    [
                                        html.H4("Top Trending Item", className="card-title"),
                                        html.Div(id="top-trending-item"),
                                    ]
                                ),
                                className="mb-4",
                            ),
                            width=4,
                        ),
                        dbc.Col(
                            dbc.Card(
                                dbc.CardBody(
                                    [
                                        html.H4("Strongest Upward Trend", className="card-title"),
                                        html.Div(id="strongest-upward-trend"),
                                    ]
                                ),
                                className="mb-4",
                            ),
                            width=4,
                        ),
                        dbc.Col(
                            dbc.Card(
                                dbc.CardBody(
                                    [
                                        html.H4("Most Significant Anomaly", className="card-title"),
                                        html.Div(id="most-significant-anomaly"),
                                    ]
                                ),
                                className="mb-4",
                            ),
                            width=4,
                        ),
                    ]
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            dbc.Card(
                                dbc.CardBody(
                                    [
                                        html.H4("Trend Score Distribution", className="card-title"),
                                        dcc.Graph(id="trend-score-distribution"),
                                    ]
                                ),
                                className="mb-4",
                            ),
                            width=12,
                        ),
                    ]
                ),
            ],
            className="mb-5",
        )
        
        # Create trending items section
        trending_section = html.Div(
            [
                html.H2("Top Trending Products and Keywords", id="trending"),
                dbc.Tabs(
                    [
                        dbc.Tab(
                            [
                                dcc.Graph(id="overall-trending-items"),
                                html.Div(id="overall-trending-table"),
                            ],
                            label="Overall Trends",
                        ),
                        dbc.Tab(
                            [
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            [
                                                html.H5("Filter by Source Type:"),
                                                dcc.Dropdown(
                                                    id="source-type-dropdown",
                                                    options=[
                                                        {"label": "Products", "value": "product"},
                                                        {"label": "Keywords", "value": "keyword"},
                                                        {"label": "Social Media", "value": "social"},
                                                    ],
                                                    value="product",
                                                ),
                                            ],
                                            width=4,
                                        ),
                                    ],
                                    className="mb-4",
                                ),
                                dcc.Graph(id="source-trending-items"),
                                html.Div(id="source-trending-table"),
                            ],
                            label="By Source Type",
                        ),
                        dbc.Tab(
                            [
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            [
                                                html.H5("Select Category:"),
                                                dcc.Dropdown(id="category-dropdown"),
                                            ],
                                            width=4,
                                        ),
                                    ],
                                    className="mb-4",
                                ),
                                dcc.Graph(id="category-trending-items"),
                                html.Div(id="category-trending-table"),
                            ],
                            label="By Category",
                        ),
                    ]
                ),
            ],
            className="mb-5",
        )
        
        # Create forecasts section
        forecasts_section = html.Div(
            [
                html.H2("Trend Forecasts (Next 30 Days)", id="forecasts"),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.H5("Select Keyword:"),
                                dcc.Dropdown(id="keyword-forecast-dropdown"),
                            ],
                            width=4,
                        ),
                    ],
                    className="mb-4",
                ),
                dcc.Graph(id="keyword-forecast-chart"),
                html.Div(id="forecast-summary-table"),
            ],
            className="mb-5",
        )
        
        # Create market segments section
        segments_section = html.Div(
            [
                html.H2("Market Segmentation", id="segments"),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.H5("Select Category:"),
                                dcc.Dropdown(id="cluster-category-dropdown"),
                            ],
                            width=4,
                        ),
                    ],
                    className="mb-4",
                ),
                dcc.Graph(id="cluster-chart"),
                html.Div(id="cluster-details-table"),
            ],
            className="mb-5",
        )
        
        # Create anomalies section
        anomalies_section = html.Div(
            [
                html.H2("Anomaly Detection", id="anomalies"),
                dbc.Tabs(
                    [
                        dbc.Tab(
                            [
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            [
                                                html.H5("Select Keyword:"),
                                                dcc.Dropdown(id="anomaly-keyword-dropdown"),
                                            ],
                                            width=4,
                                        ),
                                    ],
                                    className="mb-4",
                                ),
                                dcc.Graph(id="keyword-anomaly-chart"),
                            ],
                            label="Search Trends Anomalies",
                        ),
                        dbc.Tab(
                            [
                                dcc.Graph(id="integrated-anomalies-chart"),
                                html.Div(id="integrated-anomalies-table"),
                            ],
                            label="Integrated Data Anomalies",
                        ),
                    ]
                ),
            ],
            className="mb-5",
        )
        
        # Create footer
        footer = html.Footer(
            dbc.Container(
                [
                    html.Hr(),
                    html.P(
                        [
                            "Market Trends Dashboard • Generated by Manus AI Analytics Platform • ",
                            html.Span(datetime.now().strftime("%Y-%m-%d")),
                        ]
                    ),
                    html.P(
                        "Disclaimer: Data used in this dashboard is simulated for demonstration purposes.",
                        className="text-muted",
                    ),
                ]
            ),
            className="mt-5",
        )
        
(Content truncated due to size limit. Use line ranges to read in chunks)