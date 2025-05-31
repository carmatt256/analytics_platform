"""
Report Generator module for the Analytics Platform.
This module generates professional reports with visual aids based on analysis results.
"""

import os
import sys
import logging
import json
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Add the project root to the path so we can import the config
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.config import PRESENTATION, DATA_DIR, REPORTS_DIR, LOGS_DIR

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format=\'%(asctime)s - %(name)s - %(levelname)s - %(message)s\',
    handlers=[
        logging.FileHandler(LOGS_DIR / "report_generator.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("report_generator")

# Set plot style
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)


class ReportGenerator:
    """
    Generates professional reports with visual aids.
    """
    
    def __init__(self):
        """Initialize the Report Generator."""
        self.config = PRESENTATION
        self.analysis_results_dir = DATA_DIR / "analysis_results"
        self.features_dir = DATA_DIR / "features"
        self.reports_dir = REPORTS_DIR
        self.reports_dir.mkdir(exist_ok=True)
        self.report_content = []
        self.visuals = {}
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.report_name = f"market_trends_report_{self.timestamp}"
        self.report_path = self.reports_dir / f"{self.report_name}.md"
        self.visuals_dir = self.reports_dir / f"{self.report_name}_visuals"
        self.visuals_dir.mkdir(exist_ok=True)
    
    def generate_report(self) -> Path:
        """
        Generate the full market trends report.
        
        Returns:
            Path to the generated Markdown report
        """
        logger.info("Starting report generation")
        
        # Load the latest analysis results and features
        analysis_results = self._load_latest_analysis_results()
        features_data = self._load_latest_features()
        
        if not analysis_results:
            logger.error("No analysis results found. Cannot generate report.")
            return None
        
        # Generate report sections
        self._add_title_and_intro()
        self._add_executive_summary(analysis_results)
        self._add_trending_items_section(analysis_results)
        self._add_forecasting_section(analysis_results, features_data)
        self._add_clustering_section(analysis_results)
        self._add_anomaly_detection_section(analysis_results)
        self._add_conclusion()
        self._add_references()
        
        # Save the report
        self._save_report()
        
        logger.info(f"Report generation complete. Saved to {self.report_path}")
        return self.report_path
    
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
            with open(latest_file, \'r\') as f:
                results = json.load(f)
            logger.info(f"Loaded analysis results from {latest_file}")
            return results
        except Exception as e:
            logger.error(f"Error loading analysis results from {latest_file}: {str(e)}")
            return {}
    
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
                    df = pd.read_csv(latest_file, parse_dates=["date"] if "date" in pd.read_csv(latest_file, nrows=1).columns else None)
                    results[source_name] = df
                    logger.info(f"Loaded features data from {latest_file}")
                except Exception as e:
                    logger.error(f"Error loading features from {latest_file}: {str(e)}")
        
        return results
    
    def _add_title_and_intro(self):
        """Add the report title and introduction."""
        self.report_content.append(f"# US Market Trends Report - {datetime.now().strftime(\'%Y-%m-%d\')}")
        self.report_content.append("Generated by Manus AI Analytics Platform")
        self.report_content.append("\n---\n")
        self.report_content.append("## 1. Introduction")
        self.report_content.append("This report presents an analysis of the most popular products and services in the United States market over the last three months. The analysis leverages a combination of data sources including e-commerce platforms, search trends, social media discussions, and financial markets. Advanced AI techniques, including time series forecasting, clustering, and anomaly detection, have been employed to identify key trends, predict future movements, and uncover potential market opportunities.")
        self.report_content.append("The insights provided aim to support strategic decision-making for businesses seeking to capitalize on current market dynamics and gain a competitive edge.")
        self.report_content.append("\n")
    
    def _add_executive_summary(self, results: dict):
        """Add the executive summary section."""
        self.report_content.append("## 2. Executive Summary")
        
        summary_points = []
        
        # Top trending item
        if "trending_items" in results and "overall" in results["trending_items"] and results["trending_items"]["overall"]:
            top_item = results["trending_items"]["overall"][0]
            summary_points.append(f"- The most significant trend identified is **{top_item[\'name\']}** ({top_item[\'source_type\']}), showing a strong positive momentum with a trend score of {top_item[\'trend_score\']:.2f}.")
        
        # Key forecast
        if "time_series" in results and "google_trends" in results["time_series"]:
            forecasts = results["time_series"]["google_trends"]
            upward_trends = {k: v for k, v in forecasts.items() if v["trend_direction"] == "up"}
            if upward_trends:
                strongest_upward = max(upward_trends.items(), key=lambda item: item[1]["trend_strength"])
                summary_points.append(f"- Search interest for **{strongest_upward[0]}** is projected to see the strongest growth over the next 30 days, indicating increasing consumer demand.")
            else:
                summary_points.append("- Time series forecasting indicates generally stable or declining search interest for the tracked keywords over the next 30 days.")
        
        # Clustering insight
        if "clustering" in results and "e_commerce" in results["clustering"]:
            clusters = results["clustering"]["e_commerce"]
            high_trend_clusters = []
            for category, cat_data in clusters.items():
                for cluster in cat_data.get("clusters", []):
                    if cluster.get("avg_trend_score", 0) > 0.1: # Identify clusters with positive trend
                        high_trend_clusters.append((category, cluster))
            
            if high_trend_clusters:
                # Sort by trend score
                high_trend_clusters.sort(key=lambda x: x[1]["avg_trend_score"], reverse=True)
                top_cluster_cat, top_cluster_info = high_trend_clusters[0]
                summary_points.append(f"- Clustering analysis reveals distinct market segments. A notable segment in the **{top_cluster_cat}** category (Cluster {top_cluster_info[\'cluster_id\']}) shows strong positive trending behavior, characterized by {top_cluster_info.get(\'dominant_price_tier\', \'mixed\')} pricing and high average ratings.")
            else:
                summary_points.append("- Clustering analysis identified distinct market segments, but no single cluster showed exceptionally strong positive trending behavior across categories.")
        
        # Anomaly insight
        if "anomalies" in results and "integrated" in results["anomalies"] and results["anomalies"]["integrated"]["anomaly_items"]:
            anomalies = results["anomalies"]["integrated"]["anomaly_items"]
            # Find anomaly with highest trend score
            anomalies.sort(key=lambda x: x.get("integrated_trend_score", 0), reverse=True)
            top_anomaly = anomalies[0]
            summary_points.append(f"- Anomaly detection highlighted unusual activity for **{top_anomaly[\'name\']}** ({top_anomaly[\'source_type\']}), suggesting a potential emerging trend or market disruption that warrants further investigation.")
        elif "anomalies" in results and "google_trends" in results["anomalies"]:
             # Find keyword with most anomalies
            keyword_anomalies = results["anomalies"]["google_trends"]
            if keyword_anomalies:
                top_anomaly_kw = max(keyword_anomalies.items(), key=lambda item: item[1]["anomaly_count"])
                summary_points.append(f"- Anomaly detection highlighted unusual search interest patterns for **{top_anomaly_kw[0]}**, suggesting potential spikes or shifts in consumer attention.")
        
        if not summary_points:
            summary_points.append("- The analysis identified various trends across different data sources. Detailed findings are presented in the subsequent sections.")
            
        self.report_content.extend(summary_points)
        self.report_content.append("\n")
    
    def _add_trending_items_section(self, results: dict):
        """Add the trending items section."""
        self.report_content.append("## 3. Top Trending Products and Keywords")
        
        if "trending_items" not in results or not results["trending_items"]:
            self.report_content.append("No trending items data available.")
            self.report_content.append("\n")
            return
        
        trending_data = results["trending_items"]
        
        # Overall Top Trends
        if "overall" in trending_data and trending_data["overall"]:
            self.report_content.append("### 3.1. Overall Top Trends")
            self.report_content.append("The following table highlights the top 20 items (products or keywords) exhibiting the strongest positive trend scores based on an integrated analysis across all data sources.")
            
            df_overall = pd.DataFrame(trending_data["overall"])
            df_overall["trend_score"] = df_overall["trend_score"].round(3)
            
            # Create visualization
            try:
                plt.figure(figsize=(12, 8))
                sns.barplot(data=df_overall.head(10), y="name", x="trend_score", palette="viridis", hue="name", dodge=False, legend=False)
                plt.title("Top 10 Overall Trending Items by Integrated Trend Score")
                plt.xlabel("Integrated Trend Score")
                plt.ylabel("Item Name")
                plt.tight_layout()
                
                visual_path = self.visuals_dir / "top_10_overall_trends.png"
                plt.savefig(visual_path)
                plt.close()
                self.visuals["top_10_overall_trends"] = visual_path
                
                self.report_content.append(f"![Top 10 Overall Trends]({visual_path.relative_to(self.reports_dir)})")
                self.report_content.append("*Figure 1: Top 10 overall trending items by integrated trend score.*")
                self.report_content.append("\n")
                
            except Exception as e:
                logger.error(f"Error creating overall trends plot: {str(e)}")
            
            # Add table
            self.report_content.append(df_overall.to_markdown(index=False))
            self.report_content.append("\n")
        
        # Trends by Source Type
        self.report_content.append("### 3.2. Trends by Source Type")
        source_types = [key for key in trending_data if key.startswith("by_")]
        if source_types:
            self.report_content.append("The following tables show the top trending items within each primary data source type.")
            for source_key in source_types:
                source_type = source_key.replace("by_", "").capitalize()
                if trending_data[source_key]:
                    self.report_content.append(f"#### Top {source_type} Trends")
                    df_source = pd.DataFrame(trending_data[source_key])
                    df_source["trend_score"] = df_source["trend_score"].round(3)
                    self.report_content.append(df_source.to_markdown(index=False))
                    self.report_content.append("\n")
        else:
             self.report_content.append("No specific trends by source type available.")
             self.report_content.append("\n")
             
        # Trends by Category
        self.report_content.append("### 3.3. Trends by Category")
        category_keys = [key for key in trending_data if key.startswith("category_")]
        if category_keys:
            self.report_content.append("The following tables show the top trending items within specific product categories.")
            for category_key in category_keys:
                category_name = category_key.replace("category_", "").capitalize()
                if trending_data[category_key]:
                    self.report_content.append(f"#### Top Trends in {category_name}")
                    df_category = pd.DataFrame(trending_data[category_key])
                    df_category["trend_score"] = df_category["trend_score"].round(3)
                    self.report_content.append(df_category.to_markdown(index=False))
                    self.report_content.append("\n")
        else:
             self.report_content.append("No specific trends by category available.")
             self.report_content.append("\n")
             
        # Trending Keywords
        if "trending_keywords" in trending_data and trending_data["trending_keywords"]:
            self.report_content.append("### 3.4. Top Trending Keywords (Google Trends)")
            self.report_content.append("The following keywords show the highest positive momentum in search interest over the past 7 days.")
            df_keywords = pd.DataFrame(trending_data["trending_keywords"])
            df_keywords["interest"] = df_keywords["interest"].round(1)
            df_keywords["momentum_7d"] = df_keywords["momentum_7d"].round(1)
            df_keywords["percentile_rank"] = df_keywords["percentile_rank"].round(3)
            self.report_content.append(df_keywords.to_markdown(index=False))
            self.report_content.append("\n")
    
    def _add_forecasting_section(self, results: dict, features_data: dict):
        """Add the forecasting section."""
        self.report_content.append("## 4. Trend Forecasting (Next 30 Days)")

(Content truncated due to size limit. Use line ranges to read in chunks)