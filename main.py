"""
Main script to run the entire analytics platform pipeline.
This script orchestrates the execution of all components.
"""

import os
import sys
import logging
import time
from pathlib import Path
from datetime import datetime

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import modules
from src.config import LOGS_DIR, REPORTS_DIR
from src.data_collection.data_collector import DataCollector
from src.data_processing.etl_pipeline import ETLPipeline
from src.data_processing.feature_engineering import FeatureEngineer
from src.ai_analytics.trend_analyzer import TrendAnalyzer
from src.presentation.report_generator import ReportGenerator
from src.presentation.dashboard_generator import DashboardGenerator

# Configure logging
LOGS_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / "main.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("main")


def run_pipeline():
    """Run the complete analytics platform pipeline."""
    start_time = time.time()
    logger.info("Starting analytics platform pipeline")
    
    try:
        # Step 1: Collect data
        logger.info("Step 1: Collecting data from sources")
        collector = DataCollector()
        collection_results = collector.collect_all_data()
        logger.info(f"Data collection complete. Collected data from {len(collection_results)} sources.")
        
        # Step 2: Process data (ETL)
        logger.info("Step 2: Processing data (ETL)")
        etl = ETLPipeline()
        etl_results = etl.process_all_sources()
        logger.info(f"ETL processing complete. Processed {len(etl_results)} data sources.")
        
        # Step 3: Engineer features
        logger.info("Step 3: Engineering features")
        engineer = FeatureEngineer()
        feature_results = engineer.engineer_features(etl_results)
        logger.info(f"Feature engineering complete. Created features for {len(feature_results)} data sources.")
        
        # Step 4: Analyze trends
        logger.info("Step 4: Analyzing trends")
        analyzer = TrendAnalyzer()
        analysis_results = analyzer.analyze_trends(feature_results)
        logger.info("Trend analysis complete.")
        
        # Step 5: Generate report
        logger.info("Step 5: Generating report")
        report_generator = ReportGenerator()
        report_path = report_generator.generate_report()
        if report_path:
            logger.info(f"Report generation complete. Report saved to {report_path}")
        else:
            logger.error("Report generation failed.")
        
        # Step 6: Create dashboard
        logger.info("Step 6: Creating dashboard")
        dashboard_generator = DashboardGenerator()
        dashboard = dashboard_generator.create_dashboard()
        logger.info("Dashboard creation complete.")
        
        # Create PDF version of the report if possible
        try:
            import subprocess
            pdf_path = REPORTS_DIR / f"{report_path.stem}.pdf"
            subprocess.run(["manus-md-to-pdf", str(report_path), str(pdf_path)], check=True)
            logger.info(f"PDF report generated at {pdf_path}")
        except Exception as e:
            logger.error(f"Error generating PDF report: {str(e)}")
        
        end_time = time.time()
        duration = end_time - start_time
        logger.info(f"Pipeline execution complete. Total duration: {duration:.2f} seconds")
        
        return {
            "status": "success",
            "report_path": report_path,
            "duration": duration
        }
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}", exc_info=True)
        return {
            "status": "error",
            "error": str(e)
        }


if __name__ == "__main__":
    result = run_pipeline()
    if result["status"] == "success":
        print(f"Pipeline executed successfully in {result['duration']:.2f} seconds")
        print(f"Report generated at: {result['report_path']}")
    else:
        print(f"Pipeline execution failed: {result['error']}")
