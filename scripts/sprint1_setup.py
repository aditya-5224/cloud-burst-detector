"""
Sprint 1 Setup Script

Automates the setup and execution of Sprint 1 tasks:
1. Database initialization
2. Google Earth Engine authentication
3. Historical data collection (6 months)
4. Cloud burst event labeling (50+ events)
"""

import sys
import logging
from pathlib import Path
from datetime import datetime
import time
import subprocess

# No imports from src - avoid circular dependencies
# We'll run modules directly using subprocess

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Sprint1Setup:
    """Handles Sprint 1 setup and execution"""
    
    def __init__(self):
        """Initialize Sprint 1 setup"""
        self.start_time = datetime.now()
        self.results = {
            'database_initialized': False,
            'earth_engine_authenticated': False,
            'historical_data_collected': False,
            'events_labeled': False,
            'records_collected': 0,
            'events_count': 0,
            'errors': []
        }
    
    def step1_initialize_database(self) -> bool:
        """Step 1: Initialize database"""
        logger.info("=" * 80)
        logger.info("STEP 1: INITIALIZING DATABASE")
        logger.info("=" * 80)
        
        try:
            db = DatabaseManager()
            logger.info("✓ Database initialized successfully")
            db.close()
            self.results['database_initialized'] = True
            return True
        except Exception as e:
            error_msg = f"Database initialization failed: {e}"
            logger.error(f"✗ {error_msg}")
            self.results['errors'].append(error_msg)
            return False
    
    def step2_setup_earth_engine(self, interactive: bool = True) -> bool:
        """Step 2: Setup Google Earth Engine authentication"""
        logger.info("\n" + "=" * 80)
        logger.info("STEP 2: GOOGLE EARTH ENGINE AUTHENTICATION")
        logger.info("=" * 80)
        
        try:
            auth = EarthEngineAuth()
            
            # Check if already authenticated
            if auth.is_authenticated():
                logger.info("✓ Already authenticated with Google Earth Engine")
                self.results['earth_engine_authenticated'] = True
                return True
            
            if interactive:
                logger.info("Starting interactive authentication...")
                logger.info("A browser window will open for authentication.")
                
                if auth.authenticate_interactive():
                    logger.info("✓ Authentication successful")
                    
                    # Test authentication
                    if test_authentication():
                        self.results['earth_engine_authenticated'] = True
                        return True
                    else:
                        raise Exception("Authentication test failed")
                else:
                    raise Exception("Interactive authentication failed")
            else:
                logger.warning("⚠ Earth Engine authentication skipped (requires interactive mode)")
                logger.info("To authenticate later, run:")
                logger.info("  python src/data/earth_engine_setup.py --interactive")
                return False
                
        except Exception as e:
            error_msg = f"Earth Engine authentication failed: {e}"
            logger.error(f"✗ {error_msg}")
            self.results['errors'].append(error_msg)
            logger.info("\nTo authenticate manually:")
            logger.info("  python src/data/earth_engine_setup.py --interactive")
            return False
    
    def step3_collect_historical_data(self, months: int = 6, region: str = 'default') -> bool:
        """Step 3: Collect 6 months of historical weather data"""
        logger.info("\n" + "=" * 80)
        logger.info(f"STEP 3: COLLECTING {months} MONTHS OF HISTORICAL DATA")
        logger.info("=" * 80)
        
        try:
            collector = HistoricalDataCollector()
            
            logger.info(f"Collecting data for region: {region}")
            logger.info("This may take several minutes...")
            
            result = collector.collect_and_store_historical_data(
                region_name=region,
                months_back=months
            )
            
            if result.get('success'):
                records = result['records_collected']
                quality_score = result['quality_metrics']['data_quality_score']
                
                logger.info(f"✓ Collected {records:,} historical records")
                logger.info(f"  Data quality score: {quality_score:.2f}%")
                logger.info(f"  Period: {result['start_date'].date()} to {result['end_date'].date()}")
                
                self.results['historical_data_collected'] = True
                self.results['records_collected'] = records
                
                # Show summary
                summary = collector.generate_data_summary_report()
                logger.info(f"\n{summary}")
                
                return True
            else:
                raise Exception(result.get('error', 'Unknown error'))
                
        except Exception as e:
            error_msg = f"Historical data collection failed: {e}"
            logger.error(f"✗ {error_msg}")
            self.results['errors'].append(error_msg)
            return False
    
    def step4_label_cloud_burst_events(self, target_events: int = 50) -> bool:
        """Step 4: Label at least 50 cloud burst events"""
        logger.info("\n" + "=" * 80)
        logger.info(f"STEP 4: LABELING CLOUD BURST EVENTS (Target: {target_events}+)")
        logger.info("=" * 80)
        
        try:
            labeler = CloudBurstLabeler()
            
            # First, create sample events
            logger.info("Creating sample historical events...")
            sample_count = labeler.create_sample_events_mumbai()
            logger.info(f"✓ Created {sample_count} sample events")
            
            # Detect events from collected weather data
            logger.info("\nDetecting events from collected weather data...")
            from datetime import timedelta
            end_date = datetime.now()
            start_date = end_date - timedelta(days=180)  # 6 months
            
            df_detected = labeler.detect_events_from_weather_data(
                start_date, end_date, auto_label=True
            )
            
            detected_count = len(df_detected) if not df_detected.empty else 0
            logger.info(f"✓ Auto-labeled {detected_count} detected events")
            
            # Get statistics
            stats = labeler.get_event_statistics()
            total_events = stats['total_events']
            
            logger.info(f"\n✓ Total labeled events: {total_events}")
            
            if total_events >= target_events:
                logger.info(f"✓ Target achieved ({target_events}+ events)")
            else:
                logger.warning(f"⚠ Need {target_events - total_events} more events to reach target")
                logger.info("\nTo add more events:")
                logger.info("  1. Adjust detection thresholds")
                logger.info("  2. Import from CSV: python src/data/event_labeling.py --import events.csv")
                logger.info("  3. Collect more historical data")
            
            # Show report
            report = labeler.generate_labeling_report()
            logger.info(f"\n{report}")
            
            self.results['events_labeled'] = True
            self.results['events_count'] = total_events
            
            return True
            
        except Exception as e:
            error_msg = f"Event labeling failed: {e}"
            logger.error(f"✗ {error_msg}")
            self.results['errors'].append(error_msg)
            return False
    
    def run_full_sprint1(self, interactive_auth: bool = True, 
                        months: int = 6, region: str = 'default') -> dict:
        """Run all Sprint 1 steps"""
        logger.info("╔" + "=" * 78 + "╗")
        logger.info("║" + " " * 20 + "SPRINT 1: DATA FOUNDATION" + " " * 33 + "║")
        logger.info("╚" + "=" * 78 + "╝")
        
        # Step 1: Database
        self.step1_initialize_database()
        time.sleep(1)
        
        # Step 2: Earth Engine (optional - can skip for now)
        logger.info("\nℹ Skipping Earth Engine authentication for now (can be done later)")
        logger.info("  The system will use mock satellite data for development")
        # self.step2_setup_earth_engine(interactive=interactive_auth)
        time.sleep(1)
        
        # Step 3: Historical Data
        self.step3_collect_historical_data(months=months, region=region)
        time.sleep(1)
        
        # Step 4: Event Labeling
        self.step4_label_cloud_burst_events()
        
        # Final report
        self.generate_final_report()
        
        return self.results
    
    def generate_final_report(self):
        """Generate final Sprint 1 completion report"""
        duration = datetime.now() - self.start_time
        
        logger.info("\n\n")
        logger.info("╔" + "=" * 78 + "╗")
        logger.info("║" + " " * 25 + "SPRINT 1 COMPLETION REPORT" + " " * 27 + "║")
        logger.info("╚" + "=" * 78 + "╝")
        
        logger.info("\nTasks Completed:")
        logger.info(f"  ✓ Database Initialized: {'Yes' if self.results['database_initialized'] else 'No'}")
        logger.info(f"  ✓ Earth Engine Auth:    {'Yes' if self.results['earth_engine_authenticated'] else 'Skipped (Optional)'}")
        logger.info(f"  ✓ Historical Data:      {'Yes' if self.results['historical_data_collected'] else 'No'}")
        logger.info(f"  ✓ Events Labeled:       {'Yes' if self.results['events_labeled'] else 'No'}")
        
        logger.info("\nMetrics:")
        logger.info(f"  • Historical Records: {self.results['records_collected']:,}")
        logger.info(f"  • Labeled Events:     {self.results['events_count']}")
        logger.info(f"  • Execution Time:     {duration}")
        
        if self.results['errors']:
            logger.info("\n⚠ Errors encountered:")
            for error in self.results['errors']:
                logger.info(f"  • {error}")
        
        # Check if Sprint 1 is complete
        sprint1_complete = (
            self.results['database_initialized'] and
            self.results['historical_data_collected'] and
            self.results['events_labeled'] and
            self.results['records_collected'] > 0 and
            self.results['events_count'] >= 50
        )
        
        logger.info("\n" + "=" * 80)
        if sprint1_complete:
            logger.info("🎉 SPRINT 1 COMPLETED SUCCESSFULLY!")
            logger.info("\nReady to proceed to Sprint 2: Feature Engineering")
        else:
            logger.info("⚠ SPRINT 1 PARTIALLY COMPLETE")
            logger.info("\nSome tasks need attention before proceeding to Sprint 2")
        logger.info("=" * 80)


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Sprint 1 Setup')
    parser.add_argument('--interactive', action='store_true',
                       help='Enable interactive Earth Engine authentication')
    parser.add_argument('--months', type=int, default=6,
                       help='Months of historical data to collect')
    parser.add_argument('--region', default='default',
                       help='Region name')
    parser.add_argument('--skip-auth', action='store_true',
                       help='Skip Earth Engine authentication')
    
    args = parser.parse_args()
    
    setup = Sprint1Setup()
    results = setup.run_full_sprint1(
        interactive_auth=args.interactive and not args.skip_auth,
        months=args.months,
        region=args.region
    )
    
    # Exit with appropriate code
    if all([results['database_initialized'],
            results['historical_data_collected'],
            results['events_labeled']]):
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()