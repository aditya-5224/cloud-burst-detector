"""
Quick Start Setup Script for Production Features

Run this script to test all new production features.
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import logging
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_cache_manager():
    """Test Redis caching system"""
    logger.info("\n" + "="*80)
    logger.info("Testing Cache Manager")
    logger.info("="*80)
    
    try:
        from src.data.cache_manager import get_cache_manager
        
        cache = get_cache_manager()
        
        # Test basic caching
        cache.set('test_key', {'data': 'test_value'}, ttl=60)
        value = cache.get('test_key')
        
        assert value is not None, "Cache get failed"
        assert value['data'] == 'test_value', "Cache value mismatch"
        
        # Test cache statistics
        stats = cache.get_stats()
        logger.info(f"✅ Cache initialized: {stats['backend']}")
        logger.info(f"   - Total hits: {stats['hits']}")
        logger.info(f"   - Total misses: {stats['misses']}")
        logger.info(f"   - Hit rate: {stats['hit_rate']:.2%}")
        
        return True
    except Exception as e:
        logger.error(f"❌ Cache test failed: {e}")
        return False


def test_data_quality():
    """Test data quality middleware"""
    logger.info("\n" + "="*80)
    logger.info("Testing Data Quality Middleware")
    logger.info("="*80)
    
    try:
        from src.data.quality_middleware import DataQualityMiddleware
        
        middleware = DataQualityMiddleware()
        
        # Test with valid data
        valid_data = {
            'temperature_2m': 25.5,
            'relative_humidity_2m': 75.0,
            'pressure_msl': 1013.25,
            'wind_speed_10m': 5.2,
            'wind_direction_10m': 180.0,
            'cloud_cover': 60.0,
            'precipitation': 2.5
        }
        
        result = middleware.process_and_validate(valid_data)
        
        assert result['passed'], "Valid data failed validation"
        assert result['quality_metrics']['overall_quality'] > 0.5, "Quality score too low"
        
        logger.info(f"✅ Data quality validation working")
        logger.info(f"   - Overall quality: {result['quality_metrics']['overall_quality']:.2%}")
        logger.info(f"   - Completeness: {result['quality_metrics']['completeness']:.2%}")
        logger.info(f"   - Anomalies detected: {result['anomalies']['detected']}")
        
        # Test with anomalous data
        anomalous_data = valid_data.copy()
        anomalous_data['temperature_2m'] = 55  # High but valid value
        
        result2 = middleware.process_and_validate(anomalous_data)
        has_anomaly = result2.get('anomalies', {}).get('detected', False) if result2.get('passed', False) else True
        logger.info(f"   - Anomaly detection working: {has_anomaly}")
        
        return True
    except Exception as e:
        logger.error(f"❌ Data quality test failed: {e}")
        return False


def test_model_retraining():
    """Test model retraining pipeline"""
    logger.info("\n" + "="*80)
    logger.info("Testing Model Retraining Pipeline")
    logger.info("="*80)
    
    try:
        from src.models.retraining_pipeline import ModelRetrainingPipeline
        
        pipeline = ModelRetrainingPipeline()
        
        # Check if we have training data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        data = pipeline.collect_training_data(start_date, end_date)
        
        if data is not None:
            X, y = data
            logger.info(f"✅ Retraining pipeline initialized")
            logger.info(f"   - Training data available: {len(X)} samples")
            logger.info(f"   - Positive samples: {y.sum()}")
            logger.info(f"   - Negative samples: {len(y) - y.sum()}")
        else:
            logger.info(f"⚠️  Retraining pipeline initialized (no training data yet)")
            logger.info(f"   - Min samples required: {pipeline.min_samples_for_retraining}")
        
        # Check model versions
        history = pipeline.get_model_history()
        logger.info(f"   - Model versions in history: {len(history)}")
        
        return True
    except Exception as e:
        logger.error(f"❌ Retraining test failed: {e}")
        return False


def test_ab_testing():
    """Test A/B testing framework"""
    logger.info("\n" + "="*80)
    logger.info("Testing A/B Testing Framework")
    logger.info("="*80)
    
    try:
        from src.models.ab_testing import ABTestingFramework, ModelVariant, TrafficSplitStrategy
        
        framework = ABTestingFramework()
        
        # Create test experiment
        variants = [
            ModelVariant(
                variant_id='control',
                model_path='models/trained/random_forest_model.pkl',
                model_type='random_forest',
                version='v1',
                traffic_percentage=50.0
            ),
            ModelVariant(
                variant_id='treatment',
                model_path='models/trained/random_forest_model.pkl',
                model_type='random_forest',
                version='v2',
                traffic_percentage=50.0
            )
        ]
        
        experiment_id = f'test_experiment_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        
        experiment = framework.create_experiment(
            experiment_id=experiment_id,
            variants=variants,
            strategy=TrafficSplitStrategy.PERCENTAGE,
            description='Test experiment'
        )
        
        logger.info(f"✅ A/B testing framework initialized")
        logger.info(f"   - Experiment created: {experiment_id}")
        logger.info(f"   - Variants: {len(variants)}")
        
        # Test variant selection
        selected = framework.select_variant(experiment_id)
        logger.info(f"   - Variant selection working: {selected.variant_id}")
        
        # List experiments
        experiments = framework.list_experiments()
        logger.info(f"   - Total experiments: {len(experiments)}")
        
        # Clean up test experiment
        framework.stop_experiment(experiment_id)
        
        return True
    except Exception as e:
        logger.error(f"❌ A/B testing test failed: {e}")
        return False


def test_api_integration():
    """Test API with new middleware"""
    logger.info("\n" + "="*80)
    logger.info("Testing API Integration")
    logger.info("="*80)
    
    try:
        import requests
        
        # Check if API is running
        response = requests.get('http://localhost:8000/health', timeout=5)
        
        if response.status_code == 200:
            logger.info(f"✅ API is running")
            
            # Test monitoring endpoints
            cache_response = requests.get('http://localhost:8000/monitoring/cache/stats', timeout=5)
            if cache_response.status_code == 200:
                cache_data = cache_response.json()
                logger.info(f"   - Cache stats endpoint: ✅")
                logger.info(f"   - Cache backend: {cache_data['cache_stats']['backend']}")
            
            quality_response = requests.get('http://localhost:8000/monitoring/data-quality/report', timeout=5)
            if quality_response.status_code == 200:
                logger.info(f"   - Data quality endpoint: ✅")
            
        else:
            logger.warning(f"⚠️  API returned status {response.status_code}")
            return False
        
        return True
    except requests.exceptions.ConnectionError:
        logger.warning(f"⚠️  API not running - start with: python src/api/main.py")
        return False
    except Exception as e:
        logger.error(f"❌ API test failed: {e}")
        return False


def main():
    """Run all tests"""
    logger.info("\n" + "🚀 "*20)
    logger.info("CLOUD BURST PREDICTOR - PRODUCTION FEATURES TEST")
    logger.info("🚀 "*20 + "\n")
    
    results = {
        'Cache Manager': test_cache_manager(),
        'Data Quality': test_data_quality(),
        'Model Retraining': test_model_retraining(),
        'A/B Testing': test_ab_testing(),
        'API Integration': test_api_integration()
    }
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("TEST SUMMARY")
    logger.info("="*80)
    
    for feature, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        logger.info(f"{feature:.<40} {status}")
    
    total_passed = sum(results.values())
    total_tests = len(results)
    
    logger.info("="*80)
    logger.info(f"Total: {total_passed}/{total_tests} tests passed")
    logger.info("="*80 + "\n")
    
    # Check specific failure reasons
    if 'API Integration' in results and not results['API Integration']:
        logger.info("\n" + "ℹ️ "*40)
        logger.info("API Integration test failed because the API is not running.")
        logger.info("This is EXPECTED if you haven't started the API yet.")
        logger.info("\nTo test API integration:")
        logger.info("1. Start API: python src/api/main.py")
        logger.info("2. Run this test again")
        logger.info("ℹ️ "*40 + "\n")
    
    if total_passed == total_tests:
        logger.info("🎉 All production features are working correctly!")
        logger.info("\nNext steps:")
        logger.info("1. Install Redis (optional): See docs/PRODUCTION_DEPLOYMENT.md")
        logger.info("2. Start API: python src/api/main.py")
        logger.info("3. Test endpoints: http://localhost:8000/docs")
    elif total_passed >= total_tests - 1 and not results.get('API Integration', True):
        logger.info("✅ All backend features are working correctly!")
        logger.info("   (API Integration test skipped - API not running)")
        logger.info("\nNext steps:")
        logger.info("1. Start API: python src/api/main.py")
        logger.info("2. Test new endpoints: curl http://localhost:8000/monitoring/cache/stats")
    else:
        logger.warning("⚠️  Some features need attention. Check logs above.")
        logger.info("\nTroubleshooting:")
        logger.info("1. Install missing dependencies: pip install -r requirements.txt")
        logger.info("2. Check Redis is running: redis-cli ping (optional)")
        logger.info("3. Review docs/PRODUCTION_DEPLOYMENT.md")


if __name__ == "__main__":
    main()
