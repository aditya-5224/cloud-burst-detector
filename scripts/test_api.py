"""
API Test Script - Validate endpoints and predictions
"""

import requests
import json
import time

BASE_URL = "http://localhost:8000"

def print_section(title):
    """Print formatted section header"""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)

def test_root():
    """Test root endpoint"""
    print_section("TEST 1: Root Endpoint")
    
    try:
        response = requests.get(f"{BASE_URL}/")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"ERROR: {e}")
        return False

def test_health():
    """Test health check endpoint"""
    print_section("TEST 2: Health Check")
    
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Status Code: {response.status_code}")
        data = response.json()
        print(f"Response: {json.dumps(data, indent=2)}")
        
        if data['model_loaded']:
            print("\nMODEL STATUS: READY")
        else:
            print("\nMODEL STATUS: NOT READY")
        
        return response.status_code == 200 and data['model_loaded']
    except Exception as e:
        print(f"ERROR: {e}")
        return False

def test_model_info():
    """Test model info endpoint"""
    print_section("TEST 3: Model Information")
    
    try:
        response = requests.get(f"{BASE_URL}/model/info")
        print(f"Status Code: {response.status_code}")
        data = response.json()
        
        print(f"\nPrimary Model: {data.get('primary_model')}")
        print(f"Feature Count: {data.get('feature_count')}")
        print(f"Models Loaded: {data.get('models_loaded')}")
        
        print("\nTop 5 Important Features:")
        top_features = data.get('top_features', {})
        for i, (feature, importance) in enumerate(list(top_features.items())[:5], 1):
            print(f"  {i}. {feature}: {importance:.4f}")
        
        return response.status_code == 200
    except Exception as e:
        print(f"ERROR: {e}")
        return False

def test_get_features():
    """Test get features endpoint"""
    print_section("TEST 4: Get Required Features")
    
    try:
        response = requests.get(f"{BASE_URL}/model/features")
        print(f"Status Code: {response.status_code}")
        data = response.json()
        
        print(f"\nTotal Features Required: {data.get('feature_count')}")
        print(f"First 10 Features:")
        for i, feature in enumerate(data.get('features', [])[:10], 1):
            print(f"  {i}. {feature}")
        
        return response.status_code == 200
    except Exception as e:
        print(f"ERROR: {e}")
        return False

def test_prediction_with_dummy_data():
    """Test prediction with dummy data"""
    print_section("TEST 5: Prediction with Dummy Data")
    
    # Get required features first
    try:
        features_response = requests.get(f"{BASE_URL}/model/features")
        all_features = features_response.json()['features']
        
        # Create dummy feature dict (all zeros)
        dummy_features = {feature: 0.0 for feature in all_features}
        
        print(f"Sending prediction request with {len(dummy_features)} features...")
        
        payload = {
            "features": dummy_features,
            "model": "random_forest"
        }
        
        response = requests.post(f"{BASE_URL}/predict", json=payload)
        print(f"Status Code: {response.status_code}")
        data = response.json()
        
        print(f"\nPrediction Result:")
        print(f"  Success: {data.get('success')}")
        print(f"  Prediction: {data.get('prediction')} (0=No Cloud Burst, 1=Cloud Burst)")
        print(f"  Probability: {data.get('probability'):.4f}")
        print(f"  Risk Level: {data.get('risk_level')}")
        print(f"  Model Used: {data.get('model')}")
        print(f"  Timestamp: {data.get('timestamp')}")
        
        return response.status_code == 200 and data.get('success')
    except Exception as e:
        print(f"ERROR: {e}")
        return False

def test_prediction_with_high_risk_data():
    """Test prediction with high-risk scenario"""
    print_section("TEST 6: Prediction with High-Risk Scenario")
    
    try:
        features_response = requests.get(f"{BASE_URL}/model/features")
        all_features = features_response.json()['features']
        
        # Create high-risk feature values
        high_risk_features = {feature: 0.0 for feature in all_features}
        
        # Set high values for important features
        important_features = {
            'precipitation': 50.0,
            'precipitation_rolling_mean_3h': 40.0,
            'precipitation_div_wind_speed_10m': 10.0,
            'precipitation_rolling_std_24h': 15.0,
            'precipitation_rolling_std_12h': 12.0,
            'relative_humidity_2m': 95.0,
            'temperature_2m': 32.0
        }
        
        # Update features that exist
        for feature, value in important_features.items():
            if feature in high_risk_features:
                high_risk_features[feature] = value
        
        print("Sending prediction request with HIGH-RISK weather conditions...")
        
        payload = {
            "features": high_risk_features,
            "model": "random_forest"
        }
        
        response = requests.post(f"{BASE_URL}/predict", json=payload)
        print(f"Status Code: {response.status_code}")
        data = response.json()
        
        print(f"\nPrediction Result:")
        print(f"  Success: {data.get('success')}")
        print(f"  Prediction: {data.get('prediction')} (0=No Cloud Burst, 1=Cloud Burst)")
        print(f"  Probability: {data.get('probability'):.4f}")
        print(f"  Risk Level: {data.get('risk_level')}")
        
        if data.get('probability', 0) > 0.5:
            print("\n  ALERT: HIGH PROBABILITY OF CLOUD BURST!")
        
        return response.status_code == 200 and data.get('success')
    except Exception as e:
        print(f"ERROR: {e}")
        return False

def main():
    """Run all API tests"""
    print("\n" + "="*80)
    print("  CLOUD BURST PREDICTION API - TEST SUITE")
    print("="*80)
    print(f"\nTarget API: {BASE_URL}")
    print("Waiting for API to start...")
    
    # Wait for API to be ready
    max_retries = 10
    for i in range(max_retries):
        try:
            requests.get(f"{BASE_URL}/health", timeout=2)
            print("API is ready!")
            break
        except:
            if i < max_retries - 1:
                print(f"Waiting... ({i+1}/{max_retries})")
                time.sleep(2)
            else:
                print("\nERROR: API not responding. Please start the API first:")
                print("  python src/api/main.py")
                return
    
    # Run all tests
    results = {
        "Root Endpoint": test_root(),
        "Health Check": test_health(),
        "Model Info": test_model_info(),
        "Get Features": test_get_features(),
        "Prediction (Dummy Data)": test_prediction_with_dummy_data(),
        "Prediction (High Risk)": test_prediction_with_high_risk_data()
    }
    
    # Print summary
    print_section("TEST SUMMARY")
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    
    for test_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        symbol = "✓" if passed else "✗"
        print(f"  {symbol} {test_name}: {status}")
    
    print(f"\nTotal: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("\nALL TESTS PASSED!")
        print("API is fully operational and ready for production.")
    else:
        print(f"\n{total_tests - passed_tests} test(s) failed.")
        print("Please review the errors above.")
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
