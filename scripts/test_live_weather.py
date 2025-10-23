"""
Test script for live weather API integration
"""

import requests
import json

# API base URL
BASE_URL = "http://localhost:8000"

print("=" * 80)
print("Testing Live Weather API Integration")
print("=" * 80)

# Test 1: Health Check
print("\n1. Testing Health Endpoint...")
response = requests.get(f"{BASE_URL}/health")
print(f"Status: {response.status_code}")
print(f"Response: {json.dumps(response.json(), indent=2)}")

# Test 2: Live Weather Data
print("\n2. Testing Live Weather Endpoint...")
weather_payload = {
    "latitude": 19.0760,
    "longitude": 72.8777,
    "force_refresh": False
}
response = requests.post(f"{BASE_URL}/weather/live", json=weather_payload)
print(f"Status: {response.status_code}")
if response.status_code == 200:
    data = response.json()
    weather = data.get('weather', {})
    print("\n📍 Mumbai Weather Data:")
    print(f"  🌡️  Temperature: {weather.get('temperature')}°C")
    print(f"  💧 Humidity: {weather.get('humidity')}%")
    print(f"  🌧️  Precipitation: {weather.get('precipitation')} mm/h")
    print(f"  ☁️  Cloud Cover: {weather.get('cloud_cover')}%")
    print(f"  🌪️  Wind Speed: {weather.get('wind_speed')} km/h")
    print(f"  ⏱️  Pressure: {weather.get('pressure')} hPa")
    print(f"  📡 Source: {weather.get('source')}")
else:
    print(f"Error: {response.text}")

# Test 3: Live Prediction
print("\n3. Testing Live Prediction Endpoint...")
prediction_payload = {
    "latitude": 19.0760,
    "longitude": 72.8777,
    "model": "random_forest"
}
response = requests.post(f"{BASE_URL}/predict/live", json=prediction_payload)
print(f"Status: {response.status_code}")
if response.status_code == 200:
    data = response.json()
    print("\n🔮 Prediction Result:")
    print(f"  Prediction: {'⚠️ CLOUD BURST ALERT' if data.get('prediction') == 1 else '✅ NO RISK'}")
    print(f"  Probability: {data.get('probability', 0):.2%}")
    print(f"  Risk Level: {data.get('risk_level')}")
    print(f"  Model: {data.get('model')}")
    print(f"  Timestamp: {data.get('timestamp')}")
else:
    print(f"Error: {response.text}")

# Test 4: Cache Statistics
print("\n4. Testing Cache Statistics...")
response = requests.get(f"{BASE_URL}/weather/cache/stats")
print(f"Status: {response.status_code}")
if response.status_code == 200:
    data = response.json()
    print(f"Cache Stats: {json.dumps(data.get('cache_stats'), indent=2)}")

print("\n" + "=" * 80)
print("✅ All tests completed!")
print("=" * 80)
