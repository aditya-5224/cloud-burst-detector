"""
Streamlit Dashboard for Cloud Burst Prediction System

Interactive web dashboard displaying predictions, weather data, satellite imagery,
and real-time charts for cloud burst prediction monitoring.
"""

import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st
import pandas as pd
import numpy as np
import yaml
import requests
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from streamlit_folium import st_folium
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Cloud Burst Prediction Dashboard",
    page_icon="🌧️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load configuration
@st.cache_data
def load_config():
    """Load configuration from YAML file"""
    try:
        config_path = Path("./config/config.yaml")
        if config_path.exists():
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            # Default configuration
            return {
                'dashboard': {
                    'title': 'Cloud Burst Prediction Dashboard',
                    'refresh_interval_minutes': 15,
                    'map_center': [19.0760, 72.8777],
                    'map_zoom': 10
                },
                'regions': {
                    'default': {
                        'name': 'Mumbai Region',
                        'latitude': 19.0760,
                        'longitude': 72.8777,
                        'bbox': {'north': 19.5, 'south': 18.5, 'east': 73.2, 'west': 72.4}
                    }
                },
                'api': {'host': 'localhost', 'port': 8000}
            }
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        return {}

# API communication functions
def call_prediction_api(weather_data: Dict, satellite_data: Optional[Dict] = None) -> Dict:
    """Call the prediction API"""
    config = load_config()
    api_host = config.get('api', {}).get('host', 'localhost')
    api_port = config.get('api', {}).get('port', 8000)
    
    url = f"http://{api_host}:{api_port}/predict"
    
    payload = {
        "weather": weather_data,
        "satellite": satellite_data
    }
    
    try:
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"API call failed: {e}")
        return {"error": str(e)}

def get_api_health() -> Dict:
    """Check API health"""
    config = load_config()
    api_host = config.get('api', {}).get('host', 'localhost')
    api_port = config.get('api', {}).get('port', 8000)
    
    url = f"http://{api_host}:{api_port}/health"
    
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"status": "unhealthy", "error": str(e)}

def get_live_weather(latitude: float, longitude: float, force_refresh: bool = False) -> Optional[Dict]:
    """Fetch live weather data from API"""
    config = load_config()
    api_host = config.get('api', {}).get('host', 'localhost')
    api_port = config.get('api', {}).get('port', 8000)
    
    url = f"http://{api_host}:{api_port}/weather/live"
    
    payload = {
        "latitude": latitude,
        "longitude": longitude,
        "force_refresh": force_refresh
    }
    
    try:
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Live weather fetch failed: {e}")
        return None

def get_live_prediction(latitude: float, longitude: float, model: str = 'random_forest') -> Optional[Dict]:
    """Get prediction using live weather data"""
    config = load_config()
    api_host = config.get('api', {}).get('host', 'localhost')
    api_port = config.get('api', {}).get('port', 8000)
    
    url = f"http://{api_host}:{api_port}/predict/live"
    
    payload = {
        "latitude": latitude,
        "longitude": longitude,
        "model": model
    }
    
    try:
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        result = response.json()
        
        # Add success flag if not present
        if 'success' not in result:
            result['success'] = True
            
        return result
    except requests.exceptions.RequestException as e:
        logger.error(f"Live prediction failed: {e}")
        return {'success': False, 'error': str(e)}

# Generate sample data functions
@st.cache_data(ttl=300)  # Cache for 5 minutes
def generate_sample_weather_data(hours: int = 48) -> pd.DataFrame:
    """Generate sample weather data for demonstration"""
    np.random.seed(42)
    
    dates = pd.date_range(
        start=datetime.now() - timedelta(hours=hours),
        end=datetime.now(),
        freq='h'
    )
    
    # Generate realistic weather patterns
    temp_base = 25 + 5 * np.sin(np.arange(len(dates)) * 2 * np.pi / 24)  # Daily cycle
    
    data = pd.DataFrame({
        'datetime': dates,
        'temperature_2m': temp_base + np.random.normal(0, 2, len(dates)),
        'relative_humidity_2m': np.clip(70 + 20 * np.sin(np.arange(len(dates)) * 2 * np.pi / 24) + 
                                       np.random.normal(0, 5, len(dates)), 30, 95),
        'pressure_msl': 1013 + np.random.normal(0, 8, len(dates)),
        'wind_speed_10m': np.clip(np.random.exponential(4, len(dates)), 0, 25),
        'wind_direction_10m': np.random.uniform(0, 360, len(dates)),
        'cloud_cover': np.clip(np.random.uniform(20, 90, len(dates)), 0, 100),
        'precipitation': np.clip(np.random.exponential(0.8, len(dates)), 0, 20)
    })
    
    return data

@st.cache_data(ttl=300)
def generate_sample_predictions(weather_data: pd.DataFrame) -> pd.DataFrame:
    """Generate sample predictions based on weather conditions"""
    predictions = []
    
    for _, row in weather_data.iterrows():
        # Create conditions that increase cloud burst probability
        high_humidity = row['relative_humidity_2m'] > 80
        high_temp = row['temperature_2m'] > 28
        low_pressure = row['pressure_msl'] < 1005
        high_clouds = row['cloud_cover'] > 70
        
        # Calculate probability based on conditions
        probability = 0.1  # Base probability
        
        if high_humidity:
            probability += 0.3
        if high_temp:
            probability += 0.2
        if low_pressure:
            probability += 0.25
        if high_clouds:
            probability += 0.15
        
        # Add some randomness
        probability += np.random.normal(0, 0.1)
        probability = np.clip(probability, 0, 1)
        
        prediction = 1 if probability > 0.5 else 0
        
        predictions.append({
            'datetime': row['datetime'],
            'prediction': prediction,
            'probability': probability,
            'confidence': 'High' if probability > 0.8 else 'Medium' if probability > 0.6 else 'Low'
        })
    
    return pd.DataFrame(predictions)

# Visualization functions
def create_weather_chart(data: pd.DataFrame) -> go.Figure:
    """Create weather data time series chart"""
    fig = make_subplots(
        rows=4, cols=1,
        subplot_titles=('Temperature & Humidity', 'Pressure', 'Wind Speed', 'Cloud Cover & Precipitation'),
        vertical_spacing=0.08
    )
    
    # Temperature and Humidity
    fig.add_trace(
        go.Scatter(x=data['datetime'], y=data['temperature_2m'], 
                  name='Temperature (°C)', line=dict(color='red')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=data['datetime'], y=data['relative_humidity_2m'], 
                  name='Humidity (%)', yaxis='y2', line=dict(color='blue')),
        row=1, col=1
    )
    
    # Pressure
    fig.add_trace(
        go.Scatter(x=data['datetime'], y=data['pressure_msl'], 
                  name='Pressure (hPa)', line=dict(color='green')),
        row=2, col=1
    )
    
    # Wind Speed
    fig.add_trace(
        go.Scatter(x=data['datetime'], y=data['wind_speed_10m'], 
                  name='Wind Speed (m/s)', line=dict(color='orange')),
        row=3, col=1
    )
    
    # Cloud Cover and Precipitation
    fig.add_trace(
        go.Scatter(x=data['datetime'], y=data['cloud_cover'], 
                  name='Cloud Cover (%)', line=dict(color='gray')),
        row=4, col=1
    )
    fig.add_trace(
        go.Scatter(x=data['datetime'], y=data['precipitation'], 
                  name='Precipitation (mm)', yaxis='y8', line=dict(color='cyan')),
        row=4, col=1
    )
    
    fig.update_layout(height=800, showlegend=True, title_text="Weather Data Trends")
    return fig

def create_prediction_chart(predictions: pd.DataFrame) -> go.Figure:
    """Create prediction probability chart"""
    fig = go.Figure()
    
    # Add probability line
    fig.add_trace(go.Scatter(
        x=predictions['datetime'],
        y=predictions['probability'],
        mode='lines+markers',
        name='Cloud Burst Probability',
        line=dict(color='purple', width=2),
        marker=dict(size=4)
    ))
    
    # Add threshold line
    fig.add_hline(y=0.5, line_dash="dash", line_color="red", 
                  annotation_text="Decision Threshold (50%)")
    
    # Color background based on prediction
    for i, row in predictions.iterrows():
        if row['prediction'] == 1:
            fig.add_vrect(
                x0=row['datetime'] - timedelta(minutes=30),
                x1=row['datetime'] + timedelta(minutes=30),
                fillcolor="red", opacity=0.2,
                line_width=0
            )
    
    fig.update_layout(
        title="Cloud Burst Prediction Probability",
        xaxis_title="Time",
        yaxis_title="Probability",
        height=400
    )
    
    return fig

def create_map_visualization(config: Dict) -> folium.Map:
    """Create interactive map with region overlay"""
    region = config['regions']['default']
    center = [region['latitude'], region['longitude']]
    
    # Create map
    m = folium.Map(location=center, zoom_start=10)
    
    # Add region boundary
    bbox = region['bbox']
    bounds = [
        [bbox['south'], bbox['west']],
        [bbox['north'], bbox['east']]
    ]
    
    folium.Rectangle(
        bounds=bounds,
        popup=f"Monitoring Region: {region['name']}",
        tooltip=f"Region: {region['name']}",
        color='blue',
        fill=True,
        fillOpacity=0.2
    ).add_to(m)
    
    # Add center marker
    folium.Marker(
        center,
        popup=f"Center: {region['name']}",
        tooltip="Region Center",
        icon=folium.Icon(color='red', icon='cloud')
    ).add_to(m)
    
    return m

def create_risk_gauge(probability: float) -> go.Figure:
    """Create risk level gauge chart"""
    # Determine risk level and color
    if probability >= 0.7:
        risk_level = "HIGH"
        color = "red"
    elif probability >= 0.4:
        risk_level = "MEDIUM"
        color = "orange"
    else:
        risk_level = "LOW"
        color = "green"
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = probability * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': f"Cloud Burst Risk: {risk_level}"},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 40], 'color': "lightgray"},
                {'range': [40, 70], 'color': "gray"},
                {'range': [70, 100], 'color': "darkgray"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 70
            }
        }
    ))
    
    fig.update_layout(height=300)
    return fig

# Main dashboard function
def main():
    """Main dashboard function"""
    config = load_config()
    
    # Title and header
    st.title("🌧️ " + config.get('dashboard', {}).get('title', 'Cloud Burst Prediction Dashboard'))
    
    # Navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["🏠 Live Dashboard", "🔍 Query Validation", "� Historical Analysis"],
        label_visibility="collapsed"
    )
    
    # Show selected page
    if page == "🔍 Query Validation":
        try:
            from query_validation_page import show_query_validation
        except ImportError:
            from src.dashboard.query_validation_page import show_query_validation
        show_query_validation()
        return
    elif page == "📊 Historical Analysis":
        try:
            from historical_page import show_historical_analysis
        except ImportError:
            from src.dashboard.historical_page import show_historical_analysis
        show_historical_analysis()
        return
    
    # Sidebar for main dashboard
    st.sidebar.header("Dashboard Controls")
    
    # API Status
    st.sidebar.subheader("API Status")
    api_health = get_api_health()
    if api_health.get('status') == 'healthy':
        st.sidebar.success("✅ API Connected")
        st.sidebar.json(api_health)
    else:
        st.sidebar.error("❌ API Disconnected")
        st.sidebar.error(api_health.get('error', 'Unknown error'))
    
    # Data refresh button
    if st.sidebar.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()
    
    # Time range selector
    hours_back = st.sidebar.slider("Hours of Historical Data", 12, 168, 48)
    
    # Live Weather Section
    st.sidebar.subheader("📍 Live Weather Prediction")
    
    # Show status indicator
    if st.session_state.get('live_result') and not st.session_state.get('show_sample_data', False):
        st.sidebar.info("✅ **ACTIVE:** Displaying live prediction results in main view")
    else:
        st.sidebar.write("Get real-time predictions for any location")
    
    with st.sidebar.form("live_weather_form"):
        lat = st.number_input("Latitude", -90.0, 90.0, 19.0760, format="%.4f", help="Mumbai: 19.0760")
        lon = st.number_input("Longitude", -180.0, 180.0, 72.8777, format="%.4f", help="Mumbai: 72.8777")
        model_choice = st.selectbox("Model", ['random_forest', 'svm', 'lstm'], index=0)
        live_predict_button = st.form_submit_button("🌍 Get Live Prediction")
    
    # Manual prediction inputs
    st.sidebar.subheader("Manual Prediction")
    with st.sidebar.form("prediction_form"):
        temp = st.number_input("Temperature (°C)", -10.0, 50.0, 25.0)
        humidity = st.number_input("Humidity (%)", 0.0, 100.0, 70.0)
        pressure = st.number_input("Pressure (hPa)", 900.0, 1100.0, 1013.0)
        wind_speed = st.number_input("Wind Speed (m/s)", 0.0, 30.0, 5.0)
        wind_dir = st.number_input("Wind Direction (°)", 0.0, 360.0, 180.0)
        cloud_cover = st.number_input("Cloud Cover (%)", 0.0, 100.0, 50.0)
        precipitation = st.number_input("Precipitation (mm)", 0.0, 50.0, 0.0)
        
        predict_button = st.form_submit_button("🔮 Predict")
    
    # Initialize session state for live prediction
    if 'live_result' not in st.session_state:
        st.session_state.live_result = None
    if 'show_sample_data' not in st.session_state:
        st.session_state.show_sample_data = False
    
    # Handle live weather prediction
    if live_predict_button:
        with st.spinner(f"🌐 Fetching live weather data for ({lat}, {lon})..."):
            live_result = get_live_prediction(lat, lon, model_choice)
            
            if live_result and live_result.get('success') is not False:
                # Store in session state
                st.session_state.live_result = live_result
                st.session_state.show_sample_data = False
                st.success("✅ Live prediction complete!")
                st.rerun()
            else:
                st.session_state.live_result = None
                error_msg = live_result.get('error', 'Unknown error') if live_result else 'No response from API'
                st.error(f"❌ Failed to fetch live weather data: {error_msg}")
                st.info("💡 Make sure the API is running on http://localhost:8000")
                if live_result:
                    st.error(f"Error details: {live_result.get('error', 'Unknown error')}")
                else:
                    st.error("No response from API. Check if API server is running on http://localhost:8000")
    
    # Main content area
    # Check if we have live prediction results and not showing sample data
    if st.session_state.live_result and not st.session_state.show_sample_data:
        # Display Live Weather Results
        live_result = st.session_state.live_result
        weather_data = live_result.get('weather_data', {})
        
        # Prominent banner showing live data mode
        st.success("🌐 **LIVE WEATHER DATA MODE** - Showing real-time prediction results")
        
        # Show location name prominently in header
        location_name = weather_data.get('location_name', 'Unknown Location')
        st.header(f"🌍 Live Weather Prediction - {location_name}")
        
        # Add a button to clear and go back to sample data
        col_btn1, col_btn2 = st.columns([1, 4])
        with col_btn1:
            if st.button("🔄 Back to Sample Data", key="back_to_sample"):
                st.session_state.show_sample_data = True
                st.rerun()
        with col_btn2:
            st.caption("Click to return to the sample data demonstration view")
        
        # Display location details
        location_details = weather_data.get('location_details', {})
        if location_details:
            st.info(f"📍 **Location:** {location_details.get('city', 'N/A')}, {location_details.get('state', '')} {location_details.get('country', 'N/A')} | **Coordinates:** ({weather_data.get('location', {}).get('latitude', 'N/A')}, {weather_data.get('location', {}).get('longitude', 'N/A')})")
        
        # Display live weather data
        st.subheader("🌡️ Current Weather Conditions")
        
        col_w1, col_w2, col_w3, col_w4 = st.columns(4)
        with col_w1:
            st.metric("Temperature", f"{weather_data.get('temperature', 'N/A')}°C")
        with col_w2:
            st.metric("Humidity", f"{weather_data.get('humidity', 'N/A')}%")
        with col_w3:
            st.metric("Precipitation", f"{weather_data.get('precipitation', 'N/A')} mm/h")
        with col_w4:
            st.metric("Cloud Cover", f"{weather_data.get('cloud_cover', 'N/A')}%")
        
        col_w5, col_w6, col_w7, col_w8 = st.columns(4)
        with col_w5:
            st.metric("Pressure", f"{weather_data.get('pressure', 'N/A')} hPa")
        with col_w6:
            st.metric("Wind Speed", f"{weather_data.get('wind_speed', 'N/A')} km/h")
        with col_w7:
            st.metric("Risk Level", live_result.get('risk_level', 'N/A'))
        with col_w8:
            st.metric("Probability", f"{live_result.get('probability', 0):.1%}")
        
        # Display prediction result with gauge
        st.subheader("🔮 Prediction Result")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Risk gauge
            fig_gauge = create_risk_gauge(live_result.get('probability', 0))
            st.plotly_chart(fig_gauge, width='stretch')
        
        with col2:
            if live_result.get('prediction') == 1:
                st.error(f"⚠️ **CLOUD BURST ALERT**")
                st.error(f"Probability: {live_result.get('probability', 0):.1%}")
            else:
                st.success(f"✅ **NO IMMEDIATE RISK**")
                st.success(f"Probability: {live_result.get('probability', 0):.1%}")
            
            st.info(f"🤖 Model: {live_result.get('model', 'N/A')}")
            st.info(f"📡 Source: {weather_data.get('source', 'N/A')}")
            st.info(f"⏰ Time: {live_result.get('timestamp', 'N/A')[:19]}")
        
        # Show detailed result
        with st.expander("📊 View Detailed JSON Results"):
            st.json({
                'prediction': live_result.get('prediction'),
                'probability': live_result.get('probability'),
                'risk_level': live_result.get('risk_level'),
                'model': live_result.get('model'),
                'timestamp': live_result.get('timestamp'),
                'location': live_result.get('location'),
                'weather_data': weather_data
            })
    
    else:
        # Display Sample Data (Default View)
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Current risk level
            st.subheader("Current Risk Assessment (Sample Data)")
            
            # Get latest prediction (using sample data)
            weather_data = generate_sample_weather_data(hours_back)
            predictions = generate_sample_predictions(weather_data)
            
            if not predictions.empty:
                latest_prediction = predictions.iloc[-1]
                
                # Risk gauge
                fig_gauge = create_risk_gauge(latest_prediction['probability'])
                st.plotly_chart(fig_gauge, width='stretch')
                
                # Risk metrics
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("Risk Level", latest_prediction['confidence'])
                with col_b:
                    st.metric("Probability", f"{latest_prediction['probability']:.1%}")
                with col_c:
                    prediction_text = "⚠️ ALERT" if latest_prediction['prediction'] == 1 else "✅ SAFE"
                    st.metric("Status", prediction_text)
        
        with col2:
            # Region map
            st.subheader("Monitoring Region")
            map_obj = create_map_visualization(config)
            st_folium(map_obj, width=350, height=300)
        
        # Weather trends
        st.subheader("Weather Data Trends")
        if not weather_data.empty:
            fig_weather = create_weather_chart(weather_data)
            st.plotly_chart(fig_weather, width='stretch')
        
        # Prediction timeline
        st.subheader("Prediction Timeline")
        if not predictions.empty:
            fig_predictions = create_prediction_chart(predictions)
            st.plotly_chart(fig_predictions, width='stretch')
        
        # Recent alerts table
        st.subheader("Recent Alerts")
        alerts = predictions[predictions['prediction'] == 1].tail(10)
        if not alerts.empty:
            st.dataframe(alerts[['datetime', 'probability', 'confidence']], width='stretch')
        else:
            st.info("No recent cloud burst alerts")
        
        # Data tables (expandable)
        with st.expander("📊 Raw Weather Data"):
            st.dataframe(weather_data.tail(24), width='stretch')
        
        with st.expander("🔮 Prediction History"):
            st.dataframe(predictions.tail(24), width='stretch')
    
    # Handle manual prediction (outside of live/sample conditional)
    if predict_button:
        weather_input = {
            "temperature_2m": temp,
            "relative_humidity_2m": humidity,
            "pressure_msl": pressure,
            "wind_speed_10m": wind_speed,
            "wind_direction_10m": wind_dir,
            "cloud_cover": cloud_cover,
            "precipitation": precipitation
        }
        
        # Call API or simulate prediction
        result = call_prediction_api(weather_input)
        
        if "error" not in result:
            st.sidebar.success("✅ Prediction Complete")
            st.sidebar.json(result)
        else:
            # Simulate prediction locally
            probability = min(1.0, (humidity/100 + (temp-20)/30 + precipitation/10) / 3)
            prediction = 1 if probability > 0.5 else 0
            
            st.sidebar.success("✅ Local Prediction")
            st.sidebar.json({
                "prediction": prediction,
                "probability": probability,
                "confidence": "High" if probability > 0.7 else "Medium" if probability > 0.4 else "Low"
            })
        
        # Note: Manual predictions don't clear live predictions
        # They are shown in sidebar only
    
    # Footer with last update time
    st.markdown("---")
    st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()