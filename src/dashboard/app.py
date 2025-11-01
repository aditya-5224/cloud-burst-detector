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

# Import page modules at the top to avoid re-importing
try:
    from query_validation_page import show_query_validation
    from historical_page import show_historical_analysis
except ImportError:
    from src.dashboard.query_validation_page import show_query_validation
    from src.dashboard.historical_page import show_historical_analysis

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

# Custom CSS for elegant, smooth, and professional design
st.markdown("""
<style>
    /* Import Google Fonts for professional typography */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global smooth animations */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main container with elegant fade-in */
    .main .block-container {
        animation: fadeInUp 0.5s ease-out;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    @keyframes fadeInUp {
        from { 
            opacity: 0;
            transform: translateY(20px);
        }
        to { 
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Elegant gradient header */
    h1 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        letter-spacing: -0.5px;
        margin-bottom: 1rem;
    }
    
    h2, h3 {
        font-weight: 600;
        color: #1e293b;
        letter-spacing: -0.3px;
    }
    
    /* Smooth sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%);
        border-right: 1px solid #e2e8f0;
    }
    
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #334155;
    }
    
    /* Elegant metric cards */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 600;
        color: #0f172a;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 0.875rem;
        font-weight: 500;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Smooth card-like containers */
    .element-container {
        animation: slideIn 0.4s ease-out;
    }
    
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateX(-10px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    /* Professional button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-weight: 500;
        letter-spacing: 0.3px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(102, 126, 234, 0.2);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 12px rgba(102, 126, 234, 0.3);
    }
    
    /* Elegant form inputs */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > select {
        border-radius: 8px;
        border: 1px solid #e2e8f0;
        padding: 0.75rem;
        transition: all 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus,
    .stSelectbox > div > div > select:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Smooth data frame styling */
    .dataframe {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    }
    
    /* Clean tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    /* Elegant dividers */
    hr {
        margin: 2rem 0;
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, #e2e8f0, transparent);
    }
    
    /* Success/Error message styling */
    .stSuccess, .stError, .stWarning, .stInfo {
        border-radius: 8px;
        padding: 1rem;
        animation: fadeIn 0.3s ease;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    /* Hide Streamlit branding for professional look */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Smooth scrolling */
    html {
        scroll-behavior: smooth;
    }
    
    /* Card-like sections */
    .stMarkdown {
        line-height: 1.6;
        color: #475569;
    }
</style>
""", unsafe_allow_html=True)

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

# Visualization functions
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
    # Initialize session state FIRST (before any rendering)
    if 'live_result' not in st.session_state:
        st.session_state.live_result = None
    
    config = load_config()
    
    # Navigation (in sidebar)
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["🏠 Live Dashboard", "🔍 Query Validation", "📊 Historical Analysis"],
        label_visibility="collapsed",
        key="navigation_page"
    )
    
    # Show selected page with loading placeholder
    if page == "🔍 Query Validation":
        # Create placeholder to show loading message
        placeholder = st.empty()
        placeholder.info("⏳ Loading Query Validation page...")
        show_query_validation()
        placeholder.empty()  # Clear the loading message
        return
    elif page == "📊 Historical Analysis":
        # Create placeholder to show loading message
        placeholder = st.empty()
        placeholder.info("⏳ Loading Historical Analysis page...")
        show_historical_analysis()
        placeholder.empty()  # Clear the loading message
        return
    
    # Only set title for main dashboard page
    st.title("🌧️ " + config.get('dashboard', {}).get('title', 'Cloud Burst Prediction Dashboard'))
    
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
        # No rerun needed - cache clear will take effect on next interaction
    
    # Time range selector
    hours_back = st.sidebar.slider("Hours of Historical Data", 12, 168, 48)
    
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
    
    # Session state already initialized at top of main()
    
    # Main content area
    
    # Live Weather Prediction Form (moved from sidebar to center)
    if not st.session_state.get('live_result'):
        # Show form when no results
        st.subheader("🌍 Live Cloud Burst Prediction")
        st.markdown("Enter coordinates to get real-time **cloud burst risk assessment** (extreme precipitation events)")
        
        with st.form("live_weather_form"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                lat = st.number_input("Latitude", -90.0, 90.0, 19.0760, format="%.4f", help="Mumbai: 19.0760")
            with col2:
                lon = st.number_input("Longitude", -180.0, 180.0, 72.8777, format="%.4f", help="Mumbai: 72.8777")
            with col3:
                model_choice = st.selectbox("Model", ['random_forest', 'svm', 'lstm'], index=0)
            
            live_predict_button = st.form_submit_button("🌍 Get Live Prediction", use_container_width=True)
        
        # Handle live weather prediction
        if live_predict_button:
            with st.spinner(f"🌐 Fetching live weather data for ({lat}, {lon})..."):
                live_result = get_live_prediction(lat, lon, model_choice)
                
                if live_result and live_result.get('success') is not False:
                    # Store in session state
                    st.session_state.live_result = live_result
                    st.success("✅ Live prediction complete! Results displayed below.")
                    st.rerun()  # Refresh to show results
                else:
                    st.session_state.live_result = None
                    error_msg = live_result.get('error', 'Unknown error') if live_result else 'No response from API'
                    st.error(f"❌ Failed to fetch live weather data: {error_msg}")
                    st.info("💡 Make sure the API is running on http://localhost:8000")
                    if live_result:
                        st.error(f"Error details: {live_result.get('error', 'Unknown error')}")
                    else:
                        st.error("No response from API. Check if API server is running on http://localhost:8000")
        
        st.markdown("---")
    
    # Check if we have live prediction results
    if st.session_state.live_result:
        # Display Live Weather Results
        live_result = st.session_state.live_result
        weather_data = live_result.get('weather_data', {})
        
        # Prominent banner showing live data mode
        st.success("🌐 **LIVE WEATHER DATA MODE** - Showing real-time prediction results")
        
        # Show location name prominently in header
        location_name = weather_data.get('location_name', 'Unknown Location')
        st.header(f"🌍 Live Cloud Burst Prediction - {location_name}")
        
        # Add a button to clear prediction
        col_btn1, col_btn2 = st.columns([1, 4])
        with col_btn1:
            if st.button("🔄 Clear Prediction", key="clear_prediction"):
                st.session_state.live_result = None
        with col_btn2:
            st.caption("Click to clear current prediction and make a new one")
        
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
            st.metric("Cloud Burst Probability", f"{live_result.get('probability', 0):.1%}")
        
        st.markdown("---")
        
        # Display prediction result with gauge
        st.subheader("🔮 Cloud Burst Risk Assessment")
        st.info("ℹ️ **Note:** The probability shown is for **cloud burst occurrence**, not general rainfall. Cloud bursts are extreme precipitation events (100+ mm/hour) that can cause flash floods.")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Risk gauge
            fig_gauge = create_risk_gauge(live_result.get('probability', 0))
            st.plotly_chart(fig_gauge)
        
        with col2:
            if live_result.get('prediction') == 1:
                st.error(f"⚠️ **CLOUD BURST ALERT**")
                st.error(f"Cloud Burst Probability: {live_result.get('probability', 0):.1%}")
            else:
                st.success(f"✅ **NO IMMEDIATE RISK**")
                st.success(f"Cloud Burst Probability: {live_result.get('probability', 0):.1%}")
            
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
        # No live prediction - show instructions
        st.info("👈 **Get started:** Enter coordinates above and click '🌍 Get Live Prediction' to see real-time cloud burst risk assessment")
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📍 How to Use")
            st.markdown("""
            **Step 1:** Enter your location
            - Provide latitude and longitude coordinates
            - Or use the quick select for known events
            
            **Step 2:** Select model
            - Choose from Random Forest, SVM, or LSTM
            
            **Step 3:** Get prediction
            - Click '🌍 Get Live Prediction'
            - View real-time weather data and risk assessment
            
            **Alternative:** Use manual input
            - Enter weather parameters manually
            - Get instant prediction in sidebar
            """)
        
        with col2:
            st.subheader("🎯 Features")
            st.markdown("""
            **Live Weather Data**
            - Real-time data from Open-Meteo API
            - Current temperature, humidity, pressure
            - Precipitation and cloud cover
            
            **AI Prediction**
            - Machine learning models
            - Probability-based risk assessment
            - Alert levels and recommendations
            
            **Historical Validation**
            - Query past cloud burst events
            - Validate model accuracy
            - View event database
            """)
        
        st.markdown("---")
        
        # Quick actions
        st.subheader("🚀 Quick Actions")
        qcol1, qcol2, qcol3 = st.columns(3)
        
        with qcol1:
            st.markdown("**🌍 Live Prediction**")
            st.caption("Get real-time prediction for any location")
            st.markdown("👈 Use sidebar form")
        
        with qcol2:
            st.markdown("**🔍 Query Validation**")
            st.caption("Check historical cloud burst events")
            st.markdown("📊 Switch to Query Validation page")
        
        with qcol3:
            st.markdown("**📈 Historical Analysis**")
            st.caption("View batch validation results")
            st.markdown("📊 Switch to Historical Analysis page")
    
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