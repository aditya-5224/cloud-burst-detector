"""
Historical Data Analysis Dashboard Page

This module displays a map of India showing locations where cloud burst events have been observed.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data.historical_weather import HistoricalWeatherCollector


def show_historical_analysis():
    """Main function to display historical analysis page"""
    
    # Elegant header with description
    st.markdown("""
    <div style='text-align: center; padding: 2rem 0 1rem 0;'>
        <h1 style='font-size: 2.5rem; margin-bottom: 0.5rem;'>📍 Historical Cloud Burst Events</h1>
        <p style='font-size: 1.1rem; color: #64748b; font-weight: 400;'>
            Comprehensive mapping of observed cloud burst locations across India
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize collector
    if 'hist_collector' not in st.session_state:
        st.session_state.hist_collector = HistoricalWeatherCollector()
    
    collector = st.session_state.hist_collector
    
    # Get events with loading animation
    with st.spinner('🔄 Loading historical data...'):
        events = collector.fetch_known_cloudburst_events()
    
    # Elegant metrics cards with better spacing
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            label="Total Events Recorded",
            value=len(events),
            help="Number of documented cloud burst events in database"
        )
    with col2:
        earliest = min([e['date'] for e in events])
        st.metric(
            label="Earliest Event", 
            value=earliest,
            help="Date of the oldest recorded event"
        )
    with col3:
        latest = max([e['date'] for e in events])
        st.metric(
            label="Latest Event", 
            value=latest,
            help="Date of the most recent recorded event"
        )
    
    # Elegant spacing
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Convert to DataFrame for display
    events_df = pd.DataFrame(events)
    
    # Display map section with elegant header
    st.markdown("""
    <div style='margin-bottom: 1.5rem;'>
        <h2 style='font-size: 1.75rem; font-weight: 600; color: #1e293b;'>🗺️ Geographic Distribution</h2>
        <p style='color: #64748b; font-size: 0.95rem; margin-top: 0.5rem;'>
            Interactive map displaying cloud burst event locations across India • Hover over markers for detailed information
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create elegant map visualization
    fig = go.Figure()
    
    fig.add_trace(go.Scattermapbox(
        lat=events_df['latitude'],
        lon=events_df['longitude'],
        mode='markers',
        marker=dict(
            size=22,
            color='#dc2626',  # Professional red color
            opacity=0.85,
            symbol='circle'
        ),
        text=events_df['location'],
        customdata=events_df[['date', 'rainfall_mm', 'duration_hours']],
        hovertemplate='<b style="font-size:14px">%{text}</b><br><br>' +
                      '<b>Date:</b> %{customdata[0]}<br>' +
                      '<b>Rainfall:</b> %{customdata[1]} mm<br>' +
                      '<b>Duration:</b> %{customdata[2]} hours<br>' +
                      '<extra></extra>'
    ))
    
    # Center on India with elegant styling
    fig.update_layout(
        mapbox=dict(
            style="carto-positron",  # Cleaner, more professional map style
            center=dict(lat=22.5, lon=78.5),
            zoom=3.8
        ),
        height=750,
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    # Display map with professional configuration
    st.plotly_chart(
        fig,
        config={
            'displayModeBar': True,
            'displaylogo': False,
            'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
            'toImageButtonOptions': {
                'format': 'png',
                'filename': 'cloud_burst_locations_india',
                'height': 1080,
                'width': 1920,
                'scale': 2
            }
        }
    )
    
    # Add elegant footer information
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align: center; padding: 1.5rem; background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%); 
                border-radius: 12px; margin-top: 2rem;'>
        <p style='color: #64748b; font-size: 0.9rem; margin: 0;'>
            <b>Data Source:</b> Historical cloud burst events database • 
            <b>Coverage:</b> Pan-India monitoring • 
            <b>Last Updated:</b> 2025
        </p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    show_historical_analysis()
