"""
Historical Data Analysis Dashboard Page

This module provides an interactive interface for analyzing historical
cloud burst events and validating model predictions.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data.historical_weather import HistoricalWeatherCollector


def show_historical_analysis():
    """Main function to display historical analysis page"""
    
    st.title("📊 Historical Cloud Burst Analysis")
    st.markdown("---")
    
    # Initialize collector
    if 'hist_collector' not in st.session_state:
        st.session_state.hist_collector = HistoricalWeatherCollector()
    
    collector = st.session_state.hist_collector
    
    # Tabs for different functionalities
    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 Known Events", 
        "📅 Custom Date Range", 
        "🔍 Event Validation",
        "📈 Pattern Analysis"
    ])
    
    # Tab 1: Known Cloud Burst Events
    with tab1:
        show_known_events(collector)
    
    # Tab 2: Custom Date Range Analysis
    with tab2:
        show_custom_date_range(collector)
    
    # Tab 3: Event Validation
    with tab3:
        show_event_validation()
    
    # Tab 4: Pattern Analysis
    with tab4:
        show_pattern_analysis()


def show_known_events(collector):
    """Display known cloud burst events"""
    
    st.header("Known Cloud Burst Events Database")
    st.markdown("Historical cloud burst events from across India")
    
    # Get events
    events = collector.fetch_known_cloudburst_events()
    
    # Display count
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Events", len(events))
    with col2:
        earliest = min([e['date'] for e in events])
        st.metric("Earliest Event", earliest)
    with col3:
        latest = max([e['date'] for e in events])
        st.metric("Latest Event", latest)
    
    # Convert to DataFrame for display
    events_df = pd.DataFrame(events)
    
    # Display map of events
    st.subheader("📍 Event Locations")
    
    fig = go.Figure()
    
    fig.add_trace(go.Scattermapbox(
        lat=events_df['latitude'],
        lon=events_df['longitude'],
        mode='markers',
        marker=dict(
            size=15,
            color='red',
            opacity=0.7
        ),
        text=events_df['location'],
        hovertemplate='<b>%{text}</b><br>' +
                      'Date: ' + events_df['date'] + '<br>' +
                      'Rainfall: ' + events_df['rainfall_mm'].astype(str) + ' mm<br>' +
                      '<extra></extra>'
    ))
    
    fig.update_layout(
        mapbox=dict(
            style="open-street-map",
            center=dict(lat=28.0, lon=77.0),
            zoom=4
        ),
        height=500,
        margin=dict(l=0, r=0, t=0, b=0)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Display events table
    st.subheader("📋 Event Details")
    
    # Format for display
    display_df = events_df[[
        'date', 'location', 'rainfall_mm', 'duration_hours', 
        'casualties', 'description'
    ]].copy()
    
    display_df.columns = [
        'Date', 'Location', 'Rainfall (mm)', 'Duration (hrs)',
        'Casualties', 'Description'
    ]
    
    st.dataframe(display_df, use_container_width=True, height=400)
    
    # Rainfall distribution
    st.subheader("💧 Rainfall Distribution")
    
    fig_rainfall = px.bar(
        events_df,
        x='location',
        y='rainfall_mm',
        color='rainfall_mm',
        title='Rainfall Intensity by Location',
        labels={'rainfall_mm': 'Rainfall (mm)', 'location': 'Location'},
        color_continuous_scale='Blues'
    )
    
    fig_rainfall.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_rainfall, use_container_width=True)
    
    # Download button
    csv = events_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Events Data (CSV)",
        data=csv,
        file_name=f"cloudburst_events_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )


def show_custom_date_range(collector):
    """Allow users to fetch historical data for custom date ranges"""
    
    st.header("Custom Date Range Analysis")
    st.markdown("Fetch historical weather data for any location and time period")
    
    col1, col2 = st.columns(2)
    
    with col1:
        location_name = st.text_input("Location Name", value="Mumbai")
        latitude = st.number_input("Latitude", value=19.0760, format="%.4f")
        longitude = st.number_input("Longitude", value=72.8777, format="%.4f")
    
    with col2:
        # Date range selector (max 1 year of data at a time)
        today = datetime.now()
        start_date = st.date_input(
            "Start Date",
            value=today - timedelta(days=30),
            max_value=today
        )
        end_date = st.date_input(
            "End Date",
            value=today,
            max_value=today
        )
    
    if st.button("📊 Fetch Historical Data", type="primary"):
        if start_date >= end_date:
            st.error("❌ Start date must be before end date")
            return
        
        # Check date range (max 1 year)
        if (end_date - start_date).days > 365:
            st.warning("⚠️ Date range limited to 1 year. Adjusting...")
            end_date = start_date + timedelta(days=365)
        
        with st.spinner("Fetching historical data..."):
            df = collector.fetch_date_range_data(
                latitude=latitude,
                longitude=longitude,
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d'),
                location_name=location_name
            )
        
        if df is not None and not df.empty:
            st.success(f"✅ Fetched {len(df)} hourly records")
            
            # Store in session state
            st.session_state.custom_historical_data = df
            
            # Display summary statistics
            st.subheader("📊 Summary Statistics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Avg Temperature",
                    f"{df['temperature_2m'].mean():.1f}°C"
                )
            
            with col2:
                st.metric(
                    "Avg Humidity",
                    f"{df['relative_humidity_2m'].mean():.1f}%"
                )
            
            with col3:
                st.metric(
                    "Total Precipitation",
                    f"{df['precipitation'].sum():.1f}mm"
                )
            
            with col4:
                st.metric(
                    "Avg Pressure",
                    f"{df['pressure_msl'].mean():.0f}hPa"
                )
            
            # Temperature time series
            st.subheader("🌡️ Temperature Trend")
            fig_temp = px.line(
                df,
                x='time',
                y='temperature_2m',
                title='Temperature Over Time',
                labels={'temperature_2m': 'Temperature (°C)', 'time': 'Date/Time'}
            )
            st.plotly_chart(fig_temp, use_container_width=True)
            
            # Precipitation
            st.subheader("💧 Precipitation")
            fig_precip = px.bar(
                df,
                x='time',
                y='precipitation',
                title='Hourly Precipitation',
                labels={'precipitation': 'Precipitation (mm)', 'time': 'Date/Time'}
            )
            st.plotly_chart(fig_precip, use_container_width=True)
            
            # Download button
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download Data (CSV)",
                data=csv,
                file_name=f"historical_{location_name}_{start_date}_{end_date}.csv",
                mime="text/csv"
            )
        else:
            st.error("❌ Failed to fetch historical data")


def show_event_validation():
    """Show validation results for historical events"""
    
    st.header("Event Validation Results")
    st.markdown("Model performance on known cloud burst events")
    
    # Check if validation results exist
    results_path = Path("./data/historical/validation_results.json")
    
    if not results_path.exists():
        st.warning("⚠️ No validation results found. Run validation first.")
        
        if st.button("🔍 Run Historical Validation"):
            st.info("This will validate the model against all known events...")
            st.code("""
# Run this command in terminal:
python src/models/historical_validation.py
            """, language="bash")
        
        return
    
    # Load results
    import json
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    # Display overall metrics
    st.subheader("📊 Overall Performance")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Accuracy",
            f"{results.get('average_accuracy', 0):.1%}"
        )
    
    with col2:
        st.metric(
            "Precision",
            f"{results.get('average_precision', 0):.1%}"
        )
    
    with col3:
        st.metric(
            "Recall",
            f"{results.get('average_recall', 0):.1%}"
        )
    
    with col4:
        st.metric(
            "F1 Score",
            f"{results.get('average_f1_score', 0):.1%}"
        )
    
    # Events validated
    st.metric("Events Validated", results.get('total_events_validated', 0))
    
    if 'average_warning_time_hours' in results:
        st.metric(
            "Average Warning Time",
            f"{results['average_warning_time_hours']:.1f} hours"
        )
    
    # Individual event results
    st.subheader("📋 Individual Event Performance")
    
    if 'individual_events' in results:
        events_data = []
        
        for event_result in results['individual_events']:
            event_info = event_result.get('event', {})
            
            events_data.append({
                'Location': event_info.get('location', 'Unknown'),
                'Date': event_info.get('date', 'Unknown'),
                'Accuracy': f"{event_result.get('accuracy', 0):.1%}",
                'Precision': f"{event_result.get('precision', 0):.1%}",
                'Recall': f"{event_result.get('recall', 0):.1%}",
                'F1 Score': f"{event_result.get('f1_score', 0):.1%}",
                'Warning (hrs)': f"{event_result.get('warning_time_hours', 0):.1f}"
            })
        
        events_df = pd.DataFrame(events_data)
        st.dataframe(events_df, use_container_width=True, height=400)
    
    # Show validation report if exists
    report_path = Path("./data/historical/validation_report.txt")
    if report_path.exists():
        with st.expander("📄 View Full Validation Report"):
            with open(report_path, 'r') as f:
                st.text(f.read())


def show_pattern_analysis():
    """Analyze patterns in historical data"""
    
    st.header("Pattern Analysis")
    st.markdown("Analyze weather patterns from historical cloud burst events")
    
    # Check if historical dataset exists
    data_path = Path("./data/historical/historical_cloudburst_data.csv")
    
    if not data_path.exists():
        st.warning("⚠️ Historical dataset not found. Build it first.")
        
        if st.button("🏗️ Build Historical Dataset"):
            st.info("This will collect data for all known events...")
            st.code("""
# Run this command in terminal:
python src/data/historical_weather.py
            """, language="bash")
        
        return
    
    # Load dataset
    with st.spinner("Loading historical dataset..."):
        df = pd.read_csv(data_path)
        df['time'] = pd.to_datetime(df['time'])
    
    st.success(f"✅ Loaded {len(df)} records from {df['event_location'].nunique()} events")
    
    # Overall statistics
    st.subheader("📊 Cloud Burst Conditions")
    
    # Filter for during cloud burst periods
    cloudburst_data = df[df['during_cloudburst'] == 1]
    normal_data = df[df['during_cloudburst'] == 0]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**During Cloud Burst**")
        st.metric("Avg Temperature", f"{cloudburst_data['temperature_2m'].mean():.1f}°C")
        st.metric("Avg Humidity", f"{cloudburst_data['relative_humidity_2m'].mean():.1f}%")
        st.metric("Avg Precipitation", f"{cloudburst_data['precipitation'].mean():.1f}mm/h")
        st.metric("Avg Pressure", f"{cloudburst_data['pressure_msl'].mean():.0f}hPa")
    
    with col2:
        st.markdown("**Normal Conditions (24h before)**")
        st.metric("Avg Temperature", f"{normal_data['temperature_2m'].mean():.1f}°C")
        st.metric("Avg Humidity", f"{normal_data['relative_humidity_2m'].mean():.1f}%")
        st.metric("Avg Precipitation", f"{normal_data['precipitation'].mean():.1f}mm/h")
        st.metric("Avg Pressure", f"{normal_data['pressure_msl'].mean():.0f}hPa")
    
    # Distribution comparisons
    st.subheader("📈 Condition Distributions")
    
    # Temperature distribution
    fig_temp = go.Figure()
    fig_temp.add_trace(go.Histogram(
        x=cloudburst_data['temperature_2m'],
        name='During Cloud Burst',
        opacity=0.7,
        marker_color='red'
    ))
    fig_temp.add_trace(go.Histogram(
        x=normal_data['temperature_2m'],
        name='Normal',
        opacity=0.7,
        marker_color='blue'
    ))
    fig_temp.update_layout(
        title='Temperature Distribution',
        xaxis_title='Temperature (°C)',
        barmode='overlay'
    )
    st.plotly_chart(fig_temp, use_container_width=True)
    
    # Humidity distribution
    fig_humidity = go.Figure()
    fig_humidity.add_trace(go.Histogram(
        x=cloudburst_data['relative_humidity_2m'],
        name='During Cloud Burst',
        opacity=0.7,
        marker_color='red'
    ))
    fig_humidity.add_trace(go.Histogram(
        x=normal_data['relative_humidity_2m'],
        name='Normal',
        opacity=0.7,
        marker_color='blue'
    ))
    fig_humidity.update_layout(
        title='Humidity Distribution',
        xaxis_title='Relative Humidity (%)',
        barmode='overlay'
    )
    st.plotly_chart(fig_humidity, use_container_width=True)
    
    # Event timeline
    st.subheader("📅 Event Timeline")
    
    event_summary = df.groupby(['event_date', 'event_location']).agg({
        'event_rainfall': 'first',
        'temperature_2m': 'mean'
    }).reset_index()
    
    fig_timeline = px.scatter(
        event_summary,
        x='event_date',
        y='event_rainfall',
        size='event_rainfall',
        color='temperature_2m',
        hover_data=['event_location'],
        title='Cloud Burst Events Timeline',
        labels={
            'event_date': 'Date',
            'event_rainfall': 'Rainfall (mm)',
            'temperature_2m': 'Avg Temp (°C)'
        },
        color_continuous_scale='Reds'
    )
    st.plotly_chart(fig_timeline, use_container_width=True)


if __name__ == "__main__":
    show_historical_analysis()
