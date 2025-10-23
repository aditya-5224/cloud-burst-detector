"""
Query-Based Validation Dashboard Page

Interactive interface for querying and validating model predictions
against historical cloud burst events.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import sys
from pathlib import Path
import json

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data.events_database import CloudBurstEventsDB
from src.models.query_validator import QueryBasedValidator


def show_query_validation():
    """Main function to display query validation page"""
    
    st.title("🔍 Query-Based Validation")
    st.markdown("Test your model by querying specific dates and locations")
    st.markdown("---")
    
    # Initialize
    if 'validator' not in st.session_state:
        st.session_state.validator = QueryBasedValidator()
        st.session_state.events_db = CloudBurstEventsDB()
    
    validator = st.session_state.validator
    events_db = st.session_state.events_db
    
    # Tabs
    tab1, tab2, tab3 = st.tabs([
        "🔍 Query Validation",
        "📊 Database Events",
        "📈 Batch Results"
    ])
    
    # Tab 1: Query Validation
    with tab1:
        show_query_interface(validator, events_db)
    
    # Tab 2: Database Events
    with tab2:
        show_database_events(events_db)
    
    # Tab 3: Batch Results
    with tab3:
        show_batch_results(validator)


def show_query_interface(validator, events_db):
    """Show query interface"""
    
    st.header("Query Model Performance")
    st.markdown("Enter coordinates and date to check if a cloud burst occurred and how your model predicted it")
    
    # Quick select from database
    events = events_db.get_all_events()
    event_options = ["Custom Location"] + [f"{e['event_id']}: {e['location']} ({e['date']})" for e in events]
    selected_event = st.selectbox("Quick Select from Database", event_options, key="event_selector")
    
    # Determine default values based on selection
    if selected_event != "Custom Location":
        # Extract event
        event_id = selected_event.split(":")[0]
        event = events_db.get_event_by_id(event_id)
        default_lat = event['latitude']
        default_lon = event['longitude']
        default_date = datetime.strptime(event['date'], '%Y-%m-%d')
    else:
        default_lat = 30.7346
        default_lon = 79.0669
        default_date = datetime(2023, 7, 9)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📍 Location Input")
        latitude = st.number_input("Latitude", value=default_lat, format="%.4f", key="lat_input", 
                                   help="Enter latitude coordinate (e.g., 30.7346 for Kedarnath)")
        longitude = st.number_input("Longitude", value=default_lon, format="%.4f", key="lon_input",
                                    help="Enter longitude coordinate (e.g., 79.0669 for Kedarnath)")
        query_date = st.date_input("Date", value=default_date, key="date_input",
                                   help="Select the date to check (YYYY-MM-DD)")
        
        # Show current selection
        st.info(f"📍 Selected: ({latitude:.4f}, {longitude:.4f}) on {query_date.strftime('%Y-%m-%d')}")
    
    with col2:
        st.subheader("⚙️ Analysis Options")
        hours_before = st.slider("Hours Before Event", 1, 24, 6, key="hours_before",
                                 help="Hours of data to analyze before the event")
        hours_after = st.slider("Hours After Event", 1, 12, 3, key="hours_after",
                                help="Hours of data to analyze after the event")
        
        st.info(f"📊 Analysis window: {hours_before}h before to {hours_after}h after")
        
        # Show example coordinates
        with st.expander("💡 Example Coordinates"):
            st.markdown("""
            **Known Cloud Burst Locations:**
            - Kedarnath: 30.7346, 79.0669
            - Amarnath: 34.2268, 75.5345
            - Kullu: 31.9578, 77.1092
            - Dharamshala: 32.2190, 76.3234
            - Uttarkashi: 30.7268, 78.4354
            
            **Or use Quick Select above to auto-fill!**
            """)
    
    # Query button with clear spacing
    st.markdown("---")
    
    # Large, prominent button
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        run_query = st.button("🔍 Run Validation Query", type="primary", use_container_width=True, 
                             help="Click to validate your model's prediction for this location and date")
    
    if run_query:
        if not hasattr(st.session_state.validator, 'model') or st.session_state.validator.model is None:
            st.error("❌ Model not found! Please train your model first.")
            st.code("python src/models/train.py", language="bash")
        else:
            with st.spinner("🔄 Running validation... Please wait..."):
                try:
                    result = validator.query_and_validate(
                        latitude=latitude,
                        longitude=longitude,
                        date=query_date.strftime('%Y-%m-%d'),
                        hours_before=hours_before,
                        hours_after=hours_after
                    )
                    
                    st.session_state.query_result = result
                    st.success("✅ Validation completed!")
                except Exception as e:
                    st.error(f"❌ Error during validation: {str(e)}")
                    st.info("💡 Make sure historical weather data API is accessible")
    
    # Display results
    if 'query_result' in st.session_state:
        display_query_results(st.session_state.query_result)


def display_query_results(result):
    """Display query validation results"""
    
    st.markdown("---")
    st.header("📊 Query Results")
    
    # Query info
    st.subheader("Query Parameters")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Latitude", f"{result['query']['latitude']:.4f}")
    with col2:
        st.metric("Longitude", f"{result['query']['longitude']:.4f}")
    with col3:
        st.metric("Date", result['query']['date'])
    
    # Event found?
    st.subheader("🔍 Event Detection")
    
    if result['event_found']:
        event = result['actual_event']
        
        st.success("✅ Cloud Burst Event Found in Database!")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Event ID", event['event_id'])
            st.metric("Location", event['location'])
        
        with col2:
            st.metric("Rainfall", f"{event['rainfall_mm']}mm")
            st.metric("Duration", f"{event['duration_hours']}h")
        
        with col3:
            st.metric("Intensity", f"{event['intensity_mm_per_hour']:.0f}mm/h")
            if event.get('deaths'):
                st.metric("Deaths", event['deaths'])
        
        with st.expander("📋 Full Event Details"):
            st.json(event)
    else:
        st.info("ℹ️ No cloud burst event found in database for this date/location")
    
    # Weather data
    if result.get('weather_data'):
        st.subheader("🌤️ Historical Weather Data")
        
        weather_summary = result['weather_data']['target_date_summary']
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Avg Temperature", f"{weather_summary['avg_temperature']:.1f}°C")
        with col2:
            st.metric("Avg Humidity", f"{weather_summary['avg_humidity']:.1f}%")
        with col3:
            st.metric("Total Precipitation", f"{weather_summary['total_precipitation']:.1f}mm")
        with col4:
            st.metric("Max Hourly Rain", f"{weather_summary['max_precipitation']:.1f}mm/h")
    
    # Prediction results
    if result.get('prediction'):
        st.subheader("🤖 Model Prediction")
        
        pred = result['prediction']
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            prediction_text = "YES - ALERT 🔴" if pred['predicted_cloudburst'] else "NO - Normal 🟢"
            st.metric("Predicted Cloud Burst", prediction_text)
        
        with col2:
            st.metric("Max Probability", f"{pred['max_probability']:.1%}")
        
        with col3:
            st.metric("High-Risk Hours", f"{pred['high_risk_hours']}/{pred['total_hours']}")
        
        # Hourly predictions chart
        st.markdown("#### Hourly Predictions")
        
        hourly_data = []
        for h in pred['hourly_predictions']:
            hourly_data.append({
                'Time': h['time'][-5:],  # Get HH:MM
                'Probability': h['probability'] * 100,
                'Alert': 'ALERT' if h['predicted'] == 1 else 'Normal'
            })
        
        df_hourly = pd.DataFrame(hourly_data)
        
        fig = go.Figure()
        
        # Add probability line
        fig.add_trace(go.Scatter(
            x=df_hourly['Time'],
            y=df_hourly['Probability'],
            mode='lines+markers',
            name='Cloud Burst Probability',
            line=dict(color='red', width=2),
            marker=dict(size=8)
        ))
        
        # Add threshold line
        fig.add_hline(y=70, line_dash="dash", line_color="orange", 
                      annotation_text="Alert Threshold (70%)")
        
        fig.update_layout(
            title='Hourly Cloud Burst Probability',
            xaxis_title='Time',
            yaxis_title='Probability (%)',
            hovermode='x unified',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show alerts
        alerts = df_hourly[df_hourly['Alert'] == 'ALERT']
        if len(alerts) > 0:
            st.warning(f"🚨 Alert issued for {len(alerts)} hours: {', '.join(alerts['Time'].tolist())}")
    
    # Validation results
    if result.get('validation'):
        st.subheader("✅ Validation Results")
        
        val = result['validation']
        
        col1, col2 = st.columns(2)
        
        with col1:
            actual_text = "YES ✓" if val['actual_cloudburst'] else "NO"
            st.metric("Actual Event", actual_text)
        
        with col2:
            predicted_text = "YES ✓" if val['predicted_cloudburst'] else "NO"
            st.metric("Model Predicted", predicted_text)
        
        # Show validation type
        val_type = val.get('validation_type', 'UNKNOWN')
        
        if val_type == 'TRUE_POSITIVE':
            st.success("🎯 TRUE POSITIVE - Model successfully predicted the cloud burst!")
        elif val_type == 'FALSE_NEGATIVE':
            st.error("❌ FALSE NEGATIVE - Model FAILED to predict the cloud burst!")
        elif val_type == 'FALSE_POSITIVE':
            st.warning("⚠️ FALSE POSITIVE - Model predicted cloud burst but none occurred")
        elif val_type == 'TRUE_NEGATIVE':
            st.success("✅ TRUE NEGATIVE - Model correctly identified no cloud burst")
        
        # Warning time
        if val.get('warning_time_hours') and val['warning_time_hours'] > 0:
            st.metric("⏰ Warning Time", f"{val['warning_time_hours']:.1f} hours before event")
            st.success(f"Model provided {val['warning_time_hours']:.1f} hours advance warning!")


def show_database_events(events_db):
    """Show all events in database"""
    
    st.header("📊 Cloud Burst Events Database")
    
    # Statistics
    stats = events_db.get_statistics()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Events", stats['total_events'])
    with col2:
        st.metric("Total Deaths", int(stats['total_deaths']))
    with col3:
        st.metric("Avg Rainfall", f"{stats['average_rainfall']:.0f}mm")
    with col4:
        st.metric("Max Intensity", f"{stats['max_intensity']:.0f}mm/h")
    
    # Events table
    events = events_db.get_all_events()
    
    df = pd.DataFrame(events)
    
    display_df = df[[
        'event_id', 'date', 'location', 'rainfall_mm', 
        'duration_hours', 'intensity_mm_per_hour', 'deaths'
    ]].copy()
    
    display_df.columns = [
        'Event ID', 'Date', 'Location', 'Rainfall (mm)',
        'Duration (h)', 'Intensity (mm/h)', 'Deaths'
    ]
    
    st.dataframe(display_df, use_container_width=True, height=400)
    
    # Map
    st.subheader("📍 Event Locations")
    
    fig = go.Figure()
    
    fig.add_trace(go.Scattermapbox(
        lat=df['latitude'],
        lon=df['longitude'],
        mode='markers',
        marker=dict(
            size=df['rainfall_mm'] / 10,
            color=df['intensity_mm_per_hour'],
            colorscale='Reds',
            showscale=True,
            colorbar=dict(title="Intensity<br>(mm/h)")
        ),
        text=df['location'],
        hovertemplate='<b>%{text}</b><br>' +
                      'Date: ' + df['date'] + '<br>' +
                      'Rainfall: ' + df['rainfall_mm'].astype(str) + ' mm<br>' +
                      'Intensity: ' + df['intensity_mm_per_hour'].astype(str) + ' mm/h<br>' +
                      '<extra></extra>'
    ))
    
    fig.update_layout(
        mapbox=dict(
            style="open-street-map",
            center=dict(lat=30.0, lon=78.0),
            zoom=5
        ),
        height=500,
        margin=dict(l=0, r=0, t=0, b=0)
    )
    
    st.plotly_chart(fig, use_container_width=True)


def show_batch_results(validator):
    """Show batch validation results"""
    
    st.header("📈 Batch Validation Results")
    
    results_path = Path("./data/historical/batch_validation_summary.json")
    
    if not results_path.exists():
        st.warning("⚠️ No batch validation results found.")
        
        if st.button("🚀 Run Batch Validation Now"):
            with st.spinner("Validating all events... This may take a few minutes..."):
                results = validator.batch_validate_database()
                st.success("✅ Batch validation complete!")
                st.rerun()
        
        return
    
    # Load results
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    # Display summary
    summary = results['summary']
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Events", results['total_events'])
    with col2:
        st.metric("Accuracy", f"{summary['accuracy']:.1f}%")
    with col3:
        st.metric("Recall", f"{summary['recall']:.1f}%")
    with col4:
        if summary.get('avg_warning_time'):
            st.metric("Avg Warning Time", f"{summary['avg_warning_time']:.1f}h")
    
    # Performance breakdown
    st.subheader("📊 Performance Breakdown")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Pie chart of results
        fig = go.Figure(data=[go.Pie(
            labels=['True Positives', 'False Negatives'],
            values=[summary['true_positives'], summary['false_negatives']],
            marker=dict(colors=['green', 'red'])
        )])
        fig.update_layout(title='Detection Performance', height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Metrics
        st.metric("True Positives", summary['true_positives'])
        st.metric("False Negatives", summary['false_negatives'])
        st.metric("Correct Predictions", summary['correct_predictions'])
    
    # Individual events
    st.subheader("📋 Individual Event Results")
    
    events_data = []
    for e in results['events']:
        val = e['validation']
        events_data.append({
            'Event ID': e['event_id'],
            'Location': e['location'],
            'Date': e['date'],
            'Predicted': 'YES' if val['predicted_cloudburst'] else 'NO',
            'Correct': '✅' if val['correct_prediction'] else '❌',
            'Type': val['validation_type'],
            'Warning (h)': f"{val.get('warning_time_hours', 0):.1f}" if val.get('warning_time_hours') else 'N/A'
        })
    
    df_events = pd.DataFrame(events_data)
    st.dataframe(df_events, use_container_width=True, height=400)
    
    # Download results
    csv = df_events.to_csv(index=False)
    st.download_button(
        label="📥 Download Results (CSV)",
        data=csv,
        file_name=f"batch_validation_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )


if __name__ == "__main__":
    show_query_validation()
