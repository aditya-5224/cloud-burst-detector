"""
Sprint 1 Verification Script

Verifies that all Sprint 1 components are working correctly.
"""

import sqlite3
from pathlib import Path

def verify_sprint1():
    """Verify Sprint 1 completion"""
    db_path = Path("data/cloudburst.db")
    
    if not db_path.exists():
        print("❌ Database not found!")
        return False
    
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    print("="*80)
    print("SPRINT 1 VERIFICATION REPORT")
    print("="*80)
    
    # Check weather data
    weather_count = cursor.execute('SELECT COUNT(*) FROM weather_data').fetchone()[0]
    print(f"\n✅ Weather Records: {weather_count:,}")
    
    # Check events
    events_count = cursor.execute('SELECT COUNT(*) FROM cloud_burst_events').fetchone()[0]
    print(f"✅ Cloud Burst Events: {events_count}")
    
    # Date range
    date_range = cursor.execute('SELECT MIN(datetime), MAX(datetime) FROM weather_data').fetchone()
    print(f"✅ Date Range: {date_range[0]} to {date_range[1]}")
    
    # Average stats
    avg_stats = cursor.execute(
        'SELECT AVG(temperature_2m), AVG(relative_humidity_2m), AVG(precipitation) FROM weather_data'
    ).fetchone()
    
    print(f"\n📊 Average Temperature: {avg_stats[0]:.1f}°C")
    print(f"📊 Average Humidity: {avg_stats[1]:.1f}%")
    print(f"📊 Average Precipitation: {avg_stats[2]:.2f} mm/h")
    
    # Event distribution
    event_dist = cursor.execute(
        'SELECT intensity, COUNT(*) FROM cloud_burst_events GROUP BY intensity'
    ).fetchall()
    
    print(f"\n🌩️ Event Distribution:")
    for intensity, count in event_dist:
        print(f"   {intensity}: {count} events")
    
    # Verify data quality
    null_count = cursor.execute(
        'SELECT COUNT(*) FROM weather_data WHERE temperature_2m IS NULL'
    ).fetchone()[0]
    
    data_quality = ((weather_count - null_count) / weather_count * 100) if weather_count > 0 else 0
    print(f"\n📈 Data Completeness: {data_quality:.2f}%")
    
    print("\n" + "="*80)
    
    # Final verdict
    if weather_count >= 4000 and events_count >= 10 and data_quality >= 95:
        print("✅ SPRINT 1: COMPLETE")
        print("🚀 READY FOR SPRINT 2: FEATURE ENGINEERING")
        success = True
    else:
        print("⚠️ SPRINT 1: INCOMPLETE")
        if weather_count < 4000:
            print(f"   - Need more weather data (have {weather_count}, need 4000+)")
        if events_count < 10:
            print(f"   - Need more labeled events (have {events_count}, need 10+)")
        if data_quality < 95:
            print(f"   - Data quality too low (have {data_quality:.2f}%, need 95%+)")
        success = False
    
    print("="*80)
    
    conn.close()
    return success

if __name__ == "__main__":
    success = verify_sprint1()
    exit(0 if success else 1)
