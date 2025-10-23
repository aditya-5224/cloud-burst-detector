# Location Name Resolution Feature

## Overview
The Cloud Burst Prediction System now automatically resolves location names from latitude/longitude coordinates, making it easier to identify and understand prediction locations.

## Features

### Automatic Location Resolution
When you provide coordinates, the system automatically:
- Resolves the location name (city, state, country)
- Displays it prominently in the dashboard
- Includes it in API responses
- Caches location data for performance

### Multi-Source Geocoding with Fallback

The system uses two geocoding services with automatic fallback:

#### Primary: Nominatim (OpenStreetMap)
- **Free**: No API key required
- **Global Coverage**: Worldwide location database
- **Accuracy**: City-level precision
- **Update**: Real-time from OpenStreetMap data
- **Rate Limit**: Respectful usage (1 request per second)

#### Fallback: WeatherAPI.com Location Data
- **Integrated**: Uses existing WeatherAPI.com key
- **Accurate**: Location data from weather stations
- **Reliable**: Works when Nominatim is unavailable

### What Gets Resolved

The system extracts:
- **City**: Primary city/town/village name
- **State/Region**: State, province, or administrative region
- **Country**: Country name
- **Display Name**: Formatted as "City, State, Country"

## API Response Format

### Before (without location name):
```json
{
  "success": true,
  "weather": {
    "source": "weatherapi.com",
    "temperature": 35.4,
    "location": {
      "latitude": 19.0760,
      "longitude": 72.8777
    }
  }
}
```

### After (with location name):
```json
{
  "success": true,
  "weather": {
    "source": "weatherapi.com",
    "temperature": 35.4,
    "location": {
      "latitude": 19.0760,
      "longitude": 72.8777
    },
    "location_name": "Mumbai, Maharashtra, India",
    "location_details": {
      "city": "Mumbai",
      "state": "Maharashtra",
      "country": "India"
    }
  }
}
```

## Dashboard Display

### Live Prediction Header
The dashboard now shows:
```
🌍 Live Weather Prediction - Mumbai, Maharashtra, India
```

### Location Info Bar
```
📍 Location: Mumbai, Maharashtra India | Coordinates: (19.0760, 72.8777)
```

### Benefits
- **Immediate Recognition**: Know exactly where the prediction is for
- **Context**: Understand regional weather patterns
- **Verification**: Confirm coordinates are correct
- **User-Friendly**: No need to look up coordinates separately

## Usage Examples

### Python API
```python
from src.data.live_weather import LiveWeatherCollector

collector = LiveWeatherCollector()

# Get weather with location name
weather = collector.get_live_weather(19.0760, 72.8777)

print(f"Location: {weather['location_name']}")
# Output: Location: Mumbai, Maharashtra, India

print(f"City: {weather['location_details']['city']}")
# Output: City: Mumbai
```

### REST API
```bash
curl -X POST http://localhost:8000/weather/live \
  -H "Content-Type: application/json" \
  -d '{
    "latitude": 40.7128,
    "longitude": -74.0060
  }'
```

Response:
```json
{
  "success": true,
  "weather": {
    "location_name": "City of New York, New York, United States",
    "location_details": {
      "city": "City of New York",
      "state": "New York",
      "country": "United States"
    },
    "temperature": 11.7,
    ...
  }
}
```

### Dashboard Usage
1. Enter coordinates in sidebar (e.g., 51.5074, -0.1278)
2. Click "🌍 Get Live Prediction"
3. See result header: "🌍 Live Weather Prediction - London, England, United Kingdom"

## Global Coverage

Tested and verified for cities worldwide:

| City | Coordinates | Resolved Name |
|------|-------------|---------------|
| Mumbai, India | 19.0760, 72.8777 | Mumbai, Maharashtra, India |
| New York, USA | 40.7128, -74.0060 | City of New York, New York, United States |
| London, UK | 51.5074, -0.1278 | London, England, United Kingdom |
| Tokyo, Japan | 35.6762, 139.6503 | 杉並区, 日本 |
| Sydney, Australia | -33.8688, 151.2093 | Sydney, New South Wales, Australia |
| Paris, France | 48.8566, 2.3522 | Paris, Île-de-France, France |

## Performance Optimization

### Caching
- Location names are cached with weather data
- **Cache TTL**: 5 minutes (same as weather data)
- **Benefits**: Faster responses, reduced API calls

### Timeout Protection
- Geocoding requests timeout after 5 seconds
- Fallback to secondary source if primary fails
- System remains functional even if geocoding fails

### Graceful Degradation
If all geocoding services fail:
- System continues to work
- Shows coordinates instead: "19.0760, 72.8777"
- Weather prediction still functions normally

## Configuration

### No Additional Setup Required
Location name resolution works out-of-the-box with:
- ✅ Nominatim (no API key needed)
- ✅ WeatherAPI.com (uses existing key)

### Custom User Agent (Optional)
To customize the Nominatim user agent in `src/data/live_weather.py`:

```python
headers = {
    'User-Agent': 'YourAppName/1.0'  # Replace with your app name
}
```

## Technical Details

### Nominatim API
- **Endpoint**: https://nominatim.openstreetmap.org/reverse
- **Format**: JSON
- **Zoom Level**: 10 (city-level precision)
- **Address Details**: Full hierarchical address

### Location Extraction Logic
```python
city = address.get('city') or 
       address.get('town') or 
       address.get('village') or 
       address.get('municipality') or
       address.get('county')

state = address.get('state') or 
        address.get('state_district') or 
        address.get('region')

country = address.get('country')
```

### Error Handling
- Network errors → Fallback to secondary source
- Invalid coordinates → Return coordinate string
- Timeout → Skip geocoding, return coordinates
- All services fail → System continues with coordinates only

## Best Practices

### For Developers

1. **Always Include Coordinates**: Even with location names, coordinates are the source of truth

2. **Handle Missing Location Names**: Check if location_name exists in response
   ```python
   location = weather.get('location_name', 'Unknown Location')
   ```

3. **Use Display Name**: For UI, use `location_name` (pre-formatted)
   ```python
   title = f"Weather for {weather['location_name']}"
   ```

4. **Access Details When Needed**: For filtering/grouping, use `location_details`
   ```python
   if weather['location_details']['country'] == 'India':
       # India-specific logic
   ```

### For Users

1. **Verify Coordinates**: Check that resolved location matches your intended location

2. **Use Common Formats**: 
   - Decimal degrees: 19.0760, 72.8777 ✅
   - Not DMS: 19°4'33"N, 72°52'39"E ❌

3. **Check Location Bar**: Verify the resolved name makes sense for your prediction area

## Troubleshooting

### Issue: Location shows only coordinates
**Cause**: Geocoding services failed or timed out

**Solution**: 
1. Check internet connectivity
2. Try again (may be temporary service issue)
3. Verify coordinates are valid (-90 to 90 for lat, -180 to 180 for lon)

### Issue: Wrong location name
**Cause**: Coordinates may be imprecise or in rural area

**Solution**:
1. Verify coordinates are correct
2. Try nearby major city coordinates
3. Check if coordinates are reversed (lat/lon swapped)

### Issue: Location in wrong language
**Cause**: Some regions have names in local language (e.g., Tokyo shows Japanese)

**Impact**: This is normal behavior, showing official local name
**Note**: Country names are also in local language for authenticity

## Future Enhancements

Potential improvements for future versions:

1. **Multi-Language Support**: Option to get names in English or local language
2. **Timezone Information**: Include local timezone with location
3. **Administrative Levels**: Show district, region, postal codes
4. **Alternative Names**: Show common alternative names for locations
5. **Location Autocomplete**: Suggest locations as user types coordinates
6. **Reverse Search**: Search by location name, get coordinates
7. **Location Aliases**: Support common nicknames (e.g., "NYC" → New York City)

## Related Documentation

- [Weather Data Accuracy](./WEATHER_DATA_ACCURACY.md)
- [Live Weather Integration](./LIVE_WEATHER_INTEGRATION.md)
- [API Documentation](./API.md)

## References

- Nominatim Documentation: https://nominatim.org/release-docs/latest/api/Reverse/
- OpenStreetMap: https://www.openstreetmap.org/
- WeatherAPI.com Locations: https://www.weatherapi.com/docs/

## Conclusion

The location name resolution feature makes the Cloud Burst Prediction System more user-friendly by automatically identifying locations from coordinates. This enhancement improves usability while maintaining the system's technical precision and reliability.

**Key Benefits:**
- ✅ Automatic location identification
- ✅ Global coverage
- ✅ No additional API keys required
- ✅ Graceful fallback handling
- ✅ Cached for performance
- ✅ Seamless dashboard integration
