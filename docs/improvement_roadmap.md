# Cloud Burst Predictor - Improvement Roadmap

**Based on Perplexity AI Analysis - October 22, 2025**

## Executive Summary

**Current Status:**
- ✅ Binary detection: **TRUE POSITIVE** (88.8% confidence)
- ✅ Lead time: **14 hours** (exceeds 6-12h requirement)
- ❌ Intensity prediction: **2.5% accuracy** (2.3mm/h vs 92mm/h actual)
- ❌ Precipitation volume: **15% accuracy** (27.7mm vs 185mm actual)
- ⚠️ Alert specificity: **66.7% hours flagged** (target: <30%)

**Critical Finding:** Model detects cloudbursts well but severely underestimates severity. Single-event validation insufficient.

---

## Priority 1: CRITICAL (Data Foundation)

### 1.1 Expand Training Dataset
**Current:** 12 events  
**Required:** 150+ events minimum
- **50+ confirmed cloudbursts** (various intensities: 50-200mm/h)
- **50+ heavy rain events** (non-cloudburst: 20-50mm/h)
- **50+ normal/dry days** (baseline conditions)

**Target Regions:** Uttarakhand, Himachal Pradesh, J&K, Sikkim, Arunachal Pradesh

**Sources:**
- IMD historical archives
- NDMA disaster reports
- State disaster management databases
- Research papers on Himalayan cloudbursts
- News archives with verified events

**Expected Impact:** Enable proper precision/recall/F1 calculations, reduce overfitting

---

### 1.2 Higher Temporal Resolution
**Current:** Hourly aggregated data  
**Problem:** Misses sub-hour intensity spikes (100mm/h+ over 15 minutes)  
**Required:** 5-15 minute intervals

**Solutions:**
1. **Open-Meteo Minutely Precipitation API**
   - 1-minute intervals, 1-hour forecast
   - Best for recent events

2. **NOAA HRRR (High-Resolution Rapid Refresh)**
   - 3km spatial, 15-minute temporal
   - Requires data processing

3. **IMD Automatic Weather Stations (AWS)**
   - 15-minute reporting
   - Official source but requires partnership

**Implementation:**
```python
# src/data/high_frequency_weather.py
class HighFrequencyWeatherCollector:
    """Collect 5-15 minute interval weather data"""
    
    def fetch_minutely_data(self, lat, lon, start_time, end_time):
        """Open-Meteo minutely precipitation"""
        pass
    
    def fetch_hrrr_data(self, lat, lon, datetime):
        """NOAA HRRR 15-min data"""
        pass
    
    def aggregate_to_peaks(self, minutely_df):
        """Find peak intensity in sliding windows"""
        pass
```

**Expected Impact:** Capture 80%+ of actual peak intensity (vs. current 2.5%)

---

### 1.3 Spatial Resolution Enhancement
**Current:** Point-based (single coordinate)  
**Required:** Grid-based 20-30 km² detection

**Solutions:**
1. **GPM (Global Precipitation Measurement) Satellite**
   - 0.1° × 0.1° resolution (~11km)
   - 30-minute updates
   - Free NASA API

2. **INSAT-3D Satellite Imagery**
   - Indian geostationary satellite
   - Cloud top temperature, moisture
   - 30-minute intervals

3. **Doppler Radar Data** (if available)
   - Highest resolution for local events
   - Requires IMD partnership

**Implementation:**
```python
# src/data/satellite_collector.py
class SatelliteDataCollector:
    """Integrate satellite rainfall estimates"""
    
    def fetch_gpm_data(self, lat, lon, radius_km, datetime):
        """GPM IMERG rainfall estimates"""
        pass
    
    def fetch_insat3d_imagery(self, lat, lon, datetime):
        """INSAT-3D cloud parameters"""
        pass
    
    def create_spatial_grid(self, center_lat, center_lon, grid_size_km=30):
        """Generate 20-30km² grid for spatial analysis"""
        pass
```

**Expected Impact:** Better detection of localized events, reduced false negatives

---

## Priority 2: HIGH (Model Architecture)

### 2.1 Advanced Atmospheric Features
**Critical Missing Indicators:**

| Feature | Description | Threshold | Source |
|---------|-------------|-----------|--------|
| **CAPE** | Convective Available Potential Energy | >1500 J/kg | ERA5, Open-Meteo |
| **CIN** | Convective Inhibition | <-50 J/kg | ERA5 |
| **Lifted Index** | Atmospheric stability | <-4 | Calculated |
| **K-Index** | Thunderstorm potential | >30 | Calculated |
| **Total Totals** | Severe weather index | >50 | Calculated |
| **Vertical Velocity** | Updraft strength | >5 m/s | ERA5 |
| **Precipitable Water** | Total atmospheric moisture | >40 mm | Open-Meteo |

**Implementation:**
```python
# src/features/atmospheric_indices.py
class AtmosphericIndices:
    """Calculate advanced instability indicators"""
    
    def calculate_lifted_index(self, temp_850, temp_500, dewpoint_850):
        """LI = T500 - Tparcel_500"""
        pass
    
    def calculate_k_index(self, temp_850, temp_700, temp_500, dewpoint_850, dewpoint_700):
        """K = (T850 - T500) + Td850 - (T700 - Td700)"""
        pass
    
    def calculate_total_totals(self, temp_850, temp_500, dewpoint_850):
        """TT = VT + CT"""
        pass
    
    def fetch_cape_cin(self, lat, lon, datetime):
        """Get CAPE/CIN from ERA5 reanalysis"""
        pass
```

**Expected Impact:** Better prediction of convective severity, improved intensity estimates

---

### 2.2 Hybrid Model Architecture
**Current:** Random Forest only  
**Proposed:** LSTM + Random Forest Ensemble

**Architecture:**
```
Input Data (5-15 min intervals, 48h window)
    ↓
┌─────────────────────┬─────────────────────────┐
│   LSTM Network      │   Random Forest         │
│   (Temporal)        │   (Feature-based)       │
│                     │                         │
│ - Sequence learning │ - Non-linear relations  │
│ - Trend detection   │ - Feature importance    │
│ - Sudden changes    │ - Robust to noise       │
└─────────────────────┴─────────────────────────┘
    ↓                        ↓
    Probability_LSTM    Probability_RF
                ↓
         Ensemble (Weighted Average)
                ↓
    Final Prediction + Confidence Interval
```

**Model Components:**

1. **LSTM for Temporal Sequences**
```python
# src/models/lstm_model.py
class CloudBurstLSTM:
    """LSTM for temporal pattern recognition"""
    
    def __init__(self):
        self.model = Sequential([
            LSTM(128, return_sequences=True, input_shape=(288, 15)),  # 24h × 12 (5-min intervals), 15 features
            Dropout(0.2),
            LSTM(64, return_sequences=True),
            Dropout(0.2),
            LSTM(32),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
    
    def predict_probability(self, sequence):
        """Predict from 24h sequence"""
        pass
```

2. **Random Forest for Features**
```python
# Keep existing Random Forest but enhance features
class EnhancedRandomForest:
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=500,  # Increase from 200
            max_depth=20,
            min_samples_split=3,
            class_weight='balanced',
            max_features='sqrt'
        )
```

3. **Ensemble Combiner**
```python
# src/models/hybrid_model.py
class HybridCloudBurstPredictor:
    """Ensemble of LSTM + Random Forest"""
    
    def __init__(self):
        self.lstm_model = CloudBurstLSTM()
        self.rf_model = EnhancedRandomForest()
        self.lstm_weight = 0.6  # Tune based on validation
        self.rf_weight = 0.4
    
    def predict(self, sequence_data, feature_data):
        lstm_prob = self.lstm_model.predict_probability(sequence_data)
        rf_prob = self.rf_model.predict_proba(feature_data)[:, 1]
        
        ensemble_prob = (self.lstm_weight * lstm_prob + 
                        self.rf_weight * rf_prob)
        
        # Calculate uncertainty
        disagreement = abs(lstm_prob - rf_prob)
        confidence = 1 - disagreement
        
        return {
            'probability': ensemble_prob,
            'confidence': confidence,
            'lstm_prob': lstm_prob,
            'rf_prob': rf_prob
        }
```

**Expected Impact:** Better temporal pattern recognition, improved intensity prediction

---

### 2.3 Intensity Regression Model
**Current:** Binary classification only (cloudburst: yes/no)  
**Required:** Multi-output prediction (occurrence + intensity)

**Implementation:**
```python
# src/models/intensity_predictor.py
class IntensityPredictor:
    """Predict rainfall intensity (mm/h) not just occurrence"""
    
    def __init__(self):
        # Separate model for intensity regression
        self.intensity_model = RandomForestRegressor(
            n_estimators=500,
            max_depth=25,
            min_samples_split=3
        )
    
    def train(self, X, y_intensity):
        """Train on actual rainfall intensities"""
        self.intensity_model.fit(X, y_intensity)
    
    def predict_intensity(self, X):
        """Predict mm/h rainfall rate"""
        return self.intensity_model.predict(X)

# Combine with classification
class DualOutputModel:
    """Predict both occurrence and intensity"""
    
    def predict(self, X):
        occurrence_prob = self.classifier.predict_proba(X)
        intensity_mmh = self.regressor.predict_intensity(X)
        
        return {
            'cloudburst_probability': occurrence_prob,
            'expected_intensity_mmh': intensity_mmh,
            'severity_class': self._classify_severity(intensity_mmh)
        }
    
    def _classify_severity(self, intensity):
        if intensity < 50: return 'Heavy Rain'
        elif intensity < 100: return 'Moderate Cloudburst'
        elif intensity < 150: return 'Severe Cloudburst'
        else: return 'Extreme Cloudburst'
```

**Expected Impact:** Address the critical 2.5% intensity prediction accuracy

---

### 2.4 Regional Threshold Calibration
**Current:** Single global threshold (100mm/h IMD standard)  
**Required:** Region-specific thresholds

**Regional Settings:**
```python
# src/models/regional_config.py
REGIONAL_THRESHOLDS = {
    'uttarakhand_hills': {
        'cloudburst_threshold_mmh': 50,  # Lower for hilly terrain
        'alert_probability': 0.70,
        'description': 'Hilly terrain, lower threshold due to flash flood risk'
    },
    'himachal_mountains': {
        'cloudburst_threshold_mmh': 60,
        'alert_probability': 0.70,
    },
    'plains': {
        'cloudburst_threshold_mmh': 100,  # Standard IMD
        'alert_probability': 0.75,
    },
    'northeast_hills': {
        'cloudburst_threshold_mmh': 80,
        'alert_probability': 0.65,
    }
}

def get_regional_threshold(lat, lon):
    """Determine region from coordinates"""
    if is_uttarakhand(lat, lon):
        return REGIONAL_THRESHOLDS['uttarakhand_hills']
    # ... other regions
```

**Expected Impact:** Reduce false positives while maintaining high recall

---

## Priority 3: HIGH (Validation Framework)

### 3.1 Comprehensive Evaluation Metrics

**Target Performance (Industry Standards):**
| Metric | Target | Current | Gap |
|--------|--------|---------|-----|
| **Recall (Sensitivity)** | >90% | Unknown (1 event) | Need 50+ events |
| **Precision** | >75% | Unknown | Need testing |
| **F1-Score** | >80% | Unknown | Need testing |
| **False Positive Rate** | <25% | Unknown | Need testing |
| **Intensity RMSE** | <20 mm/h | ~90 mm/h | 350% over target |
| **Lead Time** | 6-12 hours | 14 hours ✅ | Exceeds target |
| **Alert Specificity** | <30% hours flagged | 66.7% | 123% over target |

**Implementation:**
```python
# src/models/validation_framework.py
class ComprehensiveValidator:
    """Full model validation suite"""
    
    def __init__(self, test_events):
        self.test_events = test_events  # Minimum 50 events
        self.results = []
    
    def run_validation(self, model):
        """Test on all events"""
        predictions = []
        actuals = []
        
        for event in self.test_events:
            pred = model.predict(event['features'])
            predictions.append(pred)
            actuals.append(event['actual_cloudburst'])
        
        return self.calculate_metrics(predictions, actuals)
    
    def calculate_metrics(self, y_pred, y_true):
        """Calculate all performance metrics"""
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, 
            f1_score, confusion_matrix, roc_auc_score
        )
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),  # Critical: >90%
            'f1_score': f1_score(y_true, y_pred),
            'confusion_matrix': confusion_matrix(y_true, y_pred),
            'roc_auc': roc_auc_score(y_true, y_pred),
        }
        
        # Calculate False Positive Rate
        tn, fp, fn, tp = metrics['confusion_matrix'].ravel()
        metrics['false_positive_rate'] = fp / (fp + tn)
        metrics['false_negative_rate'] = fn / (fn + tp)
        
        return metrics
    
    def k_fold_cross_validation(self, model, k=5):
        """5-fold cross-validation to prevent overfitting"""
        from sklearn.model_selection import StratifiedKFold
        
        skf = StratifiedKFold(n_splits=k, shuffle=True)
        scores = []
        
        for train_idx, test_idx in skf.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            model.fit(X_train, y_train)
            score = model.score(X_test, y_test)
            scores.append(score)
        
        return {
            'mean_accuracy': np.mean(scores),
            'std_accuracy': np.std(scores),
            'scores': scores
        }
    
    def calculate_intensity_rmse(self, predicted_intensities, actual_intensities):
        """Root Mean Square Error for intensity prediction"""
        from sklearn.metrics import mean_squared_error
        
        rmse = np.sqrt(mean_squared_error(actual_intensities, predicted_intensities))
        return rmse
    
    def analyze_lead_times(self, predictions_df):
        """Analyze consistency of warning lead times"""
        lead_times = []
        
        for event in self.test_events:
            first_alert_time = predictions_df[
                predictions_df['probability'] > 0.7
            ]['time'].min()
            
            actual_event_time = event['datetime']
            lead_time_hours = (actual_event_time - first_alert_time).total_seconds() / 3600
            lead_times.append(lead_time_hours)
        
        return {
            'mean_lead_time': np.mean(lead_times),
            'std_lead_time': np.std(lead_times),
            'min_lead_time': np.min(lead_times),
            'max_lead_time': np.max(lead_times),
            'target_range': (6, 12),
            'within_target_pct': sum(6 <= lt <= 12 for lt in lead_times) / len(lead_times) * 100
        }
```

**Expected Impact:** Establish true model performance, identify weaknesses

---

## Priority 4: MEDIUM (Operational Deployment)

### 4.1 Tiered Alert System
**Current:** Binary alerts (66.7% of hours flagged)  
**Required:** 4-tier system (<30% flagged)

**Alert Tiers:**
```python
# src/models/alert_system.py
class TieredAlertSystem:
    """Multi-level alert system to reduce alarm fatigue"""
    
    ALERT_LEVELS = {
        'NORMAL': {
            'probability_range': (0, 0.50),
            'color': 'green',
            'action': 'Monitor conditions',
            'icon': '🟢'
        },
        'LOW': {
            'probability_range': (0.50, 0.65),
            'color': 'yellow',
            'action': 'Stay informed, prepare emergency kit',
            'icon': '🟡'
        },
        'MEDIUM': {
            'probability_range': (0.65, 0.75),
            'color': 'orange',
            'action': 'Avoid travel, secure property',
            'icon': '🟠'
        },
        'HIGH': {
            'probability_range': (0.75, 0.85),
            'color': 'red',
            'action': 'Evacuate low-lying areas',
            'icon': '🔴'
        },
        'EXTREME': {
            'probability_range': (0.85, 1.0),
            'color': 'purple',
            'action': 'IMMEDIATE EVACUATION REQUIRED',
            'icon': '🟣'
        }
    }
    
    def get_alert_level(self, probability, intensity_mmh=None):
        """Determine alert level from probability and intensity"""
        for level, config in self.ALERT_LEVELS.items():
            min_prob, max_prob = config['probability_range']
            if min_prob <= probability < max_prob:
                # Upgrade level if intensity is very high
                if intensity_mmh and intensity_mmh > 150:
                    return self._upgrade_alert(level)
                return level
        return 'NORMAL'
    
    def _upgrade_alert(self, current_level):
        """Upgrade alert level for extreme intensity"""
        levels = ['NORMAL', 'LOW', 'MEDIUM', 'HIGH', 'EXTREME']
        current_idx = levels.index(current_level)
        if current_idx < len(levels) - 1:
            return levels[current_idx + 1]
        return current_level
```

**Expected Impact:** Reduce flagged hours from 66.7% to <30%, improve public trust

---

### 4.2 Real-time Nowcasting (0-3 Hour Predictions)
**Implementation:**
```python
# src/models/nowcasting.py
class NowcastingSystem:
    """Real-time 0-3 hour cloudburst predictions"""
    
    def __init__(self):
        self.update_interval = 300  # 5 minutes
        self.forecast_horizon = 180  # 3 hours
    
    async def continuous_monitoring(self, location):
        """Run continuous nowcasting"""
        while True:
            # Fetch latest 5-min data
            current_data = await self.fetch_live_weather(location)
            
            # Update model with latest observations
            prediction = self.model.predict(current_data)
            
            # Issue alert if needed
            if prediction['probability'] > 0.70:
                await self.send_alert(location, prediction)
            
            # Wait for next interval
            await asyncio.sleep(self.update_interval)
    
    def fetch_live_weather(self, location):
        """Get real-time weather data"""
        # Open-Meteo current weather + minutely precipitation
        # INSAT-3D satellite imagery
        # Doppler radar if available
        pass
```

**Expected Impact:** Operational early warning capability

---

### 4.3 Performance Monitoring Dashboard
```python
# src/dashboard/monitoring.py
class PerformanceMonitor:
    """Track model performance over time"""
    
    def __init__(self):
        self.metrics_history = []
    
    def track_prediction(self, prediction, actual_outcome):
        """Record each prediction and outcome"""
        self.metrics_history.append({
            'timestamp': datetime.now(),
            'predicted': prediction,
            'actual': actual_outcome,
            'correct': prediction == actual_outcome
        })
    
    def calculate_rolling_metrics(self, window_days=30):
        """Calculate metrics over rolling window"""
        recent = self.metrics_history[-window_days*24:]
        
        return {
            'accuracy': sum(m['correct'] for m in recent) / len(recent),
            'false_alarms': sum(1 for m in recent if m['predicted'] and not m['actual']),
            'missed_events': sum(1 for m in recent if not m['predicted'] and m['actual']),
            'true_positives': sum(1 for m in recent if m['predicted'] and m['actual'])
        }
```

---

## Implementation Timeline

### Phase 1 (Weeks 1-2): Data Foundation
- [ ] Collect 50+ additional cloud burst events
- [ ] Add 50+ heavy rain (non-cloudburst) events
- [ ] Add 50+ normal/dry day samples
- [ ] Implement high-frequency data collection (5-15 min)

### Phase 2 (Weeks 3-4): Advanced Features
- [ ] Implement CAPE, CIN, Lifted Index calculations
- [ ] Integrate ERA5 reanalysis data
- [ ] Add satellite data (GPM, INSAT-3D)
- [ ] Implement regional threshold calibration

### Phase 3 (Weeks 5-6): Model Enhancement
- [ ] Build LSTM model for temporal patterns
- [ ] Create hybrid ensemble (LSTM + RF)
- [ ] Implement intensity regression model
- [ ] Add tiered alert system

### Phase 4 (Weeks 7-8): Validation & Testing
- [ ] Implement k-fold cross-validation
- [ ] Run comprehensive validation (150+ events)
- [ ] Calculate all performance metrics
- [ ] Tune thresholds to meet targets

### Phase 5 (Weeks 9-10): Operational Deployment
- [ ] Implement real-time nowcasting
- [ ] Build performance monitoring dashboard
- [ ] Create alert distribution system
- [ ] Deploy production system

---

## Success Criteria

### Minimum Viable Performance
- ✅ Recall (Sensitivity): >90%
- ✅ Precision: >75%
- ✅ F1-Score: >80%
- ✅ False Positive Rate: <25%
- ✅ Intensity RMSE: <20 mm/h
- ✅ Lead Time: 6-12 hours (already at 14h)
- ✅ Alert Specificity: <30% hours flagged

### Current Status vs. Targets
| Metric | Current | Target | Status |
|--------|---------|--------|--------|
| Binary Detection | ✅ 100% (1 event) | >90% | Good (need more tests) |
| Intensity Prediction | ❌ 2.5% | >95% | CRITICAL GAP |
| Lead Time | ✅ 14 hours | 6-12h | Exceeds target |
| Alert Specificity | ❌ 66.7% | <30% | 123% over |
| Dataset Size | ❌ 12 events | 150+ | Need 12x more |

---

## Quick Wins (Immediate Actions)

1. **Add 10 More Events** (1-2 days)
   - Search for documented cloudbursts 2020-2025
   - Add to events_database.py
   - Retrain model

2. **Adjust Alert Threshold** (1 hour)
   - Change probability threshold from 0.50 to 0.70
   - Reduce flagged hours from 66.7% to ~30%

3. **Add Basic CAPE Data** (1 day)
   - Use Open-Meteo `cape` parameter
   - Already available in API

4. **Implement Tiered Alerts** (2 days)
   - 4-level system (Normal/Low/Medium/High/Extreme)
   - Color-coded dashboard display

5. **K-Fold Validation** (1 day)
   - Implement 5-fold cross-validation
   - Measure model stability

---

## Resources & References

### Data Sources
- **IMD (India Meteorological Department):** Historical archives, AWS data
- **ERA5 Reanalysis:** CAPE, CIN, vertical velocity - https://cds.climate.copernicus.eu/
- **Open-Meteo:** High-frequency weather data - https://open-meteo.com/
- **GPM:** Satellite precipitation - https://gpm.nasa.gov/
- **INSAT-3D:** Indian satellite imagery - https://insat3d.imd.gov.in/

### Research Papers
- Uttarakhand cloudbursts: 50mm/h threshold studies
- IMD cloudburst definition: 100mm/h
- Himalayan convective systems
- LSTM for weather prediction

### Tools & Libraries
- **TensorFlow/Keras:** LSTM implementation
- **Scikit-learn:** Enhanced RF, validation metrics
- **xarray:** ERA5 data processing
- **rasterio:** Satellite imagery processing

---

**Last Updated:** October 22, 2025  
**Version:** 1.0  
**Status:** Roadmap established, implementation pending
