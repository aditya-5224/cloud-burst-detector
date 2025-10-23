"""
Tiered Alert System for Cloud Burst Predictions

This module provides a sophisticated 5-tier alert classification system
that works universally for ALL events (past, present, and future).

Alert levels are based on probability thresholds and are designed to:
- Reduce alarm fatigue
- Provide clear, actionable guidance
- Maintain high detection rates while minimizing false positives
- Support regional customization

Created: October 22, 2025
Based on: Perplexity AI analysis recommendations
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import pandas as pd


@dataclass
class AlertLevel:
    """Data class representing an alert level"""
    name: str
    probability_min: float
    probability_max: float
    color: str
    color_code: str  # Hex color for dashboard
    emoji: str
    severity: int  # 0-4 scale
    action: str
    description: str


class TieredAlertSystem:
    """
    Universal 5-tier alert classification system.
    
    Works for ANY cloudburst prediction regardless of:
    - Location (coordinates)
    - Date (past/present/future)
    - Event (known or unknown)
    
    Alert levels based solely on probability thresholds.
    """
    
    # Define 5-tier alert levels (universal for all events)
    ALERT_LEVELS = {
        'NORMAL': AlertLevel(
            name='NORMAL',
            probability_min=0.0,
            probability_max=0.50,
            color='green',
            color_code='#28a745',
            emoji='🟢',
            severity=0,
            action='Monitor weather conditions',
            description='Low probability of cloudburst. Normal weather monitoring.'
        ),
        'LOW': AlertLevel(
            name='LOW',
            probability_min=0.50,
            probability_max=0.65,
            color='yellow',
            color_code='#ffc107',
            emoji='🟡',
            severity=1,
            action='Stay informed, prepare emergency kit',
            description='Elevated risk. Prepare basic emergency supplies.'
        ),
        'MEDIUM': AlertLevel(
            name='MEDIUM',
            probability_min=0.65,
            probability_max=0.75,
            color='orange',
            color_code='#fd7e14',
            emoji='🟠',
            severity=2,
            action='Avoid unnecessary travel, secure property',
            description='Moderate risk. Avoid travel to vulnerable areas.'
        ),
        'HIGH': AlertLevel(
            name='HIGH',
            probability_min=0.75,
            probability_max=0.85,
            color='red',
            color_code='#dc3545',
            emoji='🔴',
            severity=3,
            action='Evacuate low-lying areas, move to higher ground',
            description='High risk. Evacuation of vulnerable areas recommended.'
        ),
        'EXTREME': AlertLevel(
            name='EXTREME',
            probability_min=0.85,
            probability_max=1.0,
            color='purple',
            color_code='#6f42c1',
            emoji='🟣',
            severity=4,
            action='IMMEDIATE EVACUATION REQUIRED - Life-threatening situation',
            description='Extreme risk. Immediate evacuation mandatory.'
        )
    }
    
    # Regional threshold adjustments (optional customization)
    REGIONAL_ADJUSTMENTS = {
        'uttarakhand_hills': {
            'adjustment_factor': 0.95,  # Slightly lower thresholds for hilly terrain
            'description': 'Hilly terrain - flash flood risk elevated'
        },
        'himachal_mountains': {
            'adjustment_factor': 0.95,
            'description': 'Mountainous terrain - landslide risk'
        },
        'plains': {
            'adjustment_factor': 1.0,  # Standard thresholds
            'description': 'Plains - standard thresholds'
        },
        'northeast_hills': {
            'adjustment_factor': 0.97,
            'description': 'Northeast hills - high rainfall area'
        }
    }
    
    def __init__(self, enable_regional_adjustment: bool = False):
        """
        Initialize tiered alert system.
        
        Args:
            enable_regional_adjustment: Whether to apply regional threshold adjustments
        """
        self.enable_regional_adjustment = enable_regional_adjustment
    
    def get_alert_level(
        self, 
        probability: float,
        intensity_mmh: Optional[float] = None,
        region: Optional[str] = None
    ) -> AlertLevel:
        """
        Determine alert level from prediction probability.
        
        UNIVERSAL METHOD - works for ALL events regardless of:
        - Location (past, present, future coordinates)
        - Date (historical or real-time)
        - Event (known or unknown)
        
        Args:
            probability: Cloudburst probability (0.0 to 1.0)
            intensity_mmh: Optional rainfall intensity for severity upgrade
            region: Optional region code for threshold adjustment
            
        Returns:
            AlertLevel object with classification and guidance
        """
        # Apply regional adjustment if enabled
        adjusted_probability = probability
        if self.enable_regional_adjustment and region:
            adjustment = self.REGIONAL_ADJUSTMENTS.get(region, {}).get('adjustment_factor', 1.0)
            adjusted_probability = probability * adjustment
        
        # Intensity-based severity upgrade
        # If predicted intensity is extreme (>150mm/h), upgrade alert level
        if intensity_mmh and intensity_mmh > 150:
            adjusted_probability = min(1.0, adjusted_probability * 1.1)
        elif intensity_mmh and intensity_mmh > 100:
            adjusted_probability = min(1.0, adjusted_probability * 1.05)
        
        # Classify based on probability thresholds
        for level_name, level in self.ALERT_LEVELS.items():
            if level.probability_min <= adjusted_probability < level.probability_max:
                return level
        
        # Handle edge case: probability = 1.0
        if adjusted_probability >= 0.85:
            return self.ALERT_LEVELS['EXTREME']
        
        return self.ALERT_LEVELS['NORMAL']
    
    def classify_hourly_predictions(
        self, 
        predictions_df: pd.DataFrame,
        intensity_column: Optional[str] = None,
        region: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Classify all hourly predictions with alert levels.
        
        UNIVERSAL METHOD - processes ANY prediction DataFrame.
        
        Args:
            predictions_df: DataFrame with 'cloudburst_probability' column
            intensity_column: Optional column name for intensity values
            region: Optional region code
            
        Returns:
            DataFrame with added alert level columns
        """
        df = predictions_df.copy()
        
        # Add alert level for each prediction
        alert_data = []
        for _, row in df.iterrows():
            probability = row['cloudburst_probability']
            intensity = row[intensity_column] if intensity_column and intensity_column in df.columns else None
            
            alert_level = self.get_alert_level(probability, intensity, region)
            alert_data.append({
                'alert_level': alert_level.name,
                'alert_emoji': alert_level.emoji,
                'alert_color': alert_level.color,
                'alert_color_code': alert_level.color_code,
                'alert_severity': alert_level.severity,
                'alert_action': alert_level.action,
                'alert_description': alert_level.description
            })
        
        # Add columns to DataFrame
        alert_df = pd.DataFrame(alert_data)
        for col in alert_df.columns:
            df[col] = alert_df[col]
        
        return df
    
    def get_summary_statistics(
        self, 
        predictions_df: pd.DataFrame
    ) -> Dict[str, any]:
        """
        Calculate alert distribution statistics.
        
        Args:
            predictions_df: DataFrame with alert_level column
            
        Returns:
            Dictionary with alert statistics
        """
        if 'alert_level' not in predictions_df.columns:
            raise ValueError("DataFrame must have 'alert_level' column. Run classify_hourly_predictions first.")
        
        total_hours = len(predictions_df)
        
        # Count by severity level
        level_counts = predictions_df['alert_level'].value_counts().to_dict()
        
        # Calculate percentages
        level_percentages = {
            level: (count / total_hours * 100) 
            for level, count in level_counts.items()
        }
        
        # Count actionable alerts (MEDIUM or higher)
        actionable_count = sum(
            1 for level in predictions_df['alert_level'] 
            if level in ['MEDIUM', 'HIGH', 'EXTREME']
        )
        actionable_percentage = (actionable_count / total_hours * 100)
        
        # Highest severity
        max_severity = predictions_df['alert_severity'].max()
        max_severity_level = self.ALERT_LEVELS[
            predictions_df.loc[predictions_df['alert_severity'] == max_severity, 'alert_level'].iloc[0]
        ]
        
        return {
            'total_hours': total_hours,
            'level_counts': level_counts,
            'level_percentages': level_percentages,
            'actionable_alerts': actionable_count,
            'actionable_percentage': actionable_percentage,
            'max_severity': max_severity,
            'max_severity_level': max_severity_level.name,
            'max_severity_action': max_severity_level.action
        }
    
    def format_alert_message(
        self,
        alert_level: AlertLevel,
        probability: float,
        location: str,
        datetime_str: str
    ) -> str:
        """
        Format human-readable alert message.
        
        Args:
            alert_level: AlertLevel object
            probability: Probability value
            location: Location name/coordinates
            datetime_str: Date/time string
            
        Returns:
            Formatted alert message
        """
        message = f"""
{alert_level.emoji} {alert_level.name} ALERT - CLOUDBURST RISK

Location: {location}
Time: {datetime_str}
Probability: {probability:.1%}

Risk Level: {alert_level.severity}/4
Status: {alert_level.description}

RECOMMENDED ACTION:
{alert_level.action}
"""
        return message.strip()
    
    def get_color_for_probability(self, probability: float) -> Tuple[str, str]:
        """
        Get color information for a probability value.
        
        Args:
            probability: Cloudburst probability (0.0 to 1.0)
            
        Returns:
            Tuple of (color_name, hex_code)
        """
        alert_level = self.get_alert_level(probability)
        return (alert_level.color, alert_level.color_code)
    
    def get_alert_counts_by_level(
        self, 
        predictions_df: pd.DataFrame
    ) -> Dict[str, int]:
        """
        Get count of predictions at each alert level.
        
        Args:
            predictions_df: DataFrame with alert_level column
            
        Returns:
            Dictionary mapping alert level to count
        """
        if 'alert_level' not in predictions_df.columns:
            raise ValueError("DataFrame must have 'alert_level' column")
        
        counts = {}
        for level_name in self.ALERT_LEVELS.keys():
            counts[level_name] = (predictions_df['alert_level'] == level_name).sum()
        
        return counts


def determine_region(lat: float, lon: float) -> str:
    """
    Determine geographic region from coordinates.
    
    This is a simplified version - can be enhanced with more precise boundaries.
    
    Args:
        lat: Latitude
        lon: Longitude
        
    Returns:
        Region code
    """
    # Uttarakhand: roughly 29-31°N, 78-81°E
    if 29 <= lat <= 31 and 78 <= lon <= 81:
        return 'uttarakhand_hills'
    
    # Himachal Pradesh: roughly 30-33°N, 75-79°E
    if 30 <= lat <= 33 and 75 <= lon <= 79:
        return 'himachal_mountains'
    
    # Northeast India: roughly 23-29°N, 88-97°E
    if 23 <= lat <= 29 and 88 <= lon <= 97:
        return 'northeast_hills'
    
    # Default to plains
    return 'plains'


# Example usage and testing
if __name__ == "__main__":
    # Test the alert system
    alert_system = TieredAlertSystem()
    
    # Test different probability values
    test_probabilities = [0.25, 0.55, 0.70, 0.80, 0.90]
    
    print("="*70)
    print("TIERED ALERT SYSTEM - TEST")
    print("="*70)
    
    for prob in test_probabilities:
        alert = alert_system.get_alert_level(prob)
        print(f"\nProbability: {prob:.1%}")
        print(f"Alert Level: {alert.emoji} {alert.name}")
        print(f"Severity: {alert.severity}/4")
        print(f"Action: {alert.action}")
    
    print("\n" + "="*70)
    print("Test completed successfully!")
    print("="*70)
