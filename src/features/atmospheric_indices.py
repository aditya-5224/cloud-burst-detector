"""
Advanced Atmospheric Indices for Cloud Burst Prediction

Implements meteorological indices for atmospheric instability:
- CAPE (Convective Available Potential Energy)
- Lifted Index (LI)
- K-Index
- Total Totals Index (TT)
- Showalter Index (SI)
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AtmosphericIndices:
    """Calculate advanced atmospheric instability indices"""
    
    # Constants
    Rd = 287.05  # Gas constant for dry air (J/kg/K)
    Rv = 461.5   # Gas constant for water vapor (J/kg/K)
    Cp = 1005.0  # Specific heat at constant pressure (J/kg/K)
    L = 2.5e6    # Latent heat of vaporization (J/kg)
    g = 9.81     # Gravitational acceleration (m/s^2)
    
    def __init__(self):
        """Initialize atmospheric indices calculator"""
        pass
    
    @staticmethod
    def saturation_vapor_pressure(temp_c: float) -> float:
        """
        Calculate saturation vapor pressure using Tetens formula
        
        Args:
            temp_c: Temperature in Celsius
            
        Returns:
            Saturation vapor pressure in hPa
        """
        return 6.112 * np.exp((17.67 * temp_c) / (temp_c + 243.5))
    
    @staticmethod
    def mixing_ratio(pressure_hpa: float, temp_c: float, rh_percent: float) -> float:
        """
        Calculate mixing ratio from temperature and relative humidity
        
        Args:
            pressure_hpa: Pressure in hPa
            temp_c: Temperature in Celsius
            rh_percent: Relative humidity (0-100)
            
        Returns:
            Mixing ratio (kg/kg)
        """
        es = AtmosphericIndices.saturation_vapor_pressure(temp_c)
        e = (rh_percent / 100.0) * es
        return 0.622 * e / (pressure_hpa - e)
    
    @staticmethod
    def dewpoint(temp_c: float, rh_percent: float) -> float:
        """
        Calculate dewpoint temperature
        
        Args:
            temp_c: Temperature in Celsius
            rh_percent: Relative humidity (0-100)
            
        Returns:
            Dewpoint in Celsius
        """
        a = 17.27
        b = 237.7
        
        alpha = ((a * temp_c) / (b + temp_c)) + np.log(rh_percent / 100.0)
        return (b * alpha) / (a - alpha)
    
    @staticmethod
    def potential_temperature(temp_k: float, pressure_hpa: float, 
                             p0: float = 1000.0) -> float:
        """
        Calculate potential temperature
        
        Args:
            temp_k: Temperature in Kelvin
            pressure_hpa: Pressure in hPa
            p0: Reference pressure (default 1000 hPa)
            
        Returns:
            Potential temperature in Kelvin
        """
        R_cp = 0.286  # Rd/Cp
        return temp_k * (p0 / pressure_hpa) ** R_cp
    
    @staticmethod
    def equivalent_potential_temperature(temp_k: float, pressure_hpa: float, 
                                        mixing_ratio_kg: float) -> float:
        """
        Calculate equivalent potential temperature
        
        Args:
            temp_k: Temperature in Kelvin
            pressure_hpa: Pressure in hPa
            mixing_ratio_kg: Mixing ratio in kg/kg
            
        Returns:
            Equivalent potential temperature in Kelvin
        """
        theta = AtmosphericIndices.potential_temperature(temp_k, pressure_hpa)
        
        # Simplified Bolton (1980) formula
        exp_term = (3.376 / (temp_k - 0.00254 * mixing_ratio_kg * 1000) - 0.00254) * \
                   mixing_ratio_kg * 1000 * (1 + 0.81 * mixing_ratio_kg)
        
        return theta * np.exp(exp_term)
    
    def calculate_cape_simplified(self, surface_temp_c: float, surface_pressure_hpa: float,
                                  surface_rh: float, upper_temp_c: float = -40.0,
                                  upper_pressure_hpa: float = 500.0) -> Dict[str, float]:
        """
        Calculate simplified CAPE using surface-based approach
        
        This is a simplified version suitable for single-level data.
        For accurate CAPE, atmospheric sounding data is required.
        
        Args:
            surface_temp_c: Surface temperature (Celsius)
            surface_pressure_hpa: Surface pressure (hPa)
            surface_rh: Surface relative humidity (%)
            upper_temp_c: Upper level temperature (Celsius)
            upper_pressure_hpa: Upper level pressure (hPa)
            
        Returns:
            Dictionary with CAPE and CIN values
        """
        try:
            # Convert to Kelvin
            T_surface = surface_temp_c + 273.15
            T_upper = upper_temp_c + 273.15
            
            # Calculate mixing ratio
            w = self.mixing_ratio(surface_pressure_hpa, surface_temp_c, surface_rh)
            
            # Calculate equivalent potential temperature at surface
            theta_e_surface = self.equivalent_potential_temperature(
                T_surface, surface_pressure_hpa, w
            )
            
            # Estimate parcel temperature at upper level (adiabatic ascent)
            # Using simple dry adiabatic lapse rate (9.8 K/km)
            height_diff = (surface_pressure_hpa - upper_pressure_hpa) / 12.0  # Rough conversion
            T_parcel_upper = T_surface - (0.0098 * height_diff * 1000)  # K
            
            # Calculate CAPE (simplified)
            # CAPE = integral of (T_parcel - T_environment) * g/T_environment * dz
            if T_parcel_upper > T_upper:
                # Buoyancy exists
                dT = T_parcel_upper - T_upper
                dP = surface_pressure_hpa - upper_pressure_hpa
                
                # Simplified CAPE calculation
                cape = self.Rd * dT * np.log(surface_pressure_hpa / upper_pressure_hpa)
                cape = max(0, cape)  # CAPE cannot be negative
            else:
                cape = 0.0
            
            # Estimate CIN (negative buoyancy)
            cin = 0.0  # Simplified - would need full sounding for accurate CIN
            
            return {
                'cape': cape,
                'cin': cin,
                'theta_e_surface': theta_e_surface,
                'parcel_temp_upper': T_parcel_upper - 273.15,
                'environment_temp_upper': upper_temp_c
            }
            
        except Exception as e:
            logger.error(f"Error calculating CAPE: {e}")
            return {'cape': 0.0, 'cin': 0.0, 'theta_e_surface': 0.0, 
                   'parcel_temp_upper': 0.0, 'environment_temp_upper': 0.0}
    
    def calculate_lifted_index(self, surface_temp_c: float, surface_dewpoint_c: float,
                               temp_500mb_c: float = -5.5) -> float:
        """
        Calculate Lifted Index (LI)
        
        LI = T(500mb) - T(parcel at 500mb)
        
        Interpretation:
        - LI < -6: Extremely Unstable
        - -6 < LI < -4: Very Unstable
        - -4 < LI < 0: Unstable
        - 0 < LI < 2: Slightly Stable
        - LI > 2: Stable
        
        Args:
            surface_temp_c: Surface temperature (Celsius)
            surface_dewpoint_c: Surface dewpoint (Celsius)
            temp_500mb_c: Temperature at 500mb level (Celsius)
            
        Returns:
            Lifted Index value
        """
        try:
            # Simplified calculation assuming moist adiabatic lapse rate
            # Standard atmosphere: surface ~1000mb, 500mb ~5500m
            height_diff_km = 5.5
            
            # Average moist adiabatic lapse rate (6 K/km)
            moist_lapse_rate = 6.0
            
            # Parcel temperature at 500mb
            T_parcel_500mb = surface_temp_c - (moist_lapse_rate * height_diff_km)
            
            # Lifted Index
            li = temp_500mb_c - T_parcel_500mb
            
            return li
            
        except Exception as e:
            logger.error(f"Error calculating Lifted Index: {e}")
            return 0.0
    
    def calculate_k_index(self, temp_850mb_c: float, temp_700mb_c: float,
                         temp_500mb_c: float, dewpoint_850mb_c: float,
                         dewpoint_700mb_c: float) -> float:
        """
        Calculate K-Index for thunderstorm potential
        
        K = (T850 - T500) + Td850 - (T700 - Td700)
        
        Interpretation:
        - K < 20: Thunderstorms unlikely
        - 20 < K < 30: Isolated thunderstorms
        - 30 < K < 40: Scattered thunderstorms
        - K > 40: Numerous thunderstorms
        
        Args:
            temp_850mb_c: Temperature at 850mb (Celsius)
            temp_700mb_c: Temperature at 700mb (Celsius)
            temp_500mb_c: Temperature at 500mb (Celsius)
            dewpoint_850mb_c: Dewpoint at 850mb (Celsius)
            dewpoint_700mb_c: Dewpoint at 700mb (Celsius)
            
        Returns:
            K-Index value
        """
        try:
            k_index = (temp_850mb_c - temp_500mb_c) + dewpoint_850mb_c - \
                     (temp_700mb_c - dewpoint_700mb_c)
            return k_index
            
        except Exception as e:
            logger.error(f"Error calculating K-Index: {e}")
            return 0.0
    
    def calculate_total_totals_index(self, temp_850mb_c: float, temp_500mb_c: float,
                                     dewpoint_850mb_c: float) -> float:
        """
        Calculate Total Totals Index
        
        TT = (T850 + Td850) - 2*T500
        
        Interpretation:
        - TT < 44: Thunderstorms unlikely
        - 44 < TT < 50: Isolated thunderstorms
        - 50 < TT < 55: Scattered thunderstorms
        - TT > 55: Numerous severe thunderstorms
        
        Args:
            temp_850mb_c: Temperature at 850mb (Celsius)
            temp_500mb_c: Temperature at 500mb (Celsius)
            dewpoint_850mb_c: Dewpoint at 850mb (Celsius)
            
        Returns:
            Total Totals Index value
        """
        try:
            tt = (temp_850mb_c + dewpoint_850mb_c) - (2 * temp_500mb_c)
            return tt
            
        except Exception as e:
            logger.error(f"Error calculating Total Totals Index: {e}")
            return 0.0
    
    def calculate_showalter_index(self, temp_850mb_c: float, dewpoint_850mb_c: float,
                                  temp_500mb_c: float) -> float:
        """
        Calculate Showalter Index
        
        SI = T500 - T(parcel from 850mb lifted to 500mb)
        
        Interpretation:
        - SI < -3: Severe thunderstorms likely
        - -3 < SI < 0: Thunderstorms likely
        - 0 < SI < 3: Thunderstorms possible
        - SI > 3: Thunderstorms unlikely
        
        Args:
            temp_850mb_c: Temperature at 850mb (Celsius)
            dewpoint_850mb_c: Dewpoint at 850mb (Celsius)
            temp_500mb_c: Temperature at 500mb (Celsius)
            
        Returns:
            Showalter Index value
        """
        try:
            # Height difference from 850mb (~1500m) to 500mb (~5500m) = 4000m
            height_diff_km = 4.0
            
            # Moist adiabatic lapse rate
            moist_lapse_rate = 6.0
            
            # Parcel temperature at 500mb
            T_parcel_500mb = temp_850mb_c - (moist_lapse_rate * height_diff_km)
            
            # Showalter Index
            si = temp_500mb_c - T_parcel_500mb
            
            return si
            
        except Exception as e:
            logger.error(f"Error calculating Showalter Index: {e}")
            return 0.0
    
    def estimate_upper_levels_from_surface(self, surface_temp_c: float,
                                          surface_pressure_hpa: float = 1013.25,
                                          surface_rh: float = 70.0) -> Dict[str, float]:
        """
        Estimate upper atmospheric levels from surface data
        
        This is a rough estimation using standard atmosphere assumptions.
        For accurate values, radiosonde/reanalysis data is required.
        
        Args:
            surface_temp_c: Surface temperature (Celsius)
            surface_pressure_hpa: Surface pressure (hPa)
            surface_rh: Surface relative humidity (%)
            
        Returns:
            Dictionary with estimated upper level values
        """
        try:
            # Standard lapse rate: ~6.5 K/km in troposphere
            lapse_rate = 6.5
            
            # Estimate dewpoint
            dewpoint_surface = self.dewpoint(surface_temp_c, surface_rh)
            
            # Estimate 850mb level (~1500m above MSL)
            h_850 = 1.5  # km
            temp_850 = surface_temp_c - (lapse_rate * h_850)
            dewpoint_850 = dewpoint_surface - (2.0 * h_850)  # Dewpoint lapse ~2 K/km
            
            # Estimate 700mb level (~3000m above MSL)
            h_700 = 3.0  # km
            temp_700 = surface_temp_c - (lapse_rate * h_700)
            dewpoint_700 = dewpoint_surface - (2.0 * h_700)
            
            # Estimate 500mb level (~5500m above MSL)
            h_500 = 5.5  # km
            temp_500 = surface_temp_c - (lapse_rate * h_500)
            
            return {
                'temp_850mb': temp_850,
                'temp_700mb': temp_700,
                'temp_500mb': temp_500,
                'dewpoint_850mb': dewpoint_850,
                'dewpoint_700mb': dewpoint_700,
                'dewpoint_surface': dewpoint_surface
            }
            
        except Exception as e:
            logger.error(f"Error estimating upper levels: {e}")
            return {}
    
    def calculate_all_indices(self, surface_temp_c: float, surface_pressure_hpa: float,
                             surface_rh: float) -> Dict[str, float]:
        """
        Calculate all atmospheric indices from surface data
        
        Args:
            surface_temp_c: Surface temperature (Celsius)
            surface_pressure_hpa: Surface pressure (hPa)
            surface_rh: Surface relative humidity (%)
            
        Returns:
            Dictionary with all calculated indices
        """
        # Estimate upper levels
        upper = self.estimate_upper_levels_from_surface(
            surface_temp_c, surface_pressure_hpa, surface_rh
        )
        
        # Calculate CAPE
        cape_results = self.calculate_cape_simplified(
            surface_temp_c, surface_pressure_hpa, surface_rh,
            upper.get('temp_500mb', -40.0)
        )
        
        # Calculate Lifted Index
        li = self.calculate_lifted_index(
            surface_temp_c,
            upper.get('dewpoint_surface', surface_temp_c - 5),
            upper.get('temp_500mb', -5.5)
        )
        
        # Calculate K-Index
        k_index = self.calculate_k_index(
            upper.get('temp_850mb', surface_temp_c - 10),
            upper.get('temp_700mb', surface_temp_c - 20),
            upper.get('temp_500mb', surface_temp_c - 35),
            upper.get('dewpoint_850mb', surface_temp_c - 12),
            upper.get('dewpoint_700mb', surface_temp_c - 22)
        )
        
        # Calculate Total Totals Index
        tt = self.calculate_total_totals_index(
            upper.get('temp_850mb', surface_temp_c - 10),
            upper.get('temp_500mb', surface_temp_c - 35),
            upper.get('dewpoint_850mb', surface_temp_c - 12)
        )
        
        # Calculate Showalter Index
        si = self.calculate_showalter_index(
            upper.get('temp_850mb', surface_temp_c - 10),
            upper.get('dewpoint_850mb', surface_temp_c - 12),
            upper.get('temp_500mb', surface_temp_c - 35)
        )
        
        return {
            'cape': cape_results['cape'],
            'cin': cape_results['cin'],
            'lifted_index': li,
            'k_index': k_index,
            'total_totals': tt,
            'showalter_index': si,
            'theta_e': cape_results['theta_e_surface'],
            'dewpoint': upper.get('dewpoint_surface', 0.0)
        }


def add_atmospheric_indices_to_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add atmospheric indices to weather dataframe
    
    Args:
        df: DataFrame with weather data (must have temperature_2m, pressure_msl, relative_humidity_2m)
        
    Returns:
        DataFrame with added atmospheric indices
    """
    calculator = AtmosphericIndices()
    
    logger.info("Calculating atmospheric indices for dataset...")
    
    # Calculate indices for each row
    indices_list = []
    for idx, row in df.iterrows():
        try:
            indices = calculator.calculate_all_indices(
                surface_temp_c=row.get('temperature_2m', 25.0),
                surface_pressure_hpa=row.get('pressure_msl', 1013.25),
                surface_rh=row.get('relative_humidity_2m', 70.0)
            )
            indices_list.append(indices)
        except Exception as e:
            logger.error(f"Error calculating indices for row {idx}: {e}")
            # Append default values
            indices_list.append({
                'cape': 0.0, 'cin': 0.0, 'lifted_index': 0.0,
                'k_index': 0.0, 'total_totals': 0.0, 'showalter_index': 0.0,
                'theta_e': 300.0, 'dewpoint': 20.0
            })
    
    # Add to dataframe
    indices_df = pd.DataFrame(indices_list)
    df = pd.concat([df, indices_df], axis=1)
    
    logger.info(f"✓ Added {len(indices_df.columns)} atmospheric indices")
    
    return df


if __name__ == "__main__":
    # Test the atmospheric indices calculator
    calc = AtmosphericIndices()
    
    # Test with sample data (typical Mumbai monsoon conditions)
    test_data = {
        'temp': 28.0,      # Celsius
        'pressure': 1010,  # hPa
        'rh': 85.0         # %
    }
    
    print("="*80)
    print("ATMOSPHERIC INDICES TEST")
    print("="*80)
    print(f"\nSurface Conditions:")
    print(f"  Temperature: {test_data['temp']}°C")
    print(f"  Pressure: {test_data['pressure']} hPa")
    print(f"  Relative Humidity: {test_data['rh']}%")
    
    indices = calc.calculate_all_indices(
        test_data['temp'],
        test_data['pressure'],
        test_data['rh']
    )
    
    print(f"\nCalculated Indices:")
    print(f"  CAPE: {indices['cape']:.2f} J/kg")
    print(f"  Lifted Index: {indices['lifted_index']:.2f}")
    print(f"  K-Index: {indices['k_index']:.2f}")
    print(f"  Total Totals: {indices['total_totals']:.2f}")
    print(f"  Showalter Index: {indices['showalter_index']:.2f}")
    print(f"  Theta-E: {indices['theta_e']:.2f} K")
    print(f"  Dewpoint: {indices['dewpoint']:.2f}°C")
    
    print("\n" + "="*80)
