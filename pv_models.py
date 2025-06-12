import numpy as np
import pandas as pd
import sqlite3
from datetime import datetime, timedelta
from scipy.constants import k, e
from datetime import timezone
import logging
import os
# KST timezone
KST = timezone(timedelta(hours=9))

# Constants
G_STC = 1000  # STC irradiance (W/m^2)
T_STC = 25  # STC temperature (°C)
AM_STC = 1.5  # STC air mass
q = e  # Electron charge (C)
k_b = k  # Boltzmann constant (J/K)
ALBEDO = 0.2  # Ground reflectance for Seoul (urban area)
MAX_ZENITH_ANGLE = 85.0  # Maximum zenith angle for calculations (degrees)
GHI_MIN = 0.0  # Minimum GHI (W/m^2)
GHI_MAX = 2000.0  # Maximum GHI (W/m^2)

# Perez model coefficients for F1 and F2 (from Perez et al., 1990, as used in PVLib)
PEREZ_COEFFS = [
    {"f11": -0.008, "f12": 0.588, "f13": -0.062, "f21": -0.060, "f22": 0.072, "f23": -0.022},
    {"f11": 0.130, "f12": 0.683, "f13": -0.151, "f21": -0.019, "f22": 0.066, "f23": -0.029},
    {"f11": 0.330, "f12": 0.487, "f13": -0.221, "f21": 0.055, "f22": -0.064, "f23": -0.026},
    {"f11": 0.568, "f12": 0.187, "f13": -0.295, "f21": 0.109, "f22": -0.152, "f23": -0.014},
    {"f11": 0.873, "f12": -0.392, "f13": -0.362, "f21": 0.226, "f22": -0.462, "f23": 0.001},
    {"f11": 1.132, "f12": -1.237, "f13": -0.412, "f21": 0.288, "f22": -0.823, "f23": 0.056},
    {"f11": 1.060, "f12": -1.599, "f13": -0.359, "f21": 0.264, "f22": -1.127, "f23": 0.131},
    {"f11": 0.678, "f12": -0.327, "f13": -0.250, "f21": 0.156, "f22": -1.377, "f23": 0.251}
]

# Real PV module parameters (based on typical datasheets)
PV_MODULES = {
    "Monocrystalline": {
        "name": "SunPower SPR-X21-335",
        "Isc_ref": 6.14,
        "Voc_ref": 67.9,
        "Imp_ref": 5.89,
        "Vmp_ref": 56.9,
        "Pmp_ref": 335,
        "alpha_Isc": 0.0005,
        "beta_Voc": -0.0028,
        "gamma_Pmp": -0.0037,
        "cells": 96,
        "NOCT": 45,
        "area_ref": 1.63,
        "u0": 26.91,
        "u1": 6.2
    },
    "Polycrystalline": {
        "name": "Canadian Solar CS6K-280P",
        "Isc_ref": 9.23,
        "Voc_ref": 38.5,
        "Imp_ref": 8.78,
        "Vmp_ref": 31.9,
        "Pmp_ref": 280,
        "alpha_Isc": 0.0006,
        "beta_Voc": -0.0031,
        "gamma_Pmp": -0.0041,
        "cells": 60,
        "NOCT": 45,
        "area_ref": 1.64,
        "u0": 26.91,
        "u1": 6.2
    },
    "Thin-Film": {
        "name": "First Solar FS-4110-2",
        "Isc_ref": 2.73,
        "Voc_ref": 61.4,
        "Imp_ref": 2.45,
        "Vmp_ref": 44.9,
        "Pmp_ref": 110,
        "alpha_Isc": 0.0004,
        "beta_Voc": -0.0027,
        "gamma_Pmp": -0.0030,
        "cells": 1,
        "NOCT": 45,
        "area_ref": 0.72,
        "u0": 23.37,
        "u1": 5.44
    }
}


def calculate_zenith_angle(timestamp, latitude, longitude, standard_meridian=135):
    """Calculate zenith angle for a given timestamp and location."""
    if isinstance(timestamp, np.datetime64):
        timestamp = pd.Timestamp(timestamp)
    lat_rad = np.radians(latitude)
    day_of_year = timestamp.timetuple().tm_yday
    declination = 23.45 * np.sin(np.radians(360 * (284 + day_of_year) / 365))
    decl_rad = np.radians(declination)
    B = (360 / 365) * (day_of_year - 81)
    EOT = 9.87 * np.sin(np.radians(2 * B)) - 7.53 * np.cos(np.radians(B)) - 1.5 * np.sin(np.radians(B))
    hour = timestamp.hour + timestamp.minute / 60.0
    time_correction = (4 * (longitude - standard_meridian) + EOT) / 60.0
    solar_time = hour + time_correction
    hour_angle = 15 * (solar_time - 12)
    hour_rad = np.radians(hour_angle)
    cos_zenith = (np.sin(lat_rad) * np.sin(decl_rad) +
                  np.cos(lat_rad) * np.cos(decl_rad) * np.cos(hour_rad))
    zenith_angle = np.degrees(np.arccos(np.clip(cos_zenith, -1, 1)))
    return zenith_angle, hour_angle, declination


def calc_solar_altitude(timestamp, latitude, longitude):
    """Calculate solar altitude from zenith angle."""
    if isinstance(timestamp, np.datetime64):
        timestamp = pd.Timestamp(timestamp)
    zenith_angle, _, _ = calculate_zenith_angle(timestamp, latitude, longitude)
    solar_altitude = 90 - zenith_angle
    return solar_altitude


def calc_solar_azimuth(timestamp, latitude, longitude):
    """Calculate solar azimuth."""
    if isinstance(timestamp, np.datetime64):
        timestamp = pd.Timestamp(timestamp)
    zenith, hour_angle, declination = calculate_zenith_angle(timestamp, latitude, longitude)
    zenith_rad = np.radians(zenith)
    hour_rad = np.radians(hour_angle)
    decl_rad = np.radians(declination)
    lat_rad = np.radians(latitude)
    sin_az = np.sin(hour_rad) * np.cos(decl_rad) / np.sin(zenith_rad)
    cos_az = (np.sin(zenith_rad) * np.sin(lat_rad) - np.sin(decl_rad)) / (np.cos(zenith_rad) * np.cos(lat_rad))
    azimuth = np.degrees(np.arctan2(sin_az, cos_az))
    azimuth = (azimuth + 180) % 360
    return azimuth


def calculate_air_mass(zenith_angle):
    """Calculate air mass using the Kasten-Young (1989) formula."""
    if zenith_angle >= 90:
        return float('inf')
    zenith_rad = np.radians(zenith_angle)
    return 1 / (np.cos(zenith_rad) + 0.50572 * (96.07995 - zenith_angle) ** -1.6364)


def decompose_ghi_disc(ghi, timestamp, latitude, longitude, standard_meridian=135):
    """Decompose GHI into DNI and DHI using the DISC model."""
    if isinstance(timestamp, np.datetime64):
        timestamp = pd.Timestamp(timestamp)
    zenith_angle, _, _ = calculate_zenith_angle(timestamp, latitude, longitude, standard_meridian)
    zenith_angle = min(zenith_angle, MAX_ZENITH_ANGLE)
    ghi = np.clip(ghi, GHI_MIN, GHI_MAX)
    day_of_year = timestamp.timetuple().tm_yday
    C = np.radians(360 * (day_of_year - 1) / 365)
    re = (1.00011 + 0.034221 * np.cos(C) + 0.00128 * np.sin(C) +
          0.000719 * np.cos(2 * C) + 0.000077 * np.sin(2 * C))
    I0 = 1367 * re
    cos_z = np.cos(np.radians(zenith_angle))
    I0h = I0 * cos_z if cos_z > 0 else 0
    m = calculate_air_mass(zenith_angle)
    m = min(m, 10)
    kt = ghi / I0h if I0h > 0 else 0
    kt = np.clip(kt, 0, 1)
    a = 0.4327 - 0.1925 * m + 0.01453 * m ** 2 - 0.0005127 * m ** 3
    b = -0.3913 + 0.2779 * m - 0.02224 * m ** 2 + 0.0008042 * m ** 3
    c = 0.9585 - 0.08529 * m + 0.005923 * m ** 2 - 0.0001978 * m ** 3
    kn = a + b * kt + c * kt ** 2
    kn = np.clip(kn, 0, 1)
    dni = kn * I0 if cos_z > 0 else 0
    dni = max(dni, 0)
    dhi = ghi - dni * cos_z if cos_z > 0 else 0
    dhi = max(dhi, 0)
    return dni, dhi, I0, m


def calculate_angle_of_incidence(timestamp, latitude, longitude, tilt, orientation):
    """Calculate the angle of incidence (AOI) between the sun and the PV panel."""
    if isinstance(timestamp, np.datetime64):
        timestamp = pd.Timestamp(timestamp)
    solar_altitude = calc_solar_altitude(timestamp, latitude, longitude)
    solar_azimuth = calc_solar_azimuth(timestamp, latitude, longitude)
    tilt_rad = np.radians(tilt)
    orientation_rad = np.radians(orientation)
    solar_altitude_rad = np.radians(solar_altitude)
    solar_azimuth_rad = np.radians(solar_azimuth)
    cos_aoi = (np.sin(solar_altitude_rad) * np.cos(tilt_rad) +
               np.cos(solar_altitude_rad) * np.sin(tilt_rad) * np.cos(solar_azimuth_rad - orientation_rad))
    cos_aoi = np.clip(cos_aoi, 0, 1)
    aoi = np.degrees(np.arccos(cos_aoi))
    return aoi


def calculate_iam(aoi):
    """Calculate Incidence Angle Modifier (IAM) per IEC 61853-2."""
    b0 = 0.05
    aoi_rad = np.radians(aoi)
    if aoi >= 90:
        return 0
    iam = 1 - b0 * (1 / np.cos(aoi_rad) - 1)
    return max(iam, 0)


def calculate_perez_diffuse(dni, dhi, zenith_angle, aoi, air_mass, I0, tilt):
    """Calculate diffuse irradiance on a tilted surface using the Perez model."""
    Delta = (dhi * air_mass) / I0 if I0 > 0 else 0
    epsilon = ((dhi + dni) / dhi + 5.535e-6 * zenith_angle ** 3) / (1 + 5.535e-6 * zenith_angle ** 3) if dhi > 0 else 1
    epsilon_bins = [1.065, 1.230, 1.500, 1.950, 2.800, 4.500, 6.200, float('inf')]
    bin_idx = 0
    for i, threshold in enumerate(epsilon_bins):
        if epsilon < threshold:
            bin_idx = i
            break
    coeffs = PEREZ_COEFFS[bin_idx]
    F1 = coeffs["f11"] + coeffs["f12"] * Delta + coeffs["f13"] * zenith_angle
    F1 = max(F1, 0)
    F2 = coeffs["f21"] + coeffs["f22"] * Delta + coeffs["f23"] * zenith_angle
    a = max(0, np.cos(np.radians(aoi)))
    b = max(np.cos(np.radians(85)), np.cos(np.radians(zenith_angle)))
    tilt_rad = np.radians(tilt)
    diffuse_tilted = dhi * ((1 - F1) * (1 + np.cos(tilt_rad)) / 2 + F1 * a / b + F2 * np.sin(tilt_rad))
    return max(diffuse_tilted, 0)


def calculate_gti(ghi, timestamps, latitude, longitude, tilt, orientation):
    """Calculate Global Tilted Irradiance (GTI) with direct, diffuse (Perez model), and albedo components."""
    gti_values = []
    for i, ts in enumerate(timestamps):
        zenith_angle, _, _ = calculate_zenith_angle(ts, latitude, longitude)
        solar_altitude = calc_solar_altitude(ts, latitude, longitude)
        solar_altitude_rad = np.radians(solar_altitude)
        if solar_altitude <= 0 or zenith_angle >= MAX_ZENITH_ANGLE:
            gti_values.append(0)
            continue
        dni, dhi, I0, m = decompose_ghi_disc(ghi[i], ts, latitude, longitude)
        aoi = calculate_angle_of_incidence(ts, latitude, longitude, tilt, orientation)
        iam = calculate_iam(aoi)
        direct_tilted = dni * iam * np.cos(np.radians(aoi))
        diffuse_tilted = calculate_perez_diffuse(dni, dhi, zenith_angle, aoi, m, I0, tilt)
        tilt_rad = np.radians(tilt)
        albedo_tilted = ghi[i] * ALBEDO * (1 - np.cos(tilt_rad)) / 2
        gti = direct_tilted + diffuse_tilted + albedo_tilted
        gti = max(gti, 0)
        gti_values.append(gti)
    return np.array(gti_values)


def calculate_cell_temperature(ghi, temperature, wind_speed, u0, u1):
    """Calculate PV cell temperature using the Faiman model (IEC 61724-2)."""
    if not (len(ghi) == len(temperature) == len(wind_speed)):
        raise ValueError("Length of ghi, temperature, and wind_speed arrays must match")
    wind_speed = np.array([1.0 if ws is None else float(ws) for ws in wind_speed])
    t_cell = temperature + ghi / (u0 + u1 * wind_speed)
    return t_cell


def calculate_inverter_efficiency(pdc, pdc_max):
    """Calculate inverter efficiency as a function of load."""
    load_ratio = pdc / pdc_max if pdc_max > 0 else 0
    if load_ratio < 0.1:
        eta = 0.85
    elif load_ratio < 0.3:
        eta = 0.90
    elif load_ratio < 0.7:
        eta = 0.95
    else:
        eta = 0.96
    return eta


def simple_efficiency_model(gti, timestamps, latitude, longitude, pv_type, area, tilt, orientation, temperature,
                            wind_speed):
    """Simple Efficiency Model."""
    module = PV_MODULES[pv_type]
    efficiency = module["Pmp_ref"] / (G_STC * module["area_ref"])
    t_cell = calculate_cell_temperature(gti, temperature, wind_speed, module["u0"], module["u1"])
    temp_diff = t_cell - T_STC
    adjusted_efficiency = efficiency * (1 + module["gamma_Pmp"] * temp_diff)
    adjusted_efficiency = np.clip(adjusted_efficiency, 0, 1)
    pdc = gti * area * adjusted_efficiency
    pac = np.zeros_like(pdc)
    pdc_max = max(pdc) if max(pdc) > 0 else 1
    for i in range(len(pdc)):
        eta = calculate_inverter_efficiency(pdc[i], pdc_max)
        pac[i] = pdc[i] * eta
    I = pdc / module["Vmp_ref"]
    V = module["Vmp_ref"] * np.ones_like(I)
    return gti, pdc, pac, I, V


def single_diode_model(gti, timestamps, latitude, longitude, pv_type, area, tilt, orientation, temperature, wind_speed):
    """Single Diode Model with Newton-Raphson MPP calculation."""
    module = PV_MODULES[pv_type]
    t_cell = calculate_cell_temperature(gti, temperature, wind_speed, module["u0"], module["u1"])
    Isc_ref = module["Isc_ref"]
    Voc_ref = module["Voc_ref"]
    Imp_ref = module["Imp_ref"]
    Vmp_ref = module["Vmp_ref"]
    alpha_Isc = module["alpha_Isc"]
    beta_Voc = module["beta_Voc"]
    n_cells = module["cells"]
    T_ref = T_STC + 273.15
    Vt_ref = k_b * T_ref / q
    n = 1.0
    I0_ref = Isc_ref / (np.exp(Voc_ref / (n * n_cells * Vt_ref)) - 1)
    Rs = (Voc_ref - Vmp_ref) / Imp_ref
    Rsh_ref = Vmp_ref / (Isc_ref - Imp_ref)
    pdc = []
    pac = []
    I_mpp = []
    V_mpp = []
    for i in range(len(gti)):
        G = gti[i]
        T = t_cell[i] + 273.15
        Vt = k_b * T / q
        Isc = Isc_ref * (G / G_STC) * (1 + alpha_Isc * (T - T_ref))
        I0 = I0_ref * (T / T_ref) ** 3 * np.exp((q * Voc_ref / (n * k_b)) * (1 / T_ref - 1 / T))
        Voc = Voc_ref + beta_Voc * (T - T_ref) + n * n_cells * Vt * np.log(G / G_STC) if G > 0 else 0
        Rsh = Rsh_ref * (G_STC / G) if G > 0 else Rsh_ref
        V = Vmp_ref
        I = Imp_ref
        tol = 1e-6
        max_iter = 100
        for _ in range(max_iter):
            I = Isc - I0 * (np.exp((V + I * Rs) / (n * n_cells * Vt)) - 1) - (V + I * Rs) / Rsh
            P = V * I
            dI_dV = -I0 / (n * n_cells * Vt) * np.exp((V + I * Rs) / (n * n_cells * Vt)) * (1 + Rs * dI_dV) - 1 / Rsh
            dI_dV = dI_dV / (1 + Rs * I0 / (n * n_cells * Vt) * np.exp((V + I * Rs) / (n * n_cells * Vt)))
            dP_dV = I + V * dI_dV
            if abs(dP_dV) < tol:
                break
            V -= dP_dV / (dI_dV + I + V * dI_dV ** 2)
        I = Isc - I0 * (np.exp((V + I * Rs) / (n * n_cells * Vt)) - 1) - (V + I * Rs) / Rsh
        pdc_val = V * I
        eta = calculate_inverter_efficiency(pdc_val, pdc_val * 1.1)
        pac_val = pdc_val * eta
        pdc.append(pdc_val)
        pac.append(pac_val)
        I_mpp.append(I)
        V_mpp.append(V)
    return gti, np.array(pdc), np.array(pac), np.array(I_mpp), np.array(V_mpp)


def cec_model(gti, timestamps, latitude, longitude, pv_type, area, tilt, orientation, temperature, wind_speed):
    """CEC Model (Six Parameter Model) with Newton-Raphson MPP calculation."""
    return single_diode_model(gti, timestamps, latitude, longitude, pv_type, area, tilt, orientation, temperature,
                              wind_speed)


def sandia_array_performance_model(gti, timestamps, latitude, longitude, pv_type, area, tilt, orientation, temperature,
                                   wind_speed):
    """Sandia Array Performance Model (SAPM) - Simplified."""
    module = PV_MODULES[pv_type]
    t_cell = calculate_cell_temperature(gti, temperature, wind_speed, module["u0"], module["u1"])
    temp_diff = t_cell - T_STC
    efficiency = module["Pmp_ref"] / (G_STC * module["area_ref"])
    adjusted_efficiency = efficiency * (1 + module["gamma_Pmp"] * temp_diff)
    adjusted_efficiency = np.clip(adjusted_efficiency, 0, 1)
    pdc = gti * area * adjusted_efficiency
    pac = np.zeros_like(pdc)
    pdc_max = max(pdc) if max(pdc) > 0 else 1
    for i in range(len(pdc)):
        eta = calculate_inverter_efficiency(pdc[i], pdc_max)
        pac[i] = pdc[i] * eta
    I = pdc / module["Vmp_ref"]
    V = module["Vmp_ref"] * np.ones_like(I)
    return gti, pdc, pac, I, V


def calculate_iec_metrics(pdc_actual, pac_actual, pdc_forecasted, pac_forecasted, ghi_actual, ghi_forecasted, area,
                          pmp_ref, pv_type, hours, timestamps):
    """
    Calculate IEC performance metrics for PV system over a specified period.

    Parameters:
    - pdc_actual, pac_actual: Current DC/AC power (W, single value or array)
    - pdc_forecasted, pac_forecasted: Forecasted DC/AC power (W, array)
    - ghi_actual, ghi_forecasted: Current/forecasted GTI (W/m², single value or array)
    - area: User-input PV system area (m²)
    - pmp_ref: Reference module power at STC (W)
    - pv_type: User-selected PV module type (e.g., 'Monocrystalline')
    - hours: Total period (hours, calculated from forecast duration, for fallback)
    - timestamps: Array of forecast timestamps for accurate duration calculation

    Returns:
    - Dictionary with Yf, PR, CF, EPI, CSER, and database-compatible values (pdc_kwh, pac_kwh, ghi_kwh_per_m2, gti_kwh_per_m2)
    """
    # Setup logging
    logger = logging.getLogger(__name__)

    # Ensure inputs are arrays
    pac_forecasted = np.array(pac_forecasted, dtype=float)
    ghi_forecasted = np.array(ghi_forecasted, dtype=float)
    pdc_forecasted = np.array(pdc_forecasted, dtype=float)
    pac_actual = float(np.mean(pac_actual)) if isinstance(pac_actual, np.ndarray) else float(pac_actual)

    # Constants
    delta_t = 0.5  # Time interval (hours, 30 min)

    # Compute duration (T) from timestamps
    df = pd.DataFrame({'timestamp': timestamps})
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    T = (df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]).total_seconds() / 3600 if len(df) > 1 else hours
    T = max(T, 0.5)  # Ensure minimum period

    # Get module parameters
    module = PV_MODULES[pv_type]
    area_ref = module['area_ref']  # m²

    # Calculate rated power based on user-input area
    P_rated = (area / area_ref) * pmp_ref / 1000  # kWp

    # Calculate energies
    pdc_kwh = np.sum(pdc_forecasted * delta_t) / 1000  # kWh
    pac_kwh_raw = np.sum(pac_forecasted * delta_t) / 1000  # kWh
    pac_kwh = pac_kwh_raw * (1 - 0.10 - 0.05)  # Apply 10% inverter, 5% temperature losses
    ghi_kwh_per_m2 = np.sum(ghi_forecasted * delta_t) / 1000  # kWh/m²
    gti_kwh_per_m2 = np.sum(ghi_forecasted * delta_t) / 1000  # kWh/m² (GTI used)

    # Log intermediate values
    logger.info(
        f"pdc_kwh: {pdc_kwh} kWh, pac_kwh: {pac_kwh} kWh, ghi_kwh_per_m2: {ghi_kwh_per_m2} kWh/m², gti_kwh_per_m2: {gti_kwh_per_m2} kWh/m², T: {T} hours, P_rated: {P_rated} kWp")

    # Calculate metrics
    Yr = gti_kwh_per_m2  # kWh/m²
    Yf = pac_kwh / P_rated if P_rated > 0 else 0  # kWh/kWp
    PR = min(Yf / Yr, 1) if Yr > 0 else 0
    CF = pac_kwh / (P_rated * T) if P_rated > 0 and T > 0 else 0
    EPI = np.mean([p_f / pac_actual if pac_actual > 0 else 0 for p_f in pac_forecasted])

    # CSER with Seoul reference climate (1200 kWh/m²/year)
    H_poa_ref = 1200 * (T / 8760)  # kWh/m²
    E_expected = P_rated * H_poa_ref  # kWh
    CSER = pac_kwh / E_expected if E_expected > 0 else 0

    return {
        "Yf": round(Yf, 3),
        "PR": round(PR, 3),
        "CF": round(CF, 3),
        "EPI": round(EPI, 3),
        "CSER": round(CSER, 3),
        "pdc_kwh": round(pdc_kwh, 3),
        "pac_kwh": round(pac_kwh, 3),
        "ghi_kwh_per_m2": round(ghi_kwh_per_m2, 3),
        "gti_kwh_per_m2": round(gti_kwh_per_m2, 3)
    }

def pv_performance(ghi, timestamps, latitude, longitude, pv_type, area, tilt, orientation, temperature, wind_speed,
                   model_type="simple"):
    """Calculate PV performance metrics using SAM models."""
    gti = calculate_gti(ghi, timestamps, latitude, longitude, tilt, orientation)
    if model_type == "simple":
        return simple_efficiency_model(gti, timestamps, latitude, longitude, pv_type, area, tilt, orientation,
                                       temperature, wind_speed)
    elif model_type == "sapm":
        return sandia_array_performance_model(gti, timestamps, latitude, longitude, pv_type, area, tilt, orientation,
                                              temperature, wind_speed)
    elif model_type == "single_diode":
        return single_diode_model(gti, timestamps, latitude, longitude, pv_type, area, tilt, orientation, temperature,
                                  wind_speed)
    elif model_type == "cec":
        return cec_model(gti, timestamps, latitude, longitude, pv_type, area, tilt, orientation, temperature,
                         wind_speed)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def fetch_forecasted_data(latitude, longitude, start_time, end_time, forecast_db_path, weather_db_path, current_time):
    """Fetch forecasted GHI and weather data based on the latest forecast cycle."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(os.path.join(os.path.dirname(__file__), 'data', 'pv_performance.log')),
                  logging.StreamHandler()]
    )
    logger = logging.getLogger(__name__)

    conn_forecast = sqlite3.connect(forecast_db_path)
    conn_weather = sqlite3.connect(weather_db_path)

    current_time_str = current_time.strftime("%Y-%m-%d %H:%M:%S")

    # Debug: Fetch all available forecast data to inspect the database
    debug_query_forecast = """
                           SELECT lat, lon, timestamp, timestep, GHI
                           FROM forecast_data
                           ORDER BY timestamp DESC
                               LIMIT 10 \
                           """
    debug_forecast_df = pd.read_sql_query(debug_query_forecast, conn_forecast)
    logger.info(f"Available forecast data (top 10 rows): {debug_forecast_df.to_dict('records')}")

    # Step 1: Find the latest forecast cycle by fetching the most recent timestep = 1
    # Use a small tolerance for lat and lon comparisons to handle floating-point precision
    lat_lower = latitude - 1e-6
    lat_upper = latitude + 1e-6
    lon_lower = longitude - 1e-6
    lon_upper = longitude + 1e-6
    query_latest_cycle = """
                         SELECT timestamp
                         FROM forecast_data
                         WHERE lat BETWEEN ? \
                           AND ?
                           AND lon BETWEEN ? \
                           AND ?
                           AND timestep = 1
                           AND timestamp <= ?
                         ORDER BY timestamp DESC
                             LIMIT 1 \
                         """
    logger.info(
        f"Finding latest forecast cycle for lat={latitude}, lon={longitude}, before {current_time_str} with timestep=1.")
    latest_cycle_df = pd.read_sql_query(
        query_latest_cycle,
        conn_forecast,
        params=(lat_lower, lat_upper, lon_lower, lon_upper, current_time_str)
    )
    if latest_cycle_df.empty:
        logger.error(
            f"No forecast cycle found with timestep=1 for lat={latitude}, lon={longitude} before {current_time_str}.")
        conn_forecast.close()
        conn_weather.close()
        return None, None, None, None, None, None

    latest_cycle_timestamp = pd.to_datetime(latest_cycle_df['timestamp'].iloc[0])
    logger.info(f"Latest forecast cycle starts at {latest_cycle_timestamp} (assumed KST).")

    # Step 2: Fetch the forecast data for timesteps 1 to 6 in this cycle
    query_ghi = """
                SELECT timestamp, GHI, timestep
                FROM forecast_data
                WHERE lat BETWEEN ? \
                  AND ?
                  AND lon BETWEEN ? \
                  AND ?
                  AND timestamp >= ?
                  AND timestep BETWEEN 1 \
                  AND 6
                ORDER BY timestep \
                """
    logger.info(
        f"Fetching forecast GHI data for lat={latitude}, lon={longitude}, starting from {latest_cycle_timestamp} for timesteps 1 to 6.")
    ghi_df = pd.read_sql_query(
        query_ghi,
        conn_forecast,
        params=(lat_lower, lat_upper, lon_lower, lon_upper, latest_cycle_timestamp.strftime("%Y-%m-%d %H:%M:%S"))
    )
    logger.info(f"Forecast GHI data query result: {len(ghi_df)} rows found.")
    if not ghi_df.empty:
        logger.debug(f"GHI data sample: {ghi_df.to_dict(orient='records')}")
    # Since database timestamps are already in KST, we don't need to localize them
    ghi_df['timestamp'] = pd.to_datetime(ghi_df['timestamp'])

    # Deduplicate GHI data by keeping the most recent entry for each timestep
    if not ghi_df.empty:
        ghi_df = ghi_df.sort_values(by=['timestamp', 'timestep']).drop_duplicates(subset=['timestamp'], keep='last')
        logger.info(f"After deduplication, GHI data: {ghi_df.to_dict(orient='records')}")
    else:
        logger.warning(
            f"No forecast data found for the latest cycle starting at {latest_cycle_timestamp}. Using default values.")
        conn_forecast.close()
        conn_weather.close()
        return None, None, None, None, None, None

    # Define expected timestamps based on the GHI data
    expected_timestamps = ghi_df['timestamp'].values
    logger.info(f"Expected forecast timestamps based on GHI: {expected_timestamps}")

    # Create a DataFrame with the GHI timestamps
    ghi_full_df = pd.DataFrame({
        'timestamp': expected_timestamps,
        'GHI': ghi_df['GHI'],
        'timestep': ghi_df['timestep']
    })

    # Fetch all forecast weather data for the location to inspect available data
    debug_query_weather = """
                          SELECT forecast_time
                          FROM forecasted_weather_data
                          WHERE lat BETWEEN ? AND ?
                            AND lon BETWEEN ? AND ?
                          ORDER BY forecast_time DESC \
                          """
    lat_lower = latitude - 0.01
    lat_upper = latitude + 0.01
    lon_lower = longitude - 0.01
    lon_upper = longitude + 0.01
    debug_weather_df = pd.read_sql_query(
        debug_query_weather,
        conn_weather,
        params=(lat_lower, lat_upper, lon_lower, lon_upper)
    )
    logger.info(
        f"All available forecast weather timestamps for lat={latitude}, lon={longitude}: {debug_weather_df['forecast_time'].tolist()}")

    # Fetch weather forecast data within the range of GHI timestamps
    # Convert numpy.datetime64 to pd.Timestamp before formatting
    time_min = pd.Timestamp(expected_timestamps[0]).strftime("%Y-%m-%d %H:%M:%S")
    time_max = pd.Timestamp(expected_timestamps[-1]).strftime("%Y-%m-%d %H:%M:%S")
    query_weather = """
                    SELECT lat, lon, temperature, wind_speed, forecast_time
                    FROM forecasted_weather_data
                    WHERE lat BETWEEN ? AND ?
                      AND lon BETWEEN ? AND ?
                      AND forecast_time >= ?
                      AND forecast_time <= ?
                    ORDER BY forecast_time \
                    """
    logger.info(
        f"Fetching weather data for lat between {lat_lower} and {lat_upper}, lon between {lon_lower} and {lon_upper}, forecast_time between {time_min} and {time_max}.")
    weather_df = pd.read_sql_query(
        query_weather,
        conn_weather,
        params=(lat_lower, lat_upper, lon_lower, lon_upper, time_min, time_max)
    )
    logger.info(f"Weather data query result: {len(weather_df)} rows found.")
    if not weather_df.empty:
        logger.debug(f"Weather data sample: {weather_df.to_dict(orient='records')}")
    # Since database timestamps are already in KST, we don't need to localize them
    weather_df['forecast_time'] = pd.to_datetime(weather_df['forecast_time'])

    conn_forecast.close()
    conn_weather.close()

    # Rename forecast_time to timestamp for alignment
    weather_df = weather_df.rename(columns={'forecast_time': 'timestamp'})

    # Deduplicate weather_df by keeping the most recent entry for each (timestamp, lat, lon)
    if not weather_df.empty:
        weather_df = weather_df.sort_values(by=['timestamp', 'lat', 'lon']).drop_duplicates(
            subset=['timestamp', 'lat', 'lon'], keep='last')

    # Create a DataFrame with GHI timestamps for weather data alignment
    weather_full_df = pd.DataFrame({'timestamp': expected_timestamps})
    weather_full_df = weather_full_df.merge(weather_df, on='timestamp', how='left')

    # Interpolate weather data to fill gaps
    weather_full_df['temperature'] = weather_full_df['temperature'].interpolate(method='linear', limit_direction='both')
    weather_full_df['wind_speed'] = weather_full_df['wind_speed'].interpolate(method='linear', limit_direction='both')
    weather_full_df['lat'] = weather_full_df['lat'].fillna(latitude)
    weather_full_df['lon'] = weather_full_df['lon'].fillna(longitude)

    # Replace any remaining None or NaN values with defaults, and adjust unrealistic temperatures
    weather_full_df['temperature'] = weather_full_df['temperature'].fillna(25.0)
    weather_full_df['wind_speed'] = weather_full_df['wind_speed'].fillna(1.0)
    weather_full_df['temperature'] = weather_full_df['temperature'].apply(
        lambda x: 25.0 if x < 15.0 else x  # Adjust temperatures below 15°C to 25°C
    )

    # Merge GHI and weather data using GHI timestamps as the reference
    merged_df = pd.DataFrame(columns=['timestamp', 'GHI', 'timestep', 'temperature', 'wind_speed'])
    for idx, row in ghi_full_df.iterrows():
        ghi_time = pd.Timestamp(row['timestamp'])
        weather_row = weather_full_df[weather_full_df['timestamp'] == ghi_time]
        if not weather_row.empty:
            merged_row = {
                'timestamp': ghi_time,
                'GHI': row['GHI'],
                'timestep': row['timestep'],
                'temperature': weather_row['temperature'].iloc[0],
                'wind_speed': weather_row['wind_speed'].iloc[0]
            }
        else:
            logger.warning(f"No weather data found for {ghi_time}. Using default values.")
            merged_row = {
                'timestamp': ghi_time,
                'GHI': row['GHI'],
                'timestep': row['timestep'],
                'temperature': 25.0,
                'wind_speed': 1.0
            }
        merged_df = pd.concat([merged_df, pd.DataFrame([merged_row])], ignore_index=True)

    logger.info(f"Merged dataframe has {len(merged_df)} rows after joining forecast and weather data.")
    return (merged_df['timestamp'].values,
            merged_df['GHI'].values,
            merged_df['temperature'].values,
            merged_df['wind_speed'].values,
            merged_df['GHI'].values,
            merged_df['GHI'].values)


def fetch_historical_data(latitude, longitude, start_time, end_time, ghi_db_path, weather_db_path, current_time):
    """Fetch the most recent historical GHI and weather data before the current time."""

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(os.path.join(os.path.dirname(__file__), 'data', 'pv_performance.log')),
                  logging.StreamHandler()]
    )

    logger = logging.getLogger(__name__)

    conn_ghi = sqlite3.connect(ghi_db_path)
    conn_weather = sqlite3.connect(weather_db_path)

    current_time_str = current_time.strftime("%Y-%m-%d %H:%M:%S")

    # Debug: Fetch all available timestamps in the range to inspect the data
    lat_lower = latitude - 1e-6
    lat_upper = latitude + 1e-6
    lon_lower = longitude - 1e-6
    lon_upper = longitude + 1e-6
    debug_query_ghi = """
                      SELECT timestamp
                      FROM ghi_data
                      WHERE lat BETWEEN ? \
                        AND ?
                        AND lon BETWEEN ? \
                        AND ?
                        AND timestamp <= ?
                      ORDER BY timestamp DESC \
                      """
    debug_ghi_df = pd.read_sql_query(
        debug_query_ghi,
        conn_ghi,
        params=(lat_lower, lat_upper, lon_lower, lon_upper, current_time_str)
    )
    logger.info(
        f"Available GHI timestamps for lat={latitude}, lon={longitude} before {current_time_str}: {debug_ghi_df['timestamp'].tolist()}")

    # Fetch the most recent GHI data before the current time
    query_ghi = """
                SELECT timestamp, GHI
                FROM ghi_data
                WHERE lat BETWEEN ? \
                  AND ?
                  AND lon BETWEEN ? \
                  AND ?
                  AND timestamp <= ?
                ORDER BY timestamp DESC
                    LIMIT 1 \
                """
    try:
        ghi_df = pd.read_sql_query(
            query_ghi,
            conn_ghi,
            params=(lat_lower, lat_upper, lon_lower, lon_upper, current_time_str)
        )
        # Since database timestamps are already in KST, we don't need to localize them
        ghi_df['timestamp'] = pd.to_datetime(ghi_df['timestamp'])
    except Exception as e:
        logger.error(f"Error querying ghi_data table: {str(e)}")
        conn_ghi.close()
        conn_weather.close()
        raise

    # Debug: Fetch all available weather timestamps in the range to inspect the data
    debug_query_weather = """
                          SELECT timestamp
                          FROM weather_data
                          WHERE lat BETWEEN ? \
                            AND ?
                            AND lon BETWEEN ? \
                            AND ?
                            AND timestamp <= ?
                            AND data_type = 'historical'
                          ORDER BY timestamp DESC \
                          """
    debug_weather_df = pd.read_sql_query(
        debug_query_weather,
        conn_weather,
        params=(lat_lower, lat_upper, lon_lower, lon_upper, current_time_str)
    )
    logger.info(
        f"Available weather timestamps for lat={latitude}, lon={longitude} before {current_time_str}: {debug_weather_df['timestamp'].tolist()}")

    # Fetch the most recent weather data before the current time
    query_weather = """
                    SELECT timestamp, temperature, wind_speed
                    FROM weather_data
                    WHERE lat BETWEEN ? \
                      AND ?
                      AND lon BETWEEN ? \
                      AND ?
                      AND timestamp <= ?
                      AND data_type = 'historical'
                    ORDER BY timestamp DESC
                        LIMIT 1 \
                    """
    try:
        weather_df = pd.read_sql_query(
            query_weather,
            conn_weather,
            params=(lat_lower, lat_upper, lon_lower, lon_upper, current_time_str)
        )
        # Since database timestamps are already in KST, we don't need to localize them
        weather_df['timestamp'] = pd.to_datetime(weather_df['timestamp'])
        if weather_df['wind_speed'].isnull().any():
            logger.warning(
                f"Historical weather data contains None values for wind_speed at lat={latitude}, lon={longitude}.")
    except Exception as e:
        logger.error(f"Error querying weather_data table: {str(e)}")
        conn_ghi.close()
        conn_weather.close()
        raise

    merged_df = pd.merge(ghi_df, weather_df, on='timestamp', how='inner')
    if merged_df.empty:
        logger.warning(
            f"No historical data found for exact location (lat={latitude}, lon={longitude}) before {current_time_str}. Attempting to fetch nearest data.")
        start_time = current_time - timedelta(hours=1)
        start_time_str = start_time.strftime("%Y-%m-%d %H:%M:%S")
        query_ghi = """
                    SELECT timestamp, GHI
                    FROM ghi_data
                    WHERE lat BETWEEN ? \
                      AND ?
                      AND lon BETWEEN ? \
                      AND ?
                      AND timestamp BETWEEN ? \
                      AND ?
                      AND timestamp <= ?
                    ORDER BY ABS(strftime('%s', timestamp) - strftime('%s', ?))
                        LIMIT 1 \
                    """
        query_weather = """
                        SELECT timestamp, temperature, wind_speed
                        FROM weather_data
                        WHERE lat BETWEEN ? \
                          AND ?
                          AND lon BETWEEN ? \
                          AND ?
                          AND timestamp BETWEEN ? \
                          AND ?
                          AND data_type = 'historical'
                          AND timestamp <= ?
                        ORDER BY ABS(strftime('%s', timestamp) - strftime('%s', ?))
                            LIMIT 1 \
                        """
        try:
            ghi_df = pd.read_sql_query(
                query_ghi,
                conn_ghi,
                params=(lat_lower, lat_upper, lon_lower, lon_upper, start_time_str, current_time_str, current_time_str,
                        current_time_str)
            )
            weather_df = pd.read_sql_query(
                query_weather,
                conn_weather,
                params=(lat_lower, lat_upper, lon_lower, lon_upper, start_time_str, current_time_str, current_time_str,
                        current_time_str)
            )
            ghi_df['timestamp'] = pd.to_datetime(ghi_df['timestamp'])
            weather_df['timestamp'] = pd.to_datetime(weather_df['timestamp'])
            merged_df = pd.merge(ghi_df, weather_df, on='timestamp', how='inner')
            if weather_df['wind_speed'].isnull().any():
                logger.warning(
                    f"Fallback historical weather data contains None values for wind_speed at lat={latitude}, lon={longitude}.")
        except Exception as e:
            logger.error(f"Error in fallback query for historical data: {str(e)}")
            conn_ghi.close()
            conn_weather.close()
            raise

    conn_ghi.close()
    conn_weather.close()

    if merged_df.empty:
        logger.error(f"No historical data available for lat={latitude}, lon={longitude} before {current_time_str}.")
        raise ValueError("No historical data available for the specified location and time range.")

    # Freshness check: Ensure the data is not older than 1 hour
    latest_timestamp = pd.to_datetime(merged_df['timestamp'].iloc[0])
    time_diff = current_time - latest_timestamp
    if time_diff > pd.Timedelta(hours=1):
        logger.warning(
            f"Historical data is stale: latest timestamp is {latest_timestamp}, current time is {current_time}, difference is {time_diff}.")

    # Replace None or NaN values with defaults
    merged_df['GHI'] = merged_df['GHI'].fillna(0.0)
    merged_df['temperature'] = merged_df['temperature'].fillna(25.0)
    merged_df['wind_speed'] = merged_df['wind_speed'].fillna(1.0)
    # Adjust temperatures below 15°C (unrealistic for Seoul in June) to a more reasonable value
    merged_df['temperature'] = merged_df['temperature'].apply(
        lambda x: 25.0 if x < 15.0 else x
    )

    return (merged_df['timestamp'].values,
            merged_df['GHI'].values,
            merged_df['temperature'].values,
            merged_df['wind_speed'].values)
