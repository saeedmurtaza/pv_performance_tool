
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone, date
import sys
import os
import sqlite3
import logging
from logging.handlers import RotatingFileHandler
from scipy.spatial import cKDTree

# Add the pv_performance_tool directory to the system path
sys.path.append(r"D:\Masters thesis PV\pv_performance_tool")
from pv_models import pv_performance, calculate_iec_metrics, fetch_forecasted_data, fetch_historical_data, PV_MODULES

app = Flask(__name__)

# Configure logging
if not app.debug:
    handler = RotatingFileHandler('pv_performance.log', maxBytes=10000, backupCount=1)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    app.logger.addHandler(handler)

# Database paths
forecast_db_path = os.path.join(os.path.dirname(__file__), 'data', 'forecast_data.db')
weather_db_path = os.path.join(os.path.dirname(__file__), 'data', 'weather_data.db')
ghi_db_path = os.path.join(os.path.dirname(__file__), 'data', 'ghi_data.db')
locations_path = os.path.join(os.path.dirname(__file__), 'data', 'locations_eng.xlsx')
daily_metrics_db_path = os.path.join(os.path.dirname(__file__), 'data', 'daily_iec_metrics.db')

# KST timezone
KST = timezone(timedelta(hours=9))

# Define 20x20 grid for Seoul
coords = [(lat, lon) for lat in np.linspace(37.4, 37.7, 20) for lon in np.linspace(126.8, 127.2, 20)]
locations_df = pd.DataFrame(coords, columns=['lat', 'lon'])

# Database table creation for daily IEC metrics
def create_daily_metrics_table():
    conn = sqlite3.connect(daily_metrics_db_path)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS daily_iec_metrics (
            date TEXT PRIMARY KEY,
            lat REAL,
            lon REAL,
            pv_type TEXT,
            model_type TEXT,
            area REAL,
            tilt REAL,
            orientation REAL,
            pdc_kwh REAL,
            pac_kwh REAL,
            ghi_kwh_per_m2 REAL,
            gti_kwh_per_m2 REAL,
            yf REAL,
            yr REAL,
            pr REAL,
            cf REAL,
            epi REAL,
            cser REAL
        )
    """)
    conn.commit()
    conn.close()

# Store daily IEC metrics
def store_daily_iec_metrics(
    today, lat, lon, pv_type, model_type, area, tilt, orientation,
    pdc_kwh, pac_kwh, ghi_kwh, gti_kwh, yf, yr, pr, cf, epi, cser):
    conn = sqlite3.connect(daily_metrics_db_path)
    c = conn.cursor()
    c.execute("""
        INSERT OR REPLACE INTO daily_iec_metrics
        (date, lat, lon, pv_type, model_type, area, tilt, orientation,
         pdc_kwh, pac_kwh, ghi_kwh_per_m2, gti_kwh_per_m2, yf, yr, pr, cf, epi, cser)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        today, lat, lon, pv_type, model_type, area, tilt, orientation,
        pdc_kwh, pac_kwh, ghi_kwh, gti_kwh, yf, yr, pr, cf, epi, cser
    ))
    conn.commit()
    conn.close()

# Create database table at startup
create_daily_metrics_table()

def round_to_grid(lat, lon):
    """Round coordinates to the nearest grid point."""
    app.logger.info(f"Input coordinates: ({lat}, {lon})")
    coords = locations_df[['lat', 'lon']].values
    tree = cKDTree(coords)
    _, idx = tree.query([lat, lon])
    rounded_lat, rounded_lon = locations_df.iloc[idx][['lat', 'lon']].values
    app.logger.info(f"Rounded coordinates to grid point: ({rounded_lat}, {rounded_lon})")
    return rounded_lat, rounded_lon

def fetch_nearest_historical_data(lat, lon, target_time, ghi_db_path, weather_db_path, time_window_hours=1):
    """Fetch the nearest historical data within a time window or from the nearest location."""
    conn_ghi = sqlite3.connect(ghi_db_path)
    conn_weather = sqlite3.connect(weather_db_path)
    lat_lower = lat - 1e-6
    lat_upper = lat + 1e-6
    lon_lower = lon - 1e-6
    lon_upper = lon + 1e-6
    start_time = target_time - timedelta(hours=time_window_hours)
    end_time = target_time + timedelta(hours=time_window_hours)
    query_ghi = """
        SELECT timestamp, GHI
        FROM ghi_data
        WHERE lat BETWEEN ? AND ? 
          AND lon BETWEEN ? AND ?
          AND timestamp BETWEEN ? AND ?
        ORDER BY ABS(strftime('%s', timestamp) - strftime('%s', ?))
        LIMIT 1
    """
    query_weather = """
        SELECT timestamp, temperature, wind_speed
        FROM weather_data
        WHERE lat BETWEEN ? AND ? 
          AND lon BETWEEN ? AND ?
          AND timestamp BETWEEN ? AND ?
          AND data_type = 'historical'
        ORDER BY ABS(strftime('%s', timestamp) - strftime('%s', ?))
        LIMIT 1
    """
    try:
        ghi_df = pd.read_sql_query(
            query_ghi,
            conn_ghi,
            params=(lat_lower, lat_upper, lon_lower, lon_upper, start_time.strftime("%Y-%m-%d %H:%M:%S"), end_time.strftime("%Y-%m-%d %H:%M:%S"), target_time.strftime("%Y-%m-%d %H:%M:%S"))
        )
        weather_df = pd.read_sql_query(
            query_weather,
            conn_weather,
            params=(lat_lower, lat_upper, lon_lower, lon_upper, start_time.strftime("%Y-%m-%d %H:%M:%S"), end_time.strftime("%Y-%m-%d %H:%M:%S"), target_time.strftime("%Y-%m-%d %H:%M:%S"))
        )
        if not ghi_df.empty and not weather_df.empty:
            merged_df = pd.merge(ghi_df, weather_df, on='timestamp', how='inner')
            if not merged_df.empty:
                conn_ghi.close()
                conn_weather.close()
                return (np.array([pd.to_datetime(merged_df['timestamp'].iloc[0])]),
                        np.array([merged_df['GHI'].iloc[0]]),
                        np.array([merged_df['temperature'].iloc[0]]),
                        np.array([merged_df['wind_speed'].iloc[0]]))
        app.logger.warning(f"No historical data found at exact coordinates ({lat}, {lon}). Searching for nearest location.")
        start_time = target_time - timedelta(hours=24)
        end_time = target_time
        query_ghi = """
            SELECT timestamp, lat, lon, GHI
            FROM ghi_data
            WHERE timestamp BETWEEN ? AND ?
            ORDER BY timestamp DESC
        """
        query_weather = """
            SELECT timestamp, lat, lon, temperature, wind_speed
            FROM weather_data
            WHERE timestamp BETWEEN ? AND ?
            AND data_type = 'historical'
            ORDER BY timestamp DESC
        """
        ghi_df = pd.read_sql_query(
            query_ghi,
            conn_ghi,
            params=(start_time.strftime("%Y-%m-%d %H:%M:%S"), end_time.strftime("%Y-%m-%d %H:%M:%S"))
        )
        weather_df = pd.read_sql_query(
            query_weather,
            conn_weather,
            params=(start_time.strftime("%Y-%m-%d %H:%M:%S"), end_time.strftime("%Y-%m-%d %H:%M:%S"))
        )
        app.logger.info(f"Available GHI locations: {ghi_df[['lat', 'lon']].drop_duplicates().to_dict('records')}")
        app.logger.info(f"Available weather locations: {weather_df[['lat', 'lon']].drop_duplicates().to_dict('records')}")
        if not ghi_df.empty and not weather_df.empty:
            available_coords = ghi_df[['lat', 'lon']].drop_duplicates().values
            tree = cKDTree(available_coords)
            _, idx = tree.query([lat, lon])
            nearest_lat, nearest_lon = available_coords[idx]
            app.logger.info(f"Nearest location found: ({nearest_lat}, {nearest_lon})")
            ghi_df = ghi_df[(ghi_df['lat'] == nearest_lat) & (ghi_df['lon'] == nearest_lon)]
            weather_df = weather_df[(weather_df['lat'] == nearest_lat) & (weather_df['lon'] == nearest_lon)]
            merged_df = pd.merge(ghi_df, weather_df, on=['timestamp', 'lat', 'lon'], how='inner')
            if not merged_df.empty:
                conn_ghi.close()
                conn_weather.close()
                return (np.array([pd.to_datetime(merged_df['timestamp'].iloc[0])]),
                        np.array([merged_df['GHI'].iloc[0]]),
                        np.array([merged_df['temperature'].iloc[0]]),
                        np.array([merged_df['wind_speed'].iloc[0]]))
    except Exception as e:
        app.logger.error(f"Error fetching nearest historical data: {str(e)}")
    conn_ghi.close()
    conn_weather.close()
    return None, None, None, None

@app.route('/', methods=['GET', 'POST'])
def index():
    try:
        locations_df_excel = pd.read_excel(locations_path)
        cols = locations_df_excel.columns.str.lower().str.strip()
        app.logger.info(f"Columns in locations_eng.xlsx: {list(cols)}")
        lat_col = next(col for col in cols if 'lat' in col or 'y' in col)
        lon_col = next(col for col in cols if 'lon' in col or 'x' in col)
        step2_col = next(col for col in cols if 'step 2' in col)
        step3_col = next(col for col in cols if 'step 3' in col)
        locations_df_excel = locations_df_excel.rename(columns={
            locations_df_excel.columns[cols.tolist().index(lat_col)]: 'lat',
            locations_df_excel.columns[cols.tolist().index(lon_col)]: 'lon'
        })
        location_options = [(row['lat'], row['lon']) for _, row in locations_df_excel.iterrows()]
        location_options_with_labels = [
            (
                row['lat'],
                row['lon'],
                f"{row[locations_df_excel.columns[cols.tolist().index(step2_col)]]}{' - ' + row[locations_df_excel.columns[cols.tolist().index(step3_col)]] if pd.notna(row[locations_df_excel.columns[cols.tolist().index(step3_col)]]) else ''} (Lat: {row['lat']}, Lon: {row['lon']})"
            )
            for _, row in locations_df_excel.iterrows()
        ]
        if not location_options:
            app.logger.error("No locations found in locations_eng.xlsx.")
            return render_template('index.html', error="No locations available. Please check locations_eng.xlsx.",
                                   pv_types=list(PV_MODULES.keys()),
                                   model_types=["simple", "sapm", "single_diode", "cec"],
                                   orientation_options=[(0, "South (0°)"), (-90, "East (-90°)"), (90, "West (90°)"), (180, "North (180°)")])
        app.logger.info(f"Loaded {len(location_options)} locations from locations_eng.xlsx.")
    except Exception as e:
        app.logger.error(f"Error loading locations_eng.xlsx: {str(e)}")
        location_options = []
        location_options_with_labels = []
        return render_template('index.html', error=f"Error loading locations_eng.xlsx: {str(e)}.",
                               pv_types=list(PV_MODULES.keys()),
                               model_types=["simple", "sapm", "single_diode", "cec"],
                               orientation_options=[(0, "South (0°)"), (-90, "East (-90°)"), (90, "West (90°)"), (180, "North (180°)")])

    current_results = None
    forecast_results = None
    current_timestamp = ""
    current_ghi = 0
    current_gti = 0
    current_pac = 0
    current_I = 0
    current_V = 0
    timestamps_list = []
    ghi_list = []
    gti_list = []
    pac_list = []
    I_list = []
    V_list = []
    iec_metrics = {}
    forecast_message = ""

    if request.method == 'POST':
        app.logger.info("Received POST request to calculate PV performance.")
        lat_source = request.form.get('lat_source', 'map')
        if lat_source == 'manual':
            try:
                lat = float(request.form['manual_lat'])
                lon = float(request.form['manual_lon'])
                if not (37.4 <= lat <= 37.7 and 126.8 <= lon <= 127.2):
                    app.logger.warning(f"Invalid coordinates: Lat={lat}, Lon={lon}. Out of Seoul bounds.")
                    return render_template('index.html', error="Coordinates out of Seoul bounds (Lat: 37.4-37.7, Lon: 126.8-127.2).",
                                           location_options=location_options, location_options_with_labels=location_options_with_labels,
                                           pv_types=list(PV_MODULES.keys()),
                                           model_types=["simple", "sapm", "single_diode", "cec"],
                                           orientation_options=[(0, "South (0°)"), (-90, "East (-90°)"), (90, "West (90°)"), (180, "North (180°)")])
                lat, lon = round_to_grid(lat, lon)
            except ValueError:
                app.logger.error("Invalid latitude or longitude values in manual input.")
                return render_template('index.html', error="Invalid latitude or longitude values. Please enter valid numbers.",
                                       location_options=location_options, location_options_with_labels=location_options_with_labels,
                                       pv_types=list(PV_MODULES.keys()),
                                       model_types=["simple", "sapm", "single_diode", "cec"],
                                       orientation_options=[(0, "South (0°)"), (-90, "East (-90°)"), (90, "West (90°)"), (180, "North (180°)")])
        else:
            try:
                lat = float(request.form['latitude'])
                lon = float(request.form['longitude'])
                if not (37.4 <= lat <= 37.7 and 126.8 <= lon <= 127.2):
                    app.logger.warning(f"Invalid map-selected coordinates: Lat={lat}, Lon={lon}. Out of Seoul bounds.")
                    return render_template('index.html', error="Selected location is outside Seoul bounds (Lat: 34-40, Lon: 126-132).",
                                           location_options=location_options, location_options_with_labels=location_options_with_labels,
                                           pv_types=list(PV_MODULES.keys()),
                                           model_types=["simple", "sapm", "single_diode", "cec"],
                                           orientation_options=[(0, "South (0°)"), (-90, "East (-90°)"), (90, "West (90°)"), (180, "North (180°)")])
                lat, lon = round_to_grid(lat, lon)
            except ValueError:
                app.logger.error("Invalid latitude or longitude values from map selection.")
                return render_template('index.html', error="Please select a valid location from the map or search.",
                                       location_options=location_options, location_options_with_labels=location_options_with_labels,
                                       pv_types=list(PV_MODULES.keys()),
                                       model_types=["simple", "sapm", "single_diode", "cec"],
                                       orientation_options=[(0, "South (0°)"), (-90, "East (-90°)"), (90, "West (90°)"), (180, "North (180°)")])

        pv_type = request.form['pv_type']
        if pv_type not in PV_MODULES:
            app.logger.error(f"Invalid PV type: {pv_type}")
            return render_template('index.html', error="Invalid PV type selected.",
                                   location_options=location_options, location_options_with_labels=location_options_with_labels,
                                   pv_types=list(PV_MODULES.keys()),
                                   model_types=["simple", "sapm", "single_diode", "cec"],
                                   orientation_options=[(0, "South (0°)"), (-90, "East (-90°)"), (90, "West (90°)"), (180, "North (180°)")])

        try:
            area = float(request.form['area'])
            if area <= 0:
                raise ValueError
        except ValueError:
            app.logger.error("Invalid PV area value.")
            return render_template('index.html', error="PV area must be a positive number.",
                                   location_options=location_options, location_options_with_labels=location_options_with_labels,
                                   pv_types=list(PV_MODULES.keys()),
                                   model_types=["simple", "sapm", "single_diode", "cec"],
                                   orientation_options=[(0, "South (0°)"), (-90, "East (-90°)"), (90, "West (90°)"), (180, "North (180°)")])

        try:
            tilt = float(request.form['tilt'])
            if tilt < 0 or tilt > 90:
                raise ValueError
        except ValueError:
            app.logger.error("Invalid tilt value.")
            return render_template('index.html', error="Tilt must be between 0 and 90 degrees.",
                                   location_options=location_options, location_options_with_labels=location_options_with_labels,
                                   pv_types=list(PV_MODULES.keys()),
                                   model_types=["simple", "sapm", "single_diode", "cec"],
                                   orientation_options=[(0, "South (0°)"), (-90, "East (-90°)"), (90, "West (90°)"), (180, "North (180°)")])

        orientation_source = request.form.get('orientation_source', 'dropdown')
        if orientation_source == 'manual':
            try:
                orientation = float(request.form['manual_orientation'])
                orientation = orientation % 360
            except ValueError:
                app.logger.error("Invalid manual orientation value.")
                return render_template('index.html', error="Orientation must be a valid number.",
                                       location_options=location_options, location_options_with_labels=location_options_with_labels,
                                       pv_types=list(PV_MODULES.keys()),
                                       model_types=["simple", "sapm", "single_diode", "cec"],
                                       orientation_options=[(0, "South (0°)"), (-90, "East (-90°)"), (90, "West (90°)"), (180, "North (180°)")])
        else:
            try:
                orientation = float(request.form['orientation'])
                valid_orientations = [0, -90, 90, 180]
                if orientation not in valid_orientations:
                    app.logger.warning(f"Invalid dropdown orientation: {orientation}")
                    return render_template('index.html', error="Invalid orientation selected.",
                                           location_options=location_options, location_options_with_labels=location_options_with_labels,
                                           pv_types=list(PV_MODULES.keys()),
                                           model_types=["simple", "sapm", "single_diode", "cec"],
                                           orientation_options=[(0, "South (0°)"), (-90, "East (-90°)"), (90, "West (90°)"), (180, "North (180°)")])
            except ValueError:
                app.logger.error("Invalid orientation value from dropdown.")
                return render_template('index.html', error="Please select a valid orientation from the dropdown.",
                                       location_options=location_options, location_options_with_labels=location_options_with_labels,
                                       pv_types=list(PV_MODULES.keys()),
                                       model_types=["simple", "sapm", "single_diode", "cec"],
                                       orientation_options=[(0, "South (0°)"), (-90, "East (-90°)"), (90, "West (90°)"), (180, "North (180°)")])

        model_type = request.form['model_type']
        valid_models = ["simple", "sapm", "single_diode", "cec"]
        if model_type not in valid_models:
            app.logger.error(f"Invalid SAM model type: {model_type}")
            return render_template('index.html', error="Invalid SAM model selected.",
                                   location_options=location_options, location_options_with_labels=location_options_with_labels,
                                   pv_types=list(PV_MODULES.keys()),
                                   model_types=["simple", "sapm", "single_diode", "cec"],
                                   orientation_options=[(0, "South (0°)"), (-90, "East (-90°)"), (90, "West (90°)"), (180, "North (180°)")])

        # Fetch historical data
        now = datetime.now(KST).replace(second=0, microsecond=0)
        current_time = now
        historical_start_fetch = current_time - timedelta(hours=24)
        historical_end_fetch = current_time
        app.logger.info(f"Fetching historical data for lat={lat}, lon={lon}, from {historical_start_fetch} to {historical_end_fetch}, current time is {current_time}.")
        try:
            hist_timestamps, hist_ghi, hist_temp, hist_wind = fetch_historical_data(
                lat, lon, historical_start_fetch, historical_end_fetch, ghi_db_path, weather_db_path, current_time
            )
        except Exception as e:
            app.logger.error(f"Failed to fetch historical data: {str(e)}")
            hist_timestamps, hist_ghi, hist_temp, hist_wind = None, None, None, None

        if hist_timestamps is None or len(hist_timestamps) == 0:
            app.logger.warning("No historical data available. Attempting nearest data.")
            hist_timestamps, hist_ghi, hist_temp, hist_wind = fetch_nearest_historical_data(
                lat, lon, current_time, ghi_db_path, weather_db_path, time_window_hours=1
            )
            if hist_timestamps is None or len(hist_timestamps) == 0:
                app.logger.error("No historical data available.")
                return render_template('index.html', error="No historical data available. Try a different location.",
                                       location_options=location_options, location_options_with_labels=location_options_with_labels,
                                       pv_types=list(PV_MODULES.keys()),
                                       model_types=["simple", "sapm", "single_diode", "cec"],
                                       orientation_options=[(0, "South (0°)"), (-90, "East (-90°)"), (90, "West (90°)"), (180, "North (180°)")])

        # Use most recent timestamp for current data
        latest_idx = np.argmax(hist_timestamps)
        current_timestamp = pd.Timestamp(hist_timestamps[latest_idx], tz=KST)
        current_ghi = hist_ghi[latest_idx]
        current_temp = hist_temp[latest_idx]
        current_wind = hist_wind[latest_idx]

        # Calculate current PV performance
        app.logger.info(f"Using current timestamp {current_timestamp} for PV performance calculation.")
        try:
            current_gti, current_pdc, current_pac, current_I, current_V = pv_performance(
                np.array([current_ghi]), np.array([current_timestamp]), lat, lon, pv_type, area, tilt, orientation,
                np.array([current_temp]), np.array([current_wind]), model_type
            )
        except Exception as e:
            app.logger.error(f"Error calculating current PV performance: {str(e)}")
            return render_template('index.html', error=f"Error calculating current PV performance: {str(e)}.",
                                   location_options=location_options, location_options_with_labels=location_options_with_labels,
                                   pv_types=list(PV_MODULES.keys()),
                                   model_types=["simple", "sapm", "single_diode", "cec"],
                                   orientation_options=[(0, "South (0°)"), (-90, "East (-90°)"), (90, "West (90°)"), (180, "North (180°)")])

        # Fetch forecasted data
        app.logger.info(f"Fetching forecasted data for lat={lat}, lon={lon}, current time is {current_time}.")
        try:
            timestamps, ghi, temperature, wind_speed, ghi_actual, ghi_forecast = fetch_forecasted_data(
                lat, lon, None, None, forecast_db_path, weather_db_path, current_time
            )
        except Exception as e:
            app.logger.error(f"Error fetching forecasted data: {str(e)}")
            error_msg = str(e)
            if "cannot reindex on an axis with duplicate labels" in error_msg:
                error_msg = "Duplicate forecast data entries detected. Ensure forecast scripts run correctly."
            return render_template('index.html', error=f"Error fetching forecasted data: {error_msg}.",
                                   location_options=location_options, location_options_with_labels=location_options_with_labels,
                                   pv_types=list(PV_MODULES.keys()),
                                   model_types=["simple", "sapm", "single_diode", "cec"],
                                   orientation_options=[(0, "South (0°)"), (-90, "East (-90°)"), (90, "West (90°)"), (180, "North (180°)")])

        if timestamps is None or len(timestamps) == 0:
            app.logger.error("No forecast data available.")
            return render_template('index.html', error="No forecast data available. Ensure forecast scripts are running.",
                                   location_options=location_options, location_options_with_labels=location_options_with_labels,
                                   pv_types=list(PV_MODULES.keys()),
                                   model_types=["simple", "sapm", "single_diode", "cec"],
                                   orientation_options=[(0, "South (0°)"), (-90, "East (-90°)"), (90, "West (90°)"), (180, "North (180°)")])
        else:
            actual_start = pd.Timestamp(timestamps[0])
            actual_end = pd.Timestamp(timestamps[-1])
            delta = actual_end - actual_start
            forecast_period_seconds = delta / np.timedelta64(1, 's')
            forecast_period_hours = forecast_period_seconds / 3600.0
            if forecast_period_hours < 3:
                forecast_message = f"Forecast data from {actual_start.strftime('%Y-%m-%d %H:%M:%S')} to {actual_end.strftime('%Y-%m-%d %H:%M:%S')} ({forecast_period_hours:.1f} hours)."
            else:
                forecast_message = f"Forecast data from {actual_start.strftime('%Y-%m-%d %H:%M:%S')} to {actual_end.strftime('%Y-%m-%d %H:%M:%S')} (3 hours)."

        # Calculate forecasted PV performance
        app.logger.info("Calculating forecasted PV performance.")
        try:
            gti, pdc, pac, I, V = pv_performance(
                ghi, timestamps, lat, lon, pv_type, area, tilt, orientation, temperature, wind_speed, model_type
            )
        except Exception as e:
            app.logger.error(f"Error calculating forecasted PV performance: {str(e)}")
            return render_template('index.html', error=f"Error calculating forecasted PV performance: {str(e)}.",
                                   location_options=location_options, location_options_with_labels=location_options_with_labels,
                                   pv_types=list(PV_MODULES.keys()),
                                   model_types=["simple", "sapm", "single_diode", "cec"],
                                   orientation_options=[(0, "South (0°)"), (-90, "East (-90°)"), (90, "West (90°)"), (180, "North (180°)")])

        # Calculate IEC metrics
        app.logger.info("Calculating IEC metrics.")
        delta = timestamps[-1] - timestamps[0]
        actual_forecast_seconds = delta / np.timedelta64(1, 's')
        actual_forecast_hours = max(actual_forecast_seconds / 3600.0, 0.5)
        try:
            iec_metrics = calculate_iec_metrics(
                pdc_actual=current_pdc,
                pac_actual=current_pac,
                pdc_forecasted=pdc,
                pac_forecasted=pac,
                ghi_actual=np.array([current_ghi]),
                ghi_forecasted=ghi,
                area=area,
                pmp_ref=PV_MODULES[pv_type]["Pmp_ref"],
                pv_type=pv_type,
                hours=actual_forecast_hours,
                timestamps=timestamps
            )
        except Exception as e:
            app.logger.error(f"Error calculating IEC metrics: {str(e)}")
            return render_template('index.html', error=f"Error calculating IEC metrics: {str(e)}.",
                                   location_options=location_options, location_options_with_labels=location_options_with_labels,
                                   pv_types=list(PV_MODULES.keys()),
                                   model_types=["simple", "sapm", "single_diode", "cec"],
                                   orientation_options=[(0, "South (0°)"), (-90, "East (-90°)"), (90, "West (90°)"), (180, "North (180°)")])

        # Store daily IEC metrics
        try:
            today = date.today().isoformat()
            store_daily_iec_metrics(
                today, lat, lon, pv_type, model_type, area, tilt, orientation,
                iec_metrics['pdc_kwh'], iec_metrics['pac_kwh'],
                iec_metrics['ghi_kwh_per_m2'], iec_metrics['gti_kwh_per_m2'],
                iec_metrics['Yf'], iec_metrics['Yr'], iec_metrics['PR'],
                iec_metrics['CF'], iec_metrics['EPI'], iec_metrics['CSER']
            )
            app.logger.info(f"Stored daily IEC metrics for {today}.")
        except Exception as e:
            app.logger.error(f"Error storing daily IEC metrics: {str(e)}")

        # Prepare current data for plotting
        current_timestamp_str = current_timestamp.strftime("%Y-%m-%d %H:%M:%S")
        current_ghi = float(current_ghi if current_ghi is not None else 0.0)
        current_gti = float(current_gti[0] if current_gti[0] is not None else 0.0)
        current_pdc = float(current_pdc[0] if current_pdc[0] is not None else 0.0)
        current_pac = float(current_pac[0] if current_pac[0] is not None else 0.0)
        current_I = float(current_I[0] if current_I[0] is not None else 0.0)
        current_V = float(current_V[0] if current_V[0] is not None else 0.0)
        current_temp = float(current_temp if current_temp is not None else 25.0)
        current_wind = float(current_wind if current_wind is not None else 1.0)

        current_results = {
            'timestamp': current_timestamp_str,
            'ghi': round(current_ghi, 2),
            'gti': round(current_gti, 2),
            'pdc': round(current_pdc, 2),
            'pac': round(current_pac, 2),
            'I': round(current_I, 2),
            'V': round(current_V, 2),
            'temperature': round(current_temp, 2),
            'wind_speed': round(current_wind, 2)
        }

        # Prepare forecasted data for display and plotting
        forecast_results = []
        timestamps_list = []
        ghi_list = []
        gti_list = []
        pac_list = []
        I_list = []
        V_list = []
        for i in range(len(timestamps)):
            timestamp = pd.Timestamp(timestamps[i])
            timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
            timestamps_list.append(timestamp_str)
            ghi_val = float(ghi[i] if ghi[i] is not None else 0.0)
            gti_val = float(gti[i] if gti[i] is not None else 0.0)
            pdc_val = float(pdc[i] if pdc[i] is not None else 0.0)
            pac_val = float(pac[i] if pac[i] is not None else 0.0)
            I_val = float(I[i] if I[i] is not None else 0.0)
            V_val = float(V[i] if V[i] is not None else 0.0)
            temp_val = float(temperature[i] if temperature[i] is not None else 25.0)
            wind_val = float(wind_speed[i] if wind_speed[i] is not None else 1.0)
            ghi_list.append(round(ghi_val, 2))
            gti_list.append(round(gti_val, 2))
            pac_list.append(round(pac_val, 2))
            I_list.append(round(I_val, 2))
            V_list.append(round(V_val, 2))
            forecast_results.append({
                'timestamp': timestamp_str,
                'ghi': round(ghi_val, 2),
                'gti': round(gti_val, 2),
                'pdc': round(pdc_val, 2),
                'pac': round(pac_val, 2),
                'I': round(I_val, 2),
                'V': round(V_val, 2),
                'temperature': round(temp_val, 2),
                'wind_speed': round(wind_val, 2)
            })

        app.logger.info(f"Forecast timestamps: {timestamps_list}")
        app.logger.info(f"Length of timestamps_list: {len(timestamps_list)}")
        app.logger.info(f"Length of ghi_list: {len(ghi_list)}")
        app.logger.info(f"Length of gti_list: {len(gti_list)}")
        app.logger.info(f"Length of pac_list: {len(pac_list)}")
        app.logger.info(f"Length of I_list: {len(I_list)}")
        app.logger.info(f"Length of V_list: {len(V_list)}")

        app.logger.info("Rendering template with results.")
        return render_template('index.html',
                               current_results=current_results,
                               forecast_results=forecast_results,
                               current_timestamp=current_timestamp_str,
                               current_ghi=current_ghi,
                               current_gti=current_gti,
                               current_pac=current_pac,
                               current_I=current_I,
                               current_V=current_V,
                               timestamps=timestamps_list,
                               ghi=ghi_list,
                               gti=gti_list,
                               pac=pac_list,
                               I=I_list,
                               V=V_list,
                               iec_metrics={k: round(v, 3) for k, v in iec_metrics.items()},
                               pv_types=list(PV_MODULES.keys()),
                               model_types=["simple", "sapm", "single_diode", "cec"],
                               location_options=location_options,
                               location_options_with_labels=location_options_with_labels,
                               orientation_options=[(0, "South (0°)"), (-90, "East (-90°)"), (90, "West (90°)"), (180, "North (180°)")],
                               selected_lat=lat,
                               selected_lon=lon,
                               selected_pv_type=pv_type,
                               selected_area=area,
                               selected_tilt=tilt,
                               selected_orientation=orientation,
                               selected_model_type=model_type,
                               forecast_message=forecast_message)

    app.logger.info("Rendering initial template for GET request.")
    return render_template('index.html',
                           pv_types=list(PV_MODULES.keys()),
                           model_types=["simple", "sapm", "single_diode", "cec"],
                           location_options=location_options,
                           location_options_with_labels=location_options_with_labels,
                           orientation_options=[(0, "South (0°)"), (-90, "East (-90°)"), (90, "West (90°)"), (180, "North (180°)")])


@app.route('/annual_metrics', methods=['GET'])
def annual_metrics():
    """Calculate annual IEC metrics by querying daily_iec_metrics.db."""
    try:
        # Require mandatory parameters
        if not all(key in request.args for key in ['lat', 'lon', 'pv_type', 'area']):
            return jsonify({"error": "Missing required parameters: lat, lon, pv_type, area"}), 400

        lat = float(request.args['lat'])
        lon = float(request.args['lon'])
        pv_type = request.args['pv_type']
        area = float(request.args['area'])

        # Validate inputs
        if pv_type not in PV_MODULES:
            return jsonify({"error": f"Invalid PV type: {pv_type}"}), 400
        if area <= 0:
            return jsonify({"error": "Area must be positive"}), 400
        if not (37.4 <= lat <= 37.7 and 126.8 <= lon <= 127.2):
            return jsonify({"error": "Coordinates out of Seoul bounds (Lat: 37.4-37.7, Lon: 126.8-127.2)"}), 400

        # Optional date range (default to current year)
        start_date = request.args.get('start_date', '2025-01-01')
        end_date = request.args.get('end_date', '2025-12-31')

        # Connect to database and query aggregated metrics
        conn = sqlite3.connect(daily_metrics_db_path)
        query = """
                SELECT SUM(pdc_kwh), SUM(pac_kwh), SUM(ghi_kwh_per_m2), SUM(gti_kwh_per_m2)
                FROM daily_iec_metrics
                WHERE lat = ? \
                  AND lon = ? \
                  AND pv_type = ? \
                  AND date BETWEEN ? \
                  AND ? \
                """
        df = pd.read_sql_query(query, conn, params=(lat, lon, pv_type, start_date, end_date))
        conn.close()

        # Check for valid data
        if df.empty or df.iloc[0].isnull().all():
            return jsonify({"error": "No data available for the specified period"}), 404

        # Extract summed values
        pdc_kwh, pac_kwh, ghi_kwh_per_m2, gti_kwh_per_m2 = df.iloc[0]

        # Calculate rated power
        P_rated = (area / PV_MODULES[pv_type]['area_ref']) * PV_MODULES[pv_type]['Pmp_ref'] / 1000  # kWp

        # Set annual duration
        T = 8760  # Hours in a year

        # Calculate IEC metrics
        Yr = gti_kwh_per_m2  # kWh/m²
        Yf = pac_kwh / P_rated if P_rated > 0 else 0  # kWh/kWp
        PR = min(Yf / Yr, 1) if Yr > 0 else 0
        CF = pac_kwh / (P_rated * T) if P_rated > 0 else 0

        # Log success
        app.logger.info(f"Calculated annual metrics for lat={lat}, lon={lon}, pv_type={pv_type}, area={area}")

        return jsonify({
            "Yf": round(Yf, 3),
            "PR": round(PR, 3),
            "CF": round(CF, 3)
        })
    except ValueError as e:
        app.logger.error(f"Invalid input parameters: {str(e)}")
        return jsonify({"error": f"Invalid input: {str(e)}"}), 400
    except Exception as e:
        app.logger.error(f"Error calculating annual metrics: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
