import requests
import pandas as pd
import sqlite3
from datetime import datetime, timedelta, timezone
import time
import logging
import urllib3
import numpy as np
from scipy.spatial import cKDTree
import os

# Suppress InsecureRequestWarning
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(os.path.dirname(__file__), 'data', 'forecasted_temp.log')),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Database setup
db_path = os.path.join(os.path.dirname(__file__), 'data', 'weather_data.db')
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Drop and recreate the table with a corrected UNIQUE constraint
cursor.execute("DROP TABLE IF EXISTS forecasted_weather_data")
try:
    cursor.execute('''
        CREATE TABLE forecasted_weather_data (
            timestamp TEXT,
            nx INTEGER,
            ny INTEGER,
            lat REAL,
            lon REAL,
            temperature REAL,
            wind_speed REAL,
            forecast_time TEXT,
            data_type TEXT,
            UNIQUE(nx, ny, forecast_time, data_type)
        )
    ''')
    conn.commit()
    logger.info("Recreated forecasted_weather_data table with corrected UNIQUE constraint.")
except Exception as e:
    logger.error(f"Error setting up database: {str(e)}")
    raise

# Open-Meteo request tracking
open_meteo_requests = 0
OPEN_METEO_DAILY_LIMIT = 9500

def reset_request_counter():
    """Reset the Open-Meteo request counter daily."""
    global open_meteo_requests
    open_meteo_requests = 0
    logger.info("Reset Open-Meteo request counter for the day.")

def get_row_count(table_name, data_type):
    cursor.execute(f"SELECT COUNT(*) FROM {table_name} WHERE data_type = ?", (data_type,))
    return cursor.fetchone()[0]

# Generate 20x20 grid for Seoul (lat: 37.4–37.7, lon: 126.8–127.2)
coords = [(lat, lon) for lat in np.linspace(37.4, 37.7, 20) for lon in np.linspace(126.8, 127.2, 20)]
locations_df = pd.DataFrame(coords, columns=['lat', 'lon'])
locations_df['nx'] = locations_df.index // 20
locations_df['ny'] = locations_df.index % 20

# KST timezone
KST = timezone(timedelta(hours=9))

def fetch_open_meteo_data(locations, start_time, end_time, include_wind_speed=True):
    """Fetch forecasted temperature and wind speed from Open-Meteo."""
    global open_meteo_requests
    all_data_points = []
    for lat, lon in locations:
        if open_meteo_requests >= OPEN_METEO_DAILY_LIMIT:
            logger.warning(f"Approaching Open-Meteo daily limit ({open_meteo_requests}/{OPEN_METEO_DAILY_LIMIT}). Skipping request for lat={lat}, lon={lon}.")
            all_data_points.append((lat, lon, None))
            continue
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=temperature_2m,wind_speed_10m"
        url += f"&start_date={start_time.strftime('%Y-%m-%d')}&end_date={end_time.strftime('%Y-%m-%d')}"
        try:
            response = requests.get(url, timeout=30)
            open_meteo_requests += 1
            if response.status_code == 200:
                data = response.json()
                timestamps = pd.to_datetime(data['hourly']['time']).tz_localize(KST)
                temperatures = data['hourly']['temperature_2m']
                winds = data['hourly']['wind_speed_10m']
                data_points = []
                for ts, temp, wind in zip(timestamps, temperatures, winds):
                    if start_time <= ts <= end_time:
                        if not (-20 <= temp <= 50):
                            logger.warning(f"Invalid temperature value {temp} at lat={lat}, lon={lon}, timestamp={ts}")
                            continue
                        wind_ms = wind / 3.6
                        if not (0 <= wind_ms <= 50):
                            logger.warning(f"Invalid wind speed value {wind_ms} at lat={lat}, lon={lon}, timestamp={ts}")
                            continue
                        data_point = {'forecast_time': ts, 'temperature': temp, 'wind_speed': wind_ms}
                        data_points.append(data_point)
                all_data_points.append((lat, lon, data_points))
                logger.info(f"Fetched Open-Meteo data for lat={lat}, lon={lon} (Request count: {open_meteo_requests})")
            else:
                logger.error(f"Open-Meteo request failed for lat={lat}, lon={lon}: Status {response.status_code}")
                all_data_points.append((lat, lon, None))
        except Exception as e:
            logger.error(f"Open-Meteo error for lat={lat}, lon={lon}: {str(e)}")
            all_data_points.append((lat, lon, None))
        time.sleep(0.1)
    return all_data_points

def interpolate_to_30min(data_points, start_time, end_time):
    """Interpolate hourly forecast data to 30-minute intervals."""
    if not data_points:
        return []
    df = pd.DataFrame(data_points)
    df.set_index('forecast_time', inplace=True)
    if df.index.tz is None:
        df.index = df.index.tz_localize(KST)
    else:
        df.index = df.index.tz_convert(KST)
    time_range = pd.date_range(start=start_time, end=end_time, freq='30min', tz=KST)
    df_30min = df.reindex(time_range, method='nearest').interpolate(method='linear')
    interpolated_data = []
    for dt, row in df_30min.iterrows():
        interpolated_data.append({
            'forecast_time': dt,
            'temperature': row['temperature'],
            'wind_speed': row['wind_speed']
        })
    return interpolated_data

def interpolate_forecast(lat, lon, timestamp, conn, time_window_hours=4):
    """Interpolate missing forecast data using historical data."""
    logger.info(f"Attempting interpolation for lat={lat}, lon={lon}, timestamp={timestamp}")
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=KST)
    else:
        timestamp = timestamp.astimezone(KST)
    start_time = timestamp - timedelta(hours=time_window_hours)
    end_time = timestamp + timedelta(hours=time_window_hours)
    query = """
        SELECT timestamp, temperature, wind_speed
        FROM weather_data
        WHERE lat = ? AND lon = ? AND data_type = 'historical'
        AND timestamp BETWEEN ? AND ?
        ORDER BY timestamp
    """
    try:
        df = pd.read_sql_query(
            query, conn, params=(lat, lon, start_time.strftime("%Y-%m-%d %H:%M:%S"), end_time.strftime("%Y-%m-%d %H:%M:%S"))
        )
        if len(df) < 2:
            logger.warning(f"Insufficient data for interpolation at lat={lat}, lon={lon}, timestamp={timestamp}")
            return None
        df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(KST)
        df.set_index('timestamp', inplace=True)
        target_df = pd.DataFrame(index=[pd.Timestamp(timestamp)])
        combined_df = pd.concat([df, target_df]).sort_index()
        combined_df['temperature'] = combined_df['temperature'].interpolate(method='linear', limit_direction='both')
        combined_df['wind_speed'] = combined_df['wind_speed'].interpolate(method='linear', limit_direction='both')
        result = combined_df.loc[timestamp, ['temperature', 'wind_speed']] if timestamp in combined_df.index else None
        if result is not None and -20 <= result['temperature'] <= 50 and (result['wind_speed'] is None or 0 <= result['wind_speed'] <= 50):
            logger.info(f"Interpolated: temperature={result['temperature']:.2f}°C, wind_speed={result['wind_speed']}m/s for lat={lat}, lon={lon}, timestamp={timestamp}")
            return result.to_dict()
        else:
            logger.warning(f"Invalid interpolated values: {result} at lat={lat}, lon={lon}, timestamp={timestamp}")
            return None
    except Exception as e:
        logger.error(f"Error during interpolation: {str(e)}")
        return None

def fetch_forecasted_data(now):
    total_locations = len(locations_df)
    forecast_batch_size = 25
    start_time_forecast = now.replace(minute=0, second=0, microsecond=0)
    end_time_forecast = start_time_forecast + timedelta(hours=6)
    forecast_timestamps = pd.date_range(start=start_time_forecast, end=end_time_forecast, freq='30min', tz=KST)
    forecast_rows = []
    skipped_timestamps = 0
    logger.info(f"Fetching forecasted data from {start_time_forecast} to {end_time_forecast}")
    logger.info(f"Forecast row count before fetch: {get_row_count('forecasted_weather_data', 'forecasted')}")
    coords_grid = locations_df[['lat', 'lon']].drop_duplicates().values
    tree = cKDTree(coords_grid)
    for batch_start in range(0, total_locations, forecast_batch_size):
        batch_end = min(batch_start + forecast_batch_size, total_locations)
        batch_df = locations_df.iloc[batch_start:batch_end]
        logger.info(f"Processing forecast batch {batch_start // forecast_batch_size + 1}/{(total_locations - 1) // forecast_batch_size + 1} (points {batch_start} to {batch_end - 1})")
        locations_to_fetch = [(row['lat'], row['lon']) for _, row in batch_df.iterrows()]
        om_data = fetch_open_meteo_data(locations_to_fetch, start_time_forecast, end_time_forecast, include_wind_speed=True)
        for (lat, lon, data_points) in om_data:
            _, idx = tree.query([lat, lon])
            mapped_lat, mapped_lon = coords_grid[idx]
            if mapped_lat != lat or mapped_lon != lon:
                logger.info(f"Mapped coordinates ({lat}, {lon}) to grid point ({mapped_lat}, {mapped_lon})")
            lat, lon = mapped_lat, mapped_lon
            nx = locations_df[(locations_df['lat'] == lat) & (locations_df['lon'] == lon)]['nx'].iloc[0]
            ny = locations_df[(locations_df['lat'] == lat) & (locations_df['lon'] == lon)]['ny'].iloc[0]
            if data_points:
                interpolated_data = interpolate_to_30min(data_points, start_time_forecast, end_time_forecast)
                for dp in interpolated_data:
                    forecast_rows.append((
                        now.strftime('%Y-%m-%d %H:%M:%S'),
                        nx, ny, lat, lon,
                        dp['temperature'],
                        dp['wind_speed'],
                        dp['forecast_time'].strftime('%Y-%m-%d %H:%M:%S'),
                        'forecasted'
                    ))
            else:
                for ts in forecast_timestamps:
                    interpolated = interpolate_forecast(lat, lon, ts, conn)
                    if interpolated:
                        forecast_rows.append((
                            now.strftime('%Y-%m-%d %H:%M:%S'),
                            nx, ny, lat, lon,
                            interpolated['temperature'],
                            interpolated['wind_speed'],
                            ts.strftime('%Y-%m-%d %H:%M:%S'),
                            'interpolated'
                        ))
                    else:
                        logger.error(f"No forecast data for lat={lat}, lon={lon}, forecast_time={ts}")
                        skipped_timestamps += 1
        time.sleep(5)

    if forecast_rows:
        try:
            cursor.executemany(
                "INSERT OR REPLACE INTO forecasted_weather_data (timestamp, nx, ny, lat, lon, temperature, wind_speed, forecast_time, data_type) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                forecast_rows)
            conn.commit()
            logger.info(f"Inserted {len(forecast_rows)} forecasted weather records")
            logger.info(f"Number of skipped timestamps: {skipped_timestamps}")
            logger.info(f"Forecast row count after insert: {get_row_count('forecasted_weather_data', 'forecasted')}")
        except Exception as e:
            logger.error(f"Error inserting forecasted weather data: {str(e)}")

    for _, row in locations_df.iterrows():
        nx, ny = row["nx"], row["ny"]
        cutoff = (now - timedelta(hours=6)).strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute("""
            DELETE FROM forecasted_weather_data
            WHERE nx = ? AND ny = ? AND timestamp < ?
        """, (nx, ny, cutoff))
    conn.commit()
    logger.info(f"Cleaned up forecasted data, keeping data after {cutoff}")

def fetch_and_store_forecasted_temp():
    total_locations = len(locations_df)
    last_reset_date = datetime.now(KST).date()
    now = datetime.now(KST).replace(second=0, microsecond=0)
    current_hour = now.hour
    current_minute = now.minute
    last_forecast_fetch = now.replace(minute=5, second=0, microsecond=0)
    if current_minute < 5:
        last_forecast_fetch -= timedelta(hours=1)
    should_fetch_initial = 6 <= current_hour < 22
    if should_fetch_initial:
        logger.info(f"Performing initial fetch at {now}")
        logger.info(f"Initial forecast fetch for timestamp: {last_forecast_fetch}")
        fetch_forecasted_data(last_forecast_fetch)
    else:
        logger.info(f"Current time {now} is outside operating hours (06:00–22:00 KST). Skipping initial fetch.")
    while True:
        now = datetime.now(KST).replace(second=0, microsecond=0)
        current_hour = now.hour
        current_minute = now.minute
        current_date = now.date()
        if current_date != last_reset_date:
            reset_request_counter()
            last_reset_date = current_date
        if 6 <= current_hour < 22:
            should_fetch_forecast = current_minute == 5
            if should_fetch_forecast:
                fetch_forecasted_data(now)
        next_fetch = now.replace(second=0, microsecond=0)
        if current_minute < 5:
            next_fetch = next_fetch.replace(minute=5)
        else:
            next_fetch = next_fetch.replace(minute=5) + timedelta(hours=1)
        if next_fetch.hour >= 22:
            next_fetch = next_fetch.replace(hour=6, minute=5) + timedelta(days=1)
        elif next_fetch.hour < 6:
            next_fetch = next_fetch.replace(hour=6, minute=5)
        sleep_seconds = (next_fetch - now).total_seconds()
        if sleep_seconds > 0:
            logger.info(f"Sleeping for {sleep_seconds / 60:.2f} minutes until next fetch at {next_fetch}")
            time.sleep(sleep_seconds)
        else:
            logger.info("No sleep needed, proceeding to next iteration immediately.")

if __name__ == "__main__":
    try:
        fetch_and_store_forecasted_temp()
    except KeyboardInterrupt:
        logger.info("Stopped by user.")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
    finally:
        conn.close()