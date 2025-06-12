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
        logging.FileHandler(os.path.join(os.path.dirname(__file__), 'data', 'fetch_historical_weather_data.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# KMA API setup (historical data)
kma_auth_key = os.getenv("KMA_AUTH_KEY", "LoBU7l_-QDuAVO5f_iA7ZQ")
kma_historical_base_url = "https://apihub.kma.go.kr/api/typ01/cgi-bin/url/nph-sfc_obs_nc_pt_api"

# Database setup
db_path = os.path.join(os.path.dirname(__file__), 'data', 'weather_data.db')


conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Create table for historical data
try:
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS weather_data (
            timestamp TEXT,
            nx INTEGER,
            ny INTEGER,
            lat REAL,
            lon REAL,
            temperature REAL,
            wind_speed REAL,
            data_type TEXT,
            UNIQUE(timestamp, nx, ny, data_type)
        )
    ''')
    conn.commit()
    logger.info("Ensured weather_data table exists.")
except Exception as e:
    logger.error(f"Error setting up database: {str(e)}")
    raise

# Open-Meteo request tracking
open_meteo_requests = 0
OPEN_METEO_DAILY_LIMIT = 9500  # Cap at 9500 to leave a buffer (limit is 10,000)

def reset_request_counter():
    """Reset the Open-Meteo request counter daily."""
    global open_meteo_requests
    open_meteo_requests = 0
    logger.info("Reset Open-Meteo request counter for the day.")

# Function to get the current row count
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

# Historical Data Fetching Functions
def fetch_point_data(lat, lon, start_time, end_time, max_retries=3):
    """Fetch historical temperature data for a specific lat/lon point over a time range."""
    tm1 = start_time.strftime("%Y%m%d%H%M")
    tm2 = end_time.strftime("%Y%m%d%H%M")
    params = {
        'obs': 'ta',
        'tm1': tm1,
        'tm2': tm2,
        'itv': '10',
        'lon': lon,
        'lat': lat,
        'authKey': kma_auth_key
    }
    for attempt in range(max_retries):
        try:
            response = requests.get(kma_historical_base_url, params=params, timeout=30, verify=False)
            if response.status_code == 200:
                logger.info(f"Fetched historical data for lat={lat}, lon={lon}, {tm1} to {tm2}")
                logger.debug(f"Response content: {response.text}")
                return response.text
            elif response.status_code == 404:
                logger.warning(f"Attempt {attempt + 1}/{max_retries} - Data not available for lat={lat}, lon={lon}, {tm1} to {tm2} (404)")
                return None
            else:
                logger.warning(f"Attempt {attempt + 1}/{max_retries} - Failed to fetch data for lat={lat}, lon={lon}, {tm1} to {tm2}, Status: {response.status_code}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Attempt {attempt + 1}/{max_retries} - Error fetching data for lat={lat}, lon={lon}, {tm1} to {tm2}: {str(e)}")
        if attempt < max_retries - 1:
            time.sleep(5 * (2 ** attempt))
    logger.error(f"All {max_retries} retries failed for lat={lat}, lon={lon}, {tm1} to {tm2}")
    return None

def parse_point_data(response_text):
    """Parse the point-based API response to extract temperature values at 10-minute intervals."""
    if not response_text:
        return []
    data_points = []
    try:
        lines = response_text.strip().splitlines()
        for line in lines[1:]:
            if not line.strip():
                continue
            parts = line.split(',')
            if len(parts) < 5:
                continue
            timestamp_str = parts[0].strip()
            temp_str = parts[4].strip()
            if temp_str == '-999' or not temp_str:
                continue
            temperature = float(temp_str)
            if not (-20 <= temperature <= 50):
                logger.warning(f"Invalid temperature value {temperature} at timestamp {timestamp_str}")
                continue
            timestamp = pd.to_datetime(timestamp_str, format='%Y%m%d%H%M').tz_localize(KST)
            data_points.append({
                'datetime': timestamp,
                'temperature': temperature
            })
        return data_points
    except Exception as e:
        logger.error(f"Error parsing point data: {str(e)}")
        return []

def fetch_nearest_data(lat, lon, start_time, end_time, all_coords, max_retries=3):
    """Fetch data for the nearest coordinate if the original location has missing data."""
    coords = all_coords[['lat', 'lon']].values
    tree = cKDTree(coords)
    _, idx = tree.query([lat, lon])
    nearest_lat, nearest_lon = coords[idx]
    if nearest_lat == lat and nearest_lon == lon:
        _, indices = tree.query([lat, lon], k=2)
        if len(indices) > 1:
            nearest_lat, nearest_lon = coords[indices[1]]
        else:
            return None
    logger.info(f"Fetching historical data for nearest coordinate (lat={nearest_lat}, lon={nearest_lon}) for time range {start_time} to {end_time}")
    response_text = fetch_point_data(nearest_lat, nearest_lon, start_time, end_time, max_retries)
    if response_text:
        return parse_point_data(response_text)
    return None

def interpolate_to_30min(data_points, start_time, end_time):
    """Interpolate 10-minute interval data to 30-minute intervals."""
    if not data_points:
        return []
    df = pd.DataFrame(data_points)
    df.set_index('datetime', inplace=True)
    if df.index.tz is None:
        df.index = df.index.tz_localize(KST)
    else:
        df.index = df.index.tz_convert(KST)
    time_range = pd.date_range(start=start_time, end=end_time, freq='30min', tz=KST)
    df_30min = df.resample('30min').interpolate(method='linear')
    df_30min = df_30min.reindex(time_range, method='nearest').interpolate(method='linear')
    interpolated_data = []
    for dt, row in df_30min.iterrows():
        interpolated_data.append({
            'datetime': dt.strftime('%Y-%m-%d %H:%M:%S'),
            'temperature': row['temperature']
        })
    return interpolated_data

def interpolate_historical(lat, lon, start_time, end_time, conn, time_window_hours=4):
    """Interpolate missing historical data using stored data for the same location."""
    logger.info(f"Attempting historical interpolation for lat={lat}, lon={lon}, from {start_time} to {end_time}")
    query = """
        SELECT timestamp, temperature
        FROM weather_data
        WHERE lat = ? AND lon = ? AND data_type = 'historical'
        AND timestamp BETWEEN ? AND ?
        ORDER BY timestamp
    """
    try:
        df = pd.read_sql_query(
            query, conn,
            params=(lat, lon,
                    (start_time - timedelta(hours=time_window_hours)).strftime("%Y-%m-%d %H:%M:%S"),
                    (end_time + timedelta(hours=time_window_hours)).strftime("%Y-%m-%d %H:%M:%S"))
        )
        if len(df) < 2:
            logger.warning(f"Insufficient data for historical interpolation at lat={lat}, lon={lon}, from {start_time} to {end_time}")
            return None
        df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(KST)
        df.set_index('timestamp', inplace=True)
        time_range = pd.date_range(start=start_time, end=end_time, freq='30min', tz=KST)
        df_30min = df.reindex(time_range, method='nearest').interpolate(method='linear')
        interpolated_data = []
        for dt, row in df_30min.iterrows():
            temperature = row['temperature']
            if not (-20 <= temperature <= 50):
                logger.warning(f"Invalid interpolated temperature value {temperature} at lat={lat}, lon={lon}, timestamp={dt}")
                continue
            interpolated_data.append({
                'datetime': dt.strftime('%Y-%m-%d %H:%M:%S'),
                'temperature': temperature
            })
        logger.info(f"Successfully interpolated historical data for lat={lat}, lon={lon}")
        return interpolated_data
    except Exception as e:
        logger.error(f"Error during historical interpolation: {str(e)}")
        return None

# Open-Meteo Fallback for Historical Data
def fetch_open_meteo_data(locations, start_time, end_time, include_wind_speed=False):
    """Fetch historical data from Open-Meteo for multiple locations."""
    global open_meteo_requests
    all_data_points = []
    for lat, lon in locations:
        if open_meteo_requests >= OPEN_METEO_DAILY_LIMIT:
            logger.warning(f"Approaching Open-Meteo daily limit ({open_meteo_requests}/{OPEN_METEO_DAILY_LIMIT}). Skipping request for lat={lat}, lon={lon}.")
            all_data_points.append((lat, lon, None))
            continue
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=temperature_2m" + (",wind_speed_10m" if include_wind_speed else "")
        url += f"&start_date={start_time.strftime('%Y-%m-%d')}&end_date={end_time.strftime('%Y-%m-%d')}"
        try:
            response = requests.get(url, timeout=30)
            open_meteo_requests += 1
            if response.status_code == 200:
                data = response.json()
                timestamps = pd.to_datetime(data['hourly']['time']).tz_localize(KST)
                temperatures = data['hourly']['temperature_2m']
                winds = data['hourly'].get('wind_speed_10m', [None] * len(timestamps)) if include_wind_speed else [None] * len(timestamps)
                data_points = []
                for ts, temp, wind in zip(timestamps, temperatures, winds):
                    if start_time <= ts <= end_time:
                        if not (-20 <= temp <= 50):
                            logger.warning(f"Invalid temperature value {temp} at lat={lat}, lon={lon}, timestamp={ts}")
                            continue
                        data_point = {'forecast_time': ts, 'temperature': temp}
                        if include_wind_speed:
                            wind_ms = wind / 3.6  # Convert km/h to m/s
                            if not (0 <= wind_ms <= 50):
                                logger.warning(f"Invalid wind speed value {wind_ms} at lat={lat}, lon={lon}, timestamp={ts}")
                                continue
                            data_point['wind_speed'] = wind_ms
                        data_points.append(data_point)
                all_data_points.append((lat, lon, data_points))
                logger.info(f"Fetched Open-Meteo data for lat={lat}, lon={lon} (Request count: {open_meteo_requests})")
            else:
                logger.error(f"Open-Meteo request failed for lat={lat}, lon={lon}: Status {response.status_code}")
                all_data_points.append((lat, lon, None))
        except Exception as e:
            logger.error(f"Open-Meteo error for lat={lat}, lon={lon}: {str(e)}")
            all_data_points.append((lat, lon, None))
        time.sleep(0.1)  # Small delay to avoid rate limiting
    return all_data_points

# Fetch Function for Historical Data
def fetch_historical_data(now):
    total_locations = len(locations_df)
    historical_batch_size = 10

    # Align end_time_hist to the nearest past 30-minute mark
    minutes = (now.minute // 30) * 30
    end_time_hist = now.replace(minute=minutes, second=0, microsecond=0)
    start_time_hist = end_time_hist - timedelta(hours=4)
    historical_timestamps = pd.date_range(start=start_time_hist, end=end_time_hist, freq='30min', tz=KST)
    logger.info(f"Expected historical timestamps: {historical_timestamps}")

    historical_rows = []
    skipped_timestamps = 0

    logger.info(f"Fetching historical data from {start_time_hist} to {end_time_hist}")
    logger.info(f"Historical row count before fetch: {get_row_count('weather_data', 'historical')}")

    for batch_start in range(0, total_locations, historical_batch_size):
        batch_end = min(batch_start + historical_batch_size, total_locations)
        batch_df = locations_df.iloc[batch_start:batch_end]
        logger.info(f"Processing historical batch {batch_start // historical_batch_size + 1}/{(total_locations - 1) // historical_batch_size + 1} (points {batch_start} to {batch_end - 1})")
        for idx, row in batch_df.iterrows():
            lat, lon, nx, ny = row['lat'], row['lon'], row['nx'], row['ny']
            # Fetch from KMA API
            response_text = fetch_point_data(lat, lon, start_time_hist, end_time_hist)
            data_points = parse_point_data(response_text) if response_text else None
            if not data_points:
                # Retry mechanism with buffer
                logger.info(f"Retrying historical fetch for lat={lat}, lon={lon} after 20-sec buffer")
                time.sleep(20)
                response_text = fetch_point_data(lat, lon, start_time_hist, end_time_hist)
                data_points = parse_point_data(response_text) if response_text else None
            if not data_points:
                data_points = fetch_nearest_data(lat, lon, start_time_hist, end_time_hist, locations_df)
            if not data_points:
                interpolated_data = interpolate_historical(lat, lon, start_time_hist, end_time_hist, conn)
                if interpolated_data:
                    for data in interpolated_data:
                        historical_rows.append((
                            data['datetime'],
                            nx, ny, lat, lon,
                            data['temperature'],
                            None,
                            "historical"
                        ))
                    continue
            if not data_points:
                om_data = fetch_open_meteo_data([(lat, lon)], start_time_hist, end_time_hist, include_wind_speed=False)
                if om_data and om_data[0][2]:
                    data_points = [{'datetime': dp['forecast_time'], 'temperature': dp['temperature']} for dp in om_data[0][2]]
                    logger.info(f"Fetched Open-Meteo data points for lat={lat}, lon={lon}: {len(data_points)} timesteps")
                else:
                    logger.warning(f"No Open-Meteo data available for lat={lat}, lon={lon}")
            if not data_points:
                logger.error(f"No historical data available for lat={lat}, lon={lon} from {start_time_hist} to {end_time_hist}")
                skipped_timestamps += len(historical_timestamps)
                continue
            interpolated_data = interpolate_to_30min(data_points, start_time_hist, end_time_hist)
            logger.info(f"Interpolated data for lat={lat}, lon={lon}: {len(interpolated_data)} timesteps")
            for data in interpolated_data:
                historical_rows.append((
                    data['datetime'],
                    nx, ny, lat, lon,
                    data['temperature'],
                    None,
                    "historical"
                ))
        time.sleep(5)

    if historical_rows:
        try:
            cursor.executemany(
                "INSERT OR REPLACE INTO weather_data (timestamp, nx, ny, lat, lon, temperature, wind_speed, data_type) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                historical_rows)
            conn.commit()
            logger.info(f"Inserted {len(historical_rows)} historical weather records")
            logger.info(f"Number of skipped timestamps: {skipped_timestamps}")
            logger.info(f"Historical row count after insert: {get_row_count('weather_data', 'historical')}")
        except Exception as e:
            logger.error(f"Error inserting historical weather data: {str(e)}")

    for _, row in locations_df.iterrows():
        nx, ny = row["nx"], row["ny"]
        cutoff = (now - timedelta(hours=24)).strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute("""
            DELETE FROM weather_data
            WHERE nx = ? AND ny = ? AND timestamp < ? AND data_type = 'historical'
        """, (nx, ny, cutoff))
    conn.commit()
    logger.info(f"Cleaned up historical data, keeping data after {cutoff}")

# Main Fetching Function
def fetch_and_store_historical_weather_data():
    total_locations = len(locations_df)
    last_reset_date = datetime.now(KST).date()

    # Initial fetch for the most recent timestamp
    now = datetime.now(KST).replace(second=0, microsecond=0)
    current_hour = now.hour
    current_minute = now.minute

    # Determine the most recent historical fetch time (:05 or :35)
    if current_minute >= 35:
        last_historical_fetch = now.replace(minute=35, second=0, microsecond=0)
    else:
        last_historical_fetch = now.replace(minute=5, second=0, microsecond=0)
    if current_minute < 5:
        last_historical_fetch -= timedelta(minutes=30)

    # Check if within operating hours (06:00–22:00 KST)
    should_fetch_initial = 6 <= current_hour < 22

    if should_fetch_initial:
        logger.info(f"Performing initial fetch at {now}")
        logger.info(f"Initial historical fetch for timestamp: {last_historical_fetch}")
        fetch_historical_data(last_historical_fetch)
    else:
        logger.info(f"Current time {now} is outside operating hours (06:00–22:00 KST). Skipping initial fetch.")

    # Regular scheduling loop
    while True:
        now = datetime.now(KST).replace(second=0, microsecond=0)
        current_hour = now.hour
        current_minute = now.minute

        # Reset Open-Meteo request counter daily
        current_date = now.date()
        if current_date != last_reset_date:
            reset_request_counter()
            last_reset_date = current_date

        # Operating hours: 06:00–22:00 KST
        if 6 <= current_hour < 22:
            should_fetch_historical = current_minute == 5 or current_minute == 35
            if should_fetch_historical:
                fetch_historical_data(now)

        # Determine the next fetch time
        next_fetch = now.replace(second=0, microsecond=0)
        if current_minute < 5:
            next_fetch = next_fetch.replace(minute=5)
        elif current_minute < 35:
            next_fetch = next_fetch.replace(minute=35)
        else:
            next_fetch = next_fetch.replace(minute=5) + timedelta(hours=1)

        # Skip nighttime (22:00–06:00 KST)
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
        fetch_and_store_historical_weather_data()
    except KeyboardInterrupt:
        logger.info("Stopped by user.")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
    finally:
        conn.close()