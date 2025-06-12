import requests
import os
import netCDF4 as nc
import numpy as np
from datetime import datetime, timedelta
import pandas as pd
import sqlite3
import pyproj
import time
import logging
from pyproj import Transformer


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(os.path.dirname(__file__), 'data', 'fetch_ghi_data.log')),
        logging.StreamHandler()
    ]
)

# Set up parameters
auth_key = os.getenv("KMA_AUTH_KEY", "LoBU7l_-QDuAVO5f_iA7ZQ")  # Fallback to hard-coded if env var not set
base_url = "https://apihub.kma.go.kr/api/typ05/api/GK2A/LE2/SWRAD/KO/data"
output_dir = os.path.join(os.path.dirname(__file__), 'data', 'nc_files')
db_path = os.path.join(os.path.dirname(__file__), 'data', 'ghi_data.db')

# Create directories
os.makedirs(output_dir, exist_ok=True)

# Projection and Seoul region config
proj = "+proj=lcc +lat_0=38 +lon_0=126 +lat_1=30 +lat_2=60 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
extent = (-899000, 899000, -899000, 899000)
seoul_bbox = {"min_lon": 126.8, "max_lon": 127.2, "min_lat": 37.4, "max_lat": 37.7}

# Define the 20x20 grid directly from locations_df (consistent with other scripts)
coords = [(lat, lon) for lat in np.linspace(37.4, 37.7, 20) for lon in np.linspace(126.8, 127.2, 20)]
locations_df = pd.DataFrame(coords, columns=['lat', 'lon'])
locations_df['nx'] = locations_df.index // 20
locations_df['ny'] = locations_df.index % 20

# Transformers
wgs84 = pyproj.CRS("EPSG:4326")
lcc = pyproj.CRS(proj)
transformer_to_lcc = Transformer.from_crs(wgs84, lcc, always_xy=True)
transformer_to_wgs84 = Transformer.from_crs(lcc, wgs84, always_xy=True)

# Convert Seoul bbox to LCC
min_x, min_y = transformer_to_lcc.transform(seoul_bbox["min_lon"], seoul_bbox["max_lat"])
max_x, max_y = transformer_to_lcc.transform(seoul_bbox["max_lon"], seoul_bbox["min_lat"])

# Grid config
total_width = total_height = 900
pixel_size_x = (extent[1] - extent[0]) / total_width
pixel_size_y = (extent[3] - extent[2]) / total_height

# Align bounding box to ensure exact 20x20 grid
min_x = np.floor(min_x / pixel_size_x) * pixel_size_x
max_x = min_x + 20 * pixel_size_x
min_y = np.ceil(max_y / pixel_size_y) * pixel_size_y - 20 * pixel_size_y
max_y = min_y + 20 * pixel_size_y

# Pixel indices
col_start = int((min_x - extent[0]) / pixel_size_x)
col_end = int((max_x - extent[0]) / pixel_size_x)
row_start = int((extent[3] - max_y) / pixel_size_y)
row_end = int((extent[3] - min_y) / pixel_size_y)

logger.info(f"Seoul's LCC bounding box: min_x={min_x:.2f}, max_x={max_x:.2f}, min_y={min_y:.2f}, max_y={max_y:.2f}")
logger.info(f"Seoul pixel indices: row_start={row_start}, row_end={row_end}, col_start={col_start}, col_end={col_end}")
logger.info(f"Seoul grid size: {(row_end - row_start)} x {(col_end - col_start)}")

# Verify grid alignment
if (row_end - row_start) != 20 or (col_end - col_start) != 20:
    logger.error(f"Grid size mismatch: expected 20x20, got {(row_end - row_start)}x{(col_end - col_start)}")
    raise ValueError("Grid size mismatch")

# Set up SQLite with extended schema
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Drop and recreate the table to ensure correct schema
try:
    cursor.execute("DROP TABLE IF EXISTS ghi_data")
    cursor.execute('''
                   CREATE TABLE ghi_data
                   (
                       timestamp      TEXT,
                       GHI            REAL,
                       DNI            REAL,
                       DHI            REAL,
                       X              REAL,
                       Y              REAL,
                       lat            REAL,
                       lon            REAL,
                       Zenith_Angle   REAL,
                       Solar_Altitude REAL,
                       Solar_Azimuth  REAL,
                       UNIQUE (timestamp, lat, lon)
                   )
                   ''')
    conn.commit()
    logger.info("Recreated ghi_data table with correct schema.")
except Exception as e:
    logger.error(f"Error setting up database: {e}")
    raise


# Function to get the current row count in the database
def get_row_count():
    cursor.execute("SELECT COUNT(*) FROM ghi_data")
    count = cursor.fetchone()[0]
    return count


# Solar geometry functions (consistent with other scripts)
def calculate_zenith_angle(timestamp, latitude, longitude, standard_meridian=135):
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
    zenith_angle, _, _ = calculate_zenith_angle(timestamp, latitude, longitude)
    solar_altitude = 90 - zenith_angle
    return solar_altitude


def calc_solar_azimuth(zenith, hour_angle, declination, latitude):
    zenith_rad = np.radians(zenith)
    hour_rad = np.radians(hour_angle)
    decl_rad = np.radians(declination)
    lat_rad = np.radians(latitude)
    sin_az = np.sin(hour_rad) * np.cos(decl_rad) / np.sin(zenith_rad)
    cos_az = (np.sin(zenith_rad) * np.sin(lat_rad) - np.sin(decl_rad)) / (np.cos(zenith_rad) * np.cos(lat_rad))
    azimuth = np.degrees(np.arctan2(sin_az, cos_az))
    azimuth = (azimuth + 360) % 360
    return azimuth


# DISC Method for decomposing GHI into DNI and DHI
def disc_method(ghi, zenith_angle, timestamp):
    solar_constant = 1367  # W/m²
    standard_meridian = 135  # KST standard meridian
    day_of_year = timestamp.timetuple().tm_yday
    day_angle = 2 * np.pi / 365 * (day_of_year - 1)
    dec_solar_declination = 0.4093 * np.sin((2 * np.pi / 365) * (day_of_year + 284))
    re = (1.00011 + 0.034221 * np.cos(day_angle) + 0.00128 * np.sin(day_angle) +
          0.000719 * np.cos(2 * day_angle) + 0.000077 * np.sin(2 * day_angle))
    etr = solar_constant * re
    B = np.radians((360 / 365) * (day_of_year - 81))
    eqt = 229.18 * (0.000075 + 0.001868 * np.cos(B) - 0.032077 * np.sin(B) -
                    0.014615 * np.cos(2 * B) - 0.040849 * np.sin(2 * B)) / 60
    cos_zenith = np.cos(np.radians(zenith_angle))
    am = 1 / (cos_zenith + 0.50572 * (96.07995 - zenith_angle) ** -1.6364)
    if zenith_angle >= 90:
        am = np.nan
    i0h = etr * cos_zenith
    if cos_zenith < 0:
        i0h = 0
    kt = ghi / i0h if i0h > 0 else 0.0

    def disc_coefficients(kt):
        if not np.isfinite(kt) or kt < 0 or kt > 1.2:
            return np.nan, np.nan, np.nan
        if kt <= 0.6:
            a = 0.512 - 1.56 * kt + 2.286 * kt ** 2 - 2.222 * kt ** 3
            b = 0.37 + 0.962 * kt
            c = -0.28 + 0.932 * kt - 2.048 * kt ** 2
        else:
            a = -5.743 + 21.77 * kt - 27.49 * kt ** 2 + 11.56 * kt ** 3
            b = 41.40 - 118.5 * kt + 66.05 * kt ** 2 + 31.90 * kt ** 3
            c = -47.01 + 184.2 * kt - 222.0 * kt ** 2 + 73.81 * kt ** 3
        return a, b, c

    a, b, c = disc_coefficients(kt)
    dkn = a + b * np.exp(c * am) if not np.isnan(am) else np.nan
    knc = (0.866 - 0.122 * am + 0.0121 * am ** 2 -
           0.000653 * am ** 3 + 0.000014 * am ** 4) if not np.isnan(am) else np.nan
    dni = (knc - dkn) * etr if not np.isnan(knc) and not np.isnan(dkn) else 0.0
    dni = max(0, dni)
    dhi = ghi - dni * cos_zenith if not np.isnan(dni) else 0.0
    return dni, dhi


# Retry-safe downloader with nearest timestamp fallback
def download_file(original_time_str, search_window_minutes=30):
    file_name = f"gk2a_ami_le2_swrad_ko_{original_time_str}.nc"
    file_path = os.path.join(output_dir, file_name)
    url = f"{base_url}?date={original_time_str}&authKey={auth_key}"

    for attempt in range(5):
        try:
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                with open(file_path, 'wb') as f:
                    f.write(response.content)
                logger.info(f"Downloaded: {file_name}")
                return file_path, original_time_str
            elif response.status_code == 404:
                logger.warning(f"Attempt {attempt + 1}/5 - File not yet available: {file_name} (404)")
            else:
                logger.warning(
                    f"Attempt {attempt + 1}/5 - Failed to download: {file_name}, Status: {response.status_code}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Attempt {attempt + 1}/5 - Error downloading {file_name}: {str(e)}")

        if attempt < 4:
            time.sleep(5 * (2 ** attempt))

    logger.warning(f"All 5 retries failed for {file_name}. Attempting to fetch nearest available timestamp...")

    base_time = datetime.strptime(original_time_str, "%Y%m%d%H%M")
    search_offsets = [0, -10, 10, -20, 20, -30, 30]
    for offset in search_offsets[1:]:
        nearby_time = base_time + timedelta(minutes=offset)
        nearby_time_str = nearby_time.strftime("%Y%m%d%H%M")
        file_name = f"gk2a_ami_le2_swrad_ko_{nearby_time_str}.nc"
        file_path = os.path.join(output_dir, file_name)
        url = f"{base_url}?date={nearby_time_str}&authKey={auth_key}"

        logger.info(
            f"Attempting to download nearest file: {file_name} (offset {offset} minutes) for original timestamp {original_time_str}")
        for attempt in range(2):
            try:
                response = requests.get(url, timeout=30)
                if response.status_code == 200:
                    with open(file_path, 'wb') as f:
                        f.write(response.content)
                    logger.info(f"Downloaded nearest file: {file_name} for original timestamp {original_time_str}")
                    return file_path, original_time_str
                elif response.status_code == 404:
                    logger.warning(f"Attempt {attempt + 1}/2 - File not yet available: {file_name} (404)")
                else:
                    logger.warning(
                        f"Attempt {attempt + 1}/2 - Failed to download: {file_name}, Status: {response.status_code}")
            except requests.exceptions.RequestException as e:
                logger.error(f"Attempt {attempt + 1}/2 - Error downloading {file_name}: {str(e)}")
            if attempt < 1:
                time.sleep(5 * (2 ** attempt))

    logger.error(
        f"No nearby file found for {original_time_str} within ±{search_window_minutes} minutes. Skipping timestamp.")
    return None, original_time_str


# Process NetCDF file and compute DNI, DHI, and solar geometry
def process_file(file_path, timestamp, actual_time_str):
    try:
        dataset = nc.Dataset(file_path)
        dsr = dataset.variables['DSR'][:]
        dsr = np.flipud(dsr)

        if dsr.shape != (900, 900):
            logger.error(f"Error: Unexpected dimensions {dsr.shape}")
            return []

        data = dsr[row_start:row_end, col_start:col_end]
        if data.shape != (20, 20):
            logger.error(f"Error: Subset not 20x20: {data.shape}")
            return []

        rows_to_insert = []
        timestamp_kst = timestamp + timedelta(hours=9)  # Convert UTC to KST
        for row in range(20):
            for col in range(20):
                full_row = row_start + row
                full_col = col_start + col
                x = extent[0] + full_col * pixel_size_x + pixel_size_x / 2
                y = extent[3] - full_row * pixel_size_y - pixel_size_y / 2
                ghi = float(data[row, col]) if not np.isnan(data[row, col]) else None

                # Use lat, lon directly from locations_df
                idx = row * 20 + col
                lat = locations_df.iloc[idx]['lat']
                lon = locations_df.iloc[idx]['lon']

                if ghi is None or ghi < 0:
                    logger.warning(f"Invalid GHI value {ghi} at row={row}, col={col} in {file_path}. Skipping pixel.")
                    continue

                # Compute solar geometry
                zenith_angle, hour_angle, declination = calculate_zenith_angle(timestamp_kst, lat, lon)
                solar_altitude = calc_solar_altitude(timestamp_kst, lat, lon)
                solar_azimuth = calc_solar_azimuth(zenith_angle, hour_angle, declination, lat)

                # Validate solar geometry parameters
                if not all(np.isfinite([zenith_angle, solar_altitude, solar_azimuth])):
                    logger.warning(
                        f"Invalid solar geometry at row={row}, col={col}: zenith={zenith_angle}, altitude={solar_altitude}, azimuth={solar_azimuth}")
                    continue

                # Compute DNI and DHI using DISC method
                dni, dhi = disc_method(ghi, zenith_angle, timestamp_kst)

                # Validate DNI and DHI
                if not all(np.isfinite([dni, dhi])) or dni < 0 or dhi < 0:
                    logger.warning(f"Invalid DNI/DHI at row={row}, col={col}: DNI={dni}, DHI={dhi}")
                    continue

                rows_to_insert.append((
                    timestamp_kst.strftime("%Y-%m-%d %H:%M:%S"),
                    float(ghi), float(dni), float(dhi),
                    float(x), float(y), float(lat), float(lon),
                    float(zenith_angle), float(solar_altitude), float(solar_azimuth)
                ))
        logger.info(
            f"Processed {os.path.basename(file_path)} (used for timestamp {timestamp_kst.strftime('%Y-%m-%d %H:%M:%S')}): {len(rows_to_insert)} pixels prepared for insertion")
        return rows_to_insert
    except Exception as e:
        logger.error(f"Error processing {file_path}: {str(e)}")
        return []
    finally:
        try:
            dataset.close()
        except:
            pass


# Clean up old data (extended to 24 hours for future fine-tuning)
def cleanup_old_data():
    try:
        cutoff_time = (datetime.now() - timedelta(hours=24)).strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute("DELETE FROM ghi_data WHERE timestamp < ?", (cutoff_time,))
        conn.commit()
        logger.info(f"Cleaned up data older than {cutoff_time}. Rows deleted: {cursor.rowcount}")
    except Exception as e:
        logger.error(f"Error during database cleanup: {str(e)}")


# Fetch and update data every 30 minutes
def fetch_and_update_data():
    while True:
        now = datetime.now().replace(second=0, microsecond=0)
        minute = now.minute
        if minute < 30:
            now_aligned = now.replace(minute=0)
        else:
            now_aligned = now.replace(minute=30)

        # Skip nighttime (20:00 to 03:00 KST)
        current_hour = now.hour
        if current_hour >= 20 or current_hour < 3:
            next_run = (now + timedelta(days=1)).replace(hour=3, minute=0, second=0, microsecond=0)
            if current_hour < 3:
                next_run = now.replace(hour=3, minute=0, second=0, microsecond=0)
            sleep_seconds = (next_run - now).total_seconds()
            logger.info(
                f"Current time {now} is in restricted window (20:00-03:00). Sleeping for {sleep_seconds / 60:.2f} minutes until {next_run}")
            time.sleep(sleep_seconds)
            continue

        end_time = now_aligned - timedelta(minutes=30)
        start_time = end_time - timedelta(hours=4)

        timestamps = pd.date_range(start=start_time, end=end_time, freq='30min')
        time_strings = [t.strftime("%Y%m%d%H%M") for t in timestamps]

        logger.info(f"Fetching data from {start_time} to {end_time} at {now}")
        logger.info(f"Database row count before insert: {get_row_count()}")

        total_entries = 0
        skipped_files = 0
        for time_str, timestamp in zip(time_strings, timestamps):
            timestamp_utc = timestamp - timedelta(hours=9)
            time_str_utc = timestamp_utc.strftime("%Y%m%d%H%M")

            file_path, actual_time_str = download_file(time_str_utc)
            if file_path:
                rows = process_file(file_path, timestamp_utc, actual_time_str)
            else:
                skipped_files += 1
                logger.warning(f"No data available for {timestamp}. Skipping timestamp.")
                continue

            if rows:
                try:
                    cursor.executemany(
                        "INSERT OR REPLACE INTO ghi_data (timestamp, GHI, DNI, DHI, X, Y, lat, lon, Zenith_Angle, Solar_Altitude, Solar_Azimuth) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        rows)
                    conn.commit()
                    total_entries += cursor.rowcount
                    logger.info(f"Inserted {cursor.rowcount} new pixels for timestamp {timestamp}")
                    logger.info(f"Database row count after insert: {get_row_count()}")
                except Exception as e:
                    logger.error(f"Error inserting data for {timestamp}: {str(e)}")
            else:
                skipped_files += 1

        logger.info(f"Total number of entries inserted: {total_entries}")
        logger.info(f"Number of skipped files: {skipped_files}")
        logger.info(f"GHI data saved to SQLite DB: {db_path}")

        cleanup_old_data()

        next_update = now_aligned + timedelta(minutes=30)
        sleep_seconds = (next_update - now).total_seconds()
        logger.info(f"Sleeping for {sleep_seconds / 60:.2f} minutes until next update at {next_update}")
        time.sleep(sleep_seconds)


# Main execution
if __name__ == "__main__":
    try:
        fetch_and_update_data()
    except KeyboardInterrupt:
        logger.info("Stopped by user.")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
    finally:
        conn.close()

