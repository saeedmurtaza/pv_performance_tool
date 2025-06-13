import os
import requests
import netCDF4 as nc
import numpy as np
from datetime import datetime, timedelta
import pandas as pd
import sqlite3
import pyproj
import time
import logging
from pyproj import Transformer
from b2sdk.v1 import B2Api, InMemoryAccountInfo
from b2sdk.exception import B2Error

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(os.path.dirname(__file__), 'data', 'fetch_ghi_data.log')),
        logging.StreamHandler()
    ]
)

# Backblaze B2 credentials (set as environment variables in Heroku)
APPLICATION_KEY_ID = os.environ.get('B2_APPLICATION_KEY_ID', "004ba3eab8b5e440000000001")
APPLICATION_KEY = os.environ.get('B2_APPLICATION_KEY', "K004D2pavneAmHVb3ql+JBxcH/69DWk")
BUCKET_NAME = os.environ.get('B2_BUCKET_NAME', "pv-performance-tool-data-saeed")

# Initialize B2 API
try:
    info = InMemoryAccountInfo()
    b2_api = B2Api(info)
    b2_api.authorize_account("production", APPLICATION_KEY_ID, APPLICATION_KEY)
    bucket = b2_api.get_bucket_by_name(BUCKET_NAME)
    logger.info(f"Successfully connected to Backblaze B2 bucket: {BUCKET_NAME}")
except B2Error as e:
    logger.error(f"Failed to authorize Backblaze B2: {str(e)}")
    raise

# Set up parameters
auth_key = os.environ.get("KMA_AUTH_KEY", "LoBU7l_-QDuAVO5f_iA7ZQ")
base_url = "https://apihub.kma.go.kr/api/typ05/api/GK2A/LE2/SWRAD/KO/data"
output_dir = os.path.join(os.path.dirname(__file__), 'data', 'nc_files')
db_path = os.path.join(os.path.dirname(__file__), 'data', 'ghi_data.db')

# Create directories
os.makedirs(output_dir, exist_ok=True)

# Projection and Seoul region config
proj = "+proj=lcc +lat_0=38 +lon_0=126 +lat_1=30 +lat_2=60 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
extent = (-899000, 899000, -899000, 899000)
seoul_bbox = {"min_lon": 126.8, "max_lon": 127.2, "min_lat": 37.4, "max_lat": 37.7}

# Define the 20x20 grid directly from locations_df
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

# Upload new or updated .nc files to Backblaze B2
def upload_nc_files(local_dir):
    local_nc_dir = os.path.join(os.path.dirname(__file__), local_dir)
    if not os.path.exists(local_nc_dir):
        logger.warning(f"Local directory {local_nc_dir} does not exist, skipping upload.")
        return
    os.makedirs(local_nc_dir, exist_ok=True)
    for file_name in os.listdir(local_nc_dir):
        if file_name.endswith('.nc'):
            local_path = os.path.join(local_nc_dir, file_name)
            remote_path = f"nc_files/{file_name}"
            try:
                if not bucket.get_file_info_by_name(remote_path):
                    logger.info(f"Uploading {file_name} to Backblaze B2")
                    bucket.upload_local_file(local_path=local_path, file_name=remote_path)
                else:
                    logger.info(f"{file_name} already exists in Backblaze B2, skipping upload.")
            except B2Error as e:
                logger.error(f"Failed to upload {file_name}: {str(e)}")

# Download .nc files from Backblaze B2
def download_nc_files():
    local_nc_dir = os.path.join(os.path.dirname(__file__), 'data', 'nc_files')
    os.makedirs(local_nc_dir, exist_ok=True)
    try:
        for file in bucket.ls("nc_files/").entries:
            local_file = os.path.join(local_nc_dir, file.file_name.split('/')[-1])
            if not os.path.exists(local_file):
                logger.info(f"Downloading {file.file_name} from Backblaze B2")
                bucket.download_file_by_name(file.file_name, local_file)
    except B2Error as e:
        logger.error(f"Failed to download files from Backblaze B2: {str(e)}")

# Solar geometry functions
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