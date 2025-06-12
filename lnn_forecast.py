import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import sqlite3
from datetime import datetime, timedelta
import logging
import pickle
import time
import os
from scipy.spatial import cKDTree

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(os.path.dirname(__file__), 'data', 'lnn_forecast.log')),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# LNN Model Definition (same as training)
def get_activation(name):
    if name == "tanh":
        return torch.tanh
    elif name == "relu":
        return torch.nn.functional.relu
    elif name == "leaky_relu":
        return lambda x: torch.nn.functional.leaky_relu(x, negative_slope=0.01)
    elif name == "gelu":
        return torch.nn.functional.gelu
    else:
        raise ValueError(f"Unsupported activation: {name}")

class LiquidTimeStep(nn.Module):
    def __init__(self, input_size, hidden_size, activation_name):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.W_in = nn.Linear(input_size, hidden_size)
        self.W_h = nn.Linear(hidden_size, hidden_size)
        self.tau = nn.Parameter(torch.ones(hidden_size) * 0.1)
        self.activation = get_activation(activation_name)

    def forward(self, x, h, dt=0.1):
        dx = self.activation(self.W_in(x) + self.W_h(h))
        h_new = h + dt * (dx - h) / self.tau
        return h_new

class LiquidNeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, activation_name, dropout_rate):
        super().__init__()
        self.hidden_size = hidden_size
        self.layers = nn.ModuleList([
            LiquidTimeStep(input_size if i == 0 else hidden_size, hidden_size, activation_name)
            for i in range(num_layers)
        ])
        self.bn = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        h = torch.zeros(batch_size, self.hidden_size, device=x.device)
        for t in range(seq_len):
            h_t = x[:, t, :]
            for layer in self.layers:
                h = layer(h_t, h)
                h_t = h
        h = self.bn(h)
        h = self.dropout(h)
        return self.output_layer(h)

# Load model and scalers
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
prefix_new = 'spatiotemporal_ghi_multistep'
model_save_dir = os.path.join(os.path.dirname(__file__), 'trained_models')
save_dir = os.path.join(os.path.dirname(__file__), 'preprocessed_data')
model_path = os.path.join(model_save_dir, 'lnn_3hour_spatiotemporal_ghi_multistep.pth')
scaler_X_path = os.path.join(save_dir, 'scaler_X_spatiotemporal_ghi_multistep.pkl')
scaler_y_path = os.path.join(save_dir, 'scaler_y_spatiotemporal_ghi_multistep.pkl')
best_params_path = os.path.join(model_save_dir, 'best_params_spatiotemporal_ghi_multistep.pkl')
data_info_path = os.path.join(save_dir, 'data_info_spatiotemporal_ghi_multistep.txt')

# Load best parameters
with open(best_params_path, "rb") as f:
    best_params = pickle.load(f)

# Load data info
with open(data_info_path, "r") as f:
    data_info = f.readlines()
    timesteps = int(data_info[0].split(": ")[1])
    n_features = int(data_info[1].split(": ")[1])

# Load model with map_location to handle CPU-only machines
lnn_model = LiquidNeuralNetwork(
    input_size=n_features,
    hidden_size=best_params["hidden_size"],
    output_size=6,  # Updated to output 6 GHI values
    num_layers=best_params["num_layers"],
    activation_name=best_params["activation"],
    dropout_rate=best_params["dropout_rate"]
).to(device)
lnn_model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
lnn_model.eval()

# Load scalers
with open(scaler_X_path, "rb") as f:
    scaler_X = pickle.load(f)
with open(scaler_y_path, "rb") as f:
    scaler_y = pickle.load(f)

# Database paths
weather_db_path = os.path.join(os.path.dirname(__file__), 'data', 'weather_data.db')
ghi_db_path = os.path.join(os.path.dirname(__file__), 'data', 'ghi_data.db')
forecast_db_path = os.path.join(os.path.dirname(__file__), 'data', 'forecast_data.db')

# Connect to databases
conn_weather = sqlite3.connect(weather_db_path)
conn_ghi = sqlite3.connect(ghi_db_path)
conn_forecast = sqlite3.connect(forecast_db_path)
cursor_forecast = conn_forecast.cursor()

# Drop and recreate the forecast_data table to ensure correct schema
cursor_forecast.execute("DROP TABLE IF EXISTS forecast_data")
conn_forecast.commit()
logger.info("Dropped forecast_data table.")

# Create forecast table with explicit REAL type for GHI and add timestep column
cursor_forecast.execute('''
    CREATE TABLE forecast_data (
        timestamp TEXT,
        lat REAL,
        lon REAL,
        GHI REAL,
        timestep INTEGER,
        UNIQUE(timestamp, lat, lon, timestep)
    )
''')
conn_forecast.commit()
logger.info("Created forecast_data table with GHI as REAL and timestep column.")

# Verify the table schema
cursor_forecast.execute("PRAGMA table_info(forecast_data);")
schema = cursor_forecast.fetchall()
logger.info("Forecast_data table schema: %s", schema)

# Define the Seoul coordinates (2-km grid, 20x20 points)
coords = [(lat, lon) for lat in np.linspace(37.4, 37.7, 20) for lon in np.linspace(126.8, 127.2, 20)]
locations_df = pd.DataFrame(coords, columns=['lat', 'lon'])
locations_df = locations_df.drop_duplicates(subset=['lat', 'lon'])
locations_df['nx'] = locations_df.index // 20
locations_df['ny'] = locations_df.index % 20
logger.info(f"locations_df has {len(locations_df)} rows, unique lat/lon pairs: {len(locations_df[['lat', 'lon']].drop_duplicates())}")

# Custom solar geometry functions (as used in training)
def calculate_zenith_angle(timestamp, latitude, longitude, standard_meridian=135):
    """Calculate zenith angle, hour angle, and declination for a given timestamp and location."""
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
    zenith_angle, _, _ = calculate_zenith_angle(timestamp, latitude, longitude)
    solar_altitude = 90 - zenith_angle
    return solar_altitude

def calc_solar_azimuth(zenith, hour_angle, declination, latitude):
    """Calculate solar azimuth using zenith angle, hour angle, declination, and latitude."""
    zenith_rad = np.radians(zenith)
    hour_rad = np.radians(hour_angle)
    decl_rad = np.radians(declination)
    lat_rad = np.radians(latitude)
    sin_az = np.sin(hour_rad) * np.cos(decl_rad) / np.sin(zenith_rad)
    cos_az = (np.sin(zenith_rad) * np.sin(lat_rad) - np.sin(decl_rad)) / (np.cos(zenith_rad) * np.cos(lat_rad))
    azimuth = np.degrees(np.arctan2(sin_az, cos_az))
    azimuth = (azimuth + 360) % 360
    return azimuth

def compute_time_features(timestamp):
    """Compute cyclic time features (day of year, hour)."""
    dt = pd.to_datetime(timestamp)
    day_of_year = dt.timetuple().tm_yday
    hour = dt.hour + dt.minute / 60.0
    day_of_year_sin = np.sin(2 * np.pi * day_of_year / 365.0)
    day_of_year_cos = np.cos(2 * np.pi * day_of_year / 365.0)
    hour_sin = np.sin(2 * np.pi * hour / 24.0)
    hour_cos = np.cos(2 * np.pi * hour / 24.0)
    return day_of_year_sin, day_of_year_cos, hour_sin, hour_cos

def compute_location_features(lat, lon, lat_min, lat_max, lon_min, lon_max):
    """Compute cyclic location features (lat, lon) with the same scaling as training."""
    lat_scaled = (lat - lat_min) / (lat_max - lat_min)
    lon_scaled = (lon - lon_min) / (lon_max - lon_min)
    lat_sin = np.sin(2 * np.pi * lat_scaled)
    lat_cos = np.cos(2 * np.pi * lat_scaled)
    lon_sin = np.sin(2 * np.pi * lon_scaled)
    lon_cos = np.cos(2 * np.pi * lon_scaled)
    return lat_sin, lat_cos, lon_sin, lon_cos

def compute_kt(ghi, solar_altitude):
    """Compute clearness index (Kt)."""
    if solar_altitude <= 0:
        return 0.0
    I0 = 1367  # Solar constant (W/m^2)
    extraterrestrial = I0 * np.sin(np.radians(solar_altitude))
    if extraterrestrial <= 0:
        return 0.0
    kt = ghi / extraterrestrial
    return np.clip(kt, 0.0, 1.2)

def fetch_and_prepare_data(base_time, n_in=8):
    """Fetch and prepare historical data for LNN input."""
    start_time = base_time - timedelta(hours=4)
    end_time = base_time
    timestamps = pd.date_range(start=start_time, end=end_time, freq='30min')
    if len(timestamps) != 9:
        logger.error(f"Expected 9 timesteps, got {len(timestamps)}")
        return None

    query_weather = """
        SELECT timestamp, nx, ny, lat, lon, temperature, wind_speed
        FROM weather_data
        WHERE data_type = 'historical'
        AND timestamp BETWEEN ? AND ?
        ORDER BY nx, ny, timestamp
    """
    df_weather = pd.read_sql_query(
        query_weather,
        conn_weather,
        params=(start_time.strftime("%Y-%m-%d %H:%M:%S"), end_time.strftime("%Y-%m-%d %H:%M:%S"))
    )
    df_weather['timestamp'] = pd.to_datetime(df_weather['timestamp'])
    df_weather = df_weather.drop_duplicates(subset=['timestamp', 'nx', 'ny', 'lat', 'lon'])
    logger.info(f"Fetched weather data: {len(df_weather)} rows")
    logger.info(f"Weather timestamps: {df_weather['timestamp'].unique()}")

    query_ghi = """
        SELECT timestamp, lat, lon, GHI, DNI, DHI
        FROM ghi_data
        WHERE timestamp BETWEEN ? AND ?
        ORDER BY timestamp, lat, lon
    """
    df_ghi = pd.read_sql_query(
        query_ghi,
        conn_ghi,
        params=(start_time.strftime("%Y-%m-%d %H:%M:%S"), end_time.strftime("%Y-%m-%d %H:%M:%S"))
    )
    df_ghi['timestamp'] = pd.to_datetime(df_ghi['timestamp'])
    df_ghi = df_ghi.drop_duplicates(subset=['timestamp', 'lat', 'lon'])
    logger.info(f"Fetched GHI data: {len(df_ghi)} rows")
    logger.info(f"GHI timestamps: {df_ghi['timestamp'].unique()}")

    if df_weather.empty or df_ghi.empty:
        logger.error("No historical data available in the specified time range.")
        return None

    # Find common timestamps
    ghi_timestamps = set(df_ghi['timestamp'])
    weather_timestamps = set(df_weather['timestamp'])
    common_timestamps = ghi_timestamps.intersection(weather_timestamps)
    if not common_timestamps:
        logger.error("No common timestamps between GHI and weather data.")
        return None
    logger.info(f"Common timestamps: {common_timestamps}")

    # Adjust base_time to the latest common timestamp
    latest_common_timestamp = max(common_timestamps)
    adjusted_base_time = pd.to_datetime(latest_common_timestamp)
    time_diff = (base_time - adjusted_base_time).total_seconds() / 3600.0
    if time_diff > 2:  # Don't go back more than 2 hours
        logger.error(f"Adjusted base time {adjusted_base_time} is too far in the past (difference: {time_diff} hours).")
        return None
    start_time = adjusted_base_time - timedelta(hours=4)
    timestamps = pd.date_range(start=start_time, end=adjusted_base_time, freq='30min')
    expected_timesteps = len(timestamps)
    logger.info(f"Adjusted base time to {adjusted_base_time}, new time range: {start_time} to {adjusted_base_time}, expected timesteps: {expected_timesteps}")

    # Filter data to the adjusted time range
    df_weather = df_weather[df_weather['timestamp'].isin(timestamps)]
    df_ghi = df_ghi[df_ghi['timestamp'].isin(timestamps)]
    logger.info(f"Filtered weather data: {len(df_weather)} rows")
    logger.info(f"Filtered GHI data: {len(df_ghi)} rows")

    if df_weather.empty or df_ghi.empty:
        logger.error("No data available in the adjusted time range.")
        return None

    # Map both GHI and weather coordinates to locations_df
    coords_weather = locations_df[['lat', 'lon']].drop_duplicates().values
    tree = cKDTree(coords_weather)

    # Map GHI data
    coords_ghi = df_ghi[['lat', 'lon']].values
    _, idx_ghi = tree.query(coords_ghi)
    df_ghi['lat_mapped'] = locations_df.iloc[idx_ghi]['lat'].values
    df_ghi['lon_mapped'] = locations_df.iloc[idx_ghi]['lon'].values
    df_ghi['nx'] = locations_df.iloc[idx_ghi]['nx'].values
    df_ghi['ny'] = locations_df.iloc[idx_ghi]['ny'].values
    df_ghi = df_ghi.drop_duplicates(subset=['timestamp', 'nx', 'ny'])
    logger.info(f"After mapping, df_ghi has {len(df_ghi)} rows")
    logger.info(f"Unique nx, ny pairs in df_ghi: {len(df_ghi[['nx', 'ny']].drop_duplicates())}")

    # Map weather data
    coords_weather_data = df_weather[['lat', 'lon']].values
    _, idx_weather = tree.query(coords_weather_data)
    df_weather['lat_mapped'] = locations_df.iloc[idx_weather]['lat'].values
    df_weather['lon_mapped'] = locations_df.iloc[idx_weather]['lon'].values
    df_weather['nx'] = locations_df.iloc[idx_weather]['nx'].values
    df_weather['ny'] = locations_df.iloc[idx_weather]['ny'].values
    df_weather = df_weather.drop_duplicates(subset=['timestamp', 'nx', 'ny'])
    logger.info(f"After mapping, df_weather has {len(df_weather)} rows")
    logger.info(f"Unique nx, ny pairs in df_weather: {len(df_weather[['nx', 'ny']].drop_duplicates())}")

    # Merge GHI and weather data on timestamp, nx, ny
    df_merged = pd.merge(
        df_ghi,
        df_weather,
        on=['timestamp', 'nx', 'ny'],
        how='inner'
    )
    df_merged = df_merged.drop_duplicates(subset=['timestamp', 'nx', 'ny'])
    logger.info(f"Merged data: {len(df_merged)} rows")

    if df_merged.empty:
        logger.error("No matching GHI and weather data after merging.")
        return None

    grouped = df_merged.groupby(['nx', 'ny'])
    processed_dfs = []
    lat_min, lat_max = locations_df['lat'].min(), locations_df['lat'].max()
    lon_min, lon_max = locations_df['lon'].min(), locations_df['lon'].max()

    for (nx, ny), group in grouped:
        group = group.sort_values('timestamp')
        lat, lon = group['lat_mapped_x'].iloc[0], group['lon_mapped_x'].iloc[0]

        df_pixel = pd.DataFrame({'timestamp': timestamps})
        df_pixel = df_pixel.merge(group, on='timestamp', how='left')
        actual_timesteps = len(group['timestamp'].unique())
        logger.info(f"Location (nx={nx}, ny={ny}) has {actual_timesteps} unique timesteps, expected {expected_timesteps}")
        if len(df_pixel) != expected_timesteps:
            logger.warning(f"Location (nx={nx}, ny={ny}) has {len(df_pixel)} timesteps after merge, expected {expected_timesteps}.")
            continue
        if actual_timesteps < 5:
            logger.warning(f"Location (nx={nx}, ny={ny}) has too few timesteps ({actual_timesteps}) to forecast.")
            continue

        df_pixel['GHI'] = df_pixel['GHI'].interpolate(method='linear', limit_direction='both')
        df_pixel['DNI'] = df_pixel['DNI'].interpolate(method='linear', limit_direction='both')
        df_pixel['DHI'] = df_pixel['DHI'].interpolate(method='linear', limit_direction='both')
        df_pixel['temperature'] = df_pixel['temperature'].interpolate(method='linear', limit_direction='both')
        df_pixel['lat'] = lat
        df_pixel['lon'] = lon
        df_pixel['nx'] = nx
        df_pixel['ny'] = ny

        df_pixel['Zenith_Angle'], df_pixel['HRA'], df_pixel['DEC'] = zip(*df_pixel['timestamp'].apply(
            lambda ts: calculate_zenith_angle(ts, lat, lon)
        ))
        df_pixel['Solar_Altitude'] = df_pixel['timestamp'].apply(lambda ts: calc_solar_altitude(ts, lat, lon))
        df_pixel['Solar_Azimuth'] = df_pixel.apply(
            lambda row: calc_solar_azimuth(row['Zenith_Angle'], row['HRA'], row['DEC'], lat), axis=1
        )

        df_pixel['Kt'] = df_pixel.apply(lambda row: compute_kt(row['GHI'], row['Solar_Altitude']), axis=1)

        df_pixel[['day_of_year_sin', 'day_of_year_cos', 'hour_sin', 'hour_cos']] = df_pixel['timestamp'].apply(
            lambda ts: pd.Series(compute_time_features(ts))
        ).values
        df_pixel[['lat_sin', 'lat_cos', 'lon_sin', 'lon_cos']] = pd.Series(
            compute_location_features(lat, lon, lat_min, lat_max, lon_min, lon_max)
        ).values.repeat(len(df_pixel)).reshape(-1, 4)

        processed_dfs.append(df_pixel)

    if not processed_dfs:
        logger.error("No locations had sufficient data for forecasting.")
        return None

    return pd.concat(processed_dfs, ignore_index=True)

def forecast_ghi(df, n_forecast=6):
    """Forecast GHI using the LNN model for 6 timesteps."""
    features = [
        'GHI', 'DNI', 'DHI', 'temperature', 'lat_sin', 'lat_cos', 'lon_sin', 'lon_cos',
        'Kt', 'Solar_Altitude', 'Solar_Azimuth', 'day_of_year_sin', 'day_of_year_cos',
        'hour_sin', 'hour_cos'
    ]
    feature_columns = []
    for t in range(-8, 1):  # t-8 to t
        suffix = f"(t-{abs(t)})" if t < 0 else "(t)"
        feature_columns.extend([f"{feat}{suffix}" for feat in features])

    grouped = df.groupby(['nx', 'ny'])
    forecast_dfs = []

    for (nx, ny), group in grouped:
        group = group.sort_values('timestamp')
        lat, lon = group['lat'].iloc[0], group['lon'].iloc[0]

        # Prepare input data
        data_shifted = pd.concat([group[features].shift(i) for i in range(8, -1, -1)], axis=1)
        data_shifted.columns = feature_columns
        data_shifted['timestamp'] = group['timestamp']
        data_shifted['lat'] = lat
        data_shifted['lon'] = lon
        data_shifted = data_shifted.dropna()

        if data_shifted.empty:
            logger.warning(f"No valid data for forecasting at nx={nx}, ny={ny}")
            continue

        # Use the last row for forecasting
        X = data_shifted[feature_columns].values[-1:]  # Shape: (1, 135)
        X_scaled = scaler_X.transform(X)
        X_reshaped = X_scaled.reshape(-1, timesteps, n_features)  # Shape: (1, 9, 15)
        X_tensor = torch.tensor(X_reshaped, dtype=torch.float32).to(device)

        # Forecast GHI for 6 timesteps
        with torch.no_grad():
            forecast_scaled = lnn_model(X_tensor)  # Shape: (1, 6)
        forecast_ghi = np.zeros_like(forecast_scaled.cpu().numpy())  # Shape: (1, 6)
        for i in range(n_forecast):
            forecast_ghi[:, i] = scaler_y.inverse_transform(forecast_scaled[:, i].reshape(-1, 1)).flatten()
        forecast_ghi = forecast_ghi[0]  # Shape: (6,)

        # Generate forecast timestamps
        last_timestamp = pd.Timestamp(data_shifted['timestamp'].iloc[-1])
        forecast_timestamps = [last_timestamp + timedelta(minutes=30 * i) for i in range(1, n_forecast + 1)]

        # Create forecast DataFrame
        forecast_rows = []
        for i, ts in enumerate(forecast_timestamps):
            forecast_rows.append({
                'timestamp': ts,
                'lat': lat,
                'lon': lon,
                'GHI': float(forecast_ghi[i]),
                'timestep': i + 1  # 1 to 6 for t+1 to t+6
            })
        forecast_df = pd.DataFrame(forecast_rows)
        forecast_df['timestamp'] = pd.to_datetime(forecast_df['timestamp'])
        forecast_dfs.append(forecast_df)

    if not forecast_dfs:
        logger.error("No forecasts generated.")
        return None

    return pd.concat(forecast_dfs, ignore_index=True)

def run_forecast_attempt(base_time, n_in, n_forecast):
    try:
        df = fetch_and_prepare_data(base_time, n_in=n_in)
        if df is None:
            return None

        forecast_df = forecast_ghi(df, n_forecast=n_forecast)
        if forecast_df is None:
            logger.error("Failed to generate forecasts.")
            return None

        # Ensure GHI and timestep are appropriate types and prepare forecast rows
        forecast_df['GHI'] = forecast_df['GHI'].astype(float)  # Explicitly convert to float
        forecast_df['timestep'] = forecast_df['timestep'].astype(int)  # Explicitly convert to int
        forecast_rows = [
            (
                pd.Timestamp(row.timestamp).strftime("%Y-%m-%d %H:%M:%S"),
                float(row.lat),
                float(row.lon),
                float(row.GHI),
                int(row.timestep)
            )
            for row in forecast_df[['timestamp', 'lat', 'lon', 'GHI', 'timestep']].to_records(index=False)
        ]

        # Insert into database
        cursor_forecast.executemany(
            "INSERT OR REPLACE INTO forecast_data (timestamp, lat, lon, GHI, timestep) VALUES (?, ?, ?, ?, ?)",
            forecast_rows
        )
        conn_forecast.commit()
        logger.info(f"Inserted {len(forecast_rows)} forecast records into forecast_data.db")

        # Save to CSV, ensuring timestamp is formatted as string
        forecast_df_to_save = forecast_df.copy()
        forecast_df_to_save['timestamp'] = forecast_df_to_save['timestamp'].dt.strftime("%Y-%m-%d %H:%M:%S")
        with open(os.path.join(os.path.dirname(__file__), 'data', 'forecasted_ghi.csv'), 'w') as f:
            pass
        forecast_df_to_save.to_csv(
            r"D:\Masters thesis PV\pv_performance_tool\data\forecasted_ghi.csv",
            index=False
        )
        logger.info("Forecasted GHI data saved to CSV for web integration.")
        return forecast_df

    except Exception as e:
        logger.error(f"Error during forecast for {base_time}: {str(e)}")
        return None

def run_forecast(base_time, n_in=8, n_forecast=6, retries=3, retry_delay=30):
    """Run a single forecast cycle with retries."""
    for attempt in range(retries):
        logger.info(f"Attempt {attempt + 1}/{retries}: Running forecast for {base_time}")
        result = run_forecast_attempt(base_time, n_in, n_forecast)
        if result is not None:
            return result
        logger.info(f"Retrying in {retry_delay/60} minutes...")
        time.sleep(retry_delay)
    logger.error(f"Failed to generate forecast after {retries} attempts.")
    return None

def main_forecast_loop(n_in=8, n_forecast=6, interval_minutes=30):
    """Run forecast loop with updates every interval_minutes (30 or 60)."""
    interval = timedelta(minutes=interval_minutes)

    while True:
        now = datetime.now().replace(second=0, microsecond=0)
        logger.info(f"Checking forecast cycle at {now}")

        # Set base_time to the nearest past 30-minute mark
        minutes = (now.minute // 30) * 30
        base_time = now.replace(minute=minutes)

        # Skip nighttime (16:00 to 03:00 KST)
        current_hour = now.hour
        if current_hour >= 16 or current_hour < 3:
            next_run = (now + timedelta(days=1)).replace(hour=3, minute=0, second=0, microsecond=0)
            if current_hour < 3:
                next_run = now.replace(hour=3, minute=0, second=0, microsecond=0)
            sleep_seconds = (next_run - now).total_seconds()
            logger.info(f"Current time {now} is in restricted window (16:00-03:00). Sleeping for {sleep_seconds / 60:.2f} minutes until {next_run}")
            time.sleep(sleep_seconds)
            continue

        # Run the forecast immediately for the nearest past 30-minute mark
        run_forecast(base_time, n_in, n_forecast, retries=3, retry_delay=300)

        # Calculate the next forecast time
        next_base_time = base_time + interval
        minutes = (next_base_time.minute // 30) * 30
        next_base_time = next_base_time.replace(minute=minutes)

        # Sleep until the next forecast time
        sleep_seconds = (next_base_time - datetime.now()).total_seconds()
        if sleep_seconds > 0:
            logger.info(f"Sleeping for {sleep_seconds / 60:.2f} minutes until next forecast at {next_base_time}")
            time.sleep(sleep_seconds)

if __name__ == "__main__":
    try:
        # Run with 30-minute updates (or set interval_minutes=60 for 1-hour updates)
        main_forecast_loop(interval_minutes=30)
    except KeyboardInterrupt:
        logger.info("Forecast loop stopped by user.")
    finally:
        conn_weather.close()
        conn_ghi.close()
        conn_forecast.close()