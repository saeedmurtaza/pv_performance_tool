
# ☀️ PV Performance Prediction Tool for Seoul

A web-based solar energy performance prediction platform that integrates **satellite-based solar irradiance forecasting using Liquid Neural Networks (LNN)** with **semi-empirical PV output modeling (SAM models)**. This tool provides real-time and forecasted PV output metrics for urban environments, specifically tailored for Seoul, South Korea.

---

## 📌 Project Overview

This project combines machine learning and physical modeling to deliver accurate and actionable photovoltaic (PV) system performance predictions. It is structured in three main stages:

1. **Solar Irradiance Forecasting using LNN**  
   - Utilizes GK2A satellite data (30-min intervals, 2-km resolution)  
   - Applies the Maxwell DISC method to decompose GHI → DNI & DHI  
   - Computes Global Tilted Irradiance (GTI) using the Perez model  
   - Trains a supervised Liquid Neural Network (LNN) to forecast GTI for the next 3 hours

2. **PV Power Prediction using Semi-Empirical Models**  
   - Implements four SAM-based models:  
     1. Simple Efficiency Model  
     2. Sandia Array Performance Model (SAPM)  
     3. Single Diode Model  
     4. CEC Model (final choice due to dynamic flexibility)  
   - Uses GTI, temperature, and wind speed to compute:  
     - Pdc, Pac, I, V  
     - Performance metrics: **PR**, **Yf**, **CF**

3. **Interactive Web Dashboard (Flask + Plotly + Leaflet)**  
   - Input parameters: Location, PV type, area, tilt, orientation, SAM model  
   - Output: Real-time & forecasted plots of GHI, Pac, I-V curves, and IEC metrics  
   - Map interface with district search and auto-grid rounding for Seoul

---

## 🌐 Features

- 🌞 **3-Hour Ahead Forecasts** using spatiotemporal LNN  
- 🧮 **Semi-Empirical PV Modeling** for dynamic conditions  
- 📊 **Interactive Plots**: GHI, Power Output (Pac), Current & Voltage  
- 📍 **Location Input**: Pinpoint on map or enter coordinates manually  
- 📈 **IEC Metrics**: PR, Yf, CF, EPI – computed and benchmarked in real-time  
- ⚙️ **Model Selection**: User chooses among four SAM models for prediction  
- 🌬️ **Weather Integration**: Real-time temperature & wind forecasts via KMA & Open-Meteo APIs  

---

## 📁 Directory Structure

```
pv_performance_tool/
│
├── app.py                    # Main Flask application
├── templates/index.html      # Web dashboard UI (Bootstrap, Leaflet, Plotly)
├── static/                   # CSS, JS scripts for UI interactivity
├── data/                     # Processed datasets and SQLite DBs
│   ├── ghi_data.db           # Historical GHI data
│   ├── weather_data.db       # Historical and forecasted temperature, wind
│   ├── forecast_data.db      # Forecasted GTI using LNN
│   ├── locations_eng.xlsx    # Location metadata (Seoul)
│   └── ...                   
├── scripts/                  
│   ├── fetch_ghi_data.py     # Fetches and decomposes satellite GHI
│   ├── forecasted_temp.py    # Collects temperature and wind forecasts
│   └── lnn_forecast.py       # Performs GTI forecasting with LNN
├── pv_models.py              # Semi-empirical SAM model computations
└── lnn_model.py              # (Optional/Deprecated) Initial model utilities
```

---

## 🔍 Technologies Used

- **Python (Flask, NumPy, Pandas, SQLite)**  
- **PyTorch** for LNN modeling  
- **Optuna** for hyperparameter tuning  
- **Leaflet.js** for interactive maps  
- **Plotly.js** for real-time plots  
- **GK2A Satellite Data + KMA API** for solar and weather inputs  

---

## 📐 Standards Followed

- Solar decomposition using **DISC (Maxwell 1987)**  
- Tilted surface estimation using **Perez model**  
- PV performance benchmarking via **IEC 61724 and 61853**

---

## 📊 Sample Output Metrics

| Metric       | Description                          |
|--------------|--------------------------------------|
| PR           | Performance Ratio                    |
| CF           | Capacity Factor                      |
| Yf           | Yield Factor                         |
| Pac          | AC Output Power (W)                  |
| I, V         | Current and Voltage at MPP           |

---

## 🧪 Model Training Details

- Forecast model: **Liquid Neural Network**
- Input: GHI, zenith, Kt, air mass, timestamps, etc.
- Tuning: **Optuna** (200 trials)
- Forecast window: **3-hour ahead, 30-min intervals**
- Evaluation: MSE, RMSE, and IEC metric alignment

---

## 📦 Installation & Run

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/pv_performance_tool.git
   cd pv_performance_tool
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Start the Flask app:
   ```bash
   python app.py
   ```

---

## 🔮 Future Work

- Expand to other cities with customizable coordinate ranges  
- Add solar panel degradation factors and seasonal adjustments  
- Deploy on Google Cloud Run or Streamlit for public access  
- Optimize LNN using ensemble models or attention mechanisms

---

## 📃 Acknowledgements

- GK2A Satellite Data (KMA)  
- SAM Models via NREL  
- Solar decomposition concepts: Maxwell (1987), Perez (1990)  
- IEC 61724 and 61853 standards  
- Professor Geun Young Yun (Kyung Hee University) – Project Advisor  
