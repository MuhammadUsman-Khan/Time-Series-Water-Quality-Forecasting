# 🌊 Time Series Water Quality Forecasting

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Statsmodels](https://img.shields.io/badge/Statsmodels-Time%20Series-4B8BBE?style=for-the-badge&logo=python&logoColor=white)
![Google Colab](https://img.shields.io/badge/Google%20Colab-TPU%20v5e-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)

![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat-square)
![Dataset](https://img.shields.io/badge/Dataset-Kaggle-20BEFF?style=flat-square&logo=kaggle&logoColor=white)
![Best Model](https://img.shields.io/badge/Best%20Model-Holt--Winters-blueviolet?style=flat-square)
![MAPE](https://img.shields.io/badge/Best%20MAPE-9.01%25-blue?style=flat-square)

</div>

---

## 📌 Overview

This project applies **time series analysis and forecasting** to predict **Dissolved Oxygen (O₂)** concentrations in the **Southern Bug (Pivdennyi Booh) River, Ukraine**, using 21+ years of multivariate water quality monitoring data.

Dissolved Oxygen is one of the most vital indicators of river health — it directly reflects biological activity, water temperature dynamics, and ecosystem sustainability. This project builds a full forecasting pipeline from raw data ingestion through model comparison and future projection.

> **Goal:** Predict O₂ at time *t* using past measurements, capturing seasonal and trend dynamics through both classical statistical models and machine learning.

---

## 📊 Dataset

| Property | Detail |
|---|---|
| **Source** | [Kaggle — Southern Bug River Water Quality](https://www.kaggle.com/datasets/vbmokin/wq-southern-bug-river-01052021) |
| **Stations** | 22 monitoring stations |
| **Records** | 2,861 raw observations |
| **Period** | January 2000 – April 2021 |
| **Focus Station** | Station 3 — 262 observations over 21.1 years |
| **Target Variable** | Dissolved Oxygen (O₂) in mg/L |
| **Features** | NH4, BSK5, Suspended, NO3, NO2, SO4, PO4, CL |

---

## 🔬 Project Pipeline

```
Raw Data (Kaggle)
      │
      ▼
Data Preparation
  ├── Date parsing & sorting
  ├── Station filtering (Station 3)
  ├── Missing value imputation (median)
  └── Monthly resampling + interpolation (254 months)
      │
      ▼
Exploratory Data Analysis
  ├── Descriptive statistics
  ├── Time series visualization (all 9 parameters)
  ├── Seasonal box plots
  ├── Correlation heatmap
  └── Additive decomposition (trend + seasonality + residual)
      │
      ▼
Stationarity Testing
  ├── ADF Test  →  Original: NON-STATIONARY | Differenced: STATIONARY
  └── KPSS Test →  Original: NON-STATIONARY | Differenced: STATIONARY
      │
      ▼
ACF & PACF Analysis
  └── Seasonal spikes at lags 6, 12, 18, 24 → confirms s=12
      │
      ▼
Model Building (Train: 230m | Test: 24m)
  ├── ARIMA(1,1,1)
  ├── SARIMA(1,1,1)(1,1,1,12)
  ├── Holt-Winters Exponential Smoothing
  └── Gradient Boosting + Lag Feature Engineering
      │
      ▼
Model Evaluation & Selection
      │
      ▼
24-Month Forecast (2021–2023)
```

---

## 🧠 Models Used

| Model | Type | Seasonal Aware |
|---|---|---|
| ARIMA(1,1,1) | Statistical | ❌ |
| SARIMA(1,1,1)(1,1,1,12) | Statistical | ✅ |
| Holt-Winters Exponential Smoothing | Statistical | ✅ |
| Gradient Boosting + Lag Features | Machine Learning | ✅ (via lag_12, month_sin/cos) |

---

## 📈 Results

### Model Comparison Table

| Rank | Model | MAE | RMSE | MAPE |
|---|---|---|---|---|
| 🥇 1 | **Holt-Winters** | **1.037** | **1.417** | **9.01%** |
| 🥈 2 | SARIMA(1,1,1)(1,1,1,12) | 1.103 | 1.491 | 9.39% |
| 🥉 3 | Gradient Boosting | 1.098 | 1.466 | 9.65% |
| 4 | ARIMA(1,1,1) | 3.351 | 4.096 | 37.83% |

> ✅ **Best Model: Holt-Winters** with MAPE = **9.01%** and MAE = **1.037 mg/L**

The three seasonal-aware models all performed within a narrow band (~9% MAPE), confirming that capturing the annual cycle is the single most critical factor for accurate O₂ forecasting. The non-seasonal ARIMA baseline failed significantly, underscoring the importance of the seasonal component.

---

## 🔭 24-Month Forecast (2021–2023)

Using the best model (Holt-Winters), a 24-month forward forecast was produced with 95% confidence intervals:

| Date | Forecast (mg/L) | Lower 95% CI | Upper 95% CI |
|---|---|---|---|
| 2021-03 | 13.840 | 10.104 | 17.576 |
| 2021-07 (summer min) | 7.893 | 3.939 | 11.847 |
| 2022-01 (winter peak) | 14.041 | 9.974 | 18.109 |
| 2022-07 (summer min) | 7.892 | 3.666 | 12.118 |
| 2023-01 (winter peak) | 14.053 | 9.705 | 18.401 |
| 2023-02 | 14.330 | 9.962 | 18.698 |

O₂ levels are forecast to remain within environmentally safe ranges, with summer minima staying above ~7.9 mg/L through 2023.

---

## 🧪 Key Findings

- **Strong Seasonality** — Seasonal strength score of **0.697 (Strong)**, with O₂ peaking in winter (cold water holds more oxygen) and dipping in summer
- **Moderate Downward Trend** — Trend strength of **0.314 (Moderate)**, consistent with gradual surface water warming over the 21-year period
- **Stationarity** — The original series is non-stationary (ADF p = 0.29); first differencing achieves stationarity (ADF p ≈ 0.00)
- **ACF/PACF** — Clear annual seasonality at lag 12, motivating SARIMA with s=12
- **Seasonality is Critical** — The gap between ARIMA (37.83% MAPE) and SARIMA (9.39% MAPE) confirms that ignoring seasonality renders forecasts unreliable

---

## 🛠️ Tech Stack

```python
pandas        # Data manipulation & resampling
numpy         # Numerical operations
matplotlib    # Time series & decomposition plots
seaborn       # Correlation heatmap & box plots
statsmodels   # ARIMA, SARIMA, Holt-Winters, ADF/KPSS, ACF/PACF
scikit-learn  # Gradient Boosting, imputation, metrics
scipy         # Statistical utilities
kagglehub     # Dataset download
```

**Environment:** Google Colab with TPU v5e accelerator

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/MuhammadUsman-Khan/Time-Series-Water-Quality-Forecasting.git
cd Time-Series-Water-Quality-Forecasting
```

### 2. Install Dependencies
```bash
pip install pandas numpy matplotlib seaborn statsmodels scikit-learn scipy kagglehub
```

### 3. Run the Notebook
Open `Time_Series_Water_Quality_Forecasting.ipynb` in Jupyter or Google Colab.

The dataset is automatically downloaded from Kaggle via `kagglehub` — no manual download needed.

```python
import kagglehub
path = kagglehub.dataset_download("vbmokin/wq-southern-bug-river-01052021")
```

> ⚠️ You will need a Kaggle account and API credentials configured for `kagglehub` to work.

---

## 📁 Repository Structure

```
Time-Series-Water-Quality-Forecasting/
│
├── Time_Series_Water_Quality_Forecasting.ipynb   # Main notebook
└── README.md                                      # Project documentation
```

---

## 🔮 Future Improvements

- [ ] Extend analysis to all 22 monitoring stations for spatial comparison
- [ ] Incorporate exogenous variables (water temperature, flow rate) via SARIMAX
- [ ] Evaluate deep learning models (LSTM, Temporal Convolutional Networks)
- [ ] Build an ensemble forecasting framework to reduce confidence interval width
- [ ] Deploy a lightweight forecasting dashboard using Streamlit

---


## 🙋‍♂️ Author

**Muhammad Usman Khan**

[![GitHub](https://img.shields.io/badge/GitHub-MuhammadUsman--Khan-181717?style=flat-square&logo=github)](https://github.com/MuhammadUsman-Khan)

---

<div align="center">
  <i>If you found this project useful, consider giving it a ⭐ — it helps others discover it!</i>
</div>
