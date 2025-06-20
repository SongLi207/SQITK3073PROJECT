# Malaysia Stock Forecast Dashboard

## 📌 Overview

This Streamlit-powered dashboard allows users to compare and forecast the performance of up to 5 selected Malaysian stocks. The application retrieves 3-month historical stock data via the Yahoo Finance API and provides optional 30-day price predictions using the Random Forest machine learning model.

Developed as part of a project for data analytics learning, this tool aims to make stock trend analysis intuitive and accessible.

---

## 🔍 Features

- 📈 **Historical Trends**: View closing prices of selected stocks for the past 3 months.
- 🔮 **Forecasting**: Optional future 30-day predictions using `RandomForestRegressor`.
- 🎨 **Responsive Layout**: Dynamic layout with consistent visual proportions (3 charts in row 1, 2 charts in row 2).
- 🧠 **ML-based Forecasting**: Feature engineering with moving averages and lagged values.
- 🌐 **Interactive Web Interface**: Built with Streamlit for easy use via browser.

---

## ⚙️ Installation & Setup

### 1. Clone or Download the Repository

```bash
git clone https://github.com/your-username/stock-forecast-dashboard.git
cd stock-forecast-dashboard
