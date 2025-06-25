# 📈 Malaysia Stock Forecast & Comparison Dashboard
# 最终整合版：包含 Forecast 模式 与 Comparison 模式

import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datetime import datetime, timedelta
import io

# 页面配置
st.set_page_config(layout="wide")
st.title("📈 Malaysia Stock Forecast & Comparison Dashboard")

st.markdown("""
This dashboard allows users to select up to 5 Malaysian stocks to view their historical trends 
from the past 6 months and forecast the next 1 month using a machine learning model (Random Forest).

It is designed specifically for **short-term investors** to observe market momentum and make data-driven decisions.
""")

# 新增一次性误差说明
st.markdown("<sub>📌 MAE measures average prediction error, while RMSE penalizes larger errors more. Lower values indicate better performance.</sub>", unsafe_allow_html=True)

# 股票列表
stock_dict = {
    "Telekom Malaysia": "4863.KL",
    "CelcomDigi": "6947.KL",
    "Axiata": "6888.KL",
    "Maxis": "6012.KL",
    "YTL Power": "6742.KL",
    "Maybank": "1155.KL",
    "Public Bank": "1295.KL",
    "Petronas Chemicals": "5183.KL",
    "Tenaga Nasional": "5347.KL",
    "Nestle": "4707.KL"
}

# 时间设置
end_date = datetime.today()
plot_start_date = end_date - timedelta(days=180)  # 显示过去6个月
future_days = 30

# 页面模式选择
mode = st.sidebar.selectbox(
    "Select Display Mode",
    ["Stock Forecast", "Comparison Graph"]
)

# ==== Comparison Graph 模式 ====
if mode == "Comparison Graph":
    st.header("📊 Stock Comparison Graph")

    col1, col2 = st.columns(2)
    with col1:
        stock1 = st.selectbox("Select Stock 1", list(stock_dict.keys()), index=0)
    with col2:
        stock2 = st.selectbox("Select Stock 2", list(stock_dict.keys()), index=1)

    if stock1 == stock2:
        st.warning("Please select two different stocks for comparison.")
    else:
        df1 = yf.download(stock_dict[stock1], start=plot_start_date, end=end_date)
        df2 = yf.download(stock_dict[stock2], start=plot_start_date, end=end_date)

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(df1.index, df1['Close'], label=stock1, color='blue')
        ax.plot(df2.index, df2['Close'], label=stock2, color='orange')

        ax.set_title(f"Closing Price Comparison: {stock1} vs {stock2}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Closing Price")
        ax.legend()
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        st.image(buf)

        price1 = float(df1['Close'].iloc[-1])
        price2 = float(df2['Close'].iloc[-1])
        st.caption(f"{stock1} Closing Price: RM {price1:.2f}")
        st.caption(f"{stock2} Closing Price: RM {price2:.2f}")

# ==== Forecast 模式 ====
elif mode == "Stock Forecast":
    st.header("📈 Stock Forecast Mode")

    with st.sidebar:
        selected_stocks = st.multiselect(
            "Choose up to 5 stocks:",
            options=list(stock_dict.keys()),
            default=list(stock_dict.keys())[:5]
        )

        train_months = st.slider(
            "Select training data range (in months):",
            min_value=6,
            max_value=24,
            value=12,
            step=1
        )

        view_option = st.radio(
            "Forecast Mode:",
            ["Only Historical", "Historical + Prediction"]
        )

        feature_option = st.radio(
            "Feature Set:",
            ["Lag1 Only", "MA7 + MA14 + Lag1"],
            index=1
        )

    train_start_date = end_date - timedelta(days=train_months * 30)

    if len(selected_stocks) == 0:
        st.warning("⚠️ Please select at least one stock.")
        st.stop()
    elif len(selected_stocks) > 5:
        st.warning("⚠️ You can select up to 5 stocks only.")
        st.stop()

    row1_stocks = selected_stocks[:3]
    row2_stocks = selected_stocks[3:]

    def plot_stock_chart(name, ticker):
        df = yf.download(ticker, start=train_start_date, end=end_date)
        if df.empty:
            st.warning(f"{name} - no data available.")
            return

        plot_df = df[df.index >= plot_start_date]

        fig, ax = plt.subplots(figsize=(6, 4.5))
        ax.plot(plot_df['Close'], label='Historical Close', color='blue')

        if view_option == "Historical + Prediction":
            df['Lag1'] = df['Close'].shift(1)
            if feature_option == "MA7 + MA14 + Lag1":
                df['MA7'] = df['Close'].rolling(window=7).mean()
                df['MA14'] = df['Close'].rolling(window=14).mean()
                X = df[['MA7', 'MA14', 'Lag1']]
            else:
                X = df[['Lag1']]

            df.dropna(inplace=True)
            y = df['Close']
            X = X.loc[df.index]

            X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)

            latest_data = X.tail(1).values
            preds = []
            current = latest_data
            for _ in range(future_days):
                pred = model.predict(current)[0]
                preds.append(pred)
                if feature_option == "MA7 + MA14 + Lag1":
                    next_row = [pred, (pred + current[0][1]) / 2, current[0][0]]
                else:
                    next_row = [pred]
                current = [next_row]

            future_dates = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=future_days)
            pred_df = pd.DataFrame({'Date': future_dates, 'Predicted_Price': preds}).set_index('Date')
            ax.plot(pred_df.index, pred_df['Predicted_Price'], label='Predicted Close', color='green')

            mae = mean_absolute_error(y_test, model.predict(X_test))
            rmse = mean_squared_error(y_test, model.predict(X_test)) ** 0.5

        ax.set_title(name)
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend()
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        st.image(buf)

        closing_price = float(df['Close'].iloc[-1])
        st.caption(f"Closing Price: RM {closing_price:.2f}")

        if view_option == "Historical + Prediction":
            st.caption(f"📉 MAE (Mean Absolute Error): {mae:.4f}")
            st.caption(f"📈 RMSE (Root Mean Squared Error): {rmse:.4f}")

    row1 = st.columns(len(row1_stocks))
    for idx, name in enumerate(row1_stocks):
        with row1[idx]:
            plot_stock_chart(name, stock_dict[name])

    if row2_stocks:
        row2 = st.columns(len(row2_stocks))
        for idx, name in enumerate(row2_stocks):
            with row2[idx]:
                plot_stock_chart(name, stock_dict[name])
