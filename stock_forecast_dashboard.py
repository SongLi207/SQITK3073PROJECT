import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta
import io

# ==== Global Matplotlib Style 设置图表字体大小 ====
plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 9
})

# ==== 页面设置 / Page layout ====
st.set_page_config(layout="wide")
st.title("📈 Malaysia Stock Forecast & Trend Comparison Dashboard")

# ==== 页面说明 / Page Description ====
st.markdown("""
This dashboard allows users to select up to 5 Malaysian stocks to view their historical trends 
from the past 3 months and optionally forecast the next 30 days using a machine learning model (Random Forest).
""")

# ==== 股票池 / Stock List ====
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

# ==== 时间范围 / Time Range ====
end_date = datetime.today()
start_date = end_date - timedelta(days=90)
future_days = 30  # days to predict

# ==== Sidebar 控件 / Sidebar Controls ====
with st.sidebar:
    st.header("📋 Stock Selection & Options")

    selected_stocks = st.multiselect(
        "Choose up to 5 stocks:",
        options=list(stock_dict.keys()),
        default=list(stock_dict.keys())[:5]
    )

    view_option = st.radio(
        "Display Mode:",
        ["Only Historical", "Historical + Prediction"]
    )

# ==== 输入验证 / Validation ====
if len(selected_stocks) == 0:
    st.warning("⚠️ Please select at least one stock.")
    st.stop()
elif len(selected_stocks) > 5:
    st.warning("⚠️ You can select up to 5 stocks only.")
    st.stop()

# ==== 拆成两排 / Split to 2 Rows ====
row1_stocks = selected_stocks[:3]
row2_stocks = selected_stocks[3:]

# ==== 主函数：绘图 / Main Chart Function ====
def plot_stock_chart(name, ticker):
    df = yf.download(ticker, start=start_date, end=end_date)
    if df.empty:
        st.warning(f"{name} - no data available.")
        return

    fig, ax = plt.subplots(figsize=(6, 4.5))  # 固定图表大小
    ax.plot(df['Close'], label='Historical Close', color='blue')

    # ==== 如果选择预测，则训练模型 ====
    if view_option == "Historical + Prediction":
        df['MA7'] = df['Close'].rolling(window=7).mean()
        df['MA14'] = df['Close'].rolling(window=14).mean()
        df['Lag1'] = df['Close'].shift(1)
        df.dropna(inplace=True)

        X = df[['MA7', 'MA14', 'Lag1']]
        y = df['Close']

        X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        latest_data = X.tail(1).values
        preds = []
        current = latest_data

        for _ in range(future_days):
            pred = model.predict(current)[0]
            preds.append(pred)
            next_row = [pred, (pred + current[0][1]) / 2, current[0][0]]  # 构造下一天特征
            current = [next_row]

        future_dates = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=future_days)
        pred_df = pd.DataFrame({'Date': future_dates, 'Predicted_Price': preds}).set_index('Date')
        ax.plot(pred_df.index, pred_df['Predicted_Price'], label='Predicted Close', color='green')

    # ==== 美化图表 / Beautify Chart ====
    ax.set_title(name)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())

    # ==== 输出为图片，确保尺寸统一 / Save and display ====
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    st.image(buf)

# ==== 第一排图表 / First Row (Up to 3) ====
row1 = st.columns(len(row1_stocks))
for idx, name in enumerate(row1_stocks):
    with row1[idx]:
        plot_stock_chart(name, stock_dict[name])

# ==== 第二排图表 / Second Row (Up to 2) ====
if row2_stocks:
    row2 = st.columns(len(row2_stocks))
    for idx, name in enumerate(row2_stocks):
        with row2[idx]:
            plot_stock_chart(name, stock_dict[name])
