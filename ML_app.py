import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

from prophet import Prophet

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
try:
    from langchain_ollama import ChatOllama
    OLLAMA_AVAILABLE = True
except Exception:
    # On Streamlit Cloud, Ollama usually isn't available
    OLLAMA_AVAILABLE = False

import streamlit as st

# -----------------------------------------------------------
# Streamlit basic config
# -----------------------------------------------------------
st.set_page_config(
    page_title="Social Media Engagement Forecasting",
    page_icon="📈",
    layout="centered"
)

# -----------------------------------------------------------
# Plot style
# -----------------------------------------------------------
sns.set_style("dark")
sns.set(rc={"axes.facecolor": "#A3FFD6", "figure.facecolor": "#7BC9FF"})
palette = ["#402E7A", "#4C3BCF", "#4B70F5", "#3DC2EC"]

# -----------------------------------------------------------
# Data loading & preprocessing (CACHED)
# -----------------------------------------------------------
@st.cache_data
def load_and_prepare_data():
    # Load dataset
    df = pd.read_excel("social_media_engagement_data.xlsx")

    # Drop unnecessary columns
    drop_cols = [col for col in ["Campaign ID", "Sentiment", "Influencer ID"] if col in df.columns]
    if drop_cols:
        df.drop(columns=drop_cols, inplace=True)

    # Convert timestamp and drop rows with missing essential values
    df["Post Timestamp"] = pd.to_datetime(df["Post Timestamp"], errors="coerce")
    df = df.dropna(subset=["Post Timestamp", "Engagement Rate"])

    # Date-based features
    df["Year"] = df["Post Timestamp"].dt.year
    df["Month"] = df["Post Timestamp"].dt.month
    df["Day"] = df["Post Timestamp"].dt.day

    # Monthly engagement per platform
    monthly_engagement = (
        df.groupby(["Platform", "Year", "Month"])["Engagement Rate"]
        .mean()
        .reset_index()
        .sort_values(["Platform", "Year", "Month"])
    )

    monthly_engagement["Date"] = pd.to_datetime(
        monthly_engagement["Year"].astype(str)
        + "-"
        + monthly_engagement["Month"].astype(str)
        + "-01"
    )

    # Copy for ML
    data_ml = df.copy()
    data_ml["Post Timestamp"] = pd.to_datetime(data_ml["Post Timestamp"], errors="coerce")
    data_ml["Year"] = data_ml["Post Timestamp"].dt.year
    data_ml["Month"] = data_ml["Post Timestamp"].dt.month

    # Drop rows with missing target or important features
    data_ml = data_ml.dropna(subset=["Engagement Rate"])

    # Encode categorical variables
    le = LabelEncoder()
    for col in ["Platform", "Post Type"]:
        data_ml[col] = le.fit_transform(data_ml[col].astype(str))

    features = [
        "Platform",
        "Post Type",
        "Year",
        "Month",
        "Likes",
        "Comments",
        "Shares",
        "Impressions",
        "Reach",
    ]
    target = "Engagement Rate"

    X = data_ml[features]
    y = data_ml[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return df, monthly_engagement, data_ml, X_train, X_test, y_train, y_test, features


df, monthly_engagement, data_ml, X_train, X_test, y_train, y_test, features = load_and_prepare_data()

# -----------------------------------------------------------
# Model training (CACHED)
# -----------------------------------------------------------
@st.cache_resource
def train_all_models(X_train, y_train, X_test, y_test):
    results = {}

    # XGBoost
    xgb_advanced = XGBRegressor(
        n_estimators=300,          # Slightly reduced for speed
        learning_rate=0.05,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )
    xgb_advanced.fit(X_train, y_train)
    y_pred_xgb = xgb_advanced.predict(X_test)
    results["XGBoost"] = {
        "model": xgb_advanced,
        "mae": mean_absolute_error(y_test, y_pred_xgb),
        "r2": r2_score(y_test, y_pred_xgb),
    }

    # Random Forest
    rfr = RandomForestRegressor(n_estimators=50, random_state=42)
    rfr.fit(X_train, y_train)
    y_pred_rfr = rfr.predict(X_test)
    results["Random Forest"] = {
        "model": rfr,
        "mae": mean_absolute_error(y_test, y_pred_rfr),
        "r2": r2_score(y_test, y_pred_rfr),
    }

    # Gradient Boosting
    gbr = GradientBoostingRegressor(n_estimators=100, random_state=42)
    gbr.fit(X_train, y_train)
    y_pred_gbr = gbr.predict(X_test)
    results["Gradient Boosting"] = {
        "model": gbr,
        "mae": mean_absolute_error(y_test, y_pred_gbr),
        "r2": r2_score(y_test, y_pred_gbr),
    }

    # Decision Tree
    dtr = DecisionTreeRegressor(max_depth=15, random_state=42)
    dtr.fit(X_train, y_train)
    y_pred_dtr = dtr.predict(X_test)
    results["Decision Tree"] = {
        "model": dtr,
        "mae": mean_absolute_error(y_test, y_pred_dtr),
        "r2": r2_score(y_test, y_pred_dtr),
    }

    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    results["Linear Regression"] = {
        "model": lr,
        "mae": mean_absolute_error(y_test, y_pred_lr),
        "r2": r2_score(y_test, y_pred_lr),
    }

    return results


model_results = train_all_models(X_train, y_train, X_test, y_test)

# -----------------------------------------------------------
# Prophet model per platform (CACHED)
# -----------------------------------------------------------
@st.cache_resource
def fit_prophet_for_platform(monthly_engagement, platform: str):
    data = monthly_engagement[monthly_engagement["Platform"] == platform][
        ["Date", "Engagement Rate"]
    ]
    data = data.rename(columns={"Date": "ds", "Engagement Rate": "y"})

    model = Prophet(seasonality_mode="additive", yearly_seasonality=True)
    model.fit(data)

    future = model.make_future_dataframe(periods=12, freq="M")
    forecast = model.predict(future)

    return model, forecast


# -----------------------------------------------------------
# Sidebar Navigation
# -----------------------------------------------------------
nav = st.sidebar.radio(
    "Menu",
    ["Dashboard", "Prediction", "Advanced Application Forecasting", "Chat Bot"],
)

# -----------------------------------------------------------
# Dashboard
# -----------------------------------------------------------
if nav == "Dashboard":
    st.markdown(
        "<h3 style='text-align: center; color: white; font-size: 30px;'>Data Dashboard</h3>",
        unsafe_allow_html=True,
    )

    st.write(data_ml.head(5))

    # Engagement over time
    plt.figure(figsize=(12, 3))
    sns.lineplot(
        data=monthly_engagement,
        x="Year",
        y="Engagement Rate",
        ci=None,
        marker="o",
    )
    st.markdown(
        "<h3 style='text-align: center; color: white; font-size: 20px;'>🕒 Engagement Rate over Time</h3>",
        unsafe_allow_html=True,
    )
    st.pyplot()

    st.markdown("<hr style='border:2px solid #bbb;'>", unsafe_allow_html=True)

    # Moving average
    monthly_engagement["Monthly_Engagement_RA"] = (
        monthly_engagement["Engagement Rate"].rolling(window=3).mean()
    )
    plt.figure(figsize=(12, 3))
    sns.lineplot(
        data=monthly_engagement,
        x="Year",
        y="Monthly_Engagement_RA",
        ci=None,
        marker="o",
    )
    st.markdown(
        "<h3 style='text-align: center; color: white; font-size: 20px;'>📈 Moving Average</h3>",
        unsafe_allow_html=True,
    )
    st.pyplot()

    st.markdown("<hr style='border:2px solid #bbb;'>", unsafe_allow_html=True)

    # Monthly trend
    st.markdown(
        "<h3 style='text-align: center; color: white; font-size: 20px;'>📈 Monthly Trend of Engagement Rate</h3>",
        unsafe_allow_html=True,
    )
    plt.figure(figsize=(12, 3))
    sns.lineplot(
        data=monthly_engagement,
        x="Month",
        y="Engagement Rate",
        ci=None,
        marker="o",
    )
    st.pyplot()

# -----------------------------------------------------------
# Prediction tab
# -----------------------------------------------------------
elif nav == "Prediction":
    st.markdown(
        "<h3 style='text-align: center; color: white; font-size: 30px;'>Prediction</h3>",
        unsafe_allow_html=True,
    )

    st.markdown("### 📊 Model Performance Comparison")

    model_col, mae_col, r2_col = st.columns([2, 1, 1])

    model_col.write("**Model**")
    mae_col.write("**📉 MAE**")
    r2_col.write("**📈 R² Score**")

    icons = {
        "XGBoost": "🚀",
        "Random Forest": "🌲",
        "Gradient Boosting": "🔥",
        "Decision Tree": "🌳",
        "Linear Regression": "📏",
    }

    for name, res in model_results.items():
        model_col.write(f"{icons.get(name, '')} {name}")
        mae_col.write(f"{res['mae']:.3f}")
        r2_col.write(f"{res['r2']:.3f}")

    # Feature Importance from XGBoost
    st.markdown("### 📝 Feature Importance (XGBoost)")
    xgb_model = model_results["XGBoost"]["model"]

    fi = pd.DataFrame(
        data=xgb_model.feature_importances_,
        index=xgb_model.feature_names_in_,
        columns=["Importance"],
    )

    fi = fi.sort_values("Importance")
    plt.figure(figsize=(8, 4))
    fi.plot(kind="barh", legend=False)
    plt.xlabel("Importance")
    plt.tight_layout()
    st.pyplot(plt.gcf())

# -----------------------------------------------------------
# Advanced Application Forecasting (Prophet)
# -----------------------------------------------------------
elif nav == "Advanced Application Forecasting":
    st.markdown(
        "<h3 style='text-align: center; color: white; font-size: 30px;'>Advanced Application Forecasting</h3>",
        unsafe_allow_html=True,
    )

    platform = st.selectbox("Select Platform", ["Instagram", "Twitter", "Facebook"])

    model, forecast = fit_prophet_for_platform(monthly_engagement, platform)

    fig = model.plot(forecast)
    plt.title(f"Engagement Forecast for {platform}")
    plt.xlabel("Date")
    plt.ylabel("Predicted Engagement Rate")
    st.pyplot(fig)

# -----------------------------------------------------------
# Chat Bot (with safe fallback)
# -----------------------------------------------------------
elif nav == "Chat Bot":
    st.title("🤖 Chatbot")

    if not OLLAMA_AVAILABLE:
        st.warning(
            "Ollama integration is not available in this environment. "
            "You can still show this tab for your project explanation, "
            "but the live chatbot won't run on Streamlit Cloud."
        )
    else:
        # initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []
            st.session_state.messages.append(
                SystemMessage("Act like a time series forecasting helper.")
            )

        # display chat messages from history
        for message in st.session_state.messages:
            if isinstance(message, HumanMessage):
                with st.chat_message("user"):
                    st.markdown(message.content)
            elif isinstance(message, AIMessage):
                with st.chat_message("assistant"):
                    st.markdown(message.content)

        # chat input
        prompt = st.chat_input("Ask me anything about time series forecasting...")

        if prompt:
            with st.chat_message("user"):
                st.markdown(prompt)
            st.session_state.messages.append(HumanMessage(prompt))

            try:
                llm = ChatOllama(
                    model="llama3.2",
                    temperature=2,
                )
                result = llm.invoke(st.session_state.messages).content

                with st.chat_message("assistant"):
                    st.markdown(result)
                st.session_state.messages.append(AIMessage(result))
            except Exception as e:
                st.error(
                    "Chatbot is not responding in this deployment environment. "
                    f"Details: {e}"
                )
