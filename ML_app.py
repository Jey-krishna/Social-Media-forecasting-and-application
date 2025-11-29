import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from prophet import Prophet
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
import streamlit as st
st.set_option('deprecation.showPyplotGlobalUse', False)

st.set_page_config(
    page_title="Social Media Engagnement Forcasting",
    page_icon='📈',
    layout='centered'
    )

sns.set_style("dark")
sns.set(rc={"axes.facecolor":"#A3FFD6","figure.facecolor":"#7BC9FF"})
#sns.set_context("notebook",font_scale = .7)

palette = ['#402E7A','#4C3BCF','#4B70F5','#3DC2EC']

df = pd.read_excel('social_media_engagement_data.xlsx')
df.drop(columns=['Campaign ID', 'Sentiment', 'Influencer ID'], inplace= True)

df["Post Timestamp"] = pd.to_datetime(df["Post Timestamp"], errors='coerce')

df = df.dropna(subset=["Post Timestamp", "Engagement Rate"])

df["Year"] = df["Post Timestamp"].dt.year
df["Month"] = df["Post Timestamp"].dt.month
df["Day"] = df["Post Timestamp"].dt.day

monthly_engagement = (
    df.groupby(["Platform", "Year", "Month"])["Engagement Rate"]
    .mean()
    .reset_index()
    .sort_values(["Platform", "Year", "Month"])
)

monthly_engagement["Date"] = pd.to_datetime(
    monthly_engagement["Year"].astype(str) + "-" + monthly_engagement["Month"].astype(str) + "-01"
)


data_ml = df.copy()

# Convert timestamp
data_ml["Post Timestamp"] = pd.to_datetime(data_ml["Post Timestamp"], errors="coerce")
data_ml["Year"] = data_ml["Post Timestamp"].dt.year
data_ml["Month"] = data_ml["Post Timestamp"].dt.month

features = ["Platform", "Post Type", "Year", "Month", "Likes", "Comments", "Shares", "Impressions", "Reach"]
target = "Engagement Rate"

# Drop NaNs
data_ml.dropna()

le = LabelEncoder()
for col in ["Platform", "Post Type"]:
    data_ml[col] = le.fit_transform(data_ml[col].astype(str))

# Split data
X = data_ml[features]
y = data_ml[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



nav = st.sidebar.radio('Menu', ['Dashboard', 'Prediction', 'Advanced Application Forcasting', 'Chat Bot'])
if nav == 'Dashboard':
    st.markdown(
        "<h3 style='text-align: center; color: white; font-size: 30px;'>Data Dashboard</h3>",
        unsafe_allow_html=True
    )
    st.write(data_ml.head(5))
    plt.figure(figsize=(12,3))
    sns.lineplot(data= monthly_engagement, x=  'Year', y= 'Engagement Rate', ci = None, marker = 'o')
    st.markdown(
        "<h3 style='text-align: center; color: white; font-size: 20px;'>🕒 Engagement Rate over Time</h3>",
        unsafe_allow_html=True
    )
    st.pyplot()

    st.markdown(
    "<hr style='border:2px solid #bbb;'>",
    unsafe_allow_html=True
    )

    monthly_engagement['Monthly_engegement_RA'] = monthly_engagement['Engagement Rate'].rolling(window=3).mean()
    sns.lineplot(data= monthly_engagement, x=  'Year', y= 'Monthly_engegement_RA', ci = None, c = 'red', marker = 'o')
    st.markdown(
        "<h3 style='text-align: center; color: white; font-size: 20px;'>📈 Moving Average</h3>",
        unsafe_allow_html=True
    )
    st.pyplot()

    st.markdown(
    "<hr style='border:2px solid #bbb;'>",
    unsafe_allow_html=True
    )

    st.markdown(
        "<h3 style='text-align: center; color: white; font-size: 20px;'>📈 Monthly trend of Engagement rate</h3>",
        unsafe_allow_html=True
    )
    plt.figure(figsize=(12,3))
    sns.lineplot(data=monthly_engagement, x = 'Month', y = 'Engagement Rate', ci = None,c = 'green', marker = 'o')
    st.pyplot()

if nav == 'Prediction':
    st.markdown(
        "<h3 style='text-align: center; color: white; font-size: 30px;'>Prediction</h3>",
        unsafe_allow_html=True
    )
    
    xgb_advanced = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    # ✅ Train all models
    xgb_advanced.fit(X_train, y_train)
    y_pred_xgb = xgb_advanced.predict(X_test)

    rfr = RandomForestRegressor(n_estimators=10, random_state=42)
    rfr.fit(X_train, y_train)
    y_pred_rfr = rfr.predict(X_test)

    gbr = GradientBoostingRegressor(n_estimators=100, random_state=42)
    gbr.fit(X_train, y_train)
    y_pred_gbr = gbr.predict(X_test)

    dtr = DecisionTreeRegressor(max_depth=15, min_samples_split=2, random_state=42)
    dtr.fit(X_train, y_train)
    y_pred_dtr = dtr.predict(X_test)

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)

    # ✅ Calculate metrics
    mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
    r2_xgb = r2_score(y_test, y_pred_xgb)

    mae_rf = mean_absolute_error(y_test, y_pred_rfr)
    r2_rf = r2_score(y_test, y_pred_rfr)

    mae_gbr = mean_absolute_error(y_test, y_pred_gbr)
    r2_gbr = r2_score(y_test, y_pred_gbr)

    mae_dtr = mean_absolute_error(y_test, y_pred_dtr)
    r2_dtr = r2_score(y_test, y_pred_dtr)

    mae_lr = mean_absolute_error(y_test, y_pred_lr)
    r2_lr = r2_score(y_test, y_pred_lr)

    # ✅ Streamlit Display
    st.markdown("### 📊 Model Performance Comparison")

    model_col, mae_col, r2_col = st.columns([2, 1, 1])

    # Table Header
    model_col.write("**Model**")
    mae_col.write("**📉 MAE**")
    r2_col.write("**📈 R² Score**")

    # XGBoost
    model_col.write("🚀 XGBoost")
    mae_col.write(f"{mae_xgb:.3f}")
    r2_col.write(f"{r2_xgb:.3f}")

    # Random Forest
    model_col.write("🌲 Random Forest")
    mae_col.write(f"{mae_rf:.3f}")
    r2_col.write(f"{r2_rf:.3f}")

    # Gradient Boosting
    model_col.write("🔥 Gradient Boosting")
    mae_col.write(f"{mae_gbr:.3f}")
    r2_col.write(f"{r2_gbr:.3f}")

    # Decision Tree
    model_col.write("🌳 Decision Tree")
    mae_col.write(f"{mae_dtr:.3f}")
    r2_col.write(f"{r2_dtr:.3f}")

    # Linear Regression
    model_col.write("📏 Linear Regression")
    mae_col.write(f"{mae_lr:.3f}")
    r2_col.write(f"{r2_lr:.3f}")



    st.markdown("### 📝 Featre Importance Graph by using XGBoost")
    fi = pd.DataFrame(data = xgb_advanced.feature_importances_, index= xgb_advanced.feature_names_in_, columns= ['Importance'])
    fi.sort_values('Importance').plot(kind='barh')
    
    st.pyplot()


if nav == 'Advanced Application Forcasting':

    platform = st.selectbox('Select Platform', ['Instagram', 'Twitter', 'Facebook'])
    data = monthly_engagement[monthly_engagement["Platform"] == platform][["Date", "Engagement Rate"]]
    data = data.rename(columns={"Date": "ds", "Engagement Rate": "y"})

    model = Prophet(seasonality_mode='additive', yearly_seasonality=True)
    model.fit(data)

    future = model.make_future_dataframe(periods=12, freq='M')
    forecast = model.predict(future)

    model.plot(forecast)
    plt.title(f" Engagement Forecast for {platform}")
    plt.xlabel("Date")
    plt.ylabel("Predicted Engagement Rate")
    st.pyplot()


if nav ==  'Chat Bot':
    st.title("🤖 Chatbot")

    # initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

        st.session_state.messages.append(SystemMessage("Act like an time series forcasting helper"))

    # display chat messages from history on app rerun
    for message in st.session_state.messages:
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.markdown(message.content)
        elif isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.markdown(message.content)

    # create the bar where we can type messages
    prompt = st.chat_input("How are you?")

    # did the user submit a prompt?
    if prompt:
        # add the message from the user (prompt) to the screen with streamlit
        with st.chat_message("user"):
            st.markdown(prompt)

            st.session_state.messages.append(HumanMessage(prompt))

        # create the echo (response) and add it to the screen

        llm = ChatOllama(
            model="llama3.2",
            temperature=2
        )

        result = llm.invoke(st.session_state.messages).content

        with st.chat_message("assistant"):
            st.markdown(result)

            st.session_state.messages.append(AIMessage(result))





    

