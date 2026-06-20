import streamlit as st
import matplotlib.pyplot as plt
from model import load_data, preprocess, train_model, clustering, risk_analysis

st.set_page_config(page_title="Emergency Response System", layout="wide")

st.title("🚨 Emergency Response Optimization System")

# Load Data
df = load_data()
st.subheader("📊 Dataset")
st.dataframe(df)

# Preprocess
df = preprocess(df)

# Train Model
model = train_model(df)

# Sidebar Prediction
st.sidebar.header("🔍 Predict Accident Severity")

cases = st.sidebar.number_input("Total Cases", 0)
injured = st.sidebar.number_input("Total Injured", 0)

if st.sidebar.button("Predict Deaths"):
    prediction = model.predict([[cases, injured]])
    st.sidebar.success(f"Predicted Deaths: {prediction[0]:.2f}")

# Clustering
df = clustering(df)

st.subheader("📍 Risk Clusters")
fig, ax = plt.subplots()
ax.scatter(df['Total Traffic Accidents - Cases'],
           df['Total Traffic Accidents - Died'],
           c=df['Cluster'])
ax.set_xlabel("Cases")
ax.set_ylabel("Deaths")
st.pyplot(fig)

# Risk Analysis
st.subheader("🚨 Risk Levels")
risk = risk_analysis(df)
st.write(risk)

# Graph
st.subheader("📈 Accidents by State")
st.bar_chart(df.set_index('State/UT/City')['Total Traffic Accidents - Cases'])