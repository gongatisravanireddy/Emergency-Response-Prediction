import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans

def load_data():
    return pd.read_csv("dataset.csv")

def preprocess(df):
    df = df[['State/UT/City',
             'Total Traffic Accidents - Cases',
             'Total Traffic Accidents - Injured',
             'Total Traffic Accidents - Died']]

    df = df.dropna()

    le = LabelEncoder()
    df['State'] = le.fit_transform(df['State/UT/City'])

    return df

def train_model(df):
    X = df[['Total Traffic Accidents - Cases', 'Total Traffic Accidents - Injured']]
    y = df['Total Traffic Accidents - Died']

    model = RandomForestRegressor(n_estimators=100)
    model.fit(X, y)

    return model

def clustering(df):
    kmeans = KMeans(n_clusters=3, random_state=0)
    df['Cluster'] = kmeans.fit_predict(
        df[['Total Traffic Accidents - Cases', 'Total Traffic Accidents - Died']]
    )
    return df

def risk_analysis(df):
    def risk(row):
        if row['Total Traffic Accidents - Died'] > 8000:
            return "HIGH RISK 🚨"
        elif row['Total Traffic Accidents - Died'] > 3000:
            return "MEDIUM RISK ⚠️"
        else:
            return "LOW RISK ✅"

    df['Risk'] = df.apply(risk, axis=1)
    return df[['State/UT/City', 'Risk']]