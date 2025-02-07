import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from io import StringIO, BytesIO

# Streamlit app title
st.title("Customer Churn Analysis App")

# File uploader
uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xls", "xlsx"])

if uploaded_file:
    # Read data based on file type
    if uploaded_file.name.endswith(".csv"):
        data = pd.read_csv(uploaded_file)
    else:
        data = pd.read_excel(uploaded_file)
    
    st.write("### Data Preview")
    st.dataframe(data.head())
    
    # Check for missing values
    if data.isnull().sum().sum() > 0:
        st.warning("Data contains missing values. Consider cleaning it.")
    
    # Select target variable (churn status assumed as 'Churn')
    if 'Churn' in data.columns:
        target_col = 'Churn'
    else:
        st.error("Dataset must contain a 'Churn' column.")
        st.stop()
    
    # Convert categorical columns to numerical
    categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
    data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)
    
    # Train-test split
    X = data.drop(columns=[target_col])
    y = data[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Display accuracy
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"### Model Accuracy: {accuracy:.2f}")
    
    # Show feature importance
    feature_importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    st.write("### Feature Importance")
    st.bar_chart(feature_importance)
    
    # Visualization
    st.write("### Churn Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x=y, ax=ax)
    st.pyplot(fig)
    
    st.write("### Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(data.corr(), annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
    st.pyplot(fig)
    
    # Classification Report
    st.write("### Classification Report")
    st.text(classification_report(y_test, y_pred))
