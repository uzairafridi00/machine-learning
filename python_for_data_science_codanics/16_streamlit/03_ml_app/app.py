import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer  # Enable IterativeImputer

# Function to preprocess data
def preprocess_data(X, y, problem_type):
    # Fill missing values using Iterative Imputer
    imp = IterativeImputer()
    X_imputed = pd.DataFrame(imp.fit_transform(X), columns=X.columns)

    # Identify categorical columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns

    # Apply OneHotEncoder to categorical columns
    preprocessor = ColumnTransformer(
        transformers=[('encoder', OneHotEncoder(handle_unknown='ignore'), categorical_cols)],
        remainder='passthrough'
    )

    X_processed = preprocessor.fit_transform(X_imputed)

    # Scale numerical features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_processed)

    return X_scaled, y

# Function to train and evaluate models
def train_and_evaluate(X_train, X_test, y_train, y_test, model, problem_type):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    # Evaluation
    if problem_type == "Regression":
        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        return predictions, {"MSE": mse, "MAE": mae, "R² Score": r2}
    
    elif problem_type == "Classification":
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, average='weighted')
        recall = recall_score(y_test, predictions, average='weighted')
        f1 = f1_score(y_test, predictions, average='weighted')
        return predictions, {"Accuracy": accuracy, "Precision": precision, "Recall": recall, "F1 Score": f1}

# Main application function
def main():
    st.title("Machine Learning Application")
    st.write("Welcome to the machine learning application. Upload a dataset and train ML models.")

    # Data upload or example data selection
    data_source = st.sidebar.selectbox("Do you want to upload data or use example data?", ["Upload", "Example"])

    if data_source == "Upload":
        uploaded_file = st.sidebar.file_uploader("Choose a file", type=['csv', 'xlsx', 'tsv'])
        if uploaded_file is not None:
            if uploaded_file.type == "text/csv":
                data = pd.read_csv(uploaded_file)
            elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
                data = pd.read_excel(uploaded_file)
            elif uploaded_file.type == "text/tab-separated-values":
                data = pd.read_csv(uploaded_file, sep='\t')
    else:
        dataset_name = st.sidebar.selectbox("Select a dataset", ["titanic", "tips", "iris"])
        data = sns.load_dataset(dataset_name)

    if 'data' in locals() and not data.empty:
        st.write("### Data Preview")
        st.write(data.head())

        # Select features and target
        features = st.multiselect("Select feature columns", data.columns.tolist())
        target = st.selectbox("Select target column", data.columns.tolist())
        problem_type = st.selectbox("Problem Type", ["Classification", "Regression"])

        if features and target and problem_type:
            X = data[features]
            y = data[target]

            st.write(f"You have selected a **{problem_type}** problem.")

            # Button to start analysis
            if st.button("Run Analysis"):
                # Preprocess data
                X_processed, y_processed = preprocess_data(X, y, problem_type)

                # Train-test split
                test_size = st.slider("Select test split size", 0.1, 0.5, 0.2)
                X_train, X_test, y_train, y_test = train_test_split(X_processed, y_processed, test_size=test_size, random_state=42)

                # Model selection
                model_options = ['Linear Regression', 'Decision Tree', 'Random Forest', 'SVM'] if problem_type == 'Regression' else ['Decision Tree', 'Random Forest', 'SVM']
                selected_model = st.sidebar.selectbox("Select model", model_options)

                # Initialize model
                if selected_model == 'Linear Regression':
                    model = LinearRegression()
                elif selected_model == 'Decision Tree':
                    model = DecisionTreeRegressor() if problem_type == 'Regression' else DecisionTreeClassifier()
                elif selected_model == 'Random Forest':
                    model = RandomForestRegressor() if problem_type == 'Regression' else RandomForestClassifier()
                elif selected_model == 'SVM':
                    model = SVR() if problem_type == 'Regression' else SVC()

                # Train and evaluate model
                predictions, metrics = train_and_evaluate(X_train, X_test, y_train, y_test, model, problem_type)

                st.write("### Model Performance Metrics")
                st.write(metrics)

                # Download trained model
                model_filename = "trained_model.pkl"
                with open(model_filename, "wb") as f:
                    pickle.dump(model, f)

                st.download_button("Download Trained Model", data=open(model_filename, "rb"), file_name="trained_model.pkl", mime="application/octet-stream")

if __name__ == "__main__":
    main()
