
# Import necessary libraries
import streamlit as st
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Function to evaluate models
def evaluate_model(true, predicted):
    mae = mean_absolute_error(true, predicted)
    mse = mean_squared_error(true, predicted)
    rmse = np.sqrt(mse)
    r2_square = r2_score(true, predicted)
    return mae, rmse, r2_square

# Streamlit app
st.title("Student Performance Analysis")

# Upload dataset
uploaded_file = st.file_uploader("student-mat.csv", type=["csv"])
if uploaded_file:
    import pandas as pd
    data = pd.read_csv(uploaded_file)
    st.write("Dataset Preview:")
    st.dataframe(data.head())

    # Select target variable
    target = st.selectbox("G3:", data.columns)
    if target:
        X = data.drop(target, axis=1).values
        y = data[target].values

        # Preprocessing
        st.subheader("Data Preprocessing")
        cat_indices = st.multiselect("Select categorical columns (indices):", list(range(X.shape[1])))
        num_indices = [i for i in range(X.shape[1]) if i not in cat_indices]

        X_cat = X[:, cat_indices]
        X_num = X[:, num_indices].astype(float)

        oh_encoder = OneHotEncoder(sparse_output=False)
        X_cat_encoded = oh_encoder.fit_transform(X_cat)

        scaler = StandardScaler()
        X_num_scaled = scaler.fit_transform(X_num)

        X_preprocessed = np.hstack([X_cat_encoded, X_num_scaled])

        # Splitting the data
        X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y, test_size=0.2, random_state=42)

        st.write(f"Training set shape: {X_train.shape}, Test set shape: {X_test.shape}")

        # Model Training and Evaluation
        st.subheader("Model Training and Evaluation")
        models = {
            "Linear Regression": LinearRegression(),
            "Lasso": Lasso(),
            "Ridge": Ridge(),
            "K-Neighbors Regressor": KNeighborsRegressor(),
            "Decision Tree": DecisionTreeRegressor(),
            "Random Forest Regressor": RandomForestRegressor(),
            "CatBoost Regressor": CatBoostRegressor(verbose=False),
            "AdaBoost Regressor": AdaBoostRegressor()
        }

        model_performance = []

        for model_name, model in models.items():
            model.fit(X_train, y_train)
            y_test_pred = model.predict(X_test)

            mae, rmse, r2 = evaluate_model(y_test, y_test_pred)
            model_performance.append({
                "Model": model_name,
                "MAE": mae,
                "RMSE": rmse,
                "R2 Score": r2
            })

        # Display model performances
        performance_df = pd.DataFrame(model_performance)
        st.write("Model Performance:")
        st.dataframe(performance_df)

        # Visualization
        st.subheader("Performance Visualization")
        st.bar_chart(performance_df.set_index("Model")[["R2 Score"]])
