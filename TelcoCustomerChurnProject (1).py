# Customer Churn Prediction Project
# Import required libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import joblib
import streamlit as st

# Step 1: Problem Definition
# Goal: Predict customer churn (Yes/No) based on usage patterns and demographics

# Step 2: Load & Explore Data
def load_and_explore():
    # Load data
    df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
    
    # Initial exploration
    print("\nFirst 5 rows of data:")
    print(df.head())
    
    # Check missing values
    print("\nMissing values summary:")
    print(df.isnull().sum())
    
    # Visualize churn distribution
    plt.figure(figsize=(8, 5))
    sns.countplot(x='Churn', data=df)
    plt.title('Churn Distribution (Yes vs No)')
    plt.show()
    
    return df

# Step 3: Data Cleaning
def clean_data(df):
    # Drop irrelevant columns
    df.drop('customerID', axis=1, inplace=True)
    
    # Convert 'TotalCharges' to numeric (handling empty strings)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
    
    # Convert target to binary
    df['Churn'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)
    
    return df

# Step 4: Feature Engineering
def engineer_features(df):
    # Convert binary Yes/No to 1/0
    binary_cols = [
        'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling',
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies'
    ]
    for col in binary_cols:
        df[col] = df[col].apply(lambda x: 1 if x == 'Yes' else 0)
    
    # Convert gender to 1/0
    df['gender'] = df['gender'].apply(lambda x: 1 if x == 'Male' else 0)

    # One-hot encode remaining categoricals
    df = pd.get_dummies(df, columns=[
        'MultipleLines', 'InternetService', 'Contract', 'PaymentMethod'
    ], drop_first=True)
    
    # Create interaction feature
    df['TenureCharges'] = df['tenure'] * df['MonthlyCharges']
    
    # Scale numerical features
    scaler = StandardScaler()
    df[['tenure', 'MonthlyCharges', 'TotalCharges']] = scaler.fit_transform(
        df[['tenure', 'MonthlyCharges', 'TotalCharges']])
    
    return df

# Step 5: Model Building
def build_models(X_train, X_test, y_train, y_test):
    models = {
        "Logistic Regression": LogisticRegression(),
        "Random Forest": RandomForestClassifier()
    }
    
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        print(f"\n--- {name} ---")
        print(classification_report(y_test, y_pred))
        print(f"AUC-ROC: {roc_auc_score(y_test, y_pred):.2f}")
        
        results[name] = model
    
    return results

# Step 6: Hyperparameter Tuning
def tune_model(X_train, y_train):
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20],
        'min_samples_split': [2, 5]
    }
    
    grid = GridSearchCV(RandomForestClassifier(), param_grid, cv=5)
    grid.fit(X_train, y_train)
    
    print("\nBest parameters:", grid.best_params_)
    return grid.best_estimator_

# Step 7: Deployment (Streamlit App)
def create_app(model, X_train):
    st.title('Customer Churn Predictor')

    # Input widgets
    tenure = st.slider('Tenure (months)', 0, 100, 12)
    monthly_charges = st.number_input('Monthly Charges ($)', 0.0, 200.0, 70.0)
    contract = st.selectbox('Contract', ['Month-to-month', 'One year', 'Two year'])

    if st.button('Predict Churn'):
        contract_encoded = f"Contract_{contract.replace(' ', '')}"
        features = pd.DataFrame({
            'tenure': [tenure],
            'MonthlyCharges': [monthly_charges],
            **{f"Contract_{k}": 0 for k in ['Month-to-month', 'One year', 'Two year']},
            **{contract_encoded: 1}
        })
        
        # Ensure all columns from training are present
        missing_cols = set(X_train.columns) - set(features.columns)
        for col in missing_cols:
            features[col] = 0
            
        # Reorder columns to match training data
        features = features[X_train.columns]
        
        prediction = model.predict(features)[0]
        st.success(f"Prediction: {'Churn' if prediction == 1 else 'No Churn'}")

# Main execution
if __name__ == "__main__":
    # Load and explore data
    df = load_and_explore()
    
    # Clean data
    df = clean_data(df)
    
    # Feature engineering
    df = engineer_features(df)
    
    # Prepare data for modeling
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Build and evaluate models
    models = build_models(X_train, X_test, y_train, y_test)
    
    # Hyperparameter tuning
    best_model = tune_model(X_train, y_train)
    
    # Retrain model with the best parameters
    best_model.fit(X_train, y_train)
    
    # Save the best model
    joblib.dump(best_model, 'churn_model.pkl')
    
    # To run the Streamlit app:
    # 1. Save this code to a file (e.g., churn_prediction.py)
    # 2. Run in terminal: streamlit run churn_prediction.py
    # Note: The Streamlit app will need to be in a separate file or run conditionally
    # Uncomment the following line to run the app directly from this script:
    # create_app(best_model, X_train)
