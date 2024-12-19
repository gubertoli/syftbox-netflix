# Routines for ML
# Reference: https://syftbox.openmined.org/datasites/andrew@openmined.org/netflix_fl/example_job/job.py

import re
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

## ==================================================================================================
## Data Processing
## ==================================================================================================

def extract_features(df):
    # Extract show name and season from title
    df['show'] = df['Title'].apply(lambda x: x.split(':')[0] if ':' in x else x)
    df['season'] = df['Title'].apply(lambda x: 
        int(re.search(r'Season (\d+)', x).group(1)) if re.search(r'Season (\d+)', x) else 0)
    
    # Convert date strings to datetime objects
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
    
    # Extract temporal features
    df['day_of_week'] = df['Date'].dt.dayofweek
    # df['hour'] = df['Date'].dt.hour
    
    return df

def prepare_data(file_path):
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Process features
    df = extract_features(df)
    
    # Encode categorical variables
    le_show = LabelEncoder()
    df['show_encoded'] = le_show.fit_transform(df['show'])
    
    # Create feature matrix
    X = df[['show_encoded', 'season', 'day_of_week']].values
    
    # Create target variable (next show watched)
    y = df['show_encoded'].shift(-1).fillna(0).astype(int)
    
    return X[:-1], y[:-1], le_show


## ==================================================================================================
## Model Training
## ==================================================================================================

def train_model(dataset_location):
    # Load and prepare data
    X, y, le_show = prepare_data(dataset_location)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create and train the MLP
    mlp = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        max_iter=2000,
        random_state=42
    )
    
    mlp.fit(X_train_scaled, y_train)
    
    # Calculate accuracy
    train_accuracy = mlp.score(X_train_scaled, y_train)
    test_accuracy = mlp.score(X_test_scaled, y_test)
    
    print(f"Training accuracy: {train_accuracy:.2f}")
    print(f"Test accuracy: {test_accuracy:.2f}")

    # TODO: Train the model in the full data
    #   X_full = np.vstack((X_train, X_test))
    #   y_full = np.hstack((y_train, y_test))
    #   X_full_scaled = scaler.transform(X_full)
    #   mlp.fit(X_full_scaled, y_full)
    num_samples = X.shape[0]
    return mlp, scaler, le_show, num_samples

## ==================================================================================================
## Predictor Process
## ==================================================================================================
def train_and_save_mlp(latest_data_file, private_folder, restricted_public_folder):
    """
    Train the MLP model and save its weights and biases.

    Args:
        latest_data_file: Path to the latest data file.
        restricted_public_folder: Path to the restricted public folder.
    """
    # Train the MLP model
    mlp, _, _, num_samples = train_model(latest_data_file)

    # Define paths
    mlp_weights_file = restricted_public_folder / f"netflix_mlp_weights_{num_samples}.joblib"
    mlp_bias_file = restricted_public_folder / f"netflix_mlp_bias_{num_samples}.joblib"

    # Save MLP weights and biases
    joblib.dump(mlp.coefs_, str(mlp_weights_file))
    joblib.dump(mlp.intercepts_, str(mlp_bias_file))

## ==================================================================================================
## Inference Process
## ==================================================================================================

def get_current_day_of_week():
    """
    Returns the current day and day of the week.
    """
    current_date = datetime.now()
    return current_date, current_date.weekday()

def get_recommendation(mlp, scaler, le_show, last_watched):
    """
    Generates a show recommendation based on the last watched show.

    Args:
        mlp: Trained MLP model.
        scaler: Trained StandardScaler.
        le_show: Fitted LabelEncoder.
        last_watched: Title of the last watched show (e.g., "Show A: Season 1").

    Returns:
        Recommended show title.
    """
    # Extract features from last watched show
    show_name = last_watched.split(':')[0] if ':' in last_watched else last_watched
    season = int(re.search(r'Season (\d+)', last_watched).group(1)) if 'Season' in last_watched else 0
    _, day_of_week = get_current_day_of_week()  # Use helper function
    
    # Prepare input features
    show_encoded = le_show.transform([show_name])[0]
    features = np.array([[show_encoded, season, day_of_week]])
    features_scaled = scaler.transform(features)
    
    # Get prediction
    predicted_show_encoded = mlp.predict(features_scaled)[0]
    recommended_show = le_show.inverse_transform([predicted_show_encoded])[0]
    
    return recommended_show