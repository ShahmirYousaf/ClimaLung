import numpy as np
import pandas as pd
import os
import requests
import tempfile
from tensorflow.keras.models import load_model
import joblib
from tqdm import tqdm
import atexit

# Temporary directory for models (gets cleaned up)
TEMP_MODEL_DIR = tempfile.mkdtemp(prefix='airquality_models_')
MODEL_CACHE = {}

# Google Drive file IDs
MODEL_URL_IDS = {
    'pm25_gru_model': '1FzLvttYlus2DiccsUMb3YSMN-9BJDKar',
    'feature_scaler': '1DIS4ZC0RB43ffD8s1t78VhInSgD69K80',
    'target_scaler': '1FTPLxn5ZIwMCq6jQvB7TxqWNnsmZ9iew',
    'feature_columns': '1S8ek6JBiPEgQhReVXwX43JO2PqVXztsG'
}

# Register cleanup function
def cleanup():
    """Remove temporary directory on exit"""
    import shutil
    try:
        shutil.rmtree(TEMP_MODEL_DIR)
        print(f"Cleaned up temporary directory: {TEMP_MODEL_DIR}")
    except:
        pass

atexit.register(cleanup)

def download_to_temp(file_id, filename):
    """Download file to temporary directory with progress"""
    URL = "https://docs.google.com/uc?export=download"
    temp_path = os.path.join(TEMP_MODEL_DIR, filename)
    
    if os.path.exists(temp_path):
        return temp_path
    
    session = requests.Session()
    
    # First request to get confirmation token
    response = session.get(URL, params={'id': file_id}, stream=True)
    token = None
    
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            token = value
            break
    
    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)
    
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024 * 1024  # 1MB chunks
    
    with open(temp_path, 'wb') as f:
        for chunk in tqdm(response.iter_content(block_size),
                        total=total_size//block_size,
                        unit='MB',
                        unit_scale=True,
                        desc=f'Downloading {filename}'):
            if chunk:
                f.write(chunk)
    
    if total_size != 0 and os.path.getsize(temp_path) != total_size:
        raise Exception("Download incomplete")
    
    return temp_path

def load_models():
    """Load models into memory (cached for performance)"""
    if MODEL_CACHE:
        return MODEL_CACHE
    
    try:
        MODEL_CACHE['pm25_gru_model'] = load_model(
            download_to_temp(MODEL_URL_IDS['pm25_gru_model'], 'pm25_gru_model.h5')
        )
        MODEL_CACHE['feature_scaler'] = joblib.load(
            download_to_temp(MODEL_URL_IDS['feature_scaler'], 'feature_scaler.pkl')
        )
        MODEL_CACHE['target_scaler'] = joblib.load(
            download_to_temp(MODEL_URL_IDS['target_scaler'], 'target_scaler.pkl')
        )
        MODEL_CACHE['feature_columns'] = joblib.load(
            download_to_temp(MODEL_URL_IDS['feature_columns'], 'feature_columns.pkl')
        )
        
        return MODEL_CACHE
    except Exception as e:
        print(f"Model loading failed: {str(e)}")
        return None

def pm25_prediction(new_data):
    """Make prediction with temporary files"""
    try:
        
        models = load_models()
        if not models:
            raise Exception("Failed to load models")
            
        model = models['pm25_gru_model']
        feature_scaler = models['feature_scaler']
        target_scaler = models['target_scaler']
        feature_columns = models['feature_columns']

        processed_data = new_data.copy()

        
        # Convert any list-type columns to scalar values
        for col in processed_data.columns:
            if isinstance(processed_data[col].iloc[0], (list, np.ndarray)):
                processed_data[col] = processed_data[col].apply(lambda x: x[0] if isinstance(x, (list, np.ndarray)) else x)
        
        # Apply log1p transform to specific columns
        for col in ['PM10 (ug/m^3)', 'AQI', 'O3 (ppb)', 'SO2 (ppb)', 'NO2 (ppb)', 'CO (ppb)']:
            if col in processed_data.columns:
                processed_data[col] = np.log1p(pd.to_numeric(processed_data[col]))
        
        # Rest of your processing remains the same...
        # Handle categorical 'Day' feature
        if 'Day' in processed_data.columns:
            processed_data = pd.get_dummies(processed_data, columns=['Day'])
            for day in ['Day_Monday', 'Day_Tuesday', 'Day_Wednesday',
                       'Day_Thursday', 'Day_Friday', 'Day_Saturday', 'Day_Sunday']:
                if day not in processed_data.columns:
                    processed_data[day] = 0
        
        # Ensure all expected features are present
        for col in feature_columns:
            if col not in processed_data.columns:
                processed_data[col] = 0
        
        processed_data = processed_data[feature_columns]
        
        # Scale features and make prediction
        X_scaled = feature_scaler.transform(processed_data)
        X_gru = X_scaled.reshape(1, 1, X_scaled.shape[1])
        
        y_pred_scaled = model.predict(X_gru)
        y_pred = target_scaler.inverse_transform(y_pred_scaled)
        y_pred = np.expm1(y_pred)
        
        # Apply PM10 constraint if available
        if 'PM10 (ug/m^3)' in processed_data.columns:
            pm10_original = np.expm1(processed_data['PM10 (ug/m^3)'].iloc[0])
            return float(min(y_pred[0][0], pm10_original * 0.85))
        
        return float(y_pred[0][0])
    
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return None