from flask import Flask, request, jsonify
import requests
from dotenv import load_dotenv
import os
import pandas as pd
import numpy as np
from models.airquality import pm25_prediction
from flask_cors import CORS
import joblib
from functools import lru_cache
import argparse
import io
import tempfile
from medpy.filter.smoothing import anisotropic_diffusion
from scipy.ndimage import median_filter
from skimage import measure, morphology
import scipy.ndimage as ndimage
from sklearn.cluster import KMeans
import cv2
from io import BytesIO
from PIL import Image
from googleapiclient.discovery import build
from werkzeug.utils import secure_filename
import base64
from models.lung_cancer_processing import GetPrediction

app = Flask(__name__)

## NEEDED IF USING LOCALLY

# CORS(app, resources={
#     r"/*": {
#         "origins": ["http://localhost:3000", "http://127.0.0.1:3000"],
#         "methods": ["GET", "POST", "OPTIONS"],
#         "allow_headers": ["Content-Type"]
#     }
# })

# VERCEL NEEDED

# CORS(app, resources={r"/*": {"origins": "*", "methods": ["GET", "POST", "OPTIONS", "PUT"], "allow_headers": [
#     "X-CSRF-Token", "X-Requested-With", "Accept", "Accept-Version", "Content-Length", "Content-MD5", 
#     "Content-Type", "Date", "X-Api-Version"
# ]}})

CORS(app)


load_dotenv() 

API_KEY = os.getenv("OPEN_WEATHER_API_KEY")

MODEL_URL = "https://drive.google.com/uc?export=download&id=1SqEi4b6At-02dy76m9BHvlgq03uUDuAx" 
## FOR VERCEL
#MODEL_PATH = "/tmp/lung_health_model_new.pkl"

## FOR LOCAL
MODEL_PATH = os.path.join(os.getcwd(), 'lung_health_model_new.pkl')

# Lung Cancer Detection Things
UPLOAD_FOLDER = tempfile.mkdtemp()
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'npy'}
GOOGLE_DRIVE_URL ='https://drive.google.com/uc?export=download&id={file_id}'
META_CSV_FILE = 'https://drive.google.com/file/d/16w4XCFLNpFpQtTVyNj3JdtargHHCN1HN/view?usp=drive_link'
MASKS_FOLDER_URL = 'https://drive.google.com/drive/folders/10wdWoCIOVwYMVjknLw4P0HmplJ-ZuirE?usp=drive_link'
SEARCH_URL = "https://www.googleapis.com/drive/v3/files"
DOWNLOAD_URL = "https://drive.google.com/uc?export=download&id={file_id}"
meta_data=None

def AllowedFiles(filename):
    ext = filename.rsplit('.', 1)[-1].lower()
    return ext in ALLOWED_EXTENSIONS


# UPLOAD_FOLDER = tempfile.mkdtemp()
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'npy'}

META_CSV_ID = '16w4XCFLNpFpQtTVyNj3JdtargHHCN1HN'
DRIVE_API_URL = "https://www.googleapis.com/drive/v3/files"
DRIVE_DOWNLOAD_URL = "https://drive.google.com/uc?export=download&id={file_id}"


# API_KEY = 'AIzaSyCxONrSLHBJ2-rXRsaokrpTYmy4sC-ZSxA'  # Public API key
# DRIVE_SERVICE = build('drive', 'v3', developerKey=API_KEY)


def load_metadata():

    response = requests.get(DRIVE_DOWNLOAD_URL.format(file_id=META_CSV_ID))
    response.raise_for_status()
    return pd.read_csv(io.StringIO(response.text))

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'npy'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def validate_png(filepath):
    """Verify the file is a valid PNG image"""
    try:
        with Image.open(filepath) as img:
            img.verify()  # Verify it's a valid image
        return True
    except Exception:
        return False

DATA_FOLDER = os.path.join(app.root_path, 'data')
@app.route('/analyze', methods=['GET','POST'])
def analyze():
    try:

        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
            
        file = request.files['file']
        if not file.filename:
            return jsonify({'error': 'No filename provided'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Allowed: PNG, JPG, JPEG, NPY'}), 400 
        
        filename=secure_filename(file.filename)
        print(filename,"filename")
     
        ext = os.path.splitext(filename)[1]  
        img_file = 'LIDC-IDRI-'+filename.split('_')[0]+'/'+filename.split('.')[0]
        
    

        mask_file,label= FindMaskAndLabel(img_file)
        img_file=img_file+ext
        ct_scan_path = os.path.join(DATA_FOLDER, 'scans', img_file)
        os.makedirs(os.path.dirname(ct_scan_path), exist_ok=True)
        file.save(ct_scan_path)
        
        mask_file=os.path.normpath(mask_file)
        ct_scan_path=os.path.normpath(ct_scan_path)

        mask_file_path =os.path.join(DATA_FOLDER, 'Mask', mask_file)
        meta_file=os.path.join(DATA_FOLDER,'meta_info.csv')
        print("maskfile path ", mask_file_path)
        print("label",label)

       
        
       
        if not os.path.exists(mask_file_path):
            print(f"File not found at {mask_file_path}")
            return jsonify({'error': 'Mask file not found'}), 404
        
            
        if not validate_png(mask_file_path):
            print(f"Invalid PNG file at {mask_file_path}")
            return jsonify({'error': 'Invalid PNG file'}), 400
        

        
        results=GetPrediction(ct_scan_path,mask_file_path,label)
            
        # return send_file(mask_file_path, mimetype='image/png')
        return jsonify(results)
        
    except Exception as e:
        print(f"Error in /analyze: {str(e)}")  # Log the full error
        return jsonify({
            'error': 'Internal server error',
            'details': str(e)
        }), 500
        
    
    


def FindMaskAndLabel(full_path):
    try:
        meta_data = load_metadata()
        print(full_path, "full")
        matches = meta_data[meta_data['original_image'].str.strip() == full_path]
        if matches.empty:
            return None
            
        mask_filename = matches['mask_image'].iloc[0] + '.png'.replace('\\','/')
        label= matches['is_cancer'].iloc[0]
        
        print(mask_filename, "mask name")

        return mask_filename, label
        
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return None

    




def segment_lung(img):
    
    mean = np.mean(img)
    std = np.std(img)
    img = img-mean
    img = img/std
    
    middle = img[100:400,100:400] 
    mean = np.mean(middle)  
    max = np.max(img)
    min = np.min(img)
    #remove the underflow bins
    img[img==max]=mean
    img[img==min]=mean
    
    #apply median filter
    img= median_filter(img,size=3)
    #apply anistropic non-linear diffusion filter- This removes noise without blurring the nodule boundary
    img= anisotropic_diffusion(img)
    
    kmeans = KMeans(n_clusters=2).fit(np.reshape(middle,[np.prod(middle.shape),1]))
    centers = sorted(kmeans.cluster_centers_.flatten())
    threshold = np.mean(centers)
    thresh_img = np.where(img<threshold,1.0,0.0)  # threshold the image
    eroded = morphology.erosion(thresh_img,np.ones([4,4]))
    dilation = morphology.dilation(eroded,np.ones([10,10]))
    labels = measure.label(dilation)
    label_vals = np.unique(labels)
    regions = measure.regionprops(labels)
    good_labels = []
    for prop in regions:
        B = prop.bbox
        if B[2]-B[0]<475 and B[3]-B[1]<475 and B[0]>40 and B[2]<472:
            good_labels.append(prop.label)
    mask = np.ndarray([512,512],dtype=np.int8)
    mask[:] = 0
    
    for N in good_labels:
        mask = mask + np.where(labels==N,1,0)
    mask = morphology.dilation(mask,np.ones([10,10]))
    
    return mask*img




def PreprocessImage(image):

    if AllowedFiles(image):
        ext = image.rsplit('.', 1)[-1].lower()

        if ext=='jpg' or ext=='jpeg' or ext=='png':
            ct_scan=cv2.imread(image, cv2.IMREAD_GRAYSCALE)

def display_image_in_terminal(image_path, width=100):
    # Open and resize the image
    img = Image.open(image_path)
    aspect_ratio = img.height / img.width
    new_height = int(width * aspect_ratio)
    img = img.resize((width, new_height))
    
    # Convert to grayscale
    img = img.convert("L")  # 'L' mode = grayscale
    
    # ASCII characters (from dark to light)
    ascii_chars = "@%#*+=-:. "
    
    # Convert pixels to ASCII
    pixels = np.array(img)
    ascii_str = ""
    for row in pixels:
        for pixel in row:
            ascii_str += ascii_chars[pixel // 32]  # 256/8 = 32
        ascii_str += "\n"
    
    print(ascii_str)

# Function to download the model from Google Drive
@lru_cache(maxsize=1)
def download_model():
    """Download and save the model from Google Drive."""
    try:
        if not os.path.exists(MODEL_PATH):
            print("Downloading model from Google Drive...")
            # Send GET request to download the model
            response = requests.get(MODEL_URL)
            response.raise_for_status()  # Raise an error for bad responses
            with open(MODEL_PATH, 'wb') as f:
                f.write(response.content)
            print("Model downloaded successfully.")
        else:
            print("Model already exists.")
    except requests.exceptions.RequestException as e:
        print(f"Error downloading the model: {e}")
        raise RuntimeError("Failed to download the model.") from e

# Function to load the model using joblib
def load_model():
    """Load the trained model using joblib."""
    try:
        # Ensure the model is downloaded
        download_model()
        
        # Load the model using joblib
        model = joblib.load(MODEL_PATH)
        return model
    except Exception as e:
        print(f"Error loading the model: {e}")
        return None

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "message": "Air Quality, Patient Data API is running",
        "endpoints": {
            "predict_pm25": "POST /predict_pm25",
            "download_models": "GET /download_models",
            "predict healt": "POST /predict"
        }
    })

@app.route('/download_models', methods=['GET'])
def download_models():
    """Endpoint to manually trigger model downloads"""
    try:
        # Create properly formatted SINGLE-ROW DataFrame without list values
        test_data = pd.DataFrame({
            "Max Temperature (F)": 72.0,
            "Avg Temperature (F)": 68.5,
            "Min Temperature (F)": 65.0,
            "Max Humidity (%age)": 40,
            "Avg Humidity (%age)": 30,
            "Min Humidity (%age)": 20,
            "AQI": 12,
            "PM10 (ug/m^3)": 10.5,
            "O3 (ppb)": 40.0,
            "SO2 (ppb)": 0.5,
            "NO2 (ppb)": 5.0,
            "CO (ppb)": 0.2,
            "Day": "Wednesday"
        }, index=[0])  # index=[0] ensures single-row DataFrame
        
        # This will trigger model downloads
        prediction = pm25_prediction(test_data)
        
        return jsonify({
            'status': 'success',
            'message': 'Models are ready for use',
            'note': 'Models were downloaded automatically on first prediction'
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500
## NEW 
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get and validate input
        input_data = request.get_json()
        if not input_data:
            return jsonify({'error': 'No input data provided'}), 400
        
        app.logger.info(f"Input data: {input_data}")
        
        # Preprocess input (assuming you have preprocess_input function)
        input_array = preprocess_input(input_data)
        app.logger.info(f"Preprocessed input data: {input_array}")
        
        # Load the model
        model = load_model()
        if model is None:
            return jsonify({'error': 'Failed to load the model'}), 500
        
        # Predict using the loaded model
        prediction = model.predict(input_array)
        app.logger.info(f"Prediction result: {prediction}")
        
        # Return the result
        result = "Poor Lung Health" if prediction[0] == 1 else "Good Lung Health"
        return jsonify({'prediction': result})

    except Exception as e:
        app.logger.error(f"Error in prediction: {str(e)}")
        return jsonify({'error': 'Prediction failed', 'message': str(e)}), 500

## FUNCTION (WANIA)

@app.route('/predict_pm25', methods=['POST'])
def predict_pm25():
    try:
        input_data = request.get_json()
        new_data = pd.DataFrame([input_data])  # Single row DataFrame
        
        prediction = pm25_prediction(new_data)
        
        if prediction is not None:
            return jsonify({
                'status': 'success',
                'prediction': prediction,
                'units': 'μg/m³'
            }), 200
        else:
            return jsonify({
                'status': 'error',
                'message': 'Prediction failed'
            }), 500
            
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400


def preprocess_input(data):
    """Convert input data to numpy array (assuming pre-encoded values)"""
    features = [
        'Age',
        'Gender',    
        'Exposure_to_Occupational_Hazards',
        'Diagnosed_with_Cancer',                 
        'History_Of_Chronic_Respiratory_Diseases',
        'Lived_In_Highly_Polluted_Area',
        'Shortness_Of_breath',
        'Coughed_Blood',
        'Persistent_Cough',
        'Ever_Smoked',
        'PM2.5 (ug/m^3)',
        'PM10 (ug/m^3)',
        'AQI',
        'NO2 (ppb)',
        'SO2 (ppb)',
        'CO (ppb)'
    ]
    
    # Create array with default fallback values
    input_array = np.array([
        float(data.get(feature, 0)) if feature in ['Age', 'PM2.5 (ug/m^3)', 'PM10 (ug/m^3)', 
                                                 'AQI', 'NO2 (ppb)', 'SO2 (ppb)', 'CO (ppb)']
        else int(data.get(feature, 0))  # For categorical features (0/1/2 encoded)
        for feature in features
    ], dtype=np.float32).reshape(1, -1)
    
    return input_array


def get_aqi(lat, lon):
    url = f'http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={API_KEY}'
    response = requests.get(url)
    if response.status_code == 200:
        air_data = response.json()

        # Extract AQI from the API response (the key is 'list' -> 'main' -> 'aqi')
        aqi = air_data['list'][0]['main']['aqi']

        # Convert AQI value to a readable string
        if aqi == 1:
            aqi_text = "Good"
        elif aqi == 2:
            aqi_text = "Fair"
        elif aqi == 3:
            aqi_text = "Moderate"
        elif aqi == 4:
            aqi_text = "Poor"
        elif aqi == 5:
            aqi_text = "Very Poor"
        
        return aqi_text
    else:
        return None

@app.route('/webhook', methods=['POST'])
def webhook():
    print("Webhook endpoint hit!")
    req = request.get_json()
    parameters = req.get('queryResult').get('parameters')
    lat = parameters.get('latitude')
    lon = parameters.get('longitude')

    aqi = get_aqi(lat, lon)
    if aqi:
        fulfillment_text = f'The current Air Quality Index (AQI) at the specified location is {aqi}.'
    else:
        fulfillment_text = 'I am unable to retrieve the AQI for the specified location at this time.'

    return jsonify({'fulfillmentText': fulfillment_text})

## NEEDED IF USING LOCALLY
if __name__ == '__main__':
    app.run(port=5000,debug=True)


# if __name__ == '__main__':
#     app.run(debug=True)
    
    ## (NOT NEEDED FOR NOW)
    
    # if not os.path.exists('air_quality_models'):
    #     os.makedirs('air_quality_models')
    
    # app.run(host='0.0.0.0', port=5000, debug=True)
    # application = app  # For Vercel deployment

# Expose the Flask app for Vercel deployment
#application = app