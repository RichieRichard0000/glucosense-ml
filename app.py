from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import pickle
import os
import io
from scipy.fft import rfft, rfftfreq, irfft
from scipy.signal import find_peaks

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Constants
SENSOR_COLS = ['TGS2603', 'TGS2610', 'TGS2602', 'TGS2600', 'TGS822', 'MQ138']

# Model loading functions
def load_model(file_path):
    with open(file_path, mode='rb') as my_file:
        model = pickle.load(my_file)
    return model

# Load models at startup
try:
    models = {
        'stack': load_model('models/stack.pkl'),
        'selector': load_model('models/selector.pkl'),
        'corr_scaler': load_model('models/corr_scaler.pkl'),
        'normalizer': load_model('models/normalizer.pkl'),
        'adaboost': load_model('models/AdaBoost.pkl')
    }
    models_loaded = True
except Exception as e:
    models_loaded = False
    model_error = str(e)

# Feature engineering functions
def compute_magnitude_features(signal_samples):
    if len(signal_samples) > 1:
        mean = np.mean(signal_samples)
        std = np.std(signal_samples)
        iqr = np.percentile(signal_samples, 75) - np.percentile(signal_samples, 25)
        ptp_amplitude = np.ptp(signal_samples)
        rms = np.sqrt(np.mean(np.square(signal_samples)))
        return [mean, std, iqr, ptp_amplitude, rms]
    else:
        return [0] * 5

def compute_derivative_features(signal):
    if len(signal) > 1:
        derivative = np.gradient(signal)
        max_d = np.max(derivative)
        min_d = np.min(derivative)
        mean_d = np.mean(derivative)
        std_d = np.std(derivative)
        return [max_d, min_d, mean_d, std_d]
    else:
        return [0] * 4

def compute_integral_features(signal_samples):
    if len(signal_samples) > 1:
        integral = np.trapz(signal_samples) / len(signal_samples)
        squared_integral = np.trapz(np.square(signal_samples)) / len(signal_samples)
        return [integral, squared_integral]
    else:
        return [0] * 2

def compute_fft_features(sample, sampling_rate=100):
    if len(sample) > 1:
        length = len(sample)
        fft = rfft(sample)
        freq = rfftfreq(length, d=1 / sampling_rate)
        power_spectrum = np.square(np.abs(fft))
        energy = np.mean(power_spectrum)
        power = np.sum(power_spectrum)
        centroid = np.sum(freq * power_spectrum) / np.sum(power_spectrum)
        bandwidth = np.sum(np.square(freq - centroid) * power_spectrum) / np.sum(power_spectrum)
        return [energy, power, centroid, bandwidth]
    else:
        return [0] * 4

def denoise_signal(signal, sampling_rate=100):
    N = len(signal)
    t = np.linspace(0, N/sampling_rate, N, endpoint=False)
    fft_coefficients = rfft(signal)
    frequencies = rfftfreq(N, d=1 / sampling_rate)
    magnitude = np.abs(fft_coefficients)
    threshold = np.mean(magnitude)
    significant_indices = magnitude > threshold
    filtered_fft = np.zeros_like(fft_coefficients)
    filtered_fft[significant_indices] = fft_coefficients[significant_indices]
    filtered_signal = irfft(filtered_fft, n=N)
    return filtered_signal

def extract_feature_set(signal_data):
    comb_feature_set = []
    magnitude_features = compute_magnitude_features(signal_data)
    derivative_features = compute_derivative_features(signal_data)
    integral_features = compute_integral_features(signal_data)
    fft_features = compute_fft_features(signal_data)
    comb_feature_set += magnitude_features + derivative_features + integral_features + fft_features
    return comb_feature_set

def find_true_peak(signal, prominence_range=(0.1, 5), width_range=(20, 80)):
    peaks, properties = find_peaks(signal, prominence=prominence_range, width=width_range)
    if len(peaks) > 0:
        return peaks[np.argmax(signal[peaks])]
    else:
        return np.argmax(signal)

def find_active_point(smoothed_signal, peak_index):
    derivatives = np.gradient(smoothed_signal)
    point_B = 0
    for i in range(peak_index, 0, -1):
        if derivatives[i - 1] <= 0.0:
            point_B = i
            break
    return point_B

def find_decay_point(smoothed_signal, peak_index, decay_duration=20):
    derivatives = np.gradient(smoothed_signal)
    point_C = peak_index
    for j in range(peak_index, len(derivatives) - decay_duration):
        if all(derivatives[j: j + decay_duration] < 0):
            point_C = j
            break
    return point_C

def filter_signal(smoothed_signal):
    peak_index = find_true_peak(smoothed_signal)
    point_B = find_active_point(smoothed_signal, peak_index)
    point_C = find_decay_point(smoothed_signal, peak_index)
    filtered_signal = smoothed_signal[point_B: point_C + 1]
    return filtered_signal

def generate_features(df):
    magnitude_names = ['MEAN', 'STD', 'IQR', 'PTP', 'RMS']
    derivative_names = ['MAX_D', 'MIN_D', 'MEAN_D', 'STD_D']
    integral_names = ['INT', 'SQ_INT']
    fft_names = ['ENERGY', 'POWER', 'CD', 'BW']
    
    sensor_feature_names = []
    for n in range(len(SENSOR_COLS)):
        sensor_feature_names += [f"{SENSOR_COLS[n]}_{name}" for name in magnitude_names]
        sensor_feature_names += [f"{SENSOR_COLS[n]}_{name}" for name in derivative_names]
        sensor_feature_names += [f"{SENSOR_COLS[n]}_{name}" for name in integral_names]
        sensor_feature_names += [f"{SENSOR_COLS[n]}_{name}" for name in fft_names]
    
    feature_vector = []
    for sensor_name in SENSOR_COLS:
        smooth_signal = denoise_signal(signal=df[sensor_name].tolist())
        filtered_samples = filter_signal(smooth_signal)
        sensor_features = extract_feature_set(filtered_samples)
        feature_vector.extend(sensor_features)
    
    features = pd.DataFrame([feature_vector], columns=sensor_feature_names)
    return features

def remove_irrelevant_data(df):
    df.columns = ['H', 'MQ138', 'MQ2', 'SSID', 'T', 'TGS2600', 'TGS2602', 'TGS2603', 'TGS2610', 'TGS2611', 'TGS2620', 'TGS822', 'Device', 'Time']
    df = df.drop(['SSID', 'Device', 'H', 'T', 'Time'], axis=1)
    return df.reset_index(drop=True)

def generate_data(sensors_data, body_vitals):
    cleaned_df = remove_irrelevant_data(sensors_data)
    features_df = generate_features(df=cleaned_df)
    final_df = pd.concat([body_vitals, features_df], axis=1)
    return final_df

def perform_feature_selection(test_data):
    selected_names = test_data.columns[models['selector'].get_support()]
    features = pd.DataFrame(data=models['selector'].transform(test_data), columns=selected_names)
    X_new = models['corr_scaler'].transform(features)
    return models['normalizer'].transform(X_new)

def perform_diabetes_test(features):
    test_label = models['stack'].predict(features)
    if test_label == 0:
        return "Non-diabetic"
    elif test_label == 1:
        return "Pre-diabetic"
    else:
        return "Highly diabetic"

def perform_bgl_test(features):
    bgl_value = models['adaboost'].predict(features)[0]
    return np.round(bgl_value, 2)

# API Routes
@app.route('/')
def home():
    return "GlucoSense API is running!"

@app.route('/model-status', methods=['GET'])
def model_status():
    if models_loaded:
        return jsonify({
            "status": "success",
            "message": "Models loaded successfully",
            "models_available": list(models.keys())
        })
    else:
        return jsonify({
            "status": "error",
            "message": "Failed to load models",
            "error": model_error
        }), 500
    
@app.route('/debug-predict', methods=['POST'])
def debug_predict():
    try:
        data = request.json
        return jsonify({
            'status': 'success',
            'received_data': {
                'keys': list(data.keys()) if data else [],
                'name': data.get('name', 'missing'),
                'age': data.get('age', 'missing'),
                'has_breath_data': 'breathData' in data if data else False,
                'breath_data_length': len(data.get('breathData', '')) if data and 'breathData' in data else 0
            }
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500


@app.route('/predict', methods=['POST'])
def predict():
    if not models_loaded:
        return jsonify({
            "status": "error", 
            "message": "Models not loaded. Check /model-status for details."
        }), 500
    
    try:
        # Extract user data from request
        data = request.json
        
        # Extract user inputs
        name = data.get('name', '')
        age = data.get('age', 0)
        gender = 0 if data.get('gender', 'Male') == 'Male' else 1
        spo2 = data.get('spo2', 0)
        diastolic_bp = data.get('diastolicBP', 0)
        systolic_bp = data.get('systolicBP', 0)
        heart_rate = data.get('heartRate', 0)
        
        # Create body vitals DataFrame
        body_vitals = {
            'Age': [age], 
            'Gender': [gender], 
            'HR': [heart_rate], 
            'SPO2': [spo2], 
            'maxBP': [systolic_bp], 
            'minBP': [diastolic_bp]
        }
        body_vitals_df = pd.DataFrame(body_vitals)
        
        # Process breath data if provided
        breath_data_csv = data.get('breathData', None)
        
        if breath_data_csv:
            # Convert breath data string to DataFrame
            breath_df = pd.read_csv(io.StringIO(breath_data_csv), skiprows=3)
            
            # Generate features from breath data and body vitals
            test_data = generate_data(breath_df, body_vitals_df)
            
            # Perform feature selection
            reduced_features = perform_feature_selection(test_data)
            
            # Get diabetes classification and BGL prediction
            diabetes_result = perform_diabetes_test(reduced_features)
            bgl_result = perform_bgl_test(reduced_features)
            
            # Return prediction results
            return jsonify({
                'status': 'success',
                'prediction': diabetes_result,
                'bgl': float(bgl_result),
                'userData': {
                    'name': name,
                    'age': age,
                    'gender': 'Male' if gender == 0 else 'Female',
                    'spo2': spo2,
                    'diastolicBP': diastolic_bp,
                    'systolicBP': systolic_bp,
                    'heartRate': heart_rate
                }
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'Breath data is required for prediction'
            }), 400
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': 'Prediction failed',
            'error': str(e)
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
