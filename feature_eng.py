import numpy as np
import pandas as pd
import constants
from scipy.fft import rfft, rfftfreq, irfft
from scipy.signal import find_peaks

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

        # Power spectrum for positive frequencies
        power_spectrum = np.square(np.abs(fft))
        energy = np.mean(power_spectrum)
        power = np.sum(power_spectrum)

        # Compute centroid and bandwidth!
        centroid = np.sum(freq * power_spectrum) / np.sum(power_spectrum)
        bandwidth = np.sum(np.square(freq - centroid) * power_spectrum) / np.sum(power_spectrum)
        return [energy, power, centroid, bandwidth]
    else:
        return [0] * 4

def denoise_signal(signal, sampling_rate=100):
    """
    Apply FFT, filter the signal based on the cutoff frequency, and return the inverse RFFT.

    Parameters:
    - signal: numpy array, the input time-domain signal
    - sampling_rate: float, the sampling rate of the signal in Hz

    Returns:
    - filtered_signal: numpy array, the filtered signal in the time domain
    """
    # Perform the FFT and calculate the frequency bins
    N = len(signal)
    t = np.linspace(0, N/sampling_rate, N, endpoint=False)

    fft_coefficients = rfft(signal)
    frequencies = rfftfreq(N, d=1 / sampling_rate)

    # Calculate the magnitude and mean threshold!
    magnitude = np.abs(fft_coefficients)
    threshold = np.mean(magnitude)

    # Filter the FFT coefficients based on the threshold
    significant_indices = magnitude > threshold
    filtered_fft = np.zeros_like(fft_coefficients)
    filtered_fft[significant_indices] = fft_coefficients[significant_indices]

    # Perform the inverse FFT to get the filtered signal
    filtered_signal = irfft(filtered_fft, n=N)
    return filtered_signal

def extract_feature_set(signal_data):
    """
    Extract combined feature set from the signal data.
    :param signal_data: numpy array of the signal data
    :return: list of combined features
    """
    comb_feature_set = []
    magnitude_features = compute_magnitude_features(signal_data)
    derivative_features = compute_derivative_features(signal_data)
    integral_features = compute_integral_features(signal_data)
    fft_features = compute_fft_features(signal_data)
    comb_feature_set += magnitude_features + derivative_features + integral_features + fft_features
    return comb_feature_set

def find_true_peak(signal, prominence_range=(0.1, 5), width_range=(20, 80)):
    # Find all peaks with adjusted criteria
    peaks, properties = find_peaks(signal, prominence=prominence_range, width=width_range)

    # Get the true peak (the maximum value among valid peaks)
    if len(peaks) > 0:
        return peaks[np.argmax(signal[peaks])]
    else:
        return np.argmax(signal)

def find_active_point(smoothed_signal, peak_index):
    derivatives = np.gradient(smoothed_signal)
    point_B = 0

    # B: point of continuous rise up to the true peak!
    for i in range(peak_index, 0, -1):
        if derivatives[i - 1] <= 0.0:  # Look for where the rise starts
            point_B = i
            break
    return point_B

def find_decay_point(smoothed_signal, peak_index, decay_duration=20):
    derivatives = np.gradient(smoothed_signal)
    point_C = peak_index
    # The point after the true peak when the signal decreases for a specific duration!
    for j in range(peak_index, len(derivatives) - decay_duration):
        if all(derivatives[j: j + decay_duration] < 0):  # Continuous decay
            point_C = j
            break
    return point_C

def filter_signal(smoothed_signal):
    peak_index = find_true_peak(smoothed_signal)  # True Peak of the signal!
    point_B = find_active_point(smoothed_signal, peak_index)  # Point B (start of continuous rise) after baseline!
    point_C = find_decay_point(smoothed_signal, peak_index)  # Point C (start of continuous decay)
    filtered_signal = smoothed_signal[point_B: point_C + 1]
    return filtered_signal

def generate_features(df):
    magnitude_names = ['MEAN', 'STD', 'IQR', 'PTP', 'RMS']
    derivative_names = ['MAX_D', 'MIN_D', 'MEAN_D', 'STD_D']
    integral_names = ['INT', 'SQ_INT']
    fft_names = ['ENERGY', 'POWER', 'CD', 'BW']
    sensor_feature_names = []

    # For each sensor iteration, adding feature name!
    for n in range(len(constants.SENSOR_COLS)):
        sensor_feature_names += [f"{constants.SENSOR_COLS[n]}_{name}" for name in magnitude_names]
        sensor_feature_names += [f"{constants.SENSOR_COLS[n]}_{name}" for name in derivative_names]
        sensor_feature_names += [f"{constants.SENSOR_COLS[n]}_{name}" for name in integral_names]
        sensor_feature_names += [f"{constants.SENSOR_COLS[n]}_{name}" for name in fft_names]

    feature_vector = []
    for sensor_name in constants.SENSOR_COLS:
        smooth_signal = denoise_signal(signal=df[sensor_name].tolist())
        filtered_samples = filter_signal(smooth_signal)
        sensor_features = extract_feature_set(filtered_samples)
        feature_vector.extend(sensor_features)

    features = pd.DataFrame([feature_vector], columns=sensor_feature_names)
    return features
