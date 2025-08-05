#!/usr/bin/env python3
"""
Pan-Tompkins QRS Detection and Feature Extraction with PCA Feature Selection
This script implements the Pan-Tompkins algorithm for QRS detection and extracts
comprehensive ECG features, followed by PCA-based dimensionality reduction.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.stats import skew, kurtosis
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import os
import wfdb
from google.colab import drive
import warnings
warnings.filterwarnings('ignore')

# Define paths
base_path = '/content/drive/MyDrive/MIT_BIH_Dataset'
raw_data_path = os.path.join(base_path, 'raw_data')
denoised_data_path = os.path.join(base_path, 'denoised_data')
features_data_path = os.path.join(base_path, 'extracted_features')

# Create features directory
os.makedirs(features_data_path, exist_ok=True)

class PanTompkinsDetector:
    """
    Pan-Tompkins QRS Detection Algorithm Implementation
    """
    
    def __init__(self, sampling_rate=360):
        """
        Initialize Pan-Tompkins detector
        
        Args:
            sampling_rate (int): Sampling frequency of ECG signal
        """
        self.fs = sampling_rate
        self.detection_stats = {}
    
    def bandpass_filter(self, ecg_signal):
        """
        Apply bandpass filter (5-15 Hz) to remove noise
        
        Args:
            ecg_signal (np.array): Input ECG signal
            
        Returns:
            np.array: Filtered signal
        """
        # Design bandpass filter (5-15 Hz)
        nyquist = self.fs / 2
        low_freq = 5 / nyquist
        high_freq = 15 / nyquist
        
        # Butterworth bandpass filter
        b, a = signal.butter(2, [low_freq, high_freq], btype='band')
        filtered_signal = signal.filtfilt(b, a, ecg_signal)
        
        return filtered_signal
    
    def derivative_filter(self, signal_data):
        """
        Apply derivative filter to emphasize QRS slope
        
        Args:
            signal_data (np.array): Filtered ECG signal
            
        Returns:
            np.array: Derivative filtered signal
        """
        # Pan-Tompkins derivative filter: [1 2 0 -2 -1] * (1/8) * fs
        derivative_filter = np.array([1, 2, 0, -2, -1]) * (self.fs / 8)
        
        # Apply filter
        derivative_signal = np.convolve(signal_data, derivative_filter, mode='same')
        
        return derivative_signal
    
    def squaring_function(self, derivative_signal):
        """
        Square the derivative signal to make all data positive and emphasize higher frequencies
        
        Args:
            derivative_signal (np.array): Derivative filtered signal
            
        Returns:
            np.array: Squared signal
        """
        return derivative_signal ** 2
    
    def moving_window_integration(self, squared_signal, window_size=None):
        """
        Apply moving window integration
        
        Args:
            squared_signal (np.array): Squared signal
            window_size (int): Integration window size (default: 150ms worth of samples)
            
        Returns:
            np.array: Integrated signal
        """
        if window_size is None:
            window_size = int(0.150 * self.fs)  # 150ms window
        
        # Moving average filter
        window = np.ones(window_size) / window_size
        integrated_signal = np.convolve(squared_signal, window, mode='same')
        
        return integrated_signal
    
    def adaptive_thresholding(self, integrated_signal, learning_rate=0.125):
        """
        Implement adaptive thresholding for QRS detection
        
        Args:
            integrated_signal (np.array): Integrated signal
            learning_rate (float): Learning rate for threshold adaptation
            
        Returns:
            tuple: (qrs_peaks, thresholds_history)
        """
        # Initialize thresholds
        signal_peak = 0
        noise_peak = 0
        threshold_1 = 0
        threshold_2 = 0
        
        qrs_peaks = []
        thresholds_history = []
        
        # Minimum distance between QRS peaks (200ms)
        min_distance = int(0.2 * self.fs)
        
        # Search for peaks
        peaks, _ = signal.find_peaks(integrated_signal, distance=min_distance)
        
        if len(peaks) == 0:
            return np.array([]), np.array([])
        
        # Initialize with first few peaks
        initial_peaks = peaks[:8] if len(peaks) >= 8 else peaks
        signal_peak = np.mean(integrated_signal[initial_peaks])
        threshold_1 = 0.3125 * signal_peak  # Initial threshold
        threshold_2 = 0.5 * threshold_1
        
        # Process each potential peak
        for peak_idx in peaks:
            peak_value = integrated_signal[peak_idx]
            
            # Check if peak exceeds threshold
            if peak_value > threshold_1:
                # QRS detected
                qrs_peaks.append(peak_idx)
                
                # Update signal peak and thresholds
                signal_peak = learning_rate * peak_value + (1 - learning_rate) * signal_peak
                threshold_1 = 0.3125 * signal_peak
                threshold_2 = 0.5 * threshold_1
                
            elif peak_value > threshold_2:
                # Possible QRS, check RR interval
                if len(qrs_peaks) > 0:
                    rr_interval = peak_idx - qrs_peaks[-1]
                    avg_rr = np.mean(np.diff(qrs_peaks[-8:])) if len(qrs_peaks) >= 8 else rr_interval
                    
                    # If RR interval is reasonable, accept as QRS
                    if 0.3 * avg_rr < rr_interval < 1.5 * avg_rr:
                        qrs_peaks.append(peak_idx)
                        signal_peak = learning_rate * peak_value + (1 - learning_rate) * signal_peak
                        threshold_1 = 0.3125 * signal_peak
                        threshold_2 = 0.5 * threshold_1
                    else:
                        # Update noise peak
                        noise_peak = learning_rate * peak_value + (1 - learning_rate) * noise_peak
                else:
                    qrs_peaks.append(peak_idx)
            else:
                # Update noise peak
                noise_peak = learning_rate * peak_value + (1 - learning_rate) * noise_peak
            
            thresholds_history.append([threshold_1, threshold_2, signal_peak, noise_peak])
        
        return np.array(qrs_peaks), np.array(thresholds_history)
    
    def detect_qrs_peaks(self, ecg_signal):
        """
        Complete Pan-Tompkins QRS detection pipeline
        
        Args:
            ecg_signal (np.array): Input ECG signal
            
        Returns:
            dict: Detection results including peaks and intermediate signals
        """
        try:
            # Step 1: Bandpass filtering
            filtered_signal = self.bandpass_filter(ecg_signal)
            
            # Step 2: Derivative filtering
            derivative_signal = self.derivative_filter(filtered_signal)
            
            # Step 3: Squaring
            squared_signal = self.squaring_function(derivative_signal)
            
            # Step 4: Moving window integration
            integrated_signal = self.moving_window_integration(squared_signal)
            
            # Step 5: Adaptive thresholding
            qrs_peaks, thresholds = self.adaptive_thresholding(integrated_signal)
            
            # Calculate heart rate and RR intervals
            if len(qrs_peaks) > 1:
                rr_intervals = np.diff(qrs_peaks) / self.fs  # in seconds
                heart_rates = 60 / rr_intervals  # beats per minute
                avg_heart_rate = np.mean(heart_rates)
                hrv_metrics = self.calculate_hrv_metrics(rr_intervals)
            else:
                rr_intervals = np.array([])
                heart_rates = np.array([])
                avg_heart_rate = 0
                hrv_metrics = {}
            
            detection_result = {
                'original_signal': ecg_signal,
                'filtered_signal': filtered_signal,
                'derivative_signal': derivative_signal,
                'squared_signal': squared_signal,
                'integrated_signal': integrated_signal,
                'qrs_peaks': qrs_peaks,
                'thresholds_history': thresholds,
                'rr_intervals': rr_intervals,
                'heart_rates': heart_rates,
                'avg_heart_rate': avg_heart_rate,
                'num_beats': len(qrs_peaks),
                'hrv_metrics': hrv_metrics
            }
            
            return detection_result
            
        except Exception as e:
            print(f"Error in QRS detection: {str(e)}")
            return {'error': str(e)}
    
    def calculate_hrv_metrics(self, rr_intervals):
        """
        Calculate Heart Rate Variability metrics
        
        Args:
            rr_intervals (np.array): RR intervals in seconds
            
        Returns:
            dict: HRV metrics
        """
        if len(rr_intervals) < 2:
            return {}
        
        # Convert to milliseconds
        rr_ms = rr_intervals * 1000
        
        # Time domain metrics
        mean_rr = np.mean(rr_ms)
        sdnn = np.std(rr_ms)  # Standard deviation of NN intervals
        
        # RMSSD - Root mean square of successive differences
        successive_diffs = np.diff(rr_ms)
        rmssd = np.sqrt(np.mean(successive_diffs ** 2))
        
        # pNN50 - Percentage of intervals differing by more than 50ms
        pnn50 = (np.sum(np.abs(successive_diffs) > 50) / len(successive_diffs)) * 100
        
        # Triangular index (approximation)
        hist, _ = np.histogram(rr_ms, bins=50)
        triangular_index = len(rr_ms) / np.max(hist) if np.max(hist) > 0 else 0
        
        hrv_metrics = {
            'mean_rr_ms': mean_rr,
            'sdnn_ms': sdnn,
            'rmssd_ms': rmssd,
            'pnn50_percent': pnn50,
            'triangular_index': triangular_index,
            'heart_rate_mean': 60000 / mean_rr if mean_rr > 0 else 0,
            'heart_rate_std': np.std(60000 / rr_ms) if len(rr_ms) > 0 else 0
        }
        
        return hrv_metrics

class ECGFeatureExtractor:
    """
    Comprehensive ECG Feature Extraction
    """
    
    def __init__(self, sampling_rate=360):
        self.fs = sampling_rate
    
    def extract_morphological_features(self, ecg_signal, qrs_peaks):
        """
        Extract morphological features from ECG beats
        
        Args:
            ecg_signal (np.array): ECG signal
            qrs_peaks (np.array): QRS peak locations
            
        Returns:
            dict: Morphological features
        """
        if len(qrs_peaks) == 0:
            return {}
        
        # Define beat window (300ms before and after QRS)
        beat_window = int(0.3 * self.fs)
        
        beat_features = []
        valid_beats = []
        
        for peak in qrs_peaks:
            start_idx = max(0, peak - beat_window)
            end_idx = min(len(ecg_signal), peak + beat_window)
            
            if end_idx - start_idx >= 2 * beat_window * 0.8:  # At least 80% of window
                beat = ecg_signal[start_idx:end_idx]
                
                # Normalize beat length to standard size
                target_length = 2 * beat_window
                if len(beat) != target_length:
                    beat = signal.resample(beat, target_length)
                
                valid_beats.append(beat)
                
                # Extract features from this beat
                beat_dict = {
                    'peak_amplitude': np.max(beat),
                    'peak_to_peak': np.ptp(beat),
                    'beat_energy': np.sum(beat ** 2),
                    'beat_mean': np.mean(beat),
                    'beat_std': np.std(beat),
                    'beat_skewness': skew(beat),
                    'beat_kurtosis': kurtosis(beat),
                    'beat_rms': np.sqrt(np.mean(beat ** 2))
                }
                
                beat_features.append(beat_dict)
        
        if not beat_features:
            return {}
        
        # Average features across all beats
        morphological_features = {}
        for key in beat_features[0].keys():
            values = [beat[key] for beat in beat_features]
            morphological_features[f'avg_{key}'] = np.mean(values)
            morphological_features[f'std_{key}'] = np.std(values)
        
        # Template matching features
        if len(valid_beats) > 1:
            template = np.mean(valid_beats, axis=0)
            correlations = [np.corrcoef(beat, template)[0, 1] for beat in valid_beats]
            morphological_features['template_correlation_mean'] = np.mean(correlations)
            morphological_features['template_correlation_std'] = np.std(correlations)
            morphological_features['template_energy'] = np.sum(template ** 2)
        
        return morphological_features
    
    def extract_frequency_features(self, ecg_signal):
        """
        Extract frequency domain features
        
        Args:
            ecg_signal (np.array): ECG signal
            
        Returns:
            dict: Frequency domain features
        """
        # Power Spectral Density
        freqs, psd = signal.welch(ecg_signal, fs=self.fs, nperseg=min(1024, len(ecg_signal)//4))
        
        # Frequency bands for ECG analysis
        freq_bands = {
            'very_low': (0.003, 0.04),    # VLF
            'low': (0.04, 0.15),          # LF  
            'high': (0.15, 0.4),          # HF
            'total': (0.003, 0.4)         # Total power
        }
        
        frequency_features = {}
        
        # Calculate power in each frequency band
        for band_name, (low_freq, high_freq) in freq_bands.items():
            band_mask = (freqs >= low_freq) & (freqs <= high_freq)
            band_power = np.trapz(psd[band_mask], freqs[band_mask])
            frequency_features[f'power_{band_name}_freq'] = band_power
        
        # Calculate ratios
        if frequency_features['power_total_freq'] > 0:
            frequency_features['lf_hf_ratio'] = (frequency_features['power_low_freq'] / 
                                               frequency_features['power_high_freq'])
            frequency_features['vlf_total_ratio'] = (frequency_features['power_very_low_freq'] / 
                                                   frequency_features['power_total_freq'])
        
        # Spectral centroid and bandwidth
        frequency_features['spectral_centroid'] = np.sum(freqs * psd) / np.sum(psd)
        frequency_features['spectral_bandwidth'] = np.sqrt(np.sum(((freqs - frequency_features['spectral_centroid']) ** 2) * psd) / np.sum(psd))
        
        # Peak frequency
        peak_freq_idx = np.argmax(psd)
        frequency_features['peak_frequency'] = freqs[peak_freq_idx]
        frequency_features['peak_power'] = psd[peak_freq_idx]
        
        return frequency_features
    
    def extract_statistical_features(self, ecg_signal):
        """
        Extract statistical features from ECG signal
        
        Args:
            ecg_signal (np.array): ECG signal
            
        Returns:
            dict: Statistical features
        """
        statistical_features = {
            'signal_mean': np.mean(ecg_signal),
            'signal_std': np.std(ecg_signal),
            'signal_variance': np.var(ecg_signal),
            'signal_skewness': skew(ecg_signal),
            'signal_kurtosis': kurtosis(ecg_signal),
            'signal_rms': np.sqrt(np.mean(ecg_signal ** 2)),
            'signal_peak_to_peak': np.ptp(ecg_signal),
            'signal_energy': np.sum(ecg_signal ** 2),
            'signal_power': np.mean(ecg_signal ** 2),
            'signal_crest_factor': np.max(np.abs(ecg_signal)) / np.sqrt(np.mean(ecg_signal ** 2)),
            'signal_entropy': -np.sum(np.abs(ecg_signal) * np.log(np.abs(ecg_signal) + 1e-10))
        }
        
        # Zero crossing rate
        zero_crossings = np.where(np.diff(np.signbit(ecg_signal)))[0]
        statistical_features['zero_crossing_rate'] = len(zero_crossings) / len(ecg_signal)
        
        return statistical_features
    
    def extract_wavelet_features(self, ecg_signal):
        """
        Extract wavelet-based features
        
        Args:
            ecg_signal (np.array): ECG signal
            
        Returns:
            dict: Wavelet features
        """
        try:
            import pywt
            
            # Wavelet decomposition
            wavelet = 'db6'
            levels = 6
            coeffs = pywt.wavedec(ecg_signal, wavelet, level=levels)
            
            wavelet_features = {}
            
            # Energy in each decomposition level
            for i, coeff in enumerate(coeffs):
                level_name = 'approximation' if i == 0 else f'detail_{len(coeffs)-i}'
                wavelet_features[f'wavelet_energy_{level_name}'] = np.sum(coeff ** 2)
                wavelet_features[f'wavelet_std_{level_name}'] = np.std(coeff)
                wavelet_features[f'wavelet_mean_{level_name}'] = np.mean(coeff)
            
            # Relative wavelet energy
            total_energy = sum(np.sum(coeff ** 2) for coeff in coeffs)
            for i, coeff in enumerate(coeffs):
                level_name = 'approximation' if i == 0 else f'detail_{len(coeffs)-i}'
                wavelet_features[f'wavelet_rel_energy_{level_name}'] = (np.sum(coeff ** 2) / total_energy) if total_energy > 0 else 0
            
            return wavelet_features
            
        except ImportError:
            print("PyWavelets not available for wavelet features")
            return {}
        except Exception as e:
            print(f"Error extracting wavelet features: {str(e)}")
            return {}
    
    def extract_comprehensive_features(self, ecg_signal, qrs_peaks, hrv_metrics):
        """
        Extract all types of features
        
        Args:
            ecg_signal (np.array): ECG signal
            qrs_peaks (np.array): QRS peak locations
            hrv_metrics (dict): HRV metrics
            
        Returns:
            dict: All extracted features
        """
        features = {}
        
        # Morphological features
        morphological = self.extract_morphological_features(ecg_signal, qrs_peaks)
        features.update(morphological)
        
        # Frequency domain features
        frequency = self.extract_frequency_features(ecg_signal)
        features.update(frequency)
        
        # Statistical features
        statistical = self.extract_statistical_features(ecg_signal)
        features.update(statistical)
        
        # Wavelet features
        wavelet = self.extract_wavelet_features(ecg_signal)
        features.update(wavelet)
        
        # HRV features
        features.update(hrv_metrics)
        
        # Additional temporal features
        if len(qrs_peaks) > 0:
            features['num_beats'] = len(qrs_peaks)
            features['avg_beat_rate'] = len(qrs_peaks) / (len(ecg_signal) / self.fs) * 60  # BPM
            
            if len(qrs_peaks) > 1:
                rr_intervals = np.diff(qrs_peaks) / self.fs
                features['rr_variability'] = np.std(rr_intervals)
                features['rr_mean'] = np.mean(rr_intervals)
        
        return features

class PCAFeatureSelector:
    """
    PCA-based Feature Selection and Dimensionality Reduction
    """
    
    def __init__(self, n_components=None, variance_threshold=0.95):
        """
        Initialize PCA feature selector
        
        Args:
            n_components (int): Number of components to keep (None for auto-selection)
            variance_threshold (float): Cumulative variance threshold for auto-selection
        """
        self.n_components = n_components
        self.variance_threshold = variance_threshold
        self.pca = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.selected_features = None
        
    def fit_transform(self, feature_matrix, feature_names):
        """
        Fit PCA and transform features
        
        Args:
            feature_matrix (np.array): Feature matrix (samples x features)
            feature_names (list): List of feature names
            
        Returns:
            tuple: (transformed_features, pca_info)
        """
        # Handle missing values
        feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Standardize features
        scaled_features = self.scaler.fit_transform(feature_matrix)
        
        # Determine number of components
        if self.n_components is None:
            # Find number of components for desired variance
            pca_temp = PCA()
            pca_temp.fit(scaled_features)
            cumsum_var = np.cumsum(pca_temp.explained_variance_ratio_)
            n_components = np.argmax(cumsum_var >= self.variance_threshold) + 1
            n_components = min(n_components, scaled_features.shape[1], scaled_features.shape[0])
        else:
            n_components = min(self.n_components, scaled_features.shape[1], scaled_features.shape[0])
        
        # Fit PCA
        self.pca = PCA(n_components=n_components)
        transformed_features = self.pca.fit_transform(scaled_features)
        
        # Store feature information
        self.feature_names = feature_names
        
        # Analyze feature importance
        components = self.pca.components_
        feature_importance = np.abs(components).mean(axis=0)
        
        # Get most important original features
        importance_indices = np.argsort(feature_importance)[::-1]
        
        pca_info = {
            'n_components': n_components,
            'explained_variance_ratio': self.pca.explained_variance_ratio_,
            'cumulative_variance_ratio': np.cumsum(self.pca.explained_variance_ratio_),
            'total_variance_explained': np.sum(self.pca.explained_variance_ratio_),
            'feature_importance': feature_importance,
            'most_important_features': [feature_names[i] for i in importance_indices[:20]],
            'feature_importance_scores': feature_importance[importance_indices[:20]],
            'original_feature_count': len(feature_names),
            'reduced_feature_count': n_components,
            'dimensionality_reduction_ratio': n_components / len(feature_names)
        }
        
        return transformed_features, pca_info
    
    def transform(self, feature_matrix):
        """
        Transform new feature matrix using fitted PCA
        
        Args:
            feature_matrix (np.array): Feature matrix to transform
            
        Returns:
            np.array: Transformed features
        """
        if self.pca is None:
            raise ValueError("PCA not fitted yet. Call fit_transform first.")
        
        # Handle missing values
        feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Scale and transform
        scaled_features = self.scaler.transform(feature_matrix)
        transformed_features = self.pca.transform(scaled_features)
        
        return transformed_features

def process_record_features(record_num):
    """
    Process a single record for feature extraction
    
    Args:
        record_num (int): Record number to process
        
    Returns:
        dict: Extracted features and metadata
    """
    try:
        # Load the record
        record_path = os.path.join(raw_data_path, str(record_num))
        record = wfdb.rdrecord(record_path)
        
        if record.p_signal is None:
            print(f"No signal data found for record {record_num}")
            return None
        
        signals = record.p_signal
        
        # Initialize processors
        pan_tompkins = PanTompkinsDetector(sampling_rate=record.fs)
        feature_extractor = ECGFeatureExtractor(sampling_rate=record.fs)
        
        results = {
            'record_number': record_num,
            'sampling_rate': record.fs,
            'signal_names': record.sig_name,
            'leads': {}
        }
        
        # Process each lead
        for lead_idx, lead_name in enumerate(record.sig_name):
            if lead_idx >= signals.shape[1]:
                continue
                
            signal_data = signals[:, lead_idx]
            
            print(f"Processing Record {record_num}, Lead {lead_name}...")
            
            # Pan-Tompkins QRS detection
            detection_result = pan_tompkins.detect_qrs_peaks(signal_data)
            
            if 'error' not in detection_result:
                # Extract comprehensive features
                features = feature_extractor.extract_comprehensive_features(
                    signal_data, 
                    detection_result['qrs_peaks'], 
                    detection_result['hrv_metrics']
                )
                
                results['leads'][lead_name] = {
                    'qrs_detection': detection_result,
                    'extracted_features': features,
                    'num_features': len(features),
                    'feature_names': list(features.keys())
                }
                
                print(f"  QRS peaks detected: {len(detection_result['qrs_peaks'])}")
                print(f"  Features extracted: {len(features)}")
                if detection_result['avg_heart_rate'] > 0:
                    print(f"  Average heart rate: {detection_result['avg_heart_rate']:.1f} BPM")
            else:
                print(f"  QRS detection failed: {detection_result['error']}")
                results['leads'][lead_name] = {'error': detection_result['error']}
        
        return results
        
    except Exception as e:
        print(f"Error processing record {record_num}: {str(e)}")
        return None

def create_feature_matrix(results_list):
    """
    Create feature matrix from extracted features
    
    Args:
        results_list (list): List of feature extraction results
        
    Returns:
        tuple: (feature_matrix, feature_names, record_info)
    """
    # Collect all features
    all_features = []
    all_feature_names = set()
    record_info = []
    
    # First pass: collect all possible feature names
    for result in results_list:
        if result is None:
            continue
            
        for lead_name, lead_data in result['leads'].items():
            if 'extracted_features' in lead_data:
                all_feature_names.update(lead_data['extracted_features'].keys())
    
    all_feature_names = sorted(list(all_feature_names))
    
    # Second pass: create feature matrix
    for result in results_list:
        if result is None:
            continue
            
        record_num = result['record_number']
        
        for lead_name, lead_data in result['leads'].items():
            if 'extracted_features' in lead_data:
                features = lead_data['extracted_features']
                
                # Create feature vector with all possible features
                feature_vector = []
                for feature_name in all_feature_names:
                    feature_vector.append(features.get(feature_name, 0.0))
                
                all_features.append(feature_vector)
                record_info.append({
                    'record_number': record_num,
                    'lead_name': lead_name,
                    'num_qrs_peaks': len(lead_data['qrs_detection']['qrs_peaks']),
                    'avg_heart_rate': lead_data['qrs_detection']['avg_heart_rate']
                })
    
    feature_matrix = np.array(all_features)
    
    return feature_matrix, all_feature_names, record_info

def save_feature_results(results_list, feature_matrix, feature_names, record_info, pca_info, pca_features):
    """
    Save feature extraction and PCA results
    
    Args:
        results_list (list): Feature extraction results
        feature_matrix (np.array): Original feature matrix
        feature_names (list): Original feature names
        record_info (list): Record information
        pca_info (dict): PCA analysis information
        pca_features (np.array): PCA-transformed features
    """
    # Save original features
    features_df = pd.DataFrame(feature_matrix, columns=feature_names)
    for i, info in enumerate(record_info):
        features_df.loc[i, 'record_number'] = info['record_number']
        features_df.loc[i, 'lead_name'] = info['lead_name']
        features_df.loc[i, 'num_qrs_peaks'] = info['num_qrs_peaks']
        features_df.loc[i, 'avg_heart_rate'] = info['avg_heart_rate']
    
    # Rearrange columns to put metadata first
    metadata_cols = ['record_number', 'lead_name', 'num_qrs_peaks', 'avg_heart_rate']
    feature_cols = [col for col in features_df.columns if col not in metadata_cols]
    features_df = features_df[metadata_cols + feature_cols]
    
    # Save original features
    original_features_path = os.path.join(features_data_path, 'original_features.csv')
    features_df.to_csv(original_features_path, index=False)
    
    # Save PCA-transformed features
    pca_feature_names = [f'PC_{i+1}' for i in range(pca_features.shape[1])]
    pca_df = pd.DataFrame(pca_features, columns=pca_feature_names)
    for i, info in enumerate(record_info):
        pca_df.loc[i, 'record_number'] = info['record_number']
        pca_df.loc[i, 'lead_name'] = info['lead_name']
        pca_df.loc[i, 'num_qrs_peaks'] = info['num_qrs_peaks']
        pca_df.loc[i, 'avg_heart_rate'] = info['avg_heart_rate']
    
    # Rearrange PCA columns
    pca_df = pca_df[metadata_cols + pca_feature_names]
    pca_features_path = os.path.join(features_data_path, 'pca_features.csv')
    pca_df.to_csv(pca_features_path, index=False)
    
    # Save PCA analysis summary
    pca_summary = pd.DataFrame({
        'Component': [f'PC_{i+1}' for i in range(len(pca_info['explained_variance_ratio']))],
        'Explained_Variance_Ratio': pca_info['explained_variance_ratio'],
        'Cumulative_Variance_Ratio': pca_info['cumulative_variance_ratio']
    })
    pca_summary_path = os.path.join(features_data_path, 'pca_analysis.csv')
    pca_summary.to_csv(pca_summary_path, index=False)
    
    # Save feature importance analysis
    importance_df = pd.DataFrame({
        'Feature_Name': pca_info['most_important_features'],
        'Importance_Score': pca_info['feature_importance_scores']
    })
    importance_path = os.path.join(features_data_path, 'feature_importance.csv')
    importance_df.to_csv(importance_path, index=False)
    
    # Save overall summary
    summary = {
        'total_records_processed': len(results_list),
        'total_samples': len(record_info),
        'original_feature_count': len(feature_names),
        'pca_feature_count': pca_features.shape[1],
        'variance_explained': pca_info['total_variance_explained'],
        'dimensionality_reduction_ratio': pca_info['dimensionality_reduction_ratio']
    }
    
    summary_df = pd.DataFrame([summary])
    summary_path = os.path.join(features_data_path, 'extraction_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    
    print(f"\nFeature extraction results saved:")
    print(f"  Original features: {original_features_path}")
    print(f"  PCA features: {pca_features_path}")
    print(f"  PCA analysis: {pca_summary_path}")
    print(f"  Feature importance: {importance_path}")
    print(f"  Summary: {summary_path}")
    
    return summary

def visualize_pan_tompkins_detection(record_num, lead_name='MLII', start_sample=0, num_samples=3600):
    """
    Visualize Pan-Tompkins QRS detection results
    
    Args:
        record_num (int): Record number to visualize
        lead_name (str): Lead name to visualize
        start_sample (int): Starting sample for visualization
        num_samples (int): Number of samples to display
    """
    result = process_record_features(record_num)
    
    if result is None or lead_name not in result['leads']:
        print(f"Data not available for record {record_num}, lead {lead_name}")
        return
    
    if 'qrs_detection' not in result['leads'][lead_name]:
        print(f"QRS detection not available for record {record_num}, lead {lead_name}")
        return
    
    detection_data = result['leads'][lead_name]['qrs_detection']
    
    # Extract signals for visualization
    original = detection_data['original_signal'][start_sample:start_sample+num_samples]
    filtered = detection_data['filtered_signal'][start_sample:start_sample+num_samples]
    integrated = detection_data['integrated_signal'][start_sample:start_sample+num_samples]
    
    # Find QRS peaks in the visualization window
    qrs_peaks = detection_data['qrs_peaks']
    window_peaks = qrs_peaks[(qrs_peaks >= start_sample) & (qrs_peaks < start_sample + num_samples)] - start_sample
    
    plt.figure(figsize=(15, 12))
    
    time_axis = np.arange(len(original)) / result['sampling_rate']
    
    # Original signal with detected QRS
    plt.subplot(4, 1, 1)
    plt.plot(time_axis, original, 'b-', label='Original ECG Signal', linewidth=1)
    if len(window_peaks) > 0:
        plt.plot(time_axis[window_peaks], original[window_peaks], 'ro', 
                label=f'QRS Peaks ({len(window_peaks)} detected)', markersize=8)
    plt.title(f'Pan-Tompkins QRS Detection - Record {record_num}, Lead {lead_name}')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Filtered signal
    plt.subplot(4, 1, 2)
    plt.plot(time_axis, filtered, 'g-', label='Bandpass Filtered (5-15 Hz)', linewidth=1)
    plt.title('After Bandpass Filtering')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Integrated signal
    plt.subplot(4, 1, 3)
    plt.plot(time_axis, integrated, 'm-', label='Moving Window Integration', linewidth=1)
    if len(window_peaks) > 0:
        plt.plot(time_axis[window_peaks], integrated[window_peaks], 'ro', 
                label='Detected Peaks', markersize=6)
    plt.title('After Integration and Thresholding')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Heart rate over time
    plt.subplot(4, 1, 4)
    if len(detection_data['heart_rates']) > 0:
        rr_times = detection_data['qrs_peaks'][1:] / result['sampling_rate']
        hr_in_window = []
        hr_times = []
        
        for i, t in enumerate(rr_times):
            if start_sample/result['sampling_rate'] <= t <= (start_sample + num_samples)/result['sampling_rate']:
                hr_in_window.append(detection_data['heart_rates'][i])
                hr_times.append(t - start_sample/result['sampling_rate'])
        
        if hr_in_window:
            plt.plot(hr_times, hr_in_window, 'r.-', label=f'Heart Rate (Avg: {detection_data["avg_heart_rate"]:.1f} BPM)')
            plt.axhline(y=detection_data["avg_heart_rate"], color='r', linestyle='--', alpha=0.5)
    
    plt.title('Instantaneous Heart Rate')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Heart Rate (BPM)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print detection statistics
    print(f"\nPan-Tompkins Detection Statistics:")
    print(f"Total QRS peaks detected: {len(detection_data['qrs_peaks'])}")
    print(f"Average heart rate: {detection_data['avg_heart_rate']:.1f} BPM")
    if detection_data['hrv_metrics']:
        print(f"SDNN: {detection_data['hrv_metrics'].get('sdnn_ms', 0):.1f} ms")
        print(f"RMSSD: {detection_data['hrv_metrics'].get('rmssd_ms', 0):.1f} ms")
        print(f"pNN50: {detection_data['hrv_metrics'].get('pnn50_percent', 0):.1f}%")

def visualize_pca_analysis(pca_info, feature_matrix):
    """
    Visualize PCA analysis results
    
    Args:
        pca_info (dict): PCA analysis information
        feature_matrix (np.array): Original feature matrix
    """
    plt.figure(figsize=(15, 10))
    
    # Explained variance ratio
    plt.subplot(2, 2, 1)
    components = range(1, len(pca_info['explained_variance_ratio']) + 1)
    plt.bar(components, pca_info['explained_variance_ratio'])
    plt.title('Explained Variance Ratio by Component')
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.grid(True, alpha=0.3)
    
    # Cumulative explained variance
    plt.subplot(2, 2, 2)
    plt.plot(components, pca_info['cumulative_variance_ratio'], 'bo-')
    plt.axhline(y=0.95, color='r', linestyle='--', label='95% Variance')
    plt.axhline(y=0.90, color='orange', linestyle='--', label='90% Variance')
    plt.title('Cumulative Explained Variance')
    plt.xlabel('Principal Component')
    plt.ylabel('Cumulative Variance Ratio')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Feature importance
    plt.subplot(2, 2, 3)
    top_features = pca_info['most_important_features'][:15]
    importance_scores = pca_info['feature_importance_scores'][:15]
    
    plt.barh(range(len(top_features)), importance_scores)
    plt.yticks(range(len(top_features)), top_features)
    plt.title('Top 15 Most Important Features')
    plt.xlabel('Importance Score')
    plt.gca().invert_yaxis()
    
    # Feature correlation heatmap (sample)
    plt.subplot(2, 2, 4)
    # Show correlation for top 20 features
    top_indices = np.argsort(pca_info['feature_importance'])[-20:]
    sample_features = feature_matrix[:, top_indices]
    correlation_matrix = np.corrcoef(sample_features.T)
    
    im = plt.imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Feature Correlation Matrix (Top 20 Features)')
    plt.colorbar(im)
    
    plt.tight_layout()
    plt.show()
    
    # Print PCA summary
    print(f"\nPCA Analysis Summary:")
    print(f"Original features: {pca_info['original_feature_count']}")
    print(f"Selected components: {pca_info['n_components']}")
    print(f"Total variance explained: {pca_info['total_variance_explained']:.3f}")
    print(f"Dimensionality reduction: {pca_info['dimensionality_reduction_ratio']:.3f}")
    print(f"\nTop 10 Most Important Features:")
    for i, (feature, score) in enumerate(zip(pca_info['most_important_features'][:10], 
                                           pca_info['feature_importance_scores'][:10])):
        print(f"  {i+1:2d}. {feature:<30} ({score:.4f})")

def main_feature_extraction_pca():
    """
    Main function for feature extraction and PCA analysis
    """
    print("="*70)
    print("Pan-Tompkins Feature Extraction and PCA Feature Selection")
    print("="*70)
    
    # Define records to process
    train_records = [101, 106, 108, 109, 112, 114, 115, 113, 116, 118, 119, 
                    122, 124, 201, 203, 205, 207, 208, 209, 215, 220, 223, 230]
    test_records = [100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 
                   212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234]
    all_records = train_records + test_records
    
    print(f"Processing {len(all_records)} records for feature extraction...")
    
    # Step 1: Feature extraction using Pan-Tompkins
    print("\nStep 1: Pan-Tompkins QRS Detection and Feature Extraction")
    results = []
    for i, record_num in enumerate(all_records):
        print(f"[{i+1}/{len(all_records)}] Processing record {record_num}...")
        result = process_record_features(record_num)
        if result:
            results.append(result)
    
    if not results:
        print("❌ No records were successfully processed.")
        return None, None, None
    
    # Step 2: Create feature matrix
    print(f"\nStep 2: Creating Feature Matrix")
    feature_matrix, feature_names, record_info = create_feature_matrix(results)
    
    print(f"Feature matrix shape: {feature_matrix.shape}")
    print(f"Total features extracted: {len(feature_names)}")
    
    # Step 3: PCA feature selection
    print(f"\nStep 3: PCA Feature Selection")
    pca_selector = PCAFeatureSelector(variance_threshold=0.95)
    pca_features, pca_info = pca_selector.fit_transform(feature_matrix, feature_names)
    
    print(f"PCA components selected: {pca_info['n_components']}")
    print(f"Variance explained: {pca_info['total_variance_explained']:.3f}")
    print(f"Dimensionality reduction: {pca_info['original_feature_count']} → {pca_info['n_components']}")
    
    # Step 4: Save results
    print(f"\nStep 4: Saving Results")
    summary = save_feature_results(results, feature_matrix, feature_names, 
                                 record_info, pca_info, pca_features)
    
    print("\n" + "="*70)
    print("FEATURE EXTRACTION AND PCA COMPLETED")
    print("="*70)
    print(f"✅ Successfully processed {summary['total_records_processed']} records")
    print(f"✅ Generated {summary['total_samples']} feature samples")
    print(f"✅ Extracted {summary['original_feature_count']} original features")
    print(f"✅ Reduced to {summary['pca_feature_count']} PCA components")
    print(f"✅ Preserved {summary['variance_explained']:.1%} of variance")
    print(f"✅ Results saved to: {features_data_path}")
    print("="*70)
    
    return results, pca_info, pca_features

# Utility function to load processed features
def load_processed_features():
    """
    Load previously processed features
    
    Returns:
        tuple: (original_features_df, pca_features_df, pca_analysis_df)
    """
    try:
        original_path = os.path.join(features_data_path, 'original_features.csv')
        pca_path = os.path.join(features_data_path, 'pca_features.csv')
        analysis_path = os.path.join(features_data_path, 'pca_analysis.csv')
        
        original_df = pd.read_csv(original_path) if os.path.exists(original_path) else None
        pca_df = pd.read_csv(pca_path) if os.path.exists(pca_path) else None
        analysis_df = pd.read_csv(analysis_path) if os.path.exists(analysis_path) else None
        
        return original_df, pca_df, analysis_df
    except Exception as e:
        print(f"Error loading processed features: {str(e)}")
        return None, None, None

print("Pan-Tompkins Feature Extraction and PCA Script Ready!")
print("Run main_feature_extraction_pca() to start the complete pipeline.")
print("Use visualize_pan_tompkins_detection(record_num, lead_name) to see QRS detection.")
print("Use load_processed_features() to load previously processed results.")

# Execute the main pipeline
print("\nStarting Pan-Tompkins feature extraction and PCA pipeline...")
results, pca_info, pca_features = main_feature_extraction_pca()

if results and pca_info:
    print("\n" + "="*70)
    print("VISUALIZATION AND ANALYSIS")
    print("="*70)
    
    # Generate sample visualizations
    print("Generating Pan-Tompkins detection visualization...")
    visualize_pan_tompkins_detection(101, 'MLII', 0, 1800)
    
    print("\nGenerating PCA analysis visualization...")
    # Create a sample feature matrix for visualization
    feature_matrix, feature_names, _ = create_feature_matrix(results[:5])  # Use first 5 records for demo
    visualize_pca_analysis(pca_info, feature_matrix)
    
    print(f"\n🎉 Complete pipeline finished successfully!")
    print(f"📊 Your ECG features are extracted and ready for machine learning!")
    
else:
    print("❌ Feature extraction and PCA pipeline failed.")
