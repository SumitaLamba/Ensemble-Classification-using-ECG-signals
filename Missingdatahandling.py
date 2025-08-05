#!/usr/bin/env python3
"""
Missing Data Detection and Reconstruction for MIT-BIH Dataset
This script performs automated signal quality assessment and handles missing data
using cubic spline interpolation for ECG signals.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate, signal
from scipy.stats import zscore
import os
import wfdb
from google.colab import drive
import warnings
warnings.filterwarnings('ignore')

# Mount Google Drive to access previously downloaded data
try:
    drive.mount('/content/drive')
except:
    print("Drive already mounted or unavailable")

# Define paths
base_path = '/content/drive/MyDrive/MIT_BIH_Dataset'
raw_data_path = os.path.join(base_path, 'raw_data')
processed_data_path = os.path.join(base_path, 'processed_data')
quality_assessment_path = os.path.join(base_path, 'quality_assessment')

# Create quality assessment directory
os.makedirs(quality_assessment_path, exist_ok=True)

class SignalQualityAssessment:
    """
    Class for automated signal quality assessment and missing data detection
    """
    
    def __init__(self, sampling_rate=360):
        """
        Initialize signal quality assessment
        
        Args:
            sampling_rate (int): Sampling rate of ECG signals (default: 360 Hz for MIT-BIH)
        """
        self.fs = sampling_rate
        self.quality_metrics = {}
        
    def detect_flat_line_segments(self, signal, window_size=10, threshold=0.01):
        """
        Detect flat line segments in ECG signal
        
        Args:
            signal (np.array): ECG signal
            window_size (int): Window size for detection
            threshold (float): Threshold for variance
            
        Returns:
            np.array: Boolean array indicating flat line segments
        """
        flat_segments = np.zeros(len(signal), dtype=bool)
        
        for i in range(0, len(signal) - window_size, window_size):
            window = signal[i:i + window_size]
            if np.var(window) < threshold:
                flat_segments[i:i + window_size] = True
                
        return flat_segments
    
    def detect_outliers(self, signal, z_threshold=4):
        """
        Detect outliers using Z-score method
        
        Args:
            signal (np.array): ECG signal
            z_threshold (float): Z-score threshold for outlier detection
            
        Returns:
            np.array: Boolean array indicating outliers
        """
        try:
            z_scores = np.abs(zscore(signal, nan_policy='omit'))
            outliers = z_scores > z_threshold
            return outliers
        except:
            return np.zeros(len(signal), dtype=bool)
    
    def detect_amplitude_saturation(self, signal, saturation_threshold=0.95):
        """
        Detect amplitude saturation (clipping)
        
        Args:
            signal (np.array): ECG signal
            saturation_threshold (float): Percentage of max amplitude
            
        Returns:
            np.array: Boolean array indicating saturated samples
        """
        max_amp = np.max(np.abs(signal))
        threshold = max_amp * saturation_threshold
        saturated = np.abs(signal) >= threshold
        return saturated
    
    def detect_high_frequency_noise(self, signal, noise_freq_threshold=100):
        """
        Detect high frequency noise using spectral analysis
        
        Args:
            signal (np.array): ECG signal
            noise_freq_threshold (float): Frequency threshold for noise detection
            
        Returns:
            float: Noise ratio (0-1, where 1 is very noisy)
        """
        try:
            # Compute power spectral density
            freqs, psd = signal.welch(signal, fs=self.fs, nperseg=min(1024, len(signal)//4))
            
            # Calculate power in high frequency band vs total power
            high_freq_mask = freqs > noise_freq_threshold
            high_freq_power = np.sum(psd[high_freq_mask])
            total_power = np.sum(psd)
            
            noise_ratio = high_freq_power / (total_power + 1e-10)
            return noise_ratio
        except:
            return 0.0
    
    def detect_baseline_wander(self, signal, wander_freq_threshold=1.0):
        """
        Detect baseline wander using low frequency analysis
        
        Args:
            signal (np.array): ECG signal
            wander_freq_threshold (float): Frequency threshold for baseline wander
            
        Returns:
            float: Baseline wander ratio
        """
        try:
            # High-pass filter to remove baseline wander
            nyquist = self.fs / 2
            low_cutoff = wander_freq_threshold / nyquist
            
            if low_cutoff >= 1.0:
                return 0.0
                
            b, a = signal.butter(2, low_cutoff, btype='high')
            filtered_signal = signal.filtfilt(b, a, signal)
            
            # Calculate ratio of original to filtered signal variance
            original_var = np.var(signal)
            filtered_var = np.var(filtered_signal)
            
            wander_ratio = 1 - (filtered_var / (original_var + 1e-10))
            return max(0, wander_ratio)
        except:
            return 0.0
    
    def calculate_signal_to_noise_ratio(self, signal):
        """
        Calculate Signal-to-Noise Ratio (SNR)
        
        Args:
            signal (np.array): ECG signal
            
        Returns:
            float: SNR in dB
        """
        try:
            # Estimate signal power (using median for robustness)
            signal_power = np.median(signal ** 2)
            
            # Estimate noise power using high-pass filtered signal
            b, a = signal.butter(2, 50/(self.fs/2), btype='high')
            noise = signal.filtfilt(b, a, signal)
            noise_power = np.median(noise ** 2)
            
            if noise_power == 0:
                return float('inf')
                
            snr_db = 10 * np.log10(signal_power / (noise_power + 1e-10))
            return snr_db
        except:
            return 0.0
    
    def assess_signal_quality(self, signal, record_num):
        """
        Comprehensive signal quality assessment
        
        Args:
            signal (np.array): ECG signal
            record_num (int): Record number for identification
            
        Returns:
            dict: Quality assessment results
        """
        # Detect various types of artifacts
        flat_segments = self.detect_flat_line_segments(signal)
        outliers = self.detect_outliers(signal)
        saturated = self.detect_amplitude_saturation(signal)
        
        # Calculate quality metrics
        noise_ratio = self.detect_high_frequency_noise(signal)
        baseline_wander = self.detect_baseline_wander(signal)
        snr = self.calculate_signal_to_noise_ratio(signal)
        
        # Combine all problematic samples
        problematic_samples = flat_segments | outliers | saturated
        
        # Calculate overall quality score (0-1, where 1 is best quality)
        flat_percentage = np.sum(flat_segments) / len(signal)
        outlier_percentage = np.sum(outliers) / len(signal)
        saturation_percentage = np.sum(saturated) / len(signal)
        
        quality_score = 1.0 - (flat_percentage + outlier_percentage + 
                             saturation_percentage + noise_ratio * 0.1 + 
                             baseline_wander * 0.1)
        quality_score = max(0, min(1, quality_score))
        
        assessment = {
            'record_number': record_num,
            'signal_length': len(signal),
            'flat_segments': flat_segments,
            'outliers': outliers,
            'saturated_samples': saturated,
            'problematic_samples': problematic_samples,
            'flat_percentage': flat_percentage * 100,
            'outlier_percentage': outlier_percentage * 100,
            'saturation_percentage': saturation_percentage * 100,
            'noise_ratio': noise_ratio,
            'baseline_wander': baseline_wander,
            'snr_db': snr,
            'overall_quality_score': quality_score,
            'total_problematic_samples': np.sum(problematic_samples),
            'problematic_percentage': (np.sum(problematic_samples) / len(signal)) * 100
        }
        
        return assessment

class MissingDataReconstruction:
    """
    Class for reconstructing missing data using cubic spline interpolation
    """
    
    def __init__(self):
        self.reconstruction_stats = {}
    
    def cubic_spline_interpolation(self, signal, missing_mask):
        """
        Reconstruct missing data using cubic spline interpolation
        
        Args:
            signal (np.array): Original signal with missing data
            missing_mask (np.array): Boolean array indicating missing samples
            
        Returns:
            np.array: Reconstructed signal
        """
        if not np.any(missing_mask):
            return signal.copy()
        
        # Create indices
        indices = np.arange(len(signal))
        valid_indices = indices[~missing_mask]
        valid_values = signal[~missing_mask]
        
        if len(valid_values) < 4:  # Need at least 4 points for cubic spline
            # Fallback to linear interpolation
            return self.linear_interpolation(signal, missing_mask)
        
        try:
            # Create cubic spline interpolator
            cs = interpolate.CubicSpline(valid_indices, valid_values, 
                                       bc_type='natural', extrapolate=True)
            
            # Reconstruct missing values
            reconstructed_signal = signal.copy()
            reconstructed_signal[missing_mask] = cs(indices[missing_mask])
            
            return reconstructed_signal
        except:
            # Fallback to linear interpolation if cubic spline fails
            return self.linear_interpolation(signal, missing_mask)
    
    def linear_interpolation(self, signal, missing_mask):
        """
        Fallback linear interpolation for missing data
        
        Args:
            signal (np.array): Original signal with missing data
            missing_mask (np.array): Boolean array indicating missing samples
            
        Returns:
            np.array: Reconstructed signal
        """
        if not np.any(missing_mask):
            return signal.copy()
        
        indices = np.arange(len(signal))
        valid_indices = indices[~missing_mask]
        valid_values = signal[~missing_mask]
        
        if len(valid_values) < 2:
            # If too few valid points, use mean imputation
            mean_value = np.mean(valid_values) if len(valid_values) > 0 else 0
            reconstructed_signal = signal.copy()
            reconstructed_signal[missing_mask] = mean_value
            return reconstructed_signal
        
        # Linear interpolation
        reconstructed_signal = signal.copy()
        reconstructed_signal[missing_mask] = np.interp(
            indices[missing_mask], valid_indices, valid_values
        )
        
        return reconstructed_signal
    
    def validate_reconstruction(self, original_signal, reconstructed_signal, missing_mask):
        """
        Validate the quality of reconstruction
        
        Args:
            original_signal (np.array): Original signal
            reconstructed_signal (np.array): Reconstructed signal
            missing_mask (np.array): Missing data mask
            
        Returns:
            dict: Validation metrics
        """
        if not np.any(missing_mask):
            return {'reconstruction_needed': False}
        
        # Calculate reconstruction metrics
        reconstructed_values = reconstructed_signal[missing_mask]
        
        # Check for continuity at boundaries
        continuity_errors = []
        missing_segments = self._find_missing_segments(missing_mask)
        
        for start, end in missing_segments:
            if start > 0:
                boundary_diff_start = abs(reconstructed_signal[start] - original_signal[start-1])
                continuity_errors.append(boundary_diff_start)
            if end < len(original_signal) - 1:
                boundary_diff_end = abs(reconstructed_signal[end] - original_signal[end+1])
                continuity_errors.append(boundary_diff_end)
        
        avg_continuity_error = np.mean(continuity_errors) if continuity_errors else 0
        
        validation = {
            'reconstruction_needed': True,
            'num_reconstructed_samples': np.sum(missing_mask),
            'reconstruction_percentage': (np.sum(missing_mask) / len(original_signal)) * 100,
            'num_missing_segments': len(missing_segments),
            'avg_continuity_error': avg_continuity_error,
            'reconstructed_values_std': np.std(reconstructed_values),
            'reconstructed_values_range': np.ptp(reconstructed_values)
        }
        
        return validation
    
    def _find_missing_segments(self, missing_mask):
        """
        Find continuous segments of missing data
        
        Args:
            missing_mask (np.array): Boolean array indicating missing samples
            
        Returns:
            list: List of (start, end) tuples for missing segments
        """
        segments = []
        in_segment = False
        start = 0
        
        for i, is_missing in enumerate(missing_mask):
            if is_missing and not in_segment:
                start = i
                in_segment = True
            elif not is_missing and in_segment:
                segments.append((start, i-1))
                in_segment = False
        
        # Handle case where signal ends with missing data
        if in_segment:
            segments.append((start, len(missing_mask)-1))
        
        return segments

def process_record_quality_and_reconstruction(record_num):
    """
    Process a single record for quality assessment and missing data reconstruction
    
    Args:
        record_num (int): Record number to process
        
    Returns:
        dict: Processing results
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
        qa = SignalQualityAssessment(sampling_rate=record.fs)
        reconstructor = MissingDataReconstruction()
        
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
            
            # Quality assessment
            quality_assessment = qa.assess_signal_quality(signal_data, record_num)
            
            # Reconstruct missing data
            missing_mask = quality_assessment['problematic_samples']
            reconstructed_signal = reconstructor.cubic_spline_interpolation(
                signal_data, missing_mask
            )
            
            # Validate reconstruction
            validation = reconstructor.validate_reconstruction(
                signal_data, reconstructed_signal, missing_mask
            )
            
            results['leads'][lead_name] = {
                'original_signal': signal_data,
                'reconstructed_signal': reconstructed_signal,
                'missing_mask': missing_mask,
                'quality_assessment': quality_assessment,
                'reconstruction_validation': validation
            }
            
            print(f"Record {record_num}, Lead {lead_name}:")
            print(f"  Quality Score: {quality_assessment['overall_quality_score']:.3f}")
            print(f"  Problematic samples: {quality_assessment['problematic_percentage']:.2f}%")
            if validation['reconstruction_needed']:
                print(f"  Reconstructed: {validation['reconstruction_percentage']:.2f}%")
        
        return results
        
    except Exception as e:
        print(f"Error processing record {record_num}: {str(e)}")
        return None

def save_quality_assessment_results(results_list):
    """
    Save quality assessment and reconstruction results
    
    Args:
        results_list (list): List of processing results
    """
    # Create summary DataFrame
    summary_data = []
    detailed_data = []
    
    for result in results_list:
        if result is None:
            continue
            
        record_num = result['record_number']
        
        for lead_name, lead_data in result['leads'].items():
            qa = lead_data['quality_assessment']
            val = lead_data['reconstruction_validation']
            
            summary_row = {
                'record_number': record_num,
                'lead_name': lead_name,
                'signal_length': qa['signal_length'],
                'quality_score': qa['overall_quality_score'],
                'snr_db': qa['snr_db'],
                'problematic_percentage': qa['problematic_percentage'],
                'flat_percentage': qa['flat_percentage'],
                'outlier_percentage': qa['outlier_percentage'],
                'saturation_percentage': qa['saturation_percentage'],
                'noise_ratio': qa['noise_ratio'],
                'baseline_wander': qa['baseline_wander']
            }
            
            if val['reconstruction_needed']:
                summary_row.update({
                    'reconstruction_needed': True,
                    'reconstruction_percentage': val['reconstruction_percentage'],
                    'num_missing_segments': val['num_missing_segments'],
                    'avg_continuity_error': val['avg_continuity_error']
                })
            else:
                summary_row.update({
                    'reconstruction_needed': False,
                    'reconstruction_percentage': 0,
                    'num_missing_segments': 0,
                    'avg_continuity_error': 0
                })
            
            summary_data.append(summary_row)
    
    # Save summary
    summary_df = pd.DataFrame(summary_data)
    summary_path = os.path.join(quality_assessment_path, 'quality_assessment_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    
    print(f"\nQuality assessment summary saved to: {summary_path}")
    
    # Display overall statistics
    print("\n" + "="*60)
    print("QUALITY ASSESSMENT SUMMARY")
    print("="*60)
    print(f"Total records processed: {len(results_list)}")
    print(f"Average quality score: {summary_df['quality_score'].mean():.3f}")
    print(f"Records needing reconstruction: {summary_df['reconstruction_needed'].sum()}")
    print(f"Average reconstruction percentage: {summary_df['reconstruction_percentage'].mean():.2f}%")
    print(f"Average SNR: {summary_df['snr_db'].mean():.2f} dB")
    print("="*60)
    
    return summary_df

def main_quality_assessment():
    """
    Main function for quality assessment and missing data reconstruction
    """
    print("="*60)
    print("MIT-BIH Signal Quality Assessment and Missing Data Reconstruction")
    print("="*60)
    
    # Define records to process (same as in original script)
    train_records = [101, 106, 108, 109, 112, 114, 115, 113, 116, 118, 119, 
                    122, 124, 201, 203, 205, 207, 208, 209, 215, 220, 223, 230]
    test_records = [100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 
                   212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234]
    all_records = train_records + test_records
    
    print(f"Processing {len(all_records)} records for quality assessment...")
    
    # Process each record
    results = []
    for record_num in all_records:
        print(f"\nProcessing record {record_num}...")
        result = process_record_quality_and_reconstruction(record_num)
        if result:
            results.append(result)
    
    # Save results
    if results:
        summary_df = save_quality_assessment_results(results)
        return results, summary_df
    else:
        print("No records were successfully processed.")
        return None, None

# Utility functions
def visualize_reconstruction(record_num, lead_name='MLII', start_sample=0, num_samples=3600):
    """
    Visualize original vs reconstructed signal for a specific record
    
    Args:
        record_num (int): Record number to visualize
        lead_name (str): Lead name to visualize
        start_sample (int): Starting sample for visualization
        num_samples (int): Number of samples to display
    """
    result = process_record_quality_and_reconstruction(record_num)
    
    if result is None or lead_name not in result['leads']:
        print(f"Data not available for record {record_num}, lead {lead_name}")
        return
    
    lead_data = result['leads'][lead_name]
    original = lead_data['original_signal'][start_sample:start_sample+num_samples]
    reconstructed = lead_data['reconstructed_signal'][start_sample:start_sample+num_samples]
    missing_mask = lead_data['missing_mask'][start_sample:start_sample+num_samples]
    
    plt.figure(figsize=(15, 8))
    
    time_axis = np.arange(len(original)) / 360  # Assuming 360 Hz sampling rate
    
    plt.subplot(2, 1, 1)
    plt.plot(time_axis, original, 'b-', label='Original Signal', alpha=0.7)
    plt.plot(time_axis[missing_mask], original[missing_mask], 'ro', 
             label='Missing/Problematic Data', markersize=3)
    plt.title(f'Original Signal - Record {record_num}, Lead {lead_name}')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 1, 2)
    plt.plot(time_axis, reconstructed, 'g-', label='Reconstructed Signal', alpha=0.7)
    plt.plot(time_axis[missing_mask], reconstructed[missing_mask], 'ro', 
             label='Reconstructed Data', markersize=3)
    plt.title('Reconstructed Signal')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    qa = lead_data['quality_assessment']
    val = lead_data['reconstruction_validation']
    
    print(f"\nQuality Assessment for Record {record_num}, Lead {lead_name}:")
    print(f"Quality Score: {qa['overall_quality_score']:.3f}")
    print(f"SNR: {qa['snr_db']:.2f} dB")
    print(f"Problematic samples: {qa['problematic_percentage']:.2f}%")
    if val['reconstruction_needed']:
        print(f"Reconstruction: {val['reconstruction_percentage']:.2f}% of signal")

print("Missing Data Detection and Reconstruction Script Ready!")
print("Run main_quality_assessment() to start the quality assessment and reconstruction.")
print("Use visualize_reconstruction(record_num, lead_name) to visualize specific records.")
