#!/usr/bin/env python3
"""
DWT-based ECG Signal Denoising for MIT-BIH Dataset
This script performs denoising on quality-assessed and reconstructed ECG signals
"""

# Install required packages
import subprocess
import sys

def install_package(package):
    """Install a package using pip"""
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Install PyWavelets if not available
try:
    import pywt
    print("PyWavelets already available!")
except ImportError:
    print("Installing PyWavelets...")
    install_package("PyWavelets")
    import pywt

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
import os
import wfdb
from google.colab import drive
import warnings
warnings.filterwarnings('ignore')

# Define paths
base_path = 'your/dataset/path'
raw_data_path = os.path.join(base_path, 'raw_data')
quality_assessment_path = os.path.join(base_path, 'quality_assessment')
denoised_data_path = os.path.join(base_path, 'denoised_data')

# Create denoised data directory
os.makedirs(denoised_data_path, exist_ok=True)

class DWTDenoiser:
    """
    Class for DWT-based ECG signal denoising
    """
    
    def __init__(self, wavelet='db6', mode='symmetric'):
        """
        Initialize DWT denoiser
        
        Args:
            wavelet (str): Wavelet type (default: 'db6' - good for ECG)
            mode (str): Signal extension mode
        """
        self.wavelet = wavelet
        self.mode = mode
        self.denoising_stats = {}
        
        # Verify wavelet availability
        if wavelet not in pywt.wavelist():
            print(f"Warning: Wavelet '{wavelet}' not available. Using 'db4' instead.")
            self.wavelet = 'db4'
    
    def calculate_noise_variance(self, signal_coeffs):
        """
        Estimate noise variance using the finest detail coefficients
        
        Args:
            signal_coeffs (list): Wavelet coefficients from decomposition
            
        Returns:
            float: Estimated noise variance
        """
        # Use the finest detail coefficients (highest frequency)
        detail_coeffs = signal_coeffs[-1]  # Last level detail coefficients
        
        # Robust noise variance estimation using median absolute deviation
        sigma = np.median(np.abs(detail_coeffs)) / 0.6745
        return sigma ** 2
    
    def soft_thresholding(self, coeffs, threshold):
        """
        Apply soft thresholding to wavelet coefficients
        
        Args:
            coeffs (np.array): Wavelet coefficients
            threshold (float): Thresholding value
            
        Returns:
            np.array: Thresholded coefficients
        """
        return np.sign(coeffs) * np.maximum(np.abs(coeffs) - threshold, 0)
    
    def hard_thresholding(self, coeffs, threshold):
        """
        Apply hard thresholding to wavelet coefficients
        
        Args:
            coeffs (np.array): Wavelet coefficients
            threshold (float): Thresholding value
            
        Returns:
            np.array: Thresholded coefficients
        """
        return coeffs * (np.abs(coeffs) > threshold)
    
    def adaptive_threshold_calculation(self, signal, decomposition_level):
        """
        Calculate adaptive threshold based on signal characteristics
        
        Args:
            signal (np.array): Input signal
            decomposition_level (int): Number of decomposition levels
            
        Returns:
            float: Calculated threshold
        """
        # Perform DWT decomposition
        coeffs = pywt.wavedec(signal, self.wavelet, level=decomposition_level, mode=self.mode)
        
        # Estimate noise variance
        noise_var = self.calculate_noise_variance(coeffs)
        
        # Calculate universal threshold
        N = len(signal)
        universal_threshold = noise_var * np.sqrt(2 * np.log(N))
        
        return universal_threshold
    
    def sure_threshold(self, coeffs):
        """
        Calculate SURE (Stein's Unbiased Risk Estimate) threshold
        
        Args:
            coeffs (np.array): Wavelet coefficients
            
        Returns:
            float: SURE threshold
        """
        N = len(coeffs)
        sorted_coeffs = np.sort(np.abs(coeffs))
        
        # Calculate SURE threshold
        risks = []
        for i, t in enumerate(sorted_coeffs):
            # Count coefficients above threshold
            above_threshold = np.sum(np.abs(coeffs) > t)
            
            # SURE risk calculation
            risk = N - 2 * above_threshold + np.sum(np.minimum(np.abs(coeffs), t) ** 2)
            risks.append(risk)
        
        # Find threshold that minimizes risk
        min_risk_idx = np.argmin(risks)
        sure_threshold = sorted_coeffs[min_risk_idx]
        
        return sure_threshold
    
    def bayes_threshold(self, coeffs):
        """
        Calculate Bayes threshold using empirical Bayes approach
        
        Args:
            coeffs (np.array): Wavelet coefficients
            
        Returns:
            float: Bayes threshold
        """
        # Estimate noise variance
        sigma_n = np.median(np.abs(coeffs)) / 0.6745
        
        # Estimate signal variance
        sigma_x = max(0, np.var(coeffs) - sigma_n**2)
        
        if sigma_x == 0:
            return sigma_n * np.sqrt(2 * np.log(len(coeffs)))
        
        # Bayes threshold
        bayes_thresh = sigma_n**2 / np.sqrt(sigma_x)
        
        return bayes_thresh
    
    def denoise_signal(self, signal, method='soft', decomposition_level=6, 
                      threshold_method='adaptive'):
        """
        Denoise ECG signal using DWT
        
        Args:
            signal (np.array): Input ECG signal
            method (str): Thresholding method ('soft' or 'hard')
            decomposition_level (int): Number of decomposition levels
            threshold_method (str): Threshold calculation method
            
        Returns:
            tuple: (denoised_signal, denoising_info)
        """
        try:
            # Ensure signal length is suitable for decomposition
            min_length = 2 ** decomposition_level
            if len(signal) < min_length:
                # Pad signal if too short
                pad_length = min_length - len(signal)
                signal_padded = np.pad(signal, (0, pad_length), mode='symmetric')
                was_padded = True
            else:
                signal_padded = signal
                was_padded = False
            
            # Perform DWT decomposition
            coeffs = pywt.wavedec(signal_padded, self.wavelet, 
                                level=decomposition_level, mode=self.mode)
            
            # Calculate threshold based on selected method
            if threshold_method == 'adaptive':
                threshold = self.adaptive_threshold_calculation(signal_padded, decomposition_level)
            elif threshold_method == 'sure':
                # Apply SURE to detail coefficients
                all_details = np.concatenate(coeffs[1:])  # All detail coefficients
                threshold = self.sure_threshold(all_details)
            elif threshold_method == 'bayes':
                all_details = np.concatenate(coeffs[1:])
                threshold = self.bayes_threshold(all_details)
            else:
                # Universal threshold
                threshold = self.adaptive_threshold_calculation(signal_padded, decomposition_level)
            
            # Apply thresholding to detail coefficients only (preserve approximation)
            coeffs_thresh = coeffs.copy()
            for i in range(1, len(coeffs)):  # Skip approximation coefficients (index 0)
                if method == 'soft':
                    coeffs_thresh[i] = self.soft_thresholding(coeffs[i], threshold)
                elif method == 'hard':
                    coeffs_thresh[i] = self.hard_thresholding(coeffs[i], threshold)
            
            # Reconstruct signal
            denoised_signal_padded = pywt.waverec(coeffs_thresh, self.wavelet, mode=self.mode)
            
            # Remove padding if it was added
            if was_padded:
                denoised_signal = denoised_signal_padded[:len(signal)]
            else:
                denoised_signal = denoised_signal_padded[:len(signal)]
            
            # Calculate denoising metrics
            noise_removed = signal - denoised_signal
            noise_power = np.var(noise_removed)
            signal_power = np.var(denoised_signal)
            snr_improvement = 10 * np.log10(signal_power / (noise_power + 1e-10))
            
            # Calculate preservation of important features (QRS complexes)
            correlation = np.corrcoef(signal, denoised_signal)[0, 1]
            
            denoising_info = {
                'threshold_method': threshold_method,
                'thresholding_method': method,
                'threshold_value': threshold,
                'decomposition_level': decomposition_level,
                'wavelet_used': self.wavelet,
                'noise_power': noise_power,
                'signal_power': signal_power,
                'snr_improvement_db': snr_improvement,
                'correlation_coefficient': correlation,
                'noise_variance_estimate': self.calculate_noise_variance(coeffs),
                'coefficients_levels': len(coeffs),
                'original_length': len(signal),
                'was_padded': was_padded
            }
            
            return denoised_signal, denoising_info
            
        except Exception as e:
            print(f"Error in denoising: {str(e)}")
            return signal, {'error': str(e)}
    
    def multi_level_denoising(self, signal, levels=[4, 6, 8]):
        """
        Apply denoising at multiple decomposition levels and select best result
        
        Args:
            signal (np.array): Input signal
            levels (list): List of decomposition levels to try
            
        Returns:
            tuple: (best_denoised_signal, best_info)
        """
        results = []
        
        for level in levels:
            denoised, info = self.denoise_signal(signal, decomposition_level=level)
            if 'error' not in info:
                results.append((denoised, info, level))
        
        if not results:
            return signal, {'error': 'All denoising attempts failed'}
        
        # Select best result based on correlation and SNR improvement
        best_score = -np.inf
        best_result = results[0]
        
        for denoised, info, level in results:
            # Scoring function: balance correlation and SNR improvement
            score = 0.7 * info['correlation_coefficient'] + 0.3 * (info['snr_improvement_db'] / 20)
            if score > best_score:
                best_score = score
                best_result = (denoised, info, level)
        
        best_denoised, best_info, best_level = best_result
        best_info['selected_level'] = best_level
        best_info['selection_score'] = best_score
        
        return best_denoised, best_info

def process_record_denoising(record_num):
    """
    Process a single record for DWT denoising
    
    Args:
        record_num (int): Record number to process
        
    Returns:
        dict: Denoising results
    """
    try:
        # Load the record (reconstructed signals if available)
        record_path = os.path.join(raw_data_path, str(record_num))
        record = wfdb.rdrecord(record_path)
        
        if record.p_signal is None:
            print(f"No signal data found for record {record_num}")
            return None
        
        signals = record.p_signal
        
        # Initialize denoiser with different wavelets for comparison
        denoisers = {
            'db6': DWTDenoiser(wavelet='db6'),      # Good for ECG
            'db8': DWTDenoiser(wavelet='db8'),      # Higher order Daubechies
            'bior4.4': DWTDenoiser(wavelet='bior4.4'),  # Biorthogonal
            'coif4': DWTDenoiser(wavelet='coif4')   # Coiflets
        }
        
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
                
            original_signal = signals[:, lead_idx]
            
            print(f"Processing Record {record_num}, Lead {lead_name}...")
            
            lead_results = {}
            
            # Try different wavelets and select best
            best_wavelet = None
            best_score = -np.inf
            best_denoised = original_signal
            best_info = {}
            
            for wavelet_name, denoiser in denoisers.items():
                try:
                    # Multi-level denoising
                    denoised_signal, denoising_info = denoiser.multi_level_denoising(original_signal)
                    
                    if 'error' not in denoising_info:
                        score = denoising_info.get('selection_score', 0)
                        if score > best_score:
                            best_score = score
                            best_wavelet = wavelet_name
                            best_denoised = denoised_signal
                            best_info = denoising_info
                            
                except Exception as e:
                    print(f"  Warning: {wavelet_name} failed - {str(e)}")
                    continue
            
            # Store results for best performing wavelet
            results['leads'][lead_name] = {
                'original_signal': original_signal,
                'denoised_signal': best_denoised,
                'best_wavelet': best_wavelet,
                'denoising_info': best_info,
                'signal_length': len(original_signal)
            }
            
            # Print summary
            if 'error' not in best_info:
                print(f"  Best wavelet: {best_wavelet}")
                print(f"  SNR improvement: {best_info['snr_improvement_db']:.2f} dB")
                print(f"  Correlation: {best_info['correlation_coefficient']:.4f}")
            else:
                print(f"  Denoising failed: {best_info.get('error', 'Unknown error')}")
        
        return results
        
    except Exception as e:
        print(f"Error processing record {record_num}: {str(e)}")
        return None

def save_denoising_results(results_list):
    """
    Save denoising results and create summary
    
    Args:
        results_list (list): List of denoising results
    """
    # Create summary DataFrame
    summary_data = []
    
    for result in results_list:
        if result is None:
            continue
            
        record_num = result['record_number']
        
        for lead_name, lead_data in result['leads'].items():
            if 'error' in lead_data.get('denoising_info', {}):
                continue
                
            info = lead_data['denoising_info']
            
            summary_row = {
                'record_number': record_num,
                'lead_name': lead_name,
                'best_wavelet': lead_data['best_wavelet'],
                'decomposition_level': info.get('selected_level', info.get('decomposition_level')),
                'threshold_method': info['threshold_method'],
                'threshold_value': info['threshold_value'],
                'snr_improvement_db': info['snr_improvement_db'],
                'correlation_coefficient': info['correlation_coefficient'],
                'noise_power': info['noise_power'],
                'signal_power': info['signal_power'],
                'selection_score': info.get('selection_score', 0)
            }
            
            summary_data.append(summary_row)
    
    # Save summary
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_path = os.path.join(denoised_data_path, 'denoising_summary.csv')
        summary_df.to_csv(summary_path, index=False)
        
        print(f"\nDenoising summary saved to: {summary_path}")
        
        # Display statistics
        print("\n" + "="*60)
        print("DWT DENOISING SUMMARY")
        print("="*60)
        print(f"Total records processed: {len(results_list)}")
        print(f"Average SNR improvement: {summary_df['snr_improvement_db'].mean():.2f} dB")
        print(f"Average correlation: {summary_df['correlation_coefficient'].mean():.4f}")
        
        # Wavelet usage statistics
        wavelet_counts = summary_df['best_wavelet'].value_counts()
        print(f"\nBest performing wavelets:")
        for wavelet, count in wavelet_counts.items():
            print(f"  {wavelet}: {count} times")
        
        # Quality categories
        excellent = len(summary_df[summary_df['correlation_coefficient'] > 0.95])
        good = len(summary_df[(summary_df['correlation_coefficient'] > 0.9) & 
                             (summary_df['correlation_coefficient'] <= 0.95)])
        fair = len(summary_df[summary_df['correlation_coefficient'] <= 0.9])
        
        print(f"\nDenoising Quality Distribution:")
        print(f"Excellent (correlation > 0.95): {excellent}")
        print(f"Good (correlation 0.9-0.95): {good}")
        print(f"Fair (correlation < 0.9): {fair}")
        print("="*60)
        
        return summary_df
    else:
        print("No successful denoising results to save.")
        return None

def visualize_denoising_comparison(record_num, lead_name='MLII', start_sample=0, num_samples=3600):
    """
    Visualize original vs denoised signal comparison
    
    Args:
        record_num (int): Record number to visualize
        lead_name (str): Lead name to visualize
        start_sample (int): Starting sample for visualization
        num_samples (int): Number of samples to display
    """
    result = process_record_denoising(record_num)
    
    if result is None or lead_name not in result['leads']:
        print(f"Data not available for record {record_num}, lead {lead_name}")
        return
    
    lead_data = result['leads'][lead_name]
    original = lead_data['original_signal'][start_sample:start_sample+num_samples]
    denoised = lead_data['denoised_signal'][start_sample:start_sample+num_samples]
    noise = original - denoised
    
    plt.figure(figsize=(15, 12))
    
    time_axis = np.arange(len(original)) / 360  # Assuming 360 Hz sampling rate
    
    # Original signal
    plt.subplot(3, 1, 1)
    plt.plot(time_axis, original, 'b-', label='Original Signal', linewidth=1)
    plt.title(f'Original ECG Signal - Record {record_num}, Lead {lead_name}')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Denoised signal
    plt.subplot(3, 1, 2)
    plt.plot(time_axis, denoised, 'g-', label='Denoised Signal', linewidth=1)
    plt.title(f'Denoised Signal (Wavelet: {lead_data["best_wavelet"]})')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Removed noise
    plt.subplot(3, 1, 3)
    plt.plot(time_axis, noise, 'r-', label='Removed Noise', linewidth=1, alpha=0.7)
    plt.title('Removed Noise Component')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print denoising statistics
    info = lead_data['denoising_info']
    print(f"\nDenoising Statistics for Record {record_num}, Lead {lead_name}:")
    print(f"Best Wavelet: {lead_data['best_wavelet']}")
    print(f"Decomposition Level: {info.get('selected_level', info.get('decomposition_level'))}")
    print(f"SNR Improvement: {info['snr_improvement_db']:.2f} dB")
    print(f"Correlation: {info['correlation_coefficient']:.4f}")
    print(f"Threshold: {info['threshold_value']:.6f}")

def main_denoising():
    """
    Main function for DWT-based ECG denoising
    """
    print("="*60)
    print("DWT-based ECG Signal Denoising")
    print("="*60)
    
    # Define records to process
    train_records = [101, 106, 108, 109, 112, 114, 115, 113, 116, 118, 119, 
                    122, 124, 201, 203, 205, 207, 208, 209, 215, 220, 223, 230]
    test_records = [100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 
                   212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234]
    all_records = train_records + test_records
    
    print(f"Processing {len(all_records)} records for DWT denoising...")
    
    # Process each record
    results = []
    for i, record_num in enumerate(all_records):
        print(f"\n[{i+1}/{len(all_records)}] Processing record {record_num}...")
        result = process_record_denoising(record_num)
        if result:
            results.append(result)
    
    # Save results
    if results:
        summary_df = save_denoising_results(results)
        print(f"\n✅ DWT denoising completed successfully!")
        print(f"Results saved to: {denoised_data_path}")
        return results, summary_df
    else:
        print("❌ No records were successfully processed.")
        return None, None

print("DWT-based ECG Denoising Script Ready!")
print("Run main_denoising() to start the denoising pipeline.")
print("Use visualize_denoising_comparison(record_num, lead_name) to visualize results.")

# Execute the main denoising pipeline
print("\nStarting DWT-based denoising pipeline...")
results, summary_df = main_denoising()

if results and summary_df is not None:
    print("\n" + "="*70)
    print("DENOISING ANALYSIS COMPLETE")
    print("="*70)
    
    # Show best performing records
    print("\nTop 5 Best Denoised Records (by correlation):")
    top_denoised = summary_df.nlargest(5, 'correlation_coefficient')[['record_number', 'lead_name', 'best_wavelet', 'correlation_coefficient', 'snr_improvement_db']]
    print(top_denoised.to_string(index=False))
    
    print(f"\n✅ All signals have been denoised and are ready for the next processing step!")
    print(f"Data location: {denoised_data_path}")
    
    # Example visualization
    print("\nGenerating sample visualization...")
    visualize_denoising_comparison(101, 'MLII', 0, 1800)
    
else:
    print("❌ Denoising pipeline failed.")
