#!/usr/bin/env python3
"""
DWTFrTV-based ECG Signal Denoising for MIT-BIH Dataset
This script performs denoising on quality-assessed and reconstructed ECG signals
using DWT combined with Fractional Total Variation regularization
"""

# Install required packages
import subprocess
import sys

def install_package(package):
    """Install a package using pip"""
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Install required packages if not available
try:
    import pywt
    print("PyWavelets already available!")
except ImportError:
    print("Installing PyWavelets...")
    install_package("PyWavelets")
    import pywt

try:
    from scipy.optimize import minimize
    print("SciPy optimization available!")
except ImportError:
    print("Installing SciPy...")
    install_package("scipy")
    from scipy.optimize import minimize

try:
    from scipy.special import gamma
    print("SciPy special functions available!")
except ImportError:
    pass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, ifft, fftfreq
from scipy.ndimage import convolve1d
import os
import wfdb
import warnings
warnings.filterwarnings('ignore')

# Define paths
base_path = 'your/dataset/path'
raw_data_path = os.path.join(base_path, 'raw_data')
quality_assessment_path = os.path.join(base_path, 'quality_assessment')
denoised_data_path = os.path.join(base_path, 'denoised_data')

# Create denoised data directory
os.makedirs(denoised_data_path, exist_ok=True)

class FractionalTotalVariation:
    """
    Fractional Total Variation regularization for signal denoising
    """
    
    def __init__(self, alpha=0.5, beta=1.0):
        """
        Initialize FrTV parameters
        
        Args:
            alpha (float): Fractional order (0 < alpha <= 1)
            beta (float): Regularization parameter
        """
        self.alpha = alpha
        self.beta = beta
    
    def fractional_derivative_grunwald(self, signal, alpha, h=1.0):
        """
        Compute fractional derivative using Grünwald-Letnikov definition
        
        Args:
            signal (np.array): Input signal
            alpha (float): Fractional order
            h (float): Step size
            
        Returns:
            np.array: Fractional derivative
        """
        n = len(signal)
        result = np.zeros_like(signal)
        
        # Compute Grünwald-Letnikov weights
        weights = np.zeros(n)
        weights[0] = 1.0
        
        for k in range(1, n):
            weights[k] = weights[k-1] * (alpha - k + 1) / k
        
        # Apply convolution
        for i in range(n):
            for j in range(min(i + 1, n)):
                result[i] += weights[j] * signal[i - j]
        
        return result / (h ** alpha)
    
    def fractional_derivative_caputo(self, signal, alpha, h=1.0):
        """
        Compute Caputo fractional derivative (more suitable for signals)
        
        Args:
            signal (np.array): Input signal
            alpha (float): Fractional order
            h (float): Step size
            
        Returns:
            np.array: Caputo fractional derivative
        """
        n = len(signal)
        
        # For 0 < alpha < 1, Caputo derivative
        if alpha > 0 and alpha < 1:
            # First compute regular derivative
            diff_signal = np.gradient(signal, h)
            
            # Apply fractional integration of order (1-alpha)
            frac_int_order = 1 - alpha
            
            # Use numerical integration for fractional part
            result = np.zeros_like(signal)
            
            for i in range(1, n):
                integrand = diff_signal[:i+1] / ((i - np.arange(i+1)) * h + h) ** frac_int_order
                result[i] = np.trapz(integrand, dx=h) / gamma(frac_int_order)
            
            return result
        else:
            # For alpha = 1, return regular derivative
            return np.gradient(signal, h)
    
    def frtv_regularization_term(self, signal):
        """
        Compute fractional total variation regularization term
        
        Args:
            signal (np.array): Input signal
            
        Returns:
            float: FrTV regularization value
        """
        # Compute fractional derivative
        frac_deriv = self.fractional_derivative_caputo(signal, self.alpha)
        
        # Compute L1 norm (total variation)
        frtv = np.sum(np.abs(frac_deriv))
        
        return frtv
    
    def frtv_gradient(self, signal):
        """
        Compute gradient of FrTV regularization term
        
        Args:
            signal (np.array): Input signal
            
        Returns:
            np.array: Gradient of FrTV term
        """
        n = len(signal)
        eps = 1e-8  # Small epsilon for numerical stability
        
        # Compute fractional derivative
        frac_deriv = self.fractional_derivative_caputo(signal, self.alpha)
        
        # Compute sign of fractional derivative
        sign_frac_deriv = np.sign(frac_deriv + eps)
        
        # Compute adjoint of fractional derivative operator
        # For Caputo derivative, this involves fractional integration
        gradient = np.zeros_like(signal)
        
        # Simplified gradient computation using finite differences
        for i in range(n):
            signal_plus = signal.copy()
            signal_minus = signal.copy()
            signal_plus[i] += eps
            signal_minus[i] -= eps
            
            frtv_plus = self.frtv_regularization_term(signal_plus)
            frtv_minus = self.frtv_regularization_term(signal_minus)
            
            gradient[i] = (frtv_plus - frtv_minus) / (2 * eps)
        
        return gradient

class DWTFrTVDenoiser:
    """
    Class for DWT-based ECG signal denoising with Fractional Total Variation regularization
    """
    
    def __init__(self, wavelet='db6', mode='symmetric', frtv_alpha=0.5, frtv_beta=0.1):
        """
        Initialize DWTFrTV denoiser
        
        Args:
            wavelet (str): Wavelet type (default: 'db6' - good for ECG)
            mode (str): Signal extension mode
            frtv_alpha (float): Fractional order for FrTV
            frtv_beta (float): FrTV regularization parameter
        """
        self.wavelet = wavelet
        self.mode = mode
        self.frtv = FractionalTotalVariation(alpha=frtv_alpha, beta=frtv_beta)
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
    
    def frtv_soft_thresholding(self, coeffs, threshold, frtv_weight=0.1):
        """
        Apply FrTV-enhanced soft thresholding to wavelet coefficients
        
        Args:
            coeffs (np.array): Wavelet coefficients
            threshold (float): Primary thresholding value
            frtv_weight (float): Weight for FrTV regularization
            
        Returns:
            np.array: FrTV-enhanced thresholded coefficients
        """
        # Standard soft thresholding
        soft_thresh = self.soft_thresholding(coeffs, threshold)
        
        # Apply FrTV regularization if coefficients are long enough
        if len(coeffs) > 10:
            try:
                # Compute FrTV gradient
                frtv_grad = self.frtv.frtv_gradient(soft_thresh)
                
                # Apply FrTV correction
                frtv_enhanced = soft_thresh - frtv_weight * frtv_grad
                
                return frtv_enhanced
            except:
                # Fall back to standard soft thresholding if FrTV fails
                return soft_thresh
        else:
            return soft_thresh
    
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
    
    def frtv_enhanced_threshold(self, signal, base_threshold):
        """
        Enhance threshold calculation using FrTV characteristics
        
        Args:
            signal (np.array): Input signal
            base_threshold (float): Base threshold value
            
        Returns:
            float: FrTV-enhanced threshold
        """
        try:
            # Compute FrTV of the signal
            frtv_value = self.frtv.frtv_regularization_term(signal)
            
            # Normalize FrTV value
            normalized_frtv = frtv_value / (len(signal) * np.std(signal))
            
            # Adjust threshold based on FrTV characteristics
            # Higher FrTV (more irregular signal) -> lower threshold
            # Lower FrTV (smoother signal) -> higher threshold
            frtv_factor = 1.0 / (1.0 + normalized_frtv)
            enhanced_threshold = base_threshold * frtv_factor
            
            return enhanced_threshold
        except:
            return base_threshold
    
    def denoise_signal_dwtfrtv(self, signal, decomposition_level=6, 
                              threshold_method='adaptive', frtv_weight=0.05):
        """
        Denoise ECG signal using DWTFrTV approach
        
        Args:
            signal (np.array): Input ECG signal
            decomposition_level (int): Number of decomposition levels
            threshold_method (str): Threshold calculation method
            frtv_weight (float): Weight for FrTV regularization
            
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
            
            # Calculate base threshold
            if threshold_method == 'adaptive':
                base_threshold = self.adaptive_threshold_calculation(signal_padded, decomposition_level)
            else:
                base_threshold = self.adaptive_threshold_calculation(signal_padded, decomposition_level)
            
            # Apply DWTFrTV denoising to detail coefficients
            coeffs_frtv = coeffs.copy()
            frtv_applied_levels = 0
            
            for i in range(1, len(coeffs)):  # Skip approximation coefficients (index 0)
                if len(coeffs[i]) > 10:  # Apply FrTV only to sufficiently long coefficient arrays
                    # Enhance threshold using FrTV
                    enhanced_threshold = self.frtv_enhanced_threshold(coeffs[i], base_threshold)
                    
                    # Apply FrTV-enhanced soft thresholding
                    coeffs_frtv[i] = self.frtv_soft_thresholding(coeffs[i], enhanced_threshold, frtv_weight)
                    frtv_applied_levels += 1
                else:
                    # Apply standard soft thresholding for short coefficient arrays
                    coeffs_frtv[i] = self.soft_thresholding(coeffs[i], base_threshold)
            
            # Reconstruct signal
            denoised_signal_padded = pywt.waverec(coeffs_frtv, self.wavelet, mode=self.mode)
            
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
            
            # Calculate preservation of important features
            correlation = np.corrcoef(signal, denoised_signal)[0, 1]
            
            # Calculate FrTV metrics
            original_frtv = self.frtv.frtv_regularization_term(signal)
            denoised_frtv = self.frtv.frtv_regularization_term(denoised_signal)
            frtv_reduction = (original_frtv - denoised_frtv) / original_frtv * 100
            
            denoising_info = {
                'method': 'DWTFrTV',
                'threshold_method': threshold_method,
                'base_threshold': base_threshold,
                'decomposition_level': decomposition_level,
                'wavelet_used': self.wavelet,
                'frtv_alpha': self.frtv.alpha,
                'frtv_beta': self.frtv.beta,
                'frtv_weight': frtv_weight,
                'frtv_applied_levels': frtv_applied_levels,
                'noise_power': noise_power,
                'signal_power': signal_power,
                'snr_improvement_db': snr_improvement,
                'correlation_coefficient': correlation,
                'original_frtv': original_frtv,
                'denoised_frtv': denoised_frtv,
                'frtv_reduction_percent': frtv_reduction,
                'noise_variance_estimate': self.calculate_noise_variance(coeffs),
                'coefficients_levels': len(coeffs),
                'original_length': len(signal),
                'was_padded': was_padded
            }
            
            return denoised_signal, denoising_info
            
        except Exception as e:
            print(f"Error in DWTFrTV denoising: {str(e)}")
            return signal, {'error': str(e)}
    
    def multi_parameter_optimization(self, signal, param_ranges=None):
        """
        Optimize DWTFrTV parameters for best denoising performance
        
        Args:
            signal (np.array): Input signal
            param_ranges (dict): Parameter ranges for optimization
            
        Returns:
            tuple: (best_denoised_signal, best_params, best_info)
        """
        if param_ranges is None:
            param_ranges = {
                'decomposition_levels': [4, 6, 8],
                'frtv_alphas': [0.3, 0.5, 0.7],
                'frtv_weights': [0.01, 0.05, 0.1]
            }
        
        best_score = -np.inf
        best_result = None
        best_params = None
        
        total_combinations = (len(param_ranges['decomposition_levels']) * 
                            len(param_ranges['frtv_alphas']) * 
                            len(param_ranges['frtv_weights']))
        
        combination = 0
        
        for level in param_ranges['decomposition_levels']:
            for alpha in param_ranges['frtv_alphas']:
                for weight in param_ranges['frtv_weights']:
                    combination += 1
                    
                    try:
                        # Update FrTV parameters
                        self.frtv.alpha = alpha
                        
                        # Denoise with current parameters
                        denoised, info = self.denoise_signal_dwtfrtv(
                            signal, 
                            decomposition_level=level,
                            frtv_weight=weight
                        )
                        
                        if 'error' not in info:
                            # Scoring function combining multiple metrics
                            score = (0.4 * info['correlation_coefficient'] + 
                                    0.3 * (info['snr_improvement_db'] / 20) + 
                                    0.2 * (info['frtv_reduction_percent'] / 100) +
                                    0.1 * (1 - info['noise_power'] / np.var(signal)))
                            
                            if score > best_score:
                                best_score = score
                                best_result = (denoised, info)
                                best_params = {
                                    'decomposition_level': level,
                                    'frtv_alpha': alpha,
                                    'frtv_weight': weight
                                }
                        
                    except Exception as e:
                        print(f"Parameter combination failed: level={level}, alpha={alpha}, weight={weight}")
                        continue
        
        if best_result is None:
            # Fall back to default parameters
            self.frtv.alpha = 0.5
            denoised, info = self.denoise_signal_dwtfrtv(signal)
            best_params = {'decomposition_level': 6, 'frtv_alpha': 0.5, 'frtv_weight': 0.05}
            return denoised, best_params, info
        
        best_denoised, best_info = best_result
        best_info['optimization_score'] = best_score
        best_info['optimized_params'] = best_params
        
        return best_denoised, best_params, best_info

def process_record_dwtfrtv(record_num):
    """
    Process a single record for DWTFrTV denoising
    
    Args:
        record_num (int): Record number to process
        
    Returns:
        dict: Denoising results
    """
    try:
        # Load the record
        record_path = os.path.join(raw_data_path, str(record_num))
        record = wfdb.rdrecord(record_path)
        
        if record.p_signal is None:
            print(f"No signal data found for record {record_num}")
            return None
        
        signals = record.p_signal
        
        # Initialize DWTFrTV denoisers with different wavelets
        denoisers = {
            'db6': DWTFrTVDenoiser(wavelet='db6'),
            'db8': DWTFrTVDenoiser(wavelet='db8'),
            'bior4.4': DWTFrTVDenoiser(wavelet='bior4.4'),
            'coif4': DWTFrTVDenoiser(wavelet='coif4')
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
            
            print(f"Processing Record {record_num}, Lead {lead_name} with DWTFrTV...")
            
            # Try different wavelets and select best
            best_wavelet = None
            best_score = -np.inf
            best_denoised = original_signal
            best_info = {}
            best_params = {}
            
            for wavelet_name, denoiser in denoisers.items():
                try:
                    # Multi-parameter optimization for current wavelet
                    denoised_signal, opt_params, denoising_info = denoiser.multi_parameter_optimization(original_signal)
                    
                    if 'error' not in denoising_info:
                        score = denoising_info.get('optimization_score', 0)
                        if score > best_score:
                            best_score = score
                            best_wavelet = wavelet_name
                            best_denoised = denoised_signal
                            best_info = denoising_info
                            best_params = opt_params
                            
                except Exception as e:
                    print(f"  Warning: {wavelet_name} failed - {str(e)}")
                    continue
            
            # Store results for best performing configuration
            results['leads'][lead_name] = {
                'original_signal': original_signal,
                'denoised_signal': best_denoised,
                'best_wavelet': best_wavelet,
                'best_params': best_params,
                'denoising_info': best_info,
                'signal_length': len(original_signal)
            }
            
            # Print summary
            if 'error' not in best_info:
                print(f"  Best wavelet: {best_wavelet}")
                print(f"  Optimal params: {best_params}")
                print(f"  SNR improvement: {best_info['snr_improvement']:.2f} dB")
                print(f"  Correlation: {best_info['correlationcoefficient']:.4f}")
                print(f"  FrTV reduction: {best_info['frtv_reduction']:.1f}%")
            else:
                print(f"  Denoising failed: {best_info.get('error', 'Unknown error')}")
        
        return results
        
    except Exception as e:
        print(f"Error processing record {record_num}: {str(e)}")
        return None

def save_dwtfrtv_results(results_list):
    """
    Save DWTFrTV denoising results and create summary
    
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
            params = lead_data['best_params']
            
            summary_row = {
                'record_number': record_num,
                'lead_name': lead_name,
                'best_wavelet': lead_data['best_wavelet'],
                'decomposition_level': params.get('decomposition_level'),
                'frtv_alpha': params.get('frtv_alpha'),
                'frtv_weight': params.get('frtv_weight'),
                'base_threshold': info.get('base_threshold'),
                'snr_improvement_db': info['snr_improvement_db'],
                'correlation_coefficient': info['correlation_coefficient'],
                'frtv_reduction_percent': info['frtv_reduction_percent'],
                'noise_power': info['noise_power'],
                'signal_power': info['signal_power'],
                'optimization_score': info.get('optimization_score', 0),
                'frtv_applied_levels': info.get('frtv_applied_levels', 0)
            }
            
            summary_data.append(summary_row)
    
    # Save summary
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_path = os.path.join(denoised_data_path, 'dwtfrtv_denoising_summary.csv')
        summary_df.to_csv(summary_path, index=False)
        
        print(f"\nDWTFrTV denoising summary saved to: {summary_path}")
        
        # Display statistics
        print("\n" + "="*70)
        print("DWTFrTV DENOISING SUMMARY")
        print("="*70)
        print(f"Total records processed: {len(results_list)}")
        print(f"Average SNR improvement: {summary_df['snr_improvement_db'].mean():.2f} dB")
        print(f"Average correlation: {summary_df['correlation_coefficient'].mean():.4f}")
        print(f"Average FrTV reduction: {summary_df['frtv_reduction_percent'].mean():.1f}%")
        
        # Parameter usage statistics
        print(f"\nOptimal Parameter Distribution:")
        print(f"Best wavelets:")
        wavelet_counts = summary_df['best_wavelet'].value_counts()
        for wavelet, count in wavelet_counts.items():
            print(f"  {wavelet}: {count} times")
        
        print(f"\nFractional orders (α):")
        alpha_counts = summary_df['frtv_alpha'].value_counts().sort_index()
        for alpha, count in alpha_counts.items():
            print(f"  α = {alpha}: {count} times")
        
        print(f"\nFrTV weights:")
        weight_counts = summary_df['frtv_weight'].value_counts().sort_index()
        for weight, count in weight_counts.items():
            print(f"  λ = {weight}: {count} times")
        
        # Quality categories
        excellent = len(summary_df[summary_df['correlation_coefficient'] > 0.95])
        good = len(summary_df[(summary_df['correlation_coefficient'] > 0.9) & 
                             (summary_df['correlation_coefficient'] <= 0.95)])
        fair = len(summary_df[summary_df['correlation_coefficient'] <= 0.9])
        
        print(f"\nDenoising Quality Distribution:")
        print(f"Excellent (correlation > 0.95): {excellent}")
        print(f"Good (correlation 0.9-0.95): {good}")
        print(f"Fair (correlation < 0.9): {fair}")
        
        # FrTV effectiveness
        high_frtv = len(summary_df[summary_df['frtv_reduction_percent'] > 20])
        med_frtv = len(summary_df[(summary_df['frtv_reduction_percent'] > 10) & 
                                 (summary_df['frtv_reduction_percent'] <= 20)])
        low_frtv = len(summary_df[summary_df['frtv_reduction_percent'] <= 10])
        
        print(f"\nFrTV Regularization Effectiveness:")
        print(f"High reduction (>20%): {high_frtv}")
        print(f"Medium reduction (10-20%): {med_frtv}")
        print(f"Low reduction (≤10%): {low_frtv}")
        print("="*70)
        
        return summary_df
    else:
        print("No successful DWTFrTV denoising results to save.")
        return None
