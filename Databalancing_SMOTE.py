#!/usr/bin/env python3
"""
SMOTE Data Balancing for ECG Features
This script implements comprehensive data balancing techniques including SMOTE,
ADASYN, and other sampling methods for the extracted ECG features.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
import warnings
warnings.filterwarnings('ignore')

# Sampling techniques
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE, SVMSMOTE
from imblearn.under_sampling import RandomUnderSampler, TomekLinks, EditedNearestNeighbours
from imblearn.combine import SMOTETomek, SMOTEENN
from imblearn.metrics import classification_report_imbalanced

# Define paths (matching your original script)
base_path = '/content/drive/MyDrive/MIT_BIH_Dataset'
features_data_path = os.path.join(base_path, 'extracted_features')
balanced_data_path = os.path.join(base_path, 'balanced_data')

# Create balanced data directory
os.makedirs(balanced_data_path, exist_ok=True)

class ECGDataBalancer:
    """
    Comprehensive ECG Data Balancing using various SMOTE techniques
    """
    
    def __init__(self, random_state=42):
        """
        Initialize ECG Data Balancer
        
        Args:
            random_state (int): Random state for reproducibility
        """
        self.random_state = random_state
        self.label_encoder = LabelEncoder()
        self.sampling_techniques = {}
        self.balanced_datasets = {}
        self.performance_metrics = {}
        
        # Initialize sampling techniques
        self._initialize_sampling_techniques()
    
    def _initialize_sampling_techniques(self):
        """Initialize various sampling techniques"""
        self.sampling_techniques = {
            # Over-sampling techniques
            'SMOTE': SMOTE(random_state=self.random_state, k_neighbors=5),
            'ADASYN': ADASYN(random_state=self.random_state, n_neighbors=5),
            'BorderlineSMOTE': BorderlineSMOTE(random_state=self.random_state, k_neighbors=5),
            'SVMSMOTE': SVMSMOTE(random_state=self.random_state, k_neighbors=5),
            
            # Combined techniques (over + under sampling)
            'SMOTETomek': SMOTETomek(random_state=self.random_state),
            'SMOTEENN': SMOTEENN(random_state=self.random_state),
            
            # Under-sampling techniques
            'RandomUnder': RandomUnderSampler(random_state=self.random_state),
            'TomekLinks': TomekLinks(),
            'EditedNN': EditedNearestNeighbours()
        }
    
    def load_features_with_labels(self, features_file, labels_source='record_number'):
        """
        Load features and create labels for balancing
        
        Args:
            features_file (str): Path to features CSV file
            labels_source (str): Source for creating labels ('record_number', 'heart_rate_category', etc.)
            
        Returns:
            tuple: (features, labels, feature_names, metadata)
        """
        print(f"Loading features from: {features_file}")
        df = pd.read_csv(features_file)
        
        # Separate metadata and features
        metadata_cols = ['record_number', 'lead_name', 'num_qrs_peaks', 'avg_heart_rate']
        feature_cols = [col for col in df.columns if col not in metadata_cols]
        
        # Extract features and metadata
        features = df[feature_cols].values
        metadata = df[metadata_cols].copy()
        
        # Create labels based on the specified source
        if labels_source == 'record_number':
            # Create labels based on record number categories
            labels = self._create_record_based_labels(df['record_number'])
        elif labels_source == 'heart_rate_category':
            # Create labels based on heart rate categories
            labels = self._create_heart_rate_labels(df['avg_heart_rate'])
        elif labels_source == 'beat_density':
            # Create labels based on QRS beat density
            labels = self._create_beat_density_labels(df['num_qrs_peaks'])
        else:
            raise ValueError(f"Unknown label source: {labels_source}")
        
        # Handle missing values in features
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        print(f"Features shape: {features.shape}")
        print(f"Labels distribution: {Counter(labels)}")
        
        return features, labels, feature_cols, metadata
    
    def _create_record_based_labels(self, record_numbers):
        """Create labels based on MIT-BIH record categories"""
        labels = []
        for record_num in record_numbers:
            if record_num in [100, 101, 103, 105, 106, 107, 108, 109, 111, 112, 113, 114, 115, 116, 117, 118, 119]:
                labels.append('Normal')  # Normal rhythms
            elif record_num in [200, 201, 202, 203, 205, 207, 208, 209, 210, 212, 213, 214, 215]:
                labels.append('Arrhythmia')  # Various arrhythmias
            elif record_num in [220, 221, 222, 223, 228, 230, 231, 232, 233, 234]:
                labels.append('Abnormal')  # Other abnormalities
            else:
                labels.append('Other')
        
        return labels
    
    def _create_heart_rate_labels(self, heart_rates):
        """Create labels based on heart rate categories"""
        labels = []
        for hr in heart_rates:
            if hr < 60:
                labels.append('Bradycardia')  # Slow heart rate
            elif 60 <= hr <= 100:
                labels.append('Normal')  # Normal heart rate
            elif 100 < hr <= 150:
                labels.append('Tachycardia')  # Fast heart rate
            else:
                labels.append('Severe_Tachycardia')  # Very fast heart rate
        
        return labels
    
    def _create_beat_density_labels(self, qrs_counts):
        """Create labels based on QRS beat density"""
        labels = []
        qrs_quartiles = np.percentile(qrs_counts, [25, 50, 75])
        
        for count in qrs_counts:
            if count <= qrs_quartiles[0]:
                labels.append('Low_Density')
            elif count <= qrs_quartiles[1]:
                labels.append('Medium_Low_Density')
            elif count <= qrs_quartiles[2]:
                labels.append('Medium_High_Density')
            else:
                labels.append('High_Density')
        
        return labels
    
    def analyze_class_imbalance(self, labels):
        """
        Analyze class imbalance in the dataset
        
        Args:
            labels (list): Class labels
            
        Returns:
            dict: Imbalance analysis results
        """
        label_counts = Counter(labels)
        total_samples = len(labels)
        
        imbalance_analysis = {
            'total_samples': total_samples,
            'num_classes': len(label_counts),
            'class_counts': dict(label_counts),
            'class_percentages': {cls: (count/total_samples)*100 
                                for cls, count in label_counts.items()},
            'imbalance_ratio': max(label_counts.values()) / min(label_counts.values()),
            'majority_class': max(label_counts, key=label_counts.get),
            'minority_class': min(label_counts, key=label_counts.get)
        }
        
        return imbalance_analysis
    
    def apply_sampling_techniques(self, features, labels):
        """
        Apply various sampling techniques to balance the dataset
        
        Args:
            features (np.array): Feature matrix
            labels (list): Class labels
            
        Returns:
            dict: Balanced datasets for each technique
        """
        print("Applying sampling techniques...")
        
        # Encode labels
        encoded_labels = self.label_encoder.fit_transform(labels)
        
        balanced_results = {}
        
        for technique_name, sampler in self.sampling_techniques.items():
            try:
                print(f"  Applying {technique_name}...")
                
                # Apply sampling
                X_resampled, y_resampled = sampler.fit_resample(features, encoded_labels)
                
                # Decode labels back
                y_resampled_decoded = self.label_encoder.inverse_transform(y_resampled)
                
                balanced_results[technique_name] = {
                    'features': X_resampled,
                    'labels': y_resampled_decoded,
                    'encoded_labels': y_resampled,
                    'original_shape': features.shape,
                    'balanced_shape': X_resampled.shape,
                    'class_distribution': Counter(y_resampled_decoded)
                }
                
                print(f"    Original: {features.shape[0]} samples")
                print(f"    Balanced: {X_resampled.shape[0]} samples")
                print(f"    Distribution: {Counter(y_resampled_decoded)}")
                
            except Exception as e:
                print(f"    ❌ Failed to apply {technique_name}: {str(e)}")
                continue
        
        self.balanced_datasets = balanced_results
        return balanced_results
    
    def evaluate_sampling_techniques(self, original_features, original_labels, test_size=0.2):
        """
        Evaluate different sampling techniques using machine learning models
        
        Args:
            original_features (np.array): Original feature matrix
            original_labels (list): Original labels
            test_size (float): Test set size for evaluation
            
        Returns:
            dict: Performance metrics for each sampling technique
        """
        print("Evaluating sampling techniques...")
        
        # Split original data into train and test
        X_train_orig, X_test, y_train_orig, y_test = train_test_split(
            original_features, original_labels, test_size=test_size, 
            random_state=self.random_state, stratify=original_labels
        )
        
        # Encode test labels
        y_test_encoded = self.label_encoder.transform(y_test)
        
        # Define classifiers
        classifiers = {
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=self.random_state),
            'LogisticRegression': LogisticRegression(random_state=self.random_state, max_iter=1000),
            'SVM': SVC(random_state=self.random_state, probability=True)
        }
        
        evaluation_results = {}
        
        # Evaluate original (imbalanced) data
        print("  Evaluating original (imbalanced) dataset...")
        y_train_orig_encoded = self.label_encoder.transform(y_train_orig)
        original_results = self._evaluate_classifiers(
            classifiers, X_train_orig, y_train_orig_encoded, X_test, y_test_encoded
        )
        evaluation_results['Original'] = original_results
        
        # Evaluate each balanced dataset
        for technique_name, balanced_data in self.balanced_datasets.items():
            print(f"  Evaluating {technique_name}...")
            
            try:
                # Use balanced training data
                X_train_balanced = balanced_data['features']
                y_train_balanced = balanced_data['encoded_labels']
                
                # Evaluate classifiers
                technique_results = self._evaluate_classifiers(
                    classifiers, X_train_balanced, y_train_balanced, X_test, y_test_encoded
                )
                evaluation_results[technique_name] = technique_results
                
            except Exception as e:
                print(f"    ❌ Failed to evaluate {technique_name}: {str(e)}")
                continue
        
        self.performance_metrics = evaluation_results
        return evaluation_results
    
    def _evaluate_classifiers(self, classifiers, X_train, y_train, X_test, y_test):
        """Evaluate multiple classifiers on given data"""
        results = {}
        
        for clf_name, classifier in classifiers.items():
            try:
                # Train classifier
                classifier.fit(X_train, y_train)
                
                # Make predictions
                y_pred = classifier.predict(X_test)
                
                # Calculate metrics
                results[clf_name] = {
                    'accuracy': accuracy_score(y_test, y_pred),
                    'precision_macro': precision_score(y_test, y_pred, average='macro', zero_division=0),
                    'recall_macro': recall_score(y_test, y_pred, average='macro', zero_division=0),
                    'f1_macro': f1_score(y_test, y_pred, average='macro', zero_division=0),
                    'precision_weighted': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                    'recall_weighted': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                    'f1_weighted': f1_score(y_test, y_pred, average='weighted', zero_division=0)
                }
                
            except Exception as e:
                print(f"      ❌ Failed to evaluate {clf_name}: {str(e)}")
                results[clf_name] = {'error': str(e)}
        
        return results
    
    def visualize_class_distribution(self, original_labels, technique_name='SMOTE'):
        """
        Visualize class distribution before and after balancing
        
        Args:
            original_labels (list): Original class labels
            technique_name (str): Sampling technique to visualize
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Original distribution
        original_counts = Counter(original_labels)
        axes[0].bar(original_counts.keys(), original_counts.values(), color='skyblue', alpha=0.7)
        axes[0].set_title('Original Class Distribution (Imbalanced)')
        axes[0].set_xlabel('Class')
        axes[0].set_ylabel('Number of Samples')
        axes[0].tick_params(axis='x', rotation=45)
        
        # Add count labels on bars
        for i, (class_name, count) in enumerate(original_counts.items()):
            axes[0].text(i, count + max(original_counts.values())*0.01, str(count), 
                        ha='center', va='bottom')
        
        # Balanced distribution
        if technique_name in self.balanced_datasets:
            balanced_counts = self.balanced_datasets[technique_name]['class_distribution']
            axes[1].bar(balanced_counts.keys(), balanced_counts.values(), color='lightcoral', alpha=0.7)
            axes[1].set_title(f'Balanced Class Distribution ({technique_name})')
            axes[1].set_xlabel('Class')
            axes[1].set_ylabel('Number of Samples')
            axes[1].tick_params(axis='x', rotation=45)
            
            # Add count labels on bars
            for i, (class_name, count) in enumerate(balanced_counts.items()):
                axes[1].text(i, count + max(balanced_counts.values())*0.01, str(count), 
                            ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
    
    def visualize_performance_comparison(self):
        """Visualize performance comparison across sampling techniques"""
        if not self.performance_metrics:
            print("No performance metrics available. Run evaluate_sampling_techniques() first.")
            return
        
        # Prepare data for visualization
        techniques = []
        accuracies = []
        f1_scores = []
        precisions = []
        recalls = []
        
        for technique, results in self.performance_metrics.items():
            if 'RandomForest' in results and 'error' not in results['RandomForest']:
                techniques.append(technique)
                accuracies.append(results['RandomForest']['accuracy'])
                f1_scores.append(results['RandomForest']['f1_macro'])
                precisions.append(results['RandomForest']['precision_macro'])
                recalls.append(results['RandomForest']['recall_macro'])
        
        if not techniques:
            print("No valid performance metrics to visualize.")
            return
        
        # Create performance comparison plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy comparison
        axes[0, 0].bar(techniques, accuracies, color='skyblue', alpha=0.7)
        axes[0, 0].set_title('Accuracy Comparison (Random Forest)')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].set_ylim(0, 1)
        
        # F1-Score comparison
        axes[0, 1].bar(techniques, f1_scores, color='lightcoral', alpha=0.7)
        axes[0, 1].set_title('F1-Score Comparison (Macro Average)')
        axes[0, 1].set_ylabel('F1-Score')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].set_ylim(0, 1)
        
        # Precision comparison
        axes[1, 0].bar(techniques, precisions, color='lightgreen', alpha=0.7)
        axes[1, 0].set_title('Precision Comparison (Macro Average)')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].set_ylim(0, 1)
        
        # Recall comparison
        axes[1, 1].bar(techniques, recalls, color='gold', alpha=0.7)
        axes[1, 1].set_title('Recall Comparison (Macro Average)')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].set_ylim(0, 1)
        
        plt.tight_layout()
        plt.show()
        
        # Print best performing technique
        best_f1_idx = np.argmax(f1_scores)
        best_technique = techniques[best_f1_idx]
        print(f"\n🏆 Best performing technique: {best_technique}")
        print(f"   F1-Score: {f1_scores[best_f1_idx]:.4f}")
        print(f"   Accuracy: {accuracies[best_f1_idx]:.4f}")
    
    def save_balanced_datasets(self, original_metadata, save_top_n=3):
        """
        Save the balanced datasets to CSV files
        
        Args:
            original_metadata (pd.DataFrame): Original metadata
            save_top_n (int): Number of top-performing techniques to save
        """
        print(f"Saving balanced datasets...")
        
        # Determine top performing techniques
        if self.performance_metrics:
            technique_scores = {}
            for technique, results in self.performance_metrics.items():
                if technique != 'Original' and 'RandomForest' in results and 'error' not in results['RandomForest']:
                    technique_scores[technique] = results['RandomForest']['f1_macro']
            
            # Sort by F1-score
            top_techniques = sorted(technique_scores.items(), key=lambda x: x[1], reverse=True)[:save_top_n]
            techniques_to_save = [technique for technique, _ in top_techniques]
        else:
            # Save first few if no performance metrics
            techniques_to_save = list(self.balanced_datasets.keys())[:save_top_n]
        
        saved_files = []
        
        for technique_name in techniques_to_save:
            if technique_name in self.balanced_datasets:
                balanced_data = self.balanced_datasets[technique_name]
                
                # Create DataFrame with balanced features
                feature_names = [f'feature_{i}' for i in range(balanced_data['features'].shape[1])]
                df = pd.DataFrame(balanced_data['features'], columns=feature_names)
                
                # Add labels
                df['class_label'] = balanced_data['labels']
                
                # Add synthetic sample indicators
                original_size = balanced_data['original_shape'][0]
                df['is_synthetic'] = ['No'] * original_size + ['Yes'] * (len(df) - original_size)
                
                # Save to CSV
                filename = f'balanced_features_{technique_name.lower()}.csv'
                filepath = os.path.join(balanced_data_path, filename)
                df.to_csv(filepath, index=False)
                saved_files.append(filepath)
                
                print(f"  ✅ Saved {technique_name}: {filepath}")
                print(f"     Shape: {df.shape}")
                print(f"     Synthetic samples: {sum(df['is_synthetic'] == 'Yes')}")
        
        # Save balancing summary
        summary = {
            'original_samples': len(original_metadata),
            'techniques_applied': len(self.balanced_datasets),
            'best_technique': techniques_to_save[0] if techniques_to_save else 'None',
            'saved_datasets': len(saved_files)
        }
        
        summary_df = pd.DataFrame([summary])
        summary_path = os.path.join(balanced_data_path, 'balancing_summary.csv')
        summary_df.to_csv(summary_path, index=False)
        
        print(f"\n📊 Balancing summary saved: {summary_path}")
        return saved_files, summary

def main_smote_balancing():
    """
    Main function for SMOTE data balancing
    """
    print("="*70)
    print("SMOTE Data Balancing for ECG Features")
    print("="*70)
    
    # Initialize balancer
    balancer = ECGDataBalancer(random_state=42)
    
    # Load features (try both original and PCA features)
    original_features_path = os.path.join(features_data_path, 'original_features.csv')
    pca_features_path = os.path.join(features_data_path, 'pca_features.csv')
    
    # Choose which features to use
    features_path = pca_features_path if os.path.exists(pca_features_path) else original_features_path
    
    if not os.path.exists(features_path):
        print(f"❌ Features file not found: {features_path}")
        print("Please run the feature extraction pipeline first.")
        return None
    
    print(f"Using features from: {features_path}")
    
    # Step 1: Load features with labels
    print("\nStep 1: Loading Features and Creating Labels")
    features, labels, feature_names, metadata = balancer.load_features_with_labels(
        features_path, labels_source='record_number'
    )
    
    # Step 2: Analyze class imbalance
    print("\nStep 2: Analyzing Class Imbalance")
    imbalance_analysis = balancer.analyze_class_imbalance(labels)
    
    print(f"📊 Class Distribution Analysis:")
    print(f"   Total samples: {imbalance_analysis['total_samples']}")
    print(f"   Number of classes: {imbalance_analysis['num_classes']}")
    print(f"   Imbalance ratio: {imbalance_analysis['imbalance_ratio']:.2f}")
    print(f"   Majority class: {imbalance_analysis['majority_class']}")
    print(f"   Minority class: {imbalance_analysis['minority_class']}")
    
    for class_name, percentage in imbalance_analysis['class_percentages'].items():
        print(f"   {class_name}: {percentage:.1f}%")
    
    # Step 3: Apply sampling techniques
    print(f"\nStep 3: Applying Sampling Techniques")
    balanced_datasets = balancer.apply_sampling_techniques(features, labels)
    
    print(f"✅ Successfully applied {len(balanced_datasets)} sampling techniques")
    
    # Step 4: Evaluate sampling techniques
    print(f"\nStep 4: Evaluating Sampling Techniques")
    performance_metrics = balancer.evaluate_sampling_techniques(features, labels)
    
    # Step 5: Visualizations
    print(f"\nStep 5: Creating Visualizations")
    
    # Class distribution visualization
    balancer.visualize_class_distribution(labels, 'SMOTE')
    
    # Performance comparison visualization
    balancer.visualize_performance_comparison()
    
    # Step 6: Save results
    print(f"\nStep 6: Saving Balanced Datasets")
    saved_files, summary = balancer.save_balanced_datasets(metadata)
    
    print("\n" + "="*70)
    print("SMOTE DATA BALANCING COMPLETED")
    print("="*70)
    print(f"✅ Applied {len(balanced_datasets)} different sampling techniques")
    print(f"✅ Evaluated performance on {len(performance_metrics)} datasets")
    print(f"✅ Saved {len(saved_files)} best performing balanced datasets")
    print(f"✅ Results saved to: {balanced_data_path}")
    
    # Print performance summary
    if performance_metrics and 'Original' in performance_metrics:
        print(f"\n📈 Performance Improvement Summary:")
        original_f1 = performance_metrics['Original']['RandomForest']['f1_macro']
        print(f"   Original dataset F1-score: {original_f1:.4f}")
        
        best_technique = None
        best_f1 = 0
        for technique, results in performance_metrics.items():
            if technique != 'Original' and 'RandomForest' in results and 'error' not in results['RandomForest']:
                f1_score = results['RandomForest']['f1_macro']
                if f1_score > best_f1:
                    best_f1 = f1_score
                    best_technique = technique
        
        if best_technique:
            improvement = ((best_f1 - original_f1) / original_f1) * 100
            print(f"   Best technique: {best_technique}")
            print(f"   Best F1-score: {best_f1:.4f}")
            print(f"   Improvement: {improvement:+.1f}%")
    
    print("="*70)
    
    return balancer, balanced_datasets, performance_metrics

# Utility functions
def load_balanced_features(technique_name='smote'):
    """
    Load previously balanced features
    
    Args:
        technique_name (str): Name of the balancing technique
        
    Returns:
        pd.DataFrame: Balanced features DataFrame
    """
    filename = f'balanced_features_{technique_name.lower()}.csv'
    filepath = os.path.join(balanced_data_path, filename)
    
    if os.path.exists(filepath):
        return pd.read_csv(filepath)
    else:
        print(f"Balanced features file not found: {filepath}")
        return None

def compare_all_techniques_detailed():
    """
    Create detailed comparison of all sampling techniques
    """
    summary_path = os.path.join(balanced_data_path, 'balancing_summary.csv')
    if os.path.exists(summary_path):
        summary_df = pd.read_csv(summary_path)
        print("Balancing Summary:")
        print(summary_df.to_string(index=False))
    else:
        print("No balancing summary found. Run main_smote_balancing() first.")

print("SMOTE Data Balancing Script Ready!")
print("Run main_smote_balancing() to start the complete balancing pipeline.")
print("Use load_balanced_features('technique_name') to load specific balanced datasets.")
print("Use compare_all_techniques_detailed() for detailed comparison.")

# Execute the main balancing pipeline
if __name__ == "__main__":
    print("\nStarting SMOTE data balancing pipeline...")
    balancer, balanced_datasets, performance_metrics = main_smote_balancing()
