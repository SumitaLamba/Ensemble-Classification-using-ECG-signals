#!/usr/bin/env python3
"""
MIT-BIH Arrhythmia Dataset Acquisition and Processing
This script downloads and processes the MIT-BIH arrhythmia dataset from PhysioNet
for ECG signal analysis and arrhythmia classification.
"""

import os
import requests
import numpy as np
import pandas as pd
import wfdb
from google.colab import drive
import zipfile
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Mount Google Drive
print("Mounting Google Drive...")
drive.mount('/content/drive')

# Create directory structure in Google Drive
base_path = '/content/drive/MyDrive/MIT_BIH_Dataset'
raw_data_path = os.path.join(base_path, 'raw_data')
processed_data_path = os.path.join(base_path, 'processed_data')

os.makedirs(base_path, exist_ok=True)
os.makedirs(raw_data_path, exist_ok=True)
os.makedirs(processed_data_path, exist_ok=True)

print(f"Dataset will be stored in: {base_path}")

# Define record numbers based on your specifications
# Excluded records: 102, 104, 117, 217 (paced beats)
train_records = [101, 106, 108, 109, 112, 114, 115, 113, 116, 118, 119,
                122, 124, 201, 203, 205, 207, 208, 209, 215, 220, 223, 230]

test_records = [100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210,
               212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234]

excluded_records = [102, 104, 117, 217]  # Paced beats - excluded

all_records = train_records + test_records

print(f"Train records: {len(train_records)} records")
print(f"Test records: {len(test_records)} records")
print(f"Excluded records: {excluded_records}")
print(f"Total records to process: {len(all_records)}")

def install_requirements():
    """Install required packages for PhysioNet data processing"""
    print("Installing required packages...")
    os.system('pip install wfdb')
    os.system('pip install requests')
    print("Packages installed successfully!")

def download_record(record_num, data_path):
    """
    Download a single MIT-BIH record from PhysioNet

    Args:
        record_num (int): Record number to download
        data_path (str): Path to save the downloaded files

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # PhysioNet URL for MIT-BIH database
        base_url = "https://physionet.org/files/mitdb/1.0.0/"
        record_str = str(record_num)

        # Files to download for each record
        extensions = ['.dat', '.hea', '.atr']

        success = True
        for ext in extensions:
            filename = record_str + ext
            url = base_url + filename
            filepath = os.path.join(data_path, filename)

            # Skip if file already exists
            if os.path.exists(filepath):
                continue

            response = requests.get(url)
            if response.status_code == 200:
                with open(filepath, 'wb') as f:
                    f.write(response.content)
            else:
                print(f"Failed to download {filename}: HTTP {response.status_code}")
                success = False

        return success
    except Exception as e:
        print(f"Error downloading record {record_num}: {str(e)}")
        return False

def download_all_records():
    """Download all specified MIT-BIH records"""
    print("Starting download of MIT-BIH records...")

    successful_downloads = []
    failed_downloads = []

    for record in tqdm(all_records, desc="Downloading records"):
        if download_record(record, raw_data_path):
            successful_downloads.append(record)
        else:
            failed_downloads.append(record)

    print(f"\nDownload Summary:")
    print(f"Successful: {len(successful_downloads)} records")
    print(f"Failed: {len(failed_downloads)} records")

    if failed_downloads:
        print(f"Failed records: {failed_downloads}")

    return successful_downloads, failed_downloads

def load_record_data(record_num):
    """
    Load ECG data and annotations for a specific record

    Args:
        record_num (int): Record number to load

    Returns:
        tuple: (signals, annotations, fields) or (None, None, None) if failed
    """
    try:
        record_path = os.path.join(raw_data_path, str(record_num))

        # Read the record
        record = wfdb.rdrecord(record_path)
        annotation = wfdb.rdann(record_path, 'atr')

        # Extract signals - considering lead selection requirements
        signals = record.p_signal
        fs = record.fs  # Sampling frequency

        # Lead information
        sig_names = record.sig_name

        # Special case for record 114 (uses V5 as first lead, MLII as second)
        if record_num == 114:
            print(f"Record {record_num}: Special lead configuration detected")
            print(f"Available leads: {sig_names}")

        return signals, annotation, record.d_signal if hasattr(record, 'd_signal') else signals

    except Exception as e:
        print(f"Error loading record {record_num}: {str(e)}")
        return None, None, None

def process_record(record_num, signals, annotation):
    """
    Process a single record for arrhythmia classification

    Args:
        record_num (int): Record number
        signals (np.array): ECG signals
        annotation: Annotation object

    Returns:
        dict: Processed record information
    """
    try:
        # Extract beat annotations
        beat_annotations = annotation.sample
        beat_symbols = annotation.symbol

        # Lead selection logic
        # Modified limb lead II is typically the first lead
        # Exception: Record 114 uses V5 as first lead, MLII as second
        if record_num == 114:
            lead_1 = signals[:, 0]  # V5 lead
            lead_2 = signals[:, 1]  # MLII lead
            lead_info = "V5 (first), MLII (second)"
        else:
            lead_1 = signals[:, 0]  # MLII (Modified Limb Lead II)
            lead_2 = signals[:, 1]  # Pericardial lead (V1, V2, V4, or V5)
            lead_info = "MLII (first), Pericardial (second)"

        processed_data = {
            'record_number': record_num,
            'lead_1_signal': lead_1,
            'lead_2_signal': lead_2,
            'beat_locations': beat_annotations,
            'beat_symbols': beat_symbols,
            'signal_length': len(lead_1),
            'num_beats': len(beat_annotations),
            'lead_configuration': lead_info,
            'is_train': record_num in train_records
        }

        return processed_data

    except Exception as e:
        print(f"Error processing record {record_num}: {str(e)}")
        return None

def save_processed_data(processed_records):
    """Save processed data to files"""
    print("Saving processed data...")

    # Separate train and test data
    train_data = [r for r in processed_records if r['is_train']]
    test_data = [r for r in processed_records if not r['is_train']]

    # Save summary information
    summary = {
        'total_records': len(processed_records),
        'train_records': len(train_data),
        'test_records': len(test_data),
        'train_record_numbers': [r['record_number'] for r in train_data],
        'test_record_numbers': [r['record_number'] for r in test_data],
        'excluded_records': excluded_records
    }

    # Save summary as pandas DataFrame
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(os.path.join(processed_data_path, 'dataset_summary.csv'), index=False)

    # Save individual record information
    record_info = []
    for record in processed_records:
        info = {
            'record_number': record['record_number'],
            'signal_length': record['signal_length'],
            'num_beats': record['num_beats'],
            'lead_configuration': record['lead_configuration'],
            'dataset_split': 'train' if record['is_train'] else 'test'
        }
        record_info.append(info)

    record_df = pd.DataFrame(record_info)
    record_df.to_csv(os.path.join(processed_data_path, 'record_information.csv'), index=False)

    print(f"Summary saved to: {os.path.join(processed_data_path, 'dataset_summary.csv')}")
    print(f"Record info saved to: {os.path.join(processed_data_path, 'record_information.csv')}")

    return summary

def main():
    """Main execution function"""
    print("=" * 60)
    print("MIT-BIH Arrhythmia Dataset Acquisition and Processing")
    print("=" * 60)

    # Step 1: Install requirements
    install_requirements()

    # Step 2: Download records
    successful_records, failed_records = download_all_records()

    if not successful_records:
        print("No records were successfully downloaded. Exiting.")
        return

    # Step 3: Process downloaded records
    print("\nProcessing downloaded records...")
    processed_records = []

    for record_num in tqdm(successful_records, desc="Processing records"):
        signals, annotation, _ = load_record_data(record_num)

        if signals is not None and annotation is not None:
            processed_data = process_record(record_num, signals, annotation)
            if processed_data:
                processed_records.append(processed_data)
        else:
            print(f"Failed to process record {record_num}")

    # Step 4: Save processed data
    if processed_records:
        summary = save_processed_data(processed_records)

        print("\n" + "=" * 60)
        print("PROCESSING COMPLETE")
        print("=" * 60)
        print(f"Total processed records: {summary['total_records']}")
        print(f"Training records: {summary['train_records']}")
        print(f"Testing records: {summary['test_records']}")
        print(f"Data saved in: {base_path}")
        print("=" * 60)

        # Display sample of processed data
        print("\nSample Record Information:")
        for i, record in enumerate(processed_records[:3]):
            print(f"Record {record['record_number']}:")
            print(f"  - Signal length: {record['signal_length']} samples")
            print(f"  - Number of beats: {record['num_beats']}")
            print(f"  - Lead configuration: {record['lead_configuration']}")
            print(f"  - Dataset: {'Train' if record['is_train'] else 'Test'}")
            print()
    else:
        print("No records were successfully processed.")

# Execute the main function
if __name__ == "__main__":
    main()

# Additional utility functions for data access
def load_specific_record(record_num):
    """
    Utility function to load a specific processed record

    Args:
        record_num (int): Record number to load

    Returns:
        dict: Processed record data or None if not found
    """
    signals, annotation, _ = load_record_data(record_num)
    if signals is not None and annotation is not None:
        return process_record(record_num, signals, annotation)
    return None

def get_dataset_statistics():
    """Get basic statistics about the downloaded dataset"""
    try:
        summary_path = os.path.join(processed_data_path, 'dataset_summary.csv')
        record_info_path = os.path.join(processed_data_path, 'record_information.csv')

        if os.path.exists(summary_path) and os.path.exists(record_info_path):
            summary_df = pd.read_csv(summary_path)
            record_df = pd.read_csv(record_info_path)

            print("Dataset Statistics:")
            print(f"Total records: {len(record_df)}")
            print(f"Train records: {len(record_df[record_df['dataset_split'] == 'train'])}")
            print(f"Test records: {len(record_df[record_df['dataset_split'] == 'test'])}")
            print(f"Total beats: {record_df['num_beats'].sum()}")
            print(f"Average signal length: {record_df['signal_length'].mean():.0f} samples")

            return summary_df, record_df
        else:
            print("Dataset summary files not found. Please run the main processing first.")
            return None, None
    except Exception as e:
        print(f"Error getting dataset statistics: {str(e)}")
        return None, None

print("MIT-BIH Dataset Acquisition Script Ready!")
print("Run main() to start the download and processing pipeline.")
print("Use load_specific_record(record_num) to load individual records.")
print("Use get_dataset_statistics() to view dataset information.")
