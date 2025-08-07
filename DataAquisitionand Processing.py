#!/usr/bin/env python3
"""
ECG Dataset Access and Processing from Google Drive Folders
"""

import os
import numpy as np
import pandas as pd
import wfdb
from google.colab import drive
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Mount Google Drive
print("Mounting Google Drive...")
drive.mount('/content/drive')

# Define dataset paths
base_drive_path = '/your/path/for/ECG_Datasets'

dataset_paths = {
    'MIT_BIH': os.path.join(base_drive_path, 'MIT_BIH'),
    'INCART': os.path.join(base_drive_path, 'INCART_DB'),
    'NSR': os.path.join(base_drive_path, 'NSR_DB'),
}

processed_data_path = os.path.join(base_drive_path, 'processed_data')
os.makedirs(processed_data_path, exist_ok=True)

# MIT-BIH train/test split
train_records = [101, 106, 108, 109, 112, 114, 115, 113, 116, 118, 119,
                 122, 124, 201, 203, 205, 207, 208, 209, 215, 220, 223, 230]
test_records = [100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210,
                212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234]
excluded_records = [102, 104, 117, 217]

all_records = train_records + test_records

print(f"\nMIT-BIH Arrhythmia Dataset from Drive:")
print(f"Train records: {len(train_records)}")
print(f"Test records: {len(test_records)}")
print(f"Excluded: {excluded_records}")

def load_record_data(record_num, dataset='MIT_BIH'):
    """Load ECG data and annotations from local folder"""
    try:
        record_path = os.path.join(dataset_paths[dataset], str(record_num))
        record = wfdb.rdrecord(record_path)
        annotation = wfdb.rdann(record_path, 'atr')
        signals = record.p_signal
        return signals, annotation, record.d_signal if hasattr(record, 'd_signal') else signals
    except Exception as e:
        print(f"Error loading record {record_num} from {dataset}: {str(e)}")
        return None, None, None

def process_record(record_num, signals, annotation):
    """Process individual record"""
    try:
        beat_annotations = annotation.sample
        beat_symbols = annotation.symbol
        if record_num == 114:
            lead_1 = signals[:, 0]
            lead_2 = signals[:, 1]
            lead_info = "V5 (first), MLII (second)"
        else:
            lead_1 = signals[:, 0]
            lead_2 = signals[:, 1]
            lead_info = "MLII (first), Pericardial (second)"

        return {
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
    except Exception as e:
        print(f"Error processing record {record_num}: {str(e)}")
        return None

def save_processed_data(processed_records):
    """Save summaries"""
    train_data = [r for r in processed_records if r['is_train']]
    test_data = [r for r in processed_records if not r['is_train']]

    summary = {
        'total_records': len(processed_records),
        'train_records': len(train_data),
        'test_records': len(test_data),
        'train_record_numbers': [r['record_number'] for r in train_data],
        'test_record_numbers': [r['record_number'] for r in test_data],
        'excluded_records': excluded_records
    }

    pd.DataFrame([summary]).to_csv(os.path.join(processed_data_path, 'dataset_summary.csv'), index=False)

    record_info = [{
        'record_number': r['record_number'],
        'signal_length': r['signal_length'],
        'num_beats': r['num_beats'],
        'lead_configuration': r['lead_configuration'],
        'dataset_split': 'train' if r['is_train'] else 'test'
    } for r in processed_records]

    pd.DataFrame(record_info).to_csv(os.path.join(processed_data_path, 'record_information.csv'), index=False)
    print("Processed data saved.")

def main():
    print("=" * 60)
    print("ECG Dataset Processing from Google Drive")
    print("=" * 60)

    processed_records = []
    for record_num in tqdm(all_records, desc="Processing MIT-BIH"):
        signals, annotation, _ = load_record_data(record_num, dataset='MIT_BIH')
        if signals is not None and annotation is not None:
            data = process_record(record_num, signals, annotation)
            if data: processed_records.append(data)

    if processed_records:
        save_processed_data(processed_records)
        print(f"Processed {len(processed_records)} records")
    else:
        print("No records were successfully processed.")

def load_specific_record(record_num, dataset='MIT_BIH'):
    """Load and process a specific record from any dataset"""
    signals, annotation, _ = load_record_data(record_num, dataset=dataset)
    if signals is not None and annotation is not None:
        return process_record(record_num, signals, annotation)
    return None

def get_dataset_statistics():
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
            print(f"Avg signal length: {record_df['signal_length'].mean():.0f} samples")
            return summary_df, record_df
        else:
            print("Summary files not found. Run `main()` first.")
            return None, None
    except Exception as e:
        print(f"Error loading dataset stats: {str(e)}")
        return None, None

print("Script ready! Run `main()` to begin processing.")
print("Use `load_specific_record(record_num, dataset)` for any dataset.")
print("Use `get_dataset_statistics()` to view summary.")
