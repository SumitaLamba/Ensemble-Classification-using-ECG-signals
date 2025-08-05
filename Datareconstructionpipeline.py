# Execute the main quality assessment and reconstruction pipeline
print("Starting MIT-BIH Signal Quality Assessment and Missing Data Reconstruction...")
print("This will process all records and generate quality reports.")
print("Please wait, this may take several minutes...")

# Run the main processing function
results, summary_df = main_quality_assessment()

# Display additional insights if processing was successful
if results and summary_df is not None:
    print("\n" + "="*70)
    print("DETAILED ANALYSIS COMPLETE")
    print("="*70)
    
    # Show top 5 highest quality records
    print("\nTop 5 Highest Quality Records:")
    top_quality = summary_df.nlargest(5, 'quality_score')[['record_number', 'lead_name', 'quality_score', 'snr_db']]
    print(top_quality.to_string(index=False))
    
    # Show records that needed most reconstruction
    print("\nRecords Requiring Most Reconstruction:")
    most_reconstruction = summary_df.nlargest(5, 'reconstruction_percentage')[['record_number', 'lead_name', 'reconstruction_percentage', 'quality_score']]
    print(most_reconstruction.to_string(index=False))
    
    # Quality distribution
    print(f"\nQuality Score Distribution:")
    print(f"Excellent (>0.9): {len(summary_df[summary_df['quality_score'] > 0.9])} records")
    print(f"Good (0.7-0.9): {len(summary_df[(summary_df['quality_score'] > 0.7) & (summary_df['quality_score'] <= 0.9)])} records")
    print(f"Fair (0.5-0.7): {len(summary_df[(summary_df['quality_score'] > 0.5) & (summary_df['quality_score'] <= 0.7)])} records")
    print(f"Poor (<0.5): {len(summary_df[summary_df['quality_score'] <= 0.5])} records")
    
    print(f"\nData saved to: /content/drive/MyDrive/MIT_BIH_Dataset/quality_assessment/")
    print("✅ Quality assessment and missing data reconstruction completed successfully!")
    
    # Offer to visualize a sample record
    print("\n" + "="*70)
    print("NEXT STEPS:")
    print("1. Review the quality assessment summary CSV file")
    print("2. Use visualize_reconstruction(record_num, 'MLII') to see specific examples")
    print("3. The reconstructed signals are ready for further processing")
    print("="*70)
    
else:
    print("❌ Quality assessment failed. Please check your data directory and try again.")
