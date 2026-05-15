import os
import pandas as pd

def main():
    videos_dir = "/home/user3/AI_VQA(10000)/AI-VQA_Full_dataset"
    splits_dir = "/home/user3/AI_VQA(10000)/10Ksplits"
    
    print(f"Looking for CSV files in: {splits_dir}")
    if not os.path.exists(splits_dir):
        print("Error: Splits directory does not exist.")
        return

    # 1. List all CSV files
    csv_files = [f for f in os.listdir(splits_dir) if f.endswith('.csv')]
    print("\n--- CSV Files Found ---")
    for f in csv_files:
        print(f)
        
    summary_stats = []
    
    columns_to_check = [
        'mos', 'quality_accuracy', 'spatial_flag', 'hallucination_flag', 
        'lighting_flag', 'rendering_flag', 'physics_violation_flag', 
        'object_flicker_flag', 'motion_inconsistency_flag'
    ]

    for csv_file in csv_files:
        print(f"\n{'='*50}")
        print(f"Analyzing File: {csv_file}")
        print(f"{'='*50}")
        
        df = pd.read_csv(os.path.join(splits_dir, csv_file))
        
        # 2. Print columns, number of rows, and sample rows
        print("\n--- File Info ---")
        print(f"Columns: {', '.join(df.columns)}")
        print(f"Number of rows: {len(df)}")
        print("\nSample Rows (First 3):")
        print(df.head(3).to_string())
        
        # 4. Identify Split
        lower_name = csv_file.lower()
        split_type = "Unknown"
        if "train" in lower_name:
            split_type = "Train"
        elif "val" in lower_name:
            split_type = "Validation"
        elif "test" in lower_name:
            split_type = "Test"
            
        print(f"\nIdentified Split: {split_type}")
        
        # 3. Check video filenames
        print("\n--- Video File Matching ---")
        video_col = None
        for col in ['video', 'video_name', 'filename', 'file_name', 'name']:
            if col in df.columns:
                video_col = col
                break
                
        if video_col:
            matches = 0
            missing = 0
            if os.path.exists(videos_dir):
                for vid in df[video_col]:
                    vid_path = os.path.join(videos_dir, str(vid))
                    if os.path.exists(vid_path):
                        matches += 1
                    else:
                        # Try stripping _standardized
                        alt_vid = str(vid).replace("_standardized", "")
                        alt_vid_path = os.path.join(videos_dir, alt_vid)
                        if os.path.exists(alt_vid_path):
                            matches += 1
                        else:
                            missing += 1
                print(f"Matches: {matches}")
                print(f"Missing: {missing}")
            else:
                print(f"Video directory {videos_dir} not found.")
        else:
            print("Could not find a video filename column. Looked for: 'video', 'video_name', 'filename', 'file_name', 'name'")
            
        # 6. Check for null/missing values
        print("\n--- Missing Values Check ---")
        for col in columns_to_check:
            if col in df.columns:
                nulls = df[col].isnull().sum()
                print(f"{col}: {nulls} missing")
            else:
                print(f"{col}: <Column not in CSV>")
                
        # Gather info for summary
        mos_stats = {"min": None, "max": None, "mean": None}
        if 'mos' in df.columns:
            mos_stats['min'] = df['mos'].min()
            mos_stats['max'] = df['mos'].max()
            mos_stats['mean'] = df['mos'].mean()
            
        summary_stats.append({
            'split': split_type,
            'file': csv_file,
            'total_rows': len(df),
            'mos_stats': mos_stats
        })

    # 5. Print a clean summary report
    print(f"\n{'#'*50}")
    print("FINAL SUMMARY REPORT")
    print(f"{'#'*50}")
    for stat in summary_stats:
        print(f"\nSplit: {stat['split']} ({stat['file']})")
        print(f"Total Videos: {stat['total_rows']}")
        ms = stat['mos_stats']
        if ms['mean'] is not None:
            print(f"MOS Distribution -> Min: {ms['min']:.4f}, Max: {ms['max']:.4f}, Mean: {ms['mean']:.4f}")
        else:
            print("MOS Distribution -> 'mos' column not found.")
            
if __name__ == "__main__":
    main()
