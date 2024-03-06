import os
import json
import glob
import ntpath
from datetime import datetime

from trajectory import process_gt

# BASE_PATH = "/storage/extraction/UnderwaterMapping/Data-collection"
# IN_GT_FOLDER = "/storage/extraction/UnderwaterMapping/Data-collection/csv_data"
# OUT_GT_FOLDER = "/storage/data/gt"
BASE_PATH = "/RUD-PT/rudpt_ws/src/UnderwaterMapping/Data-collection"
IN_GT_FOLDER = "/RUD-PT/rudpt_ws/src/UnderwaterMapping/Data-collection/csv_data"
OUT_GT_FOLDER = "/RUD-PT/rudpt_ws/src/UnderwaterMapping/Data-collection/txt_data"

def convert_gt(base_path, in_gt_folder, out_gt_folder):
    # GT files to convert
    in_gt_files = sorted([file for file in glob.glob(in_gt_folder + "/*")])
    
    # Opening JSON file with times
    times_path = base_path + '/new_times.json'
    with open(times_path) as json_file:
        times = json.load(json_file)
    
    # Create log file
    log_file = base_path + '/log.txt'
    
    processed = 0
    for in_gt_file in in_gt_files:
        ### Prechecks to ensure correct video pair ###
        in_gt_base = ntpath.basename(in_gt_file)
        
        output_file = out_gt_folder + "/" + in_gt_base[:7] + ".txt"
        if os.path.isfile(output_file):
            log_print("Skipping", in_gt_base[:7], "as txt file already exists", log_file=log_file)
            continue
        
        log_print("CONVERTING:", in_gt_base, log_file=log_file)
        
        process_gt(in_gt_file, output_file, times)
        
        processed += 1
    
    log_print(f"Conversion complete. Processed {processed} videos", log_file=log_file)

    
def log_print(*args, log_file, sep=' ', end='\n'):
    # Get the current time
    current_time = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    
    # Open the log file in append mode if it exists, or create it if it doesn't
    with open(log_file, 'a') as f:
        # Print to console
        print(*args, sep=sep, end=end)
        
        # Print to log file
        print(current_time, *args, sep=sep, end=end, file=f)


if __name__ == "__main__":
    convert_gt(base_path=BASE_PATH, in_gt_folder=IN_GT_FOLDER, out_gt_folder=OUT_GT_FOLDER)