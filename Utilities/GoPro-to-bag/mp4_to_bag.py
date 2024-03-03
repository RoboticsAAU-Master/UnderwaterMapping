import os
import json
import glob
import ntpath
import subprocess

from sync_audio import get_offset
from extract_frames import save_all_frames
from gopro_to_bag import CreateBag
from view_bag import show_topics

# BASE_PATH = "/storage/extraction/UnderwaterMapping/Utilities"
# VIDEOS_FOLDER = "/storage/data/videos"
# OUTPUT_FOLDER = "/storage/data/bags"
BASE_PATH = "/RUD-PT/rudpt_ws/src/UnderwaterMapping/Utilities"
VIDEOS_FOLDER = "/RUD-PT/rudpt_ws/src/UnderwaterMapping/Utilities/GoPro-to-bag/videos"
OUTPUT_FOLDER = "/RUD-PT/rudpt_ws/src/UnderwaterMapping/Utilities/GoPro-to-bag/bags"

def mp4_to_bag(base_path, videos_folder, output_folder):
    # Mp4 files to convert
    videos_left = [file for file in glob.glob(videos_folder + "/left/*")]
    videos_right = [file for file in glob.glob(videos_folder + "/right/*")]
    
    # Opening JSON file
    with open(base_path + '/GoPro-to-bag/times.json') as json_file:
        times = json.load(json_file)
        new_times = times.copy()
    
    # Create empty log file
    log_file = base_path + '/GoPro-to-bag/log.txt'
    open(log_file, 'w').close()
    
    processed = 0
    for video_left, video_right in zip(videos_left, videos_right):
        ### Prechecks to ensure correct video pair ###
        video_left_base = ntpath.basename(video_left)
        video_right_base = ntpath.basename(video_right)
        if video_left_base[:7] != video_right_base[:7]:
            log_print("Videos do not match", video_left_base, video_right_base, log_file=log_file)
            continue
        
        output_file = output_folder + "/" + video_left_base[:7] + ".bag"
        if os.path.isfile(output_file):
            log_print("Skipping", video_left_base[:7], "as bag file already exists", log_file=log_file)
            continue
        
        log_print("CONVERTING:", video_left_base, video_right_base, log_file=log_file)
        
        T_START = times[video_left_base[:7]]["left_time"][0]
        T_END = times[video_left_base[:7]]["left_time"][1]

        ### Obtaining synchronisation offset between videos ###
        output_file_offset = base_path + "/GoPro-synchronise-audio/Output/offsets.txt"
        offset = get_offset(video_left, video_right, output_file_offset, overwrite=True)
        
        new_times[video_left_base[:7]]["left_crop"] = offset
        
        
        ### Extracting the frames from the videos ###
        output_folder_images = base_path + "/GoPro-data-extraction/Output/" + video_left_base[:7] + "/Images"
        save_all_frames(video_left, output_folder_images, "img_left", "png", T_START, T_END, gray=True, frame_skip=1, downscale=0.5)
        save_all_frames(video_right, output_folder_images, "img_right", "png", T_START - offset, T_END - offset, gray=True, frame_skip=1, downscale=0.5)
        
        
        ### Extracting the metadata from the videos ###
        output_folder_metadata = base_path + "/GoPro-data-extraction/Output/" + video_left_base[:7] + "/Metadata"
        subprocess.run([base_path + "/GoPro-to-bag/extract_metadata.sh", base_path, video_left, output_folder_metadata])

        from determine_onset import determine_onset
        try :
            onset_time = determine_onset(output_folder_metadata + "/outputAccl.csv")
            new_times[video_left_base[:7]]["left_onset"] = onset_time
        except ValueError as e:
            log_print(e, "for", video_left_base[:7], log_file=log_file)
        

        ### Converting the images and metadata to a bag file ###
        CreateBag(args=[
            output_folder_images,
            output_file,
            output_folder_metadata,
            "euroc_stereo",
            T_START,
            T_END,
        ])
        
        show_topics(output_file)
        
        processed += 1

    log_print(f"Conversion complete. Processed {processed} videos", log_file=log_file)


    # Convert and write JSON object to file with new times
    with open(base_path + "/GoPro-to-bag/new_times.json", "w") as outfile: 
        json.dump(new_times, outfile)


def log_print(*args, log_file, sep=' ', end='\n'):
    # Open the log file in append mode if it exists, or create it if it doesn't
    with open(log_file, 'a') as f:
        # Print to console
        print(*args, sep=sep, end=end, file=f)
        
        # Print to log file
        print(*args, sep=sep, end=end)

if __name__ == "__main__":
    mp4_to_bag(base_path=BASE_PATH, videos_folder=VIDEOS_FOLDER, output_folder=OUTPUT_FOLDER)
    
