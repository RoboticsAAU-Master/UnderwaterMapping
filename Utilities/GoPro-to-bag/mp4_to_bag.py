import os
import sys
import json
import glob
import ntpath
import subprocess

# Mp4 files to convert
base_path = "/RUD-PT/rudpt_ws/src/UnderwaterMapping/Utilities/"
videos_left = [file for file in glob.glob(base_path + "GoPro-to-bag/videos/left/*")]
videos_right = [file for file in glob.glob(base_path + "GoPro-to-bag/videos/right/*")]

# Opening JSON file
with open(base_path + 'GoPro-to-bag/times.json') as json_file:
    times = json.load(json_file)
    new_times = times.copy()

for video_left, video_right in zip(videos_left, videos_right):
    ### Prechecks to ensure correct video pair ###
    video_left_base = ntpath.basename(video_left)
    video_right_base = ntpath.basename(video_right)
    if video_left_base[:7] != video_right_base[:7]:
        print("Videos do not match")
        continue
    
    print("CONVERTING:", video_left_base, video_right_base)
    
    T_START = times[video_left_base[:7]]["left_time"][0]
    T_END = times[video_left_base[:7]]["left_time"][1]
    
    ### Obtaining synchronisation offset between videos ###
    from sync_audio import get_offset

    output_file = base_path + "GoPro-synchronise-audio/Output/offsets.txt"
    offset = get_offset(video_left, video_right, output_file, overwrite=True)
    
    new_times[video_left_base[:7]]["left_crop"] = offset
    
    
    ### Extracting the frames from the videos ###
    from extract_frames import save_all_frames
    
    output_folder = base_path + "GoPro-data-extraction/Output/" + video_left_base[:7] + "/Images"
    save_all_frames(video_left, output_folder, "img_left", "png", T_START, T_END, gray=True, frame_skip=2, downscale=0.5)
    save_all_frames(video_right, output_folder, "img_right", "png", T_START - offset, T_END - offset, gray=True, frame_skip=2, downscale=0.5)
    
    
    ### Extracting the metadata from the videos ###
    output_folder = base_path + "GoPro-data-extraction/Output/" + video_left_base[:7] + "/Metadata"
    subprocess.run([base_path + "GoPro-to-bag/extract_metadata.sh", video_left, output_folder])

    from determine_onset import determine_onset
    try :
        onset_time = determine_onset(output_folder + "/outputAccl.csv")
        new_times[video_left_base[:7]]["left_onset"] = onset_time
    except ValueError as e:
        print(e)
        continue
    

    ### Converting the images and metadata to a bag file ###
    from gopro_to_bag import CreateBag

    CreateBag(args=[
        base_path + "GoPro-data-extraction/Output/1,1_0_0_10/Images",
        base_path + "Bag-conversion/Output/1,1_0_0_10.bag",
        base_path + "GoPro-data-extraction/Output/1,1_0_0_10/Metadata",
        "euroc_stereo",
        T_START,
        T_END,
    ])

print("Conversion complete")

# Convert and write JSON object to file with new times
with open(base_path + "GoPro-to-bag/new_times.json", "w") as outfile: 
    json.dump(new_times, outfile)
    