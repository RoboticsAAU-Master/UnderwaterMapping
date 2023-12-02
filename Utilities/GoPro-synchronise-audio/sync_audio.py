from syncstart import file_offset
from datetime import datetime, timedelta
import os
import cv2
from math import ceil

# Parameters (video2_input should be the video that is cut)
video1_input = "1,1_0_0_10_left.MP4"  # Specify path with "/" or "\\" not "\"
video2_input = "1,1_0_0_10_right.MP4"
video1_output = "apriltag_2torch_targmove_left_cut.MP4"
video2_output = "apriltag_2torch_targmove_right_cut.MP4"
offset_output = "Utilities/GoPro-synchronise-audio/Output/apriltag_torch.text"
start_time = "00:00:50.000"  # HH:MM:SS.mmm # Keep at whole seconds
end_time = "00:01:45.000"  # HH:MM:SS.mmm
# duration = "00:00:50.000"  # HH:MM:SS.mmm


def get_offset(video1, video2, output_file=None):
    args = {
        "in1": video1,
        "in2": video2,
        "take": 30,
        "show": False,
    }  # Seconds of videos to keep (20 is default)

    file_ahead, offset = file_offset(**args)

    # Return if no output file is specified
    if output_file is not None:
        # Write offset to file
        with open(output_file, "w") as file:
            # Write the value to the file
            file.write(
                os.path.basename(file_ahead)
                + f" is started {offset} [s] before the other.\n"
            )

    if video1.split("/")[-1] == file_ahead:  # Check which video is ahead
        return offset  # Positive offset means video1 is ahead of video2
    else:
        return -offset  # Negative offset means video2 is ahead of video1


if __name__ == "__main__":
    # Input: path_video1  path_video2  output_file
    offset = get_offset(video1_input, video2_input, offset_output)

    # Output gives how much the given video should be cut to for the two videos to be synchronised

    # if offset > 0:
    #     video1_input, video2_input = video2_input, video1_input
    #     video1_output, video2_output = video2_output, video1_output
    #     offset = -offset

    # # Crop video1
    # os.system(
    #     f'ffmpeg -i "{video1_input}" -ss {start_time} -to {end_time} -map_metadata 0 -map 0:u -c copy "{video1_output}" -y'
    # )
    # # os.system(
    # #     f'ffmpeg -i "{video1_input}" -ss {start_time} -t {end_time} -movflags use_metadata_tags -map_metadata 0 -map 0:u -c copy "{video1_output}" -y'
    # # )

    # # # Crop video2
    # start_time_dt = datetime.strptime(start_time, "%H:%M:%S.%f")  # Convert to datetime
    # start_time_dt += timedelta(
    #     seconds=-offset
    # )  # Add offset to start time (offset is negative if video2 is ahead of video1)
    # start_time = start_time_dt.strftime("%H:%M:%S.%f")  # Convert back to string

    # end_time_dt = datetime.strptime(end_time, "%H:%M:%S.%f")  # Convert to datetime
    # end_time_dt += timedelta(
    #     seconds=-offset
    # )  # Add offset to start time (offset is negative if video2 is ahead of video1)
    # end_time = end_time_dt.strftime("%H:%M:%S.%f")  # Convert back to string

    # os.system(
    #     f'ffmpeg -i "{video2_input}" -ss {start_time} -to {end_time} -map_metadata 0 -map 0:u -c copy "{video2_output}" -y'
    # )
    # # os.system(
    # #     f'ffmpeg -i "{video2_input}" -ss {start_time} -t {end_time} -movflags use_metadata_tags -map_metadata 0 -map 0:u -c copy "{video2_output}" -y'
    # # )

    # cap1 = cv2.VideoCapture(video1_output)
    # cap2 = cv2.VideoCapture(video2_output)

    # print(f"Cropped video 1 framelength: {int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))}")
    # print(f"Cropped video 2 framelength: {int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))}")
