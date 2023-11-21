from syncstart import file_offset
from datetime import datetime, timedelta
import os

# Parameters (video2_input should be the video that is cut)
video1_input = "RightCalib_UW_3.MP4"  # Specify path with "/" or "\\" not "\"
video2_input = "LeftCalib_UW_3.MP4"
video1_output = "Utilities/GoPro-synchronise-audio/Output/RightCalib_UW_3_Cut.MP4"
video2_output = "Utilities/GoPro-synchronise-audio/Output/LeftCalib_UW_3_Cut.MP4"
offset_output = "Utilities/GoPro-synchronise-audio/Output/Calib_UW_offset.text"
start_time = "00:00:07.000"  # HH:MM:SS.mmm
end_time = "00:01:00.000"  # HH:MM:SS.mmm


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

    # Crop video1
    os.system(
        f'ffmpeg -i "{video1_input}" -ss {start_time} -to {end_time} -map_metadata 0 -map 0:u -c copy "{video1_output}" -y'
    )

    # Crop video2
    start_time_dt = datetime.strptime(start_time, "%H:%M:%S.%f")  # Convert to datetime
    start_time_dt += timedelta(
        seconds=-offset
    )  # Add offset to start time (offset is negative if video2 is ahead of video1)
    start_time = start_time_dt.strftime("%H:%M:%S.%f")  # Convert back to string
    os.system(
        f'ffmpeg -i "{video2_input}" -ss {start_time} -to {end_time} -map_metadata 0 -map 0:u -c copy "{video2_output}" -y'
    )
