from syncstart import file_offset
from datetime import datetime, timedelta
import os
import cv2
from math import ceil

# Parameters (video2_input should be the video that is cut)
video1_input = "1,1_0_0_10_left.MP4"  # Specify path with "/" or "\\" not "\"
video2_input = "1,1_0_0_10_right.MP4"
offset_output = "Utilities/GoPro-synchronise-audio/Output/apriltag_torch.text"


def get_offset(video1, video2, output_file=None, overwrite=False):
    args = {
        "in1": video1,
        "in2": video2,
        "take": 30,
        "show": False,
    }  # Seconds of videos to keep (20 is default)

    file_ahead, offset = file_offset(**args)

    if video1.split("/")[-1] != file_ahead:  # Change sign of offset if video2 is ahead
         offset = -offset

    # Return if no output file is specified
    if output_file is not None:
        # Check if the file exists
        file_mode = "a" if (os.path.exists(output_file) and not overwrite) else "w"
        
        # Write offset to file
        with open(output_file, file_mode) as file:
            # Write the value to the file
            file.write(
                os.path.basename(video1)
                + f" is started {offset} [s] before the other.\n"
            )

    return offset  # Positive offset means video1 is ahead of video2


if __name__ == "__main__":
    # Input: path_video1  path_video2  output_file
    get_offset(video1_input, video2_input, offset_output)
