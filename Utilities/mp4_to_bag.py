import os
import sys

from sync_audio import get_offset

# Mp4 files to convert
video_left = "rudpt_ws/src/UnderwaterMapping/1,1_0_0_10_left.MP4"
video_right = "rudpt_ws/src/UnderwaterMapping/1,1_0_0_10_right.MP4"
output_file = "rudpt_ws/src/UnderwaterMapping/Utilities/GoPro-synchronise-audio/Output/offsets.text"

offset = get_offset(video_left, video_right, output_file)



