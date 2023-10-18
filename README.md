# UnderwaterMapping


## GoPro to bag
It is possible to crop the .mp4 GoPro video while keeping metadata using the following command:
`ffmpeg -i INPUT_FILE -ss MM:SS -to MM:SS -map_metadata 0 -map 0:u -c copy OUTPUT_FILE`

The conversion from GoPro to bag is done in two main step: 1) Data extraction, 2) Bag conversion

### 1) Data extraction (Folder: "Utilities/GoPro-data-extraction")
- Extract the individual frames from the video using "extract_frames.py". Changeable parameters for the output are: frames to remove (framerate), downscaling, color space
- Extract the metadata from the video using "gopro_data_extraction.c". Compile the file using `make` and run the file.
    - Note: If submodule "gpmf-parser" is empty, you first need to initialise it using `git submodule update --init`

The output of the data should be in the "Output/$CHOSEN_NAME$/Images" and "Output/$CHOSEN_NAME$/Metadata" folder, respectively. 

### 2) Bag conversion (Folder: "Utilities/Bag-conversion")
- Convert the previously extracted data to a .bag file using "gopro_to_bag.py". Remember to specify topics according to used format in SVO. To check that the conversion worked as intended, you can run "view_bag.py" with the obtained .bag file.
    - Note: If the topics don't match the required format, use "remap_bag.py" to change the topic names.

The output of the .bag file should be in the "Output" folder.