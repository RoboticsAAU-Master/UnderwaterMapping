from syncstart import file_offset

def get_offset(video1, video2, output_file):
    args = {'in1': video1,
            'in2': video2,
            'take': 20} # Seconds of videos to keep (20 is default)
    
    file_ahead, offset = file_offset(**args)
    
    # Write offset to file
    with open(output_file, "w") as file:
        # Write the value to the file
        file.write(file_ahead + f" is ahead by: {offset} [s]\n")

if __name__ == "__main__":
    # Input: path_video1  path_video2  output_file
    get_offset("Utilities/GoPro-synchronise-audio/Output/GoPro1_Clap.MP4",
               "Utilities/GoPro-synchronise-audio/Output/GoPro2_Clap.MP4",
               "Utilities/GoPro-synchronise-audio/Output/Kridtgraven-20-09-23.txt")
    
    # Output gives how much the given video should be cut to for the two videos to be synchronised