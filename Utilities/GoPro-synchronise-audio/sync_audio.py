from syncstart import file_offset
import os


def get_offset(video1, video2, output_file=None):
    args = {
        "in1": video1,
        "in2": video2,
        "take": 20,
    }  # Seconds of videos to keep (20 is default)

    file_ahead, offset = file_offset(**args)

    # Return if no output file is specified
    if output_file is None:
        return

    # Write offset to file
    with open(output_file, "w") as file:
        # Write the value to the file
        file.write(
            os.path.basename(file_ahead)
            + f" is started {offset} [s] before the other.\n"
        )


if __name__ == "__main__":
    # Input: path_video1  path_video2  output_file
    get_offset(
        "GoPro1_Clap.MP4",
        "GoPro2_Clap.MP4",
        "Utilities/GoPro-synchronise-audio/Output/Kridtgraven-20-09-23.txt",
    )

    # Output gives how much the given video should be cut to for the two videos to be synchronised
