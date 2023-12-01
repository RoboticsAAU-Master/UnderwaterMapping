import numpy as np
import pandas as pd


IMU_SAMPLE_RATE = 200.0


def determine_onset(axis_data):
    # Get the maximum index and value of the absolute gradient
    max_index = np.argmax(axis_data)
    max_val = axis_data[max_index]

    # Check if the trajectory is stationary at the beginning
    prev_window = axis_data[max(0, max_index - 60) : max_index + 1]
    threshold = 0.1 * max_val
    if np.median(prev_window) > threshold:
        raise ValueError("The trajectory is not stationary at the beginning")

    # Set start time to half a second before the maximum
    start_time = (max_index - 0.5 * IMU_SAMPLE_RATE) * IMU_SAMPLE_RATE

    return start_time


if __name__ == "__main__":
    # Read the data
    accl = pd.read_csv("Output/XXX/Metadata/outputAccl.csv", delimiter=";", header=None)

    # Determine the onset
    onset_time = determine_onset(accl.iloc[:, 0])

    # Print the onset
    print("The onset is at {} seconds".format(onset_time))
