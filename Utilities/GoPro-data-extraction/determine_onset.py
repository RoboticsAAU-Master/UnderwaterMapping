import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


IMU_SAMPLE_RATE = 200.0


def determine_onset(accl_file):
    # Read the data
    accl = pd.read_csv(accl_file, delimiter=",", header=None)
    accl.drop(accl.columns[-1], axis=1, inplace=True)
    
    axis_data = accl.iloc[:, 0].to_numpy()
    
    data = np.abs(axis_data[:len(axis_data)//2])
    
    # Get the maximum index and value of the absolute gradient
    max_index = np.argmax(data)
    max_val = data[max_index]

    # Check if the trajectory is stationary at the beginning
    prev_window = data[max(0, max_index - 60) : max_index + 1]
    threshold = 0.1 * max_val
    if np.median(prev_window) > threshold:
        raise ValueError("The trajectory is not stationary at the beginning")

    # Set start time to half a second before the maximum
    start_time = max_index * IMU_SAMPLE_RATE

    return start_time


if __name__ == "__main__":
    # Determine the onset
    onset_time = determine_onset("Utilities/GoPro-data-extraction/Output/1,1_0_0_10/Metadata/outputAccl.csv")

    # Print the onset
    print("The onset is at {} seconds".format(onset_time))
