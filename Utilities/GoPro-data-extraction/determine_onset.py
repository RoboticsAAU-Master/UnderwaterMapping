import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as signal

IMU_SAMPLE_RATE = 200.0


def determine_onset(accl_file):
    # Read the data
    accl = pd.read_csv(accl_file, delimiter=",", header=None)
    accl.drop(accl.columns[-1], axis=1, inplace=True)
    
    axis_data = accl.iloc[:, 0].to_numpy()
    
    data = np.abs(np.gradient(axis_data[:len(axis_data)//2]))
    
    # Cut out the first 20 seconds of the data
    start_index = int(IMU_SAMPLE_RATE * 20)
    data = data[start_index:]
    
    # Determine the peaks
    peak_idxs = signal.argrelextrema(data, np.greater, order=50)[0]
    sorted_peak_idxs = sorted(peak_idxs, key=lambda x: data[x], reverse=True) 
    
    for peak_idx in sorted_peak_idxs[:min(3, len(sorted_peak_idxs))]:
        # Check if the trajectory is stationary at the beginning
        prev_window = data[max(0, peak_idx - int(1*IMU_SAMPLE_RATE)) : peak_idx + 1]
        threshold = 0.01
        if np.median(prev_window) < threshold:
            # Return the start time from the imu sample rate
            return (start_index + peak_idx) / IMU_SAMPLE_RATE
    
    raise ValueError("The trajectory is not stationary at the beginning")


if __name__ == "__main__":
    # Determine the onset
    onset_time = determine_onset("/storage/extraction/UnderwaterMapping/Utilities/GoPro-data-extraction/Output/2,1_0_0/Metadata/outputAccl.csv")

    # Print the onset
    print("The onset is at {} seconds".format(onset_time))
