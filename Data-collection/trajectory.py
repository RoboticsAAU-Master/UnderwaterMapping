import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import transforms3d as t3d


class DataLoader:
    def __init__(self, csv_file, delimiter) -> None:
        self.delimiter = delimiter
        self.df = pd.read_csv(csv_file, delimiter=self.delimiter)

    def drop_columns(self, columns):
        self.df = self.df.drop(columns, axis=1)

    def drop_invalid_rows(self, column_subset):
        # Drop rows with 0.0 or NaN values in the specified columns
        self.df = self.df.dropna(subset=column_subset, how="all")
        self.df = self.df[(self.df[column_subset] != 0).all(axis=1)]

        # Reset the index
        self.df = self.df.reset_index(drop=True)

    def datetime_to_seconds(self, column):
        # Convert the 'Timestamp' column to datetime
        self.df[column] = pd.to_datetime(self.df[column], format="%Y-%m-%d %H:%M:%S.%f")

        # Convert the datetime values to total seconds
        self.df[column + "_seconds"] = (
            self.df[column] - self.df[column][0]
        ) / pd.Timedelta("1s")

    def save_to_csv(self, output_file):
        self.df.to_csv(output_file, index=False, sep=self.delimiter)


class Trajectory3D:
    def __init__(self) -> None:
        self.x = []
        self.y = []
        self.z = []
        self.rot_x = []
        self.rot_y = []
        self.rot_z = []
        self.timestamps_seconds = []
        self.data_loader = None

    def load_trajectory(
        self, csv_file, delimiter, drop_columns, pose_columns, timestamp_column
    ):
        # Load the csv data and do some preprocessing
        self.data_loader = DataLoader(csv_file, delimiter)
        self.data_loader.drop_columns(drop_columns)
        self.data_loader.drop_invalid_rows(pose_columns)
        self.data_loader.datetime_to_seconds(timestamp_column)

        # Save the trajectory data
        self.x = self.data_loader.df[pose_columns[0]].to_numpy()
        self.y = self.data_loader.df[pose_columns[1]].to_numpy()
        self.z = self.data_loader.df[pose_columns[2]].to_numpy()
        self.rot_x = self.data_loader.df[pose_columns[3]].to_numpy()
        self.rot_y = self.data_loader.df[pose_columns[4]].to_numpy()
        self.rot_z = self.data_loader.df[pose_columns[5]].to_numpy()
        self.timestamps_seconds = self.data_loader.df[
            timestamp_column + "_seconds"
        ].to_numpy()

    def remove_initial_offset(self):
        # Remove the initial offset from the trajectory
        self.x = self.x - self.x[0]
        self.y = self.y - self.y[0]
        self.z = self.z - self.z[0]
        self.rot_x = self.rot_x - self.rot_x[0]
        self.rot_y = self.rot_y - self.rot_y[0]
        self.rot_z = self.rot_z - self.rot_z[0]

    def convert_rotation_to_rad(self):
        # Convert the rotation values to radians
        self.rot_x = np.deg2rad(self.rot_x)
        self.rot_y = np.deg2rad(self.rot_y)
        self.rot_z = np.deg2rad(self.rot_z)

    def _get_trajectory(self):
        return np.array([self.x, self.y, self.z, self.rot_x, self.rot_y, self.rot_z]).T

    def _get_trajectory_time_seconds(self):
        return self.timestamps_seconds[-1]

    def _euler_to_direction_vectors(self):
        # Convert the euler angles to direction vectors
        direction_vectors = np.zeros((len(self.rot_x), 3))
        for i in range(len(self.rot_x)):
            direction_vectors[i] = t3d.euler.euler2mat(
                self.rot_x[i], self.rot_y[i], self.rot_z[i], "sxyz"
            )[:, 0]

        return direction_vectors

    def plot(self, simulate=False, update_time=None):
        # Create a 3d figure
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        # Set labels for the axes
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        # Set axis limits
        ax.set_xlim3d(min(self.x), max(self.x))
        ax.set_ylim3d(min(self.y), max(self.y))
        ax.set_zlim3d(min(self.z), max(self.z))

        # Convert the euler angles to direction vectors
        direction_vectors = self._euler_to_direction_vectors()

        # Plot the trajectory simulated in real time
        if simulate:
            # Calculate the number of samples to skip between updates
            if update_time is None:
                skip_num = 0
            else:
                if update_time < (
                    self.timestamps_seconds[1] - self.timestamps_seconds[0]
                ):
                    raise ValueError(
                        "The update frequency cannot be greater than the sampling frequency"
                    )

                skip_num = (
                    int(
                        update_time
                        / (self.timestamps_seconds[1] - self.timestamps_seconds[0])
                    )
                    - 1
                )

            # Simulate the trajectory
            plt.ion()
            t_prev = -1e-6
            for i, t in enumerate(self.timestamps_seconds):
                if i % (skip_num + 1) != 0:
                    continue

                ax.plot(self.x[i], self.y[i], self.z[i], "bo")
                ax.quiver(
                    self.x[i],
                    self.y[i],
                    self.z[i],
                    self.x[i] + direction_vectors[i, 0],
                    self.y[i] + direction_vectors[i, 1],
                    self.z[i] + direction_vectors[i, 2],
                    color="red",
                    length=0.1,
                )

                plt.show()
                plt.pause(t - t_prev)
                t_prev = t
        # Plot the entire final trajectory
        else:
            ax.plot(self.x, self.y, self.z, "bo")
            ax.quiver(
                self.x,
                self.y,
                self.z,
                self.x + direction_vectors[:, 0],
                self.y + direction_vectors[:, 1],
                self.z + direction_vectors[:, 2],
                color="red",
                length=0.1,
            )

            plt.show()


if __name__ == "__main__":
    # Create a trajectory object
    trajectory = Trajectory3D()

    # Load the trajectory data
    trajectory.load_trajectory(
        csv_file="Data-collection/log_2023_11_10_16_14_15_4840_Sample.csv",
        delimiter=";",
        drop_columns=["Email", "Framecount"],
        pose_columns=[
            "RightControllerPosWorldX",
            "RightControllerPosWorldY",
            "RightControllerPosWorldZ",
            "RightControllerRotEulerX",
            "RightControllerRotEulerY",
            "RightControllerRotEulerZ",
        ],
        timestamp_column="Timestamp",
    )

    # Remove the initial offset from the trajectory
    trajectory.remove_initial_offset()
    trajectory.convert_rotation_to_rad()

    # Print the trajectory time
    trajectory_time = trajectory._get_trajectory_time_seconds()
    print("Trajectory time: " + str(trajectory_time) + " seconds")

    # Plot the trajectory
    trajectory.plot(simulate=True, update_time=0.1)
