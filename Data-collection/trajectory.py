import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import transforms3d as t3d
from enum import Enum


class OrientationType(Enum):
    EULER = 0
    ROTATION_MATRIX = 1
    QUATERNION = 2


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
    def __init__(self, orientation_type: OrientationType) -> None:
        self.position = np.empty([3], dtype=np.float32)
        if orientation_type == OrientationType.EULER:
            self.orientation = np.empty([3], dtype=np.float32)
        elif orientation_type == OrientationType.ROTATION_MATRIX:
            self.orientation = np.empty([9], dtype=np.float32)
        elif orientation_type == OrientationType.QUATERNION:
            self.orientation = np.empty([4], dtype=np.float32)
        self.orientation_type = orientation_type

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
        self.position = self.data_loader.df[pose_columns[0:3]].to_numpy()
        self.orientation = self.data_loader.df[pose_columns[3:]].to_numpy()

        self.timestamps_seconds = self.data_loader.df[
            timestamp_column + "_seconds"
        ].to_numpy()

    def remove_initial_transformation(self):
        # Rotation matrix and translation offset of the controller w.r.t. the world frame
        R = self._euler_to_rotation_matrix()[0, :, :]
        t = self.position[0, :].T.reshape(3, 1)

        # Change the coordinate from the world frame to the controller frame
        self.position = (R.T @ (self.position.T - t)).T
        self.orientation = (R.T @ self.orientation.T).T

    def convert_degree_to_rad(self):
        if self.orientation_type != OrientationType.EULER:
            raise ValueError(
                "The orientation type must be Euler angles to convert to radians"
            )
        # Convert the rotation values to radians
        self.orientation = np.deg2rad(self.orientation)

    def convert_orientation(self, new_orientation_type: OrientationType):
        if self.orientation_type == new_orientation_type:
            return

        match self.orientation_type:
            case OrientationType.EULER:
                if new_orientation_type == OrientationType.ROTATION_MATRIX:
                    self.orientation = (
                        self._euler_to_rotation_matrix()
                        .T.flatten()
                        .reshape(len(self.orientation), 9)
                    )
                elif new_orientation_type == OrientationType.QUATERNION:
                    self.orientation = self._euler_to_quaternions().reshape(
                        len(self.orientation), 4
                    )
            # TODO: Add the other cases
            # case OrientationType.ROTATION_MATRIX:
            #     if new_orientation_type == OrientationType.EULER:
            #         self.orientation = self._rotation_matrix_to_euler()
            #     elif new_orientation_type == OrientationType.QUATERNION:
            #         self.orientation = self._rotation_matrix_to_quaternions()
            # case OrientationType.QUATERNION:
            #     if new_orientation_type == OrientationType.EULER:
            #         self.orientation = self._quaternions_to_euler()
            #     elif new_orientation_type == OrientationType.ROTATION_MATRIX:
            #         self.orientation = self._quaternions_to_rotation_matrix()

        self.orientation_type = new_orientation_type

    def _get_trajectory(self):
        return (
            self.orientation_type,
            np.concatenate((self.position, self.orientation), axis=1),
        )

    def _get_trajectory_time_seconds(self):
        return self.timestamps_seconds[-1]

    def _euler_to_rotation_matrix(self):
        # Convert the euler angles to rotation matrix
        rotation_matrices = np.zeros((len(self.orientation), 3, 3))
        for i in range(len(self.orientation)):
            rotation_matrices[i] = t3d.euler.euler2mat(
                self.orientation[i, 0],
                self.orientation[i, 1],
                self.orientation[i, 2],
                "sxyz",
            )

        return rotation_matrices

    def _euler_to_quaternions(self):
        # Convert the euler angles to quaternions
        quaternions = np.zeros((len(self.orientation), 4))
        for i in range(len(self.orientation)):
            quaternions[i] = t3d.euler.euler2quat(
                self.orientation[i, 0],
                self.orientation[i, 1],
                self.orientation[i, 2],
                "sxyz",
            )

        return quaternions

    def output_as_txt(self, output_file):
        # Create a header row
        match self.orientation_type:
            case OrientationType.EULER:
                header_row = ["timestamp", "tx", "ty", "tz", "rx", "ry", "rz"]
            case OrientationType.ROTATION_MATRIX:
                header_row = [
                    "timestamp",
                    "tx",
                    "ty",
                    "tz",
                    "r11",
                    "r21",
                    "r31",
                    "r12",
                    "r22",
                    "r32",
                    "r13",
                    "r23",
                    "r33",
                ]
            case OrientationType.QUATERNION:
                header_row = ["timestamp", "tx", "ty", "tz", "qx", "qy", "qz", "qw"]

        # Save the trajectory data with reference in the header row
        trajectory_txt = np.concatenate(
            (self.timestamps_seconds.reshape(-1, 1), self.position, self.orientation),
            axis=1,
        )

        # Add the header row to the trajectory data
        trajectory_txt = np.vstack((header_row, trajectory_txt))

        # Save the trajectory data as a txt file
        np.savetxt(output_file, trajectory_txt, delimiter=" ", fmt="%s")

    def plot(self, simulate=False, update_time=None):
        # Create a 3d figure
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        # Set labels for the axes
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        # Set axis limits
        ax.set_xlim3d(min(self.position[:, 0]), max(self.position[:, 0]))
        ax.set_ylim3d(min(self.position[:, 1]), max(self.position[:, 1]))
        ax.set_zlim3d(min(self.position[:, 2]), max(self.position[:, 2]))
        smallest_range = min(
            max(self.position[:, 0]) - min(self.position[:, 0]),
            max(self.position[:, 1]) - min(self.position[:, 1]),
            max(self.position[:, 2]) - min(self.position[:, 2]),
        )

        # Convert the euler angles to direction vectors (x-direction in rotation matrix)
        direction_vectors = self._euler_to_rotation_matrix()[:, :, 0]

        # Calculate the number of samples to skip between updates
        if update_time is None:
            skip_num = 0
        else:
            if update_time < (self.timestamps_seconds[1] - self.timestamps_seconds[0]):
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

        # Plot the trajectory simulated in real time
        if simulate:
            # Simulate the trajectory
            plt.ion()
            t_prev = -1e-6
            for i, t in enumerate(self.timestamps_seconds):
                if i % (skip_num + 1) != 0:
                    continue

                ax.plot(*(self.position[i, :]), "bo")
                ax.quiver(
                    *(self.position[i, :]),
                    *(self.position[i, :] + direction_vectors[i, :]),
                    color="red",
                    length=0.1 * smallest_range,
                )

                plt.show()
                plt.pause(t - t_prev)
                t_prev = t
        # Plot the entire final trajectory
        else:
            ax.plot(
                *(self.position[:: (skip_num + 1), :].T),
                "bo",
            )
            ax.quiver(
                *(self.position[:: (skip_num + 1), :].T),
                *(
                    (
                        self.position[:: (skip_num + 1), :]
                        + direction_vectors[:: (skip_num + 1), :]
                    ).T
                ),
                color="red",
                length=0.1 * smallest_range,
            )

            plt.show()


if __name__ == "__main__":
    # Create a trajectory object
    trajectory = Trajectory3D(orientation_type=OrientationType.EULER)

    # Load the trajectory data
    trajectory.load_trajectory(
        csv_file="Data-collection/1,1_0_0_1.csv",
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
    trajectory.convert_degree_to_rad()
    trajectory.remove_initial_transformation()

    # Print the trajectory time
    trajectory_time = trajectory._get_trajectory_time_seconds()
    print("Trajectory time: " + str(trajectory_time) + " seconds")

    # Plot the trajectory
    trajectory.plot(simulate=True, update_time=1)

    # Output converted trajectory as txt file
    # trajectory.convert_orientation(new_orientation_type=OrientationType.QUATERNION)
    # trajectory.output_as_txt("Data-collection/trajectory.txt")
