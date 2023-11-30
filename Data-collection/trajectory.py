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


class OrientationConversion:
    # All axes are in the order of x, y, z
    def convert(
        orientations,
        start_orientation_type,
        end_orientation_type,
        end_orientation_dim=1,
    ):
        orientations = orientations.copy()

        # Add an index dimension if the input is single dimensional
        if orientations.ndim == 1:
            orientations = orientations.reshape(1, -1)

        # Return the original orientations if the start and end orientation types are the same
        if start_orientation_type == end_orientation_type:
            # Change the dimension of the rotation matrix if needed
            if end_orientation_type == OrientationType.ROTATION_MATRIX:
                orientations = OrientationConversion.change_rot_matrix_dim(
                    orientations, end_orientation_dim
                )

            return orientations

        match start_orientation_type:
            case OrientationType.EULER:
                # Convert the euler angles to rotation matrix
                if end_orientation_type == OrientationType.ROTATION_MATRIX:
                    rotation_matrices = np.zeros((len(orientations), 3, 3))
                    for i in range(len(orientations)):
                        rotation_matrices[i] = t3d.euler.euler2mat(*orientations[i, :])
                    return OrientationConversion.change_rot_matrix_dim(
                        rotation_matrices, end_orientation_dim
                    )
                # Convert the euler angles to quaternions
                elif end_orientation_type == OrientationType.QUATERNION:
                    quaternions = np.zeros((len(orientations), 4))
                    for i in range(len(orientations)):
                        quaternions[i] = t3d.euler.euler2quat(*orientations[i, :])
                    return np.roll(quaternions, shift=-1, axis=1)
            case OrientationType.ROTATION_MATRIX:
                # Make sure the rotation matrices are 2d
                orientations = OrientationConversion.change_rot_matrix_dim(
                    orientations, 2
                )

                # Convert the rotation matrices to euler angles
                if end_orientation_type == OrientationType.EULER:
                    euler_angles = np.zeros((len(orientations), 3))
                    for i in range(len(orientations)):
                        euler_angles[i] = t3d.euler.mat2euler(orientations[i, :, :])
                    return euler_angles
                # Convert the rotation matrices to quaternions
                elif end_orientation_type == OrientationType.QUATERNION:
                    quaternions = np.zeros((len(orientations), 4))
                    for i in range(len(orientations)):
                        quaternions[i] = t3d.quaternions.mat2quat(orientations[i, :, :])
                    return np.roll(quaternions, shift=-1, axis=1)
            case OrientationType.QUATERNION:
                # Since the function expects the order w, x, y, z
                orientations = np.roll(orientations, shift=1, axis=1)

                # Convert the quaternions to euler angles
                if end_orientation_type == OrientationType.EULER:
                    euler_angles = np.zeros((len(orientations), 3))
                    for i in range(len(orientations)):
                        euler_angles[i] = t3d.euler.quat2euler(orientations[i, :])
                    return euler_angles
                # Convert the quaternions to rotation matrices
                elif end_orientation_type == OrientationType.ROTATION_MATRIX:
                    rotation_matrices = np.zeros((len(orientations), 3, 3))
                    for i in range(len(orientations)):
                        rotation_matrices[i] = t3d.quaternions.quat2mat(
                            orientations[i, :]
                        )
                    return OrientationConversion.change_rot_matrix_dim(
                        rotation_matrices, end_orientation_dim
                    )

    def change_rot_matrix_dim(rotation_matrix, end_dim):
        match end_dim:
            # Conversion from 2d to 1d
            case 1:
                # Return the original rotation matrix if it is already 1d
                if rotation_matrix.shape[-1] == 9:
                    return rotation_matrix.reshape(-1, 9)
                else:
                    return np.transpose(rotation_matrix, axes=(0, 2, 1)).reshape(-1, 9)
            # Conversion from 1d to 2d
            case 2:
                # Return the original rotation matrix if it is already 2d
                if rotation_matrix.shape[-1] == 3:
                    return rotation_matrix.reshape(-1, 3, 3)
                else:
                    return np.transpose(
                        rotation_matrix.reshape(-1, 3, 3), axes=(0, 2, 1)
                    )


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

    def synchronise_initial_time(self, plot=False):
        # Save the x position and time of the trajectory in case of plotting
        x = self.position[:, 0].copy()
        t = self.timestamps_seconds.copy()

        # Compute the absolute gradient of the x position
        abs_grad_x = np.abs(np.gradient(x))

        # Get the maximum index and value of the absolute gradient
        max_index = np.argmax(abs_grad_x)
        max_val = abs_grad_x[max_index]

        # Check if the trajectory is stationary at the beginning
        prev_window = abs_grad_x[max(0, max_index - 30) : max_index + 1]
        threshold = 0.1 * max_val
        if np.median(prev_window) > threshold:
            raise ValueError("The trajectory is not stationary at the beginning")

        # Synchronise the initial time of the trajectory to 0
        self.timestamps_seconds -= self.timestamps_seconds[max_index]

        # Remove the times before the initial time
        self.timestamps_seconds = self.timestamps_seconds[max_index:]
        self.position = self.position[max_index:, :]
        self.orientation = self.orientation[max_index:, :]

        if plot:
            # Plot the absolute gradient
            plt.plot(t, x, "--r", label="Old")
            plt.plot(self.timestamps_seconds, self.position[:, 0], "-k", label="New")
            plt.xlabel("Time [s]")
            plt.ylabel("X-position [m]")
            plt.legend()
            plt.show()

    def make_right_handed(self):
        # Fix the position
        self.position[:, 1], self.position[:, 2] = (
            self.position[:, 2].copy(),
            self.position[:, 1].copy(),
        )

        # Convert the orientation to Euler angles temporarily
        # NOTE: This doesn't work if the orientation is not already Euler angles, since the conversion
        # cannot understand left-handed coordinate systems
        old_orientations = OrientationConversion.convert(
            self.orientation,
            self.orientation_type,
            OrientationType.EULER,
        )

        # Fix the orientation
        def rotX(angles):
            # Create the rotation matrices along the first axis
            rotations = np.array(
                [
                    [
                        [1, 0, 0],
                        [0, cos, -sin],
                        [0, sin, cos],
                    ]
                    for cos, sin in zip(np.cos(angles), np.sin(angles))
                ]
            )
            return rotations

        def rotY(angles):
            # Create the rotation matrices along the first axis
            rotations = np.array(
                [
                    [
                        [cos, 0, sin],
                        [0, 1, 0],
                        [-sin, 0, cos],
                    ]
                    for cos, sin in zip(np.cos(angles), np.sin(angles))
                ]
            )
            return rotations

        def rotZ(angles):
            # Create the rotation matrices along the first axis
            rotations = np.array(
                [
                    [
                        [cos, -sin, 0],
                        [sin, cos, 0],
                        [0, 0, 1],
                    ]
                    for cos, sin in zip(np.cos(angles), np.sin(angles))
                ]
            )
            return rotations

        RX = rotX(-old_orientations[:, 0])
        RY = rotY(-old_orientations[:, 2])
        RZ = rotZ(-old_orientations[:, 1])

        R = RZ @ RX @ RY

        self.orientation = OrientationConversion.convert(
            R,
            OrientationType.ROTATION_MATRIX,
            self.orientation_type,
        )

    def remove_initial_transformation(self):
        # Rotation matrix and translation offset of the controller w.r.t. the world frame
        t = self.position[0, :].T.reshape(3, 1)
        old_orientations = OrientationConversion.convert(
            self.orientation,
            self.orientation_type,
            OrientationType.ROTATION_MATRIX,
            end_orientation_dim=2,
        )
        R = old_orientations[0, :, :]

        # Change the coordinate from the world frame to the controller frame
        self.position = (R.T @ (self.position.T - t)).T
        new_orientations = R.T @ old_orientations

        self.orientation = OrientationConversion.convert(
            new_orientations,
            OrientationType.ROTATION_MATRIX,
            self.orientation_type,
        )

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

        # Convert the orientation to the new orientation type as 1d
        self.orientation = OrientationConversion.convert(
            self.orientation,
            self.orientation_type,
            new_orientation_type,
            end_orientation_dim=1,
        )

        # Update the orientation type
        self.orientation_type = new_orientation_type

    def apply_transformation(self, transform: np.ndarray((4, 4)), right_hand=True):
        # Store the old orientation and position
        old_orientations = OrientationConversion.convert(
            self.orientation,
            self.orientation_type,
            OrientationType.ROTATION_MATRIX,
            end_orientation_dim=2,
        )
        old_positions = self.position.copy().reshape(-1, 3, 1)

        # Define the old transformations
        old_transformations = np.concatenate((old_orientations, old_positions), axis=2)
        old_transformations = np.concatenate(
            (
                old_transformations,
                np.tile(np.array([0, 0, 0, 1]), (len(old_transformations), 1, 1)),
            ),
            axis=1,
        )

        # Define the new transformation
        if right_hand:
            new_transformations = old_transformations @ transform
        else:
            new_transformations = transform @ old_transformations

        # Update the orientation and position
        self.position = new_transformations[:, 0:3, 3]
        self.orientation = OrientationConversion.convert(
            new_transformations[:, 0:3, 0:3],
            OrientationType.ROTATION_MATRIX,
            self.orientation_type,
        )

    def crop_time(self, start_time, end_time):
        # Get start and end indices
        start_index = np.argmax(self.timestamps_seconds >= start_time)
        end_index = np.argmax(self.timestamps_seconds >= end_time)

        # Set the start time of the trajectory to 0
        self.timestamps_seconds -= self.timestamps_seconds[start_index]

        # Crop the trajectory
        self.timestamps_seconds = self.timestamps_seconds[start_index:end_index]
        self.position = self.position[start_index:end_index, :]
        self.orientation = self.orientation[start_index:end_index, :]

    def _get_trajectory(self):
        return (
            self.orientation_type,
            np.concatenate((self.position, self.orientation), axis=1),
        )

    def _get_trajectory_time_seconds(self):
        return self.timestamps_seconds[-1]

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

    def plot(self, simulate=False, update_time=None, orientation_axes=[1, 0, 0]):
        # Create a 3d figure
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        # Set labels for the axes
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        # Set axis limits
        ax_bound = 1
        ax.set_xlim3d(-ax_bound, ax_bound)
        ax.set_ylim3d(-ax_bound, ax_bound)
        ax.set_zlim3d(-ax_bound, ax_bound)
        smallest_range = ax_bound * 2

        # Convert the euler angles to direction vectors (corresponding to rotation matrices)
        direction_vectors = OrientationConversion.convert(
            self.orientation,
            self.orientation_type,
            OrientationType.ROTATION_MATRIX,
            end_orientation_dim=2,
        )

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
        axis_colors = ["red", "green", "blue"]
        if simulate:
            # Simulate the trajectory
            plt.ion()
            t_prev = -1e-6
            for i, t in enumerate(self.timestamps_seconds):
                if i % (skip_num + 1) != 0:
                    continue

                # Plot position
                ax.plot(*(self.position[i, :]), "ko")

                # Plot orientation axes
                for j, display_axis in enumerate(orientation_axes):
                    if display_axis != 1:
                        continue

                    ax.quiver(
                        *(self.position[i, :]),
                        *(direction_vectors[i, :, j]),
                        color=axis_colors[j],
                        length=0.1 * smallest_range,
                    )

                plt.show()
                plt.pause(t - t_prev)
                t_prev = t
        # Plot the entire final trajectory
        else:
            # Plot position
            ax.plot(
                *(self.position[:: (skip_num + 1), :].T),
                "ko",
            )
            # Plot orientation axes
            for j, display_axis in enumerate(orientation_axes):
                if display_axis != 1:
                    continue

                print(self.position[0, :] + direction_vectors[0, :, j])
                ax.quiver(
                    *(self.position[:: (skip_num + 1), :].T),
                    *((direction_vectors[:: (skip_num + 1), :, j]).T),
                    color=axis_colors[j],
                    length=0.1 * smallest_range,
                )

            plt.show()


if __name__ == "__main__":
    # Create a trajectory object
    trajectory = Trajectory3D(orientation_type=OrientationType.EULER)

    # Load the trajectory data
    trajectory.load_trajectory(
        csv_file="Data-collection/1,1_0_0_10.csv",
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
    trajectory.make_right_handed()
    trajectory.synchronise_initial_time(plot=True)
    trajectory.remove_initial_transformation()

    # Print the trajectory time
    trajectory_time = trajectory._get_trajectory_time_seconds()
    print("Trajectory time: " + str(trajectory_time) + " seconds")

    # Plot the trajectory
    trajectory.plot(simulate=False, update_time=1000, orientation_axes=[1, 1, 1])

    # Apply the transformation to the imu to the trajectory
    T_ctrl_lcam = np.array(
        [
            [-0.68, 0.01, -0.73, -0.0278],
            [-0.73, 0.00, -0.68, -0.9697],
            [0.01, 1.00, 0.00, -0.1036],
            [0, 0, 0, 1],
        ],
    )
    trajectory.apply_transformation(T_ctrl_lcam, right_hand=True)

    # Crop the trajectory in time
    trajectory.crop_time(0, 10)

    # Output converted trajectory as txt file
    # trajectory.convert_orientation(new_orientation_type=OrientationType.QUATERNION)
    # trajectory.output_as_txt("Data-collection/trajectory.txt")
