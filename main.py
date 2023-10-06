# Standard imports


# Own imports
from Utilities.Calibration import calib
from Utilities import load, video_handler
from PoseEstimation.pose_estimation import PoseEstimator
from Mapping.mapping import Mapper

if __name__ == "__main__":
    # Load video(s) and corresponding imu data
    cap, _ = load.load_monocular("Datasets/Kridtgraven/Normal", imu=False)

    # Calibrate stereo cameras
    #calib_params = calib.calibrate_stereo("Calibration/CheckerImages", board_size=(9,6))
    
    # Initialise map object
    mapper = Mapper()
    pose_estimator = PoseEstimator()
    
    while cap.isOpened():
        # TODO: Should maybe use threading to separate uw_vo and mapping
        pose_estimator.update()
        mapper.update()
        