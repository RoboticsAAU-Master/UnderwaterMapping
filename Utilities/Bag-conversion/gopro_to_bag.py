import time, sys, os
from ros import rosbag
import roslib
import rospy

roslib.load_manifest("sensor_msgs")
from sensor_msgs.msg import Image, Imu
from cv_bridge import CvBridge
import pandas as pd
from tqdm import tqdm

from PIL import ImageFile

import cv2

CAM_SAMPLE_RATE = 60.0  # Hz
IMU_SAMPLE_RATE = 197.720721  # Hz
TIME_START = 0.0
TIME_END = 1000.0

STANDARD_TO_TOPIC = {
    "euroc_mono": ["/cam0/image_raw", "/imu0"],
    "euroc_stereo": ["/cam0/image_raw", "/cam1/image_raw", "/imu0"],
    "fla_mono": ["/sync/cam0/image_raw", "/sync/imu/imu"],
    "fla_stereo": ["/sync/cam0/image_raw", "/sync/cam1/image_raw", "/sync/imu/imu"],
}


def GetFilesFromDir(dir):
    """Generates a list of files from the directory"""
    print("Searching directory %s" % dir)
    all = []
    left_files = []
    right_files = []
    if os.path.exists(dir):
        for path, names, files in os.walk(dir):
            for f in sorted(files):
                if os.path.splitext(f)[1] in [".bmp", ".png", ".jpg", ".ppm"]:
                    if "left" in f or "left" in path:
                        left_files.append(os.path.join(path, f))
                    elif "right" in f or "right" in path:
                        right_files.append(os.path.join(path, f))
                    all.append(os.path.join(path, f))
    return all, left_files, right_files


def CreateStereoBag(left_imgs, right_imgs, bagname, standard):
    """Creates a bag file containing stereo image pairs"""
    bag = rosbag.Bag(bagname, "w")

    timer = 0

    try:
        for i in tqdm(range(len(left_imgs))):
            # print("Adding %s" % left_imgs[i])
            img_left = cv2.imread(left_imgs[i], cv2.IMREAD_GRAYSCALE)
            img_right = cv2.imread(right_imgs[i], cv2.IMREAD_GRAYSCALE)

            bridge = CvBridge()

            # Stamp = rospy.Time.from_sec(time.time())

            # Since image acquisition frequency depends on write speed, the timestamp is synthesized
            Stamp = rospy.Time.from_sec(timer)
            timer += 1 / CAM_SAMPLE_RATE

            img_msg_left = Image()
            img_msg_left = bridge.cv2_to_imgmsg(img_left, "mono8")
            img_msg_left.header.seq = i
            img_msg_left.header.stamp = Stamp
            img_msg_left.header.frame_id = "camera/left"

            img_msg_right = Image()
            img_msg_right = bridge.cv2_to_imgmsg(img_right, "mono8")
            img_msg_right.header.seq = i
            img_msg_right.header.stamp = Stamp
            img_msg_right.header.frame_id = "camera/right"

            bag.write(STANDARD_TO_TOPIC[standard][0], img_msg_left, Stamp)
            bag.write(STANDARD_TO_TOPIC[standard][1], img_msg_right, Stamp)

    finally:
        bag.close()


def CreateMonoBag(imgs, bagname, standard):
    """Creates a bag file with camera images"""
    bag = rosbag.Bag(bagname, "w")

    timer = 0

    try:
        for i in tqdm(range(len(imgs))):
            # print("Adding %s" % imgs[i])
            img = cv2.imread(imgs[i], cv2.IMREAD_GRAYSCALE)
            bridge = CvBridge()

            # Stamp = rospy.rostime.Time.from_sec(time.time())

            # Since image acquisition frequency depends on write speed, the timestamp is synthesized
            Stamp = rospy.Time.from_sec(timer)
            timer += 1 / CAM_SAMPLE_RATE

            img_msg = Image()
            img_msg = bridge.cv2_to_imgmsg(img, "mono8")
            img_msg.header.seq = i
            img_msg.header.stamp = Stamp
            img_msg.header.frame_id = "camera"

            bag.write(STANDARD_TO_TOPIC[standard][0], img_msg, Stamp)
    finally:
        bag.close()


def CreateBag(args):
    """Creates the actual bag file by successively adding images"""
    all_imgs, left_imgs, right_imgs = GetFilesFromDir(args[0])
    if len(all_imgs) <= 0:
        print("No images found in %s" % args[0])
        exit()

    if len(left_imgs) > 0 and len(right_imgs) > 0:
        if not "stereo" in args[3]:
            raise Exception("Stereo data provided does not match standard")
        # create bagfile with stereo camera image pairs
        CreateStereoBag(left_imgs, right_imgs, args[1], args[3])
    else:
        if not "mono" in args[3]:
            raise Exception("Mono data provided does not match standard")
        # create bagfile with mono camera image stream
        CreateMonoBag(all_imgs, args[1], args[3])

    # Check if imu path has been specified
    if len(args) > 2:
        imu_data = GetImuData(args[2])
        AddImuToBag(args[1], imu_data, args[3])


def GetImuData(dir: str):
    """Read accelerometer and gyroscope data from path"""
    print("Searching directory %s" % dir)

    if os.path.exists(dir):
        for path, names, files in os.walk(dir):
            for f in files:
                if os.path.splitext(f)[1] in [".csv", ".ods"]:
                    if "Accl" in f or "Accl" in path:
                        accl_data = pd.read_csv(os.path.join(path, f), header=None)
                    elif "Gyro" in f or "Gyro" in path:
                        gyro_data = pd.read_csv(os.path.join(path, f), header=None)

    # Delete last column, corresponding to NaN.
    accl_data.drop(accl_data.columns[-1], axis=1, inplace=True)
    gyro_data.drop(gyro_data.columns[-1], axis=1, inplace=True)

    imu_data = pd.concat([accl_data, gyro_data], axis=1)

    return imu_data


def AddImuToBag(bagname: str, imu_data, standard):
    with rosbag.Bag(bagname, "a") as bag:
        timer = 0

        for index, row in imu_data.iterrows():
            if (
                index < TIME_START * IMU_SAMPLE_RATE
                or index > TIME_END * IMU_SAMPLE_RATE
            ):
                continue

            row = row.tolist()
            timestamp = rospy.Time.from_sec(timer)
            timer += 1 / IMU_SAMPLE_RATE

            imu_msg = Imu()
            imu_msg.header.stamp = timestamp
            imu_msg.linear_acceleration.x = row[0]
            imu_msg.linear_acceleration.y = row[1]
            imu_msg.linear_acceleration.z = row[2]
            imu_msg.angular_velocity.x = row[3]
            imu_msg.angular_velocity.y = row[4]
            imu_msg.angular_velocity.z = row[5]

            bag.write(STANDARD_TO_TOPIC[standard][-1], imu_msg, timestamp)


if __name__ == "__main__":
    # if len( sys.argv ) == 3:
    #     CreateBag( sys.argv[1:])
    # else:
    #     print( "Usage: SCRIPT_NAME.py imagedir bagfilename imudir data_standard")

    # OBS: Convert input images to grayscale
    CreateBag(
        args=[
            "Utilities/GoPro-data-extraction/Output/C4/Images",
            "Utilities/Bag-conversion/Output/C4.bag",
            "Utilities/GoPro-data-extraction/Output/C4/Metadata",
            "euroc_stereo",
        ]
    )
