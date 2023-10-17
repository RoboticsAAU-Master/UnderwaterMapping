import time, sys, os
from ros import rosbag
import roslib
import rospy
roslib.load_manifest('sensor_msgs')
from sensor_msgs.msg import Image, Imu
from cv_bridge import CvBridge
import pandas as pd

from PIL import ImageFile

import cv2

CAM_SAMPLE_RATE = 60.0 # Hz
IMU_SAMPLE_RATE = 200.0 # Hz

def GetFilesFromDir(dir):
    '''Generates a list of files from the directory'''
    print( "Searching directory %s" % dir )
    all = []
    left_files = []
    right_files = []
    if os.path.exists(dir):
        for path, names, files in os.walk(dir):
            for f in sorted(files):
                if os.path.splitext(f)[1] in ['.bmp', '.png', '.jpg', '.ppm']:
                    if 'left' in f or 'left' in path:
                        left_files.append( os.path.join( path, f ) )
                    elif 'right' in f or 'right' in path:
                        right_files.append( os.path.join( path, f ) )
                    all.append( os.path.join( path, f ) )
    return all, left_files, right_files


def CreateStereoBag(left_imgs, right_imgs, bagname):
    '''Creates a bag file containing stereo image pairs'''
    bag =rosbag.Bag(bagname, 'w')

    timer = 0

    try:
        for i in range(len(left_imgs)):
            print("Adding %s" % left_imgs[i])
            img_left = cv2.imread(left_imgs[i])
            img_right = cv2.imread(right_imgs[i])

            bridge = CvBridge()

            # Stamp = rospy.Time.from_sec(time.time())

            # Since image acquisition frequency depends on write speed, the timestamp is synthesized
            Stamp = rospy.Time.from_sec(timer)
            timer += 1/CAM_SAMPLE_RATE

            img_msg_left = Image()
            img_msg_left = bridge.cv2_to_imgmsg(img_left, "bgr8")
            img_msg_left.header.seq = i
            img_msg_left.header.stamp = Stamp
            img_msg_left.header.frame_id = "camera/left"

            img_msg_right = Image()
            img_msg_right = bridge.cv2_to_imgmsg(img_right, "bgr8")
            img_msg_right.header.seq = i
            img_msg_right.header.stamp = Stamp
            img_msg_right.header.frame_id = "camera/right"

            bag.write('camera/left/image_raw', img_msg_left, Stamp)
            bag.write('camera/right/image_raw', img_msg_right, Stamp)

            # Adding IMU data
    finally:
        bag.close()


def CreateMonoBag(imgs, bagname):
    '''Creates a bag file with camera images'''
    bag =rosbag.Bag(bagname, 'w')

    try:
        for i in range(len(imgs)):
            print("Adding %s" % imgs[i])
            img = cv2.imread(imgs[i])
            bridge = CvBridge()

            Stamp = rospy.rostime.Time.from_sec(time.time())
            img_msg = Image()
            img_msg = bridge.cv2_to_imgmsg(img, "bgr8")
            img_msg.header.seq = i
            img_msg.header.stamp = Stamp
            img_msg.header.frame_id = "camera"

            bag.write('camera/image_raw', img_msg, Stamp)
    finally:
        bag.close()


def CreateBag(args):
    '''Creates the actual bag file by successively adding images'''
    all_imgs, left_imgs, right_imgs = GetFilesFromDir(args[0])
    if len(all_imgs) <= 0:
        print("No images found in %s" % args[0])
        exit()

    if len(left_imgs) > 0 and len(right_imgs) > 0:
        # create bagfile with stereo camera image pairs
        CreateStereoBag(left_imgs, right_imgs, args[1])
    else:
        # create bagfile with mono camera image stream
        CreateMonoBag(all_imgs, args[1])

    # Check if imu path has been specified
    if len(args) > 2:
        imu_data = GetImuData(args[2])
        AddImuToBag(args[1], imu_data)

def GetImuData(dir : str):
    '''Read accelerometer and gyroscope data from path'''
    print( "Searching directory %s" % dir )

    if os.path.exists(dir):
        for path, names, files in os.walk(dir):
            for f in files:
                if os.path.splitext(f)[1] in ['.csv', '.ods']:
                    if 'accl' in f or 'accl' in path:
                        accl_data = pd.read_csv(os.path.join( path, f ), header=None)
                    elif 'gyro' in f or 'gyro' in path:
                        gyro_data = pd.read_csv(os.path.join( path, f ), header=None)

    # Delete last column, corresponding to NaN
    # accl_data.drop(accl_data.columns[-1], axis=1, inplace=True)
    # gyro_data.drop(gyro_data.columns[-1], axis=1, inplace=True)
    
    imu_data = pd.concat([accl_data, gyro_data], axis=1)

    return imu_data

def AddImuToBag(bagname : str, imu_data):

    with rosbag.Bag(bagname, 'a') as bag:

        timer = 0

        for index, row in imu_data.iterrows():
            row = row.tolist()
            timestamp = rospy.Time.from_sec(timer)
            timer += 1/IMU_SAMPLE_RATE
            imu_msg = Imu()
            imu_msg.header.stamp = timestamp
            imu_msg.linear_acceleration.x = row[0]
            imu_msg.linear_acceleration.y = row[1]
            imu_msg.linear_acceleration.z = row[2]
            imu_msg.angular_velocity.x = row[3]
            imu_msg.angular_velocity.y = row[4]
            imu_msg.angular_velocity.z = row[5]

            # Populate the data elements for IMU
            # e.g. imu_msg.angular_velocity.x = df['a_v_x'][row]

            bag.write("/imu", imu_msg, timestamp)

            # gps_msg = NavSatFix()
            # gps_msg.header.stamp = timestamp

            # # Populate the data elements for GPS

            # bag.write("/gps", gpu_msg, timestamp)

if __name__ == "__main__":
    # if len( sys.argv ) == 3:
    #     CreateBag( sys.argv[1:])
    # else:
    #     print( "Usage: SCRIPT_NAME.py imagedir bagfilename imudir")

    CreateBag(args=["haud_test_img", "gopro_test.bag", "haud_test_imu"])