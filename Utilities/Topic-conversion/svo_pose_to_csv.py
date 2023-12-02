import rospy
import csv
import os
from geometry_msgs.msg import PoseStamped

def topic_to_csv(pose_topic, csv_name, output_folder):
    
    print("\nWhen recording is done, clean exit node by pressing ctrl + c")

    msgs_recv = 0 # Counter

    # Open csv-file (throw exception if it exists)
    with open(os.path.join(output_folder, csv_name), 'x') as csv_file:

        csv_obj = csv.writer(csv_file)
        csv_obj.writerow(["timestamp", "tx", "ty", "tz", "qx", "qy", "qz", "qw"])
        
        rospy.init_node('svo_to_csv')
        
        def callback_pose(data : PoseStamped):
            x = data.pose.position.x
            y = data.pose.position.y      
            z = data.pose.position.z
            qx = data.pose.orientation.x
            qy = data.pose.orientation.y
            qz = data.pose.orientation.z
            qw = data.pose.orientation.w
            timestamp = data.header.stamp.to_time()
            csv_obj.writerow([timestamp, x, y, z, qx, qy, qz, qw])

            nonlocal msgs_recv
            msgs_recv += 1
            print(f"Stored {msgs_recv} msgs", end='\r')

        # Specify subscribers
        rospy.Subscriber(pose_topic, PoseStamped, callback_pose, queue_size=100)

        # spin() simply keeps python from exiting until this node is stopped
        rospy.spin()


if __name__ == "__main__":
    pose_topic = "/svo/pose_cam/0" # Pose of camera 0. Message type is assumed to be geometry_msgs/PoseStamped
    output_name = "svo_cam0_trajectory.csv" # Name of csv-file to be created (with extension).
    output_folder = "Utilities/Topic-conversion" # Output folder to store csv-file

    # topic_to_csv writes any messages received on the pose_topic into the csv. 
    # OBS: If csv_name exists in folder, it will overwrite its contents 
    topic_to_csv(pose_topic, output_name, output_folder)
    
    
    
    
    


