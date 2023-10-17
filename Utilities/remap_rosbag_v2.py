from rosbag import Bag
from bagpy import bagreader
from tqdm import tqdm
import os

num_files = 1
filename = "archaeo_sequence_5.bag" # Name of the initial bag file
path = "../svo_ws/src/bagfiles/AQUALOC/" + filename

b = bagreader(path)

# Remove ".bag" part
os.rmdir(path[:-4])

# get the list of topics
print(b.topic_table)

# For euroc mono
remap_dict = {
    "/camera/image_raw" : "/cam0/image_raw", # left
    "/rtimulib_node/imu" : "/imu0"
}

# For FLA stereo
# remap_dict = {
#     "/alphasense_driver_ros/cam0" : "/sync/cam0/image_raw", # left
#     "/alphasense_driver_ros/cam1" : "/sync/cam1/image_raw", # right
#     "/alphasense_driver_ros/imu" : "/sync/imu/imu"
# }

# Print topics for overview
# topics = bag.get_type_and_topic_info()[1].keys() # All the topics info
# print("\nAvailable topics in bag file are:{}".format(topics))

topics_to_remap = set(remap_dict.keys())

with Bag(path.replace(filename,"m_" + filename), 'w') as mapped_bag:
    # Loop through each bag 
    for i in range(num_files):
        path = path.replace(f"{i-1}.bag",f"{i}.bag")
        print(path)
        # Open the bag
        bag = Bag(path, "r")
        # Loop through each message in the source bag
        for topic, msg, t in tqdm(bag.read_messages(topics=topics_to_remap)):
            new_topic = remap_dict[topic]
            mapped_bag.write(new_topic, msg, t)


bag.close()

