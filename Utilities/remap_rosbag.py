from rosbag import Bag
from tqdm import tqdm

path = "../svo_ws/src/bagfiles/MCUVID/sequence_09.bag"

bag = Bag(path, "r")

# Dictionary that tells what topics from source maps to what topics in destination
remap_dict = {
    "/zed2/zed_node/left_raw/image_raw_color" : "/sync/cam0/image_raw", # left
    "/zed2/zed_node/right_raw/image_raw_color" : "/sync/cam1/image_raw", # left
    "/zed2/zed_node/imu/data" : "/sync/imu/imu"
}

# Print topics of source for overview
topics = bag.get_type_and_topic_info()[1].keys() # All the topics info
print("\nAvailable topics in bag file are:{}".format(topics))


with Bag('sequence_09_m.bag', 'w') as mapped_bag:
    # Loop through each topic in source bag
    for topic, msg, t in tqdm(bag):
        # Loop through each key-value pair in dict
        for key, value in remap_dict.items():
            if topic == key:
                mapped_bag.write(value, msg, t)
            else:
                pass
                # mapped_bag.write(topic, msg, t)


