from rosbag import Bag
from bagpy import bagreader
import os

path = "gopro_test.bag"

##### USING rosbag #####
# bag = Bag(path, "r")
# get the list of topics
# topics = bag.get_type_and_topic_info()[1].keys() # All the topics info
# print("\nAvailable topics in bag file are:{}".format(topics))

##### USING bagpy #####
bag = bagreader(path)

# Remove ".bag" part
os.rmdir(path[:-4])

# get the list of topics
print(bag.topic_table)
