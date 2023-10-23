from rosbag import Bag
from bagpy import bagreader
import os

path = "Utilities/Bag-conversion/Output/C2.bag"

##### USING rosbag #####
# bag = Bag(path, "r")
# topics = bag.get_type_and_topic_info()[1].keys() # All the topics info
# print("\nAvailable topics in bag file are:{}".format(topics))

##### USING bagpy #####
bag = bagreader(path)
os.rmdir(path[:-4]) # Remove folder created by bagpy
print(bag.topic_table)
