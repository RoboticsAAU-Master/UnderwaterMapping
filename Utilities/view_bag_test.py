from rosbag import Bag

path = "../svo_ws/src/bagfiles/AQUALOC/archaeo_sequence_5.bag"

bag = Bag(path, "r")

# get the list of topics
topics = bag.get_type_and_topic_info()[1].keys() # All the topics info
print("\nAvailable topics in bag file are:{}".format(topics))

# get all the messages of type velocity
# velmsgs   = b.vel_data()
# veldf = pd.read_csv(velmsgs[0])
# plt.plot(veldf['Time'], veldf['linear.x'])

# # quickly plot velocities
# b.plot_vel(save_fig=True)

# # you can animate a timeseries data
# bagpy.animate_timeseries(veldf['Time'], veldf['linear.x'], title='Velocity Timeseries Plot')