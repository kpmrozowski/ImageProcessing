python3 -m rosbags.convert ~/dev/bags_turtle/rosbag2_2024_04_18-13_06_28/ --dst ~/dev/bags_turtle/rosbag2_2024_04_18-13_06_28.bag
# for f in $(find ~/dev/bags_turtle/ -type f); do python3 -m rosbags.convert --src $f --dst $f.bag

# ros2 pkg create --build-type ament_cmake --node-name stereo2depth offline_tools
