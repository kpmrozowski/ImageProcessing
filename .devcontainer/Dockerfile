FROM ubuntu:jammy-20240416

LABEL maintainer="Kornel Mrozowski kornel.mrozowski@samsung.com"

RUN apt update && \
    apt upgrade -y && \
# tzdata
    ln -fs /usr/share/zoneinfo/Europe/Warsaw /etc/localtime && \
    apt-get install -y tzdata && \
    dpkg-reconfigure --frontend noninteractive tzdata && \
    apt install -y curl gnupg lsb-release sudo vim git ninja-build wget build-essential ffmpeg libeigen3-dev libglew-dev && \
# adding user jetbot
    adduser --disabled-password --gecos '' jetbot && \
    adduser jetbot sudo && \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
WORKDIR /home/jetbot
USER jetbot

RUN mkdir /home/jetbot/sdk && \
    sudo apt update && sudo apt install -y locales && \
    sudo locale-gen en_US en_US.UTF-8 && \
    sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 && \
    export LANG=en_US.UTF-8 && \
    echo "export LANG=en_US.UTF-8" >> /home/jetbot/.bashrc
WORKDIR /home/jetbot/sdk

# install ROS 2 Humble
RUN sudo apt update &&\
    sudo apt install -y software-properties-common &&\
    sudo add-apt-repository universe &&\
    sudo apt update &&\
    sudo apt install -y curl gnupg lsb-release &&\
    sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg &&\
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null &&\
    sudo apt update &&\
    sudo apt upgrade -y &&\
    sudo rm /bin/sh &&\
    sudo ln -s /usr/bin/bash /bin/sh &&\
    sudo apt-get update -y && \
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y keyboard-configuration &&\
    sudo apt install -y ros-humble-desktop &&\
    echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc

RUN sudo apt update &&\
    sudo apt install -y python3-catkin-pkg \
    python3-colcon-common-extensions \
    python3-rosdep \
    python3-argcomplete \
    python3-pip \
    python-setuptools &&\
    rm -rf /var/lib/apt/lists/*rm

RUN sudo add-apt-repository --update -y ppa:ubuntu-toolchain-r/test &&\
    wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | sudo apt-key add - &&\
    sudo add-apt-repository --update -y "deb http://apt.llvm.org/focal/ llvm-toolchain-focal-15 main" &&\
    sudo apt install -y \
    clang-15 \
    lldb-15 \
    lld-15 \
    clangd-15 \
    clang-tidy-15 \
    clang-format-15 \
    clang-tools-15 \
    llvm-15-dev \
    llvm-15-tools \
    libc++-15-dev \
    libc++abi-15-dev \
    libclang-common-15-dev \
    libclang-15-dev \
    libclang-cpp15-dev \
    libunwind-15-dev \
    gcc-11 \
    g++-11 \
    libfmt-dev \
    bash-completion \
    dirmngr \
    htop \
    gdb \
    git-lfs \
    x11-apps \
    libgmp3-dev \
    pybind11-dev \
    gitk \
    meld \
    terminator

RUN sudo apt update &&\
    sudo apt install -y \
    ros-humble-xacro \
    ros-humble-angles \
    ros-humble-ros2-control \
    ros-humble-realtime-tools \
    ros-humble-control-toolbox \
    ros-humble-ros2-controllers \
    ros-humble-joint-state-publisher \
    ros-humble-joint-state-publisher-gui \
    ros-humble-ament-cmake-clang-format \
    ros-humble-dynamic-edt-3d \
    ros-humble-rviz-visual-tools \
    ros-humble-control-msgs \
    ros-humble-sensor-msgs \
    ros-humble-vision-msgs \
    ros-humble-rosbridge-msgs
# install realsense - only if you will run on real hardware
# Instructions from https://github.com/IntelRealSense/librealsense/blob/master/doc/distribution_linux.md
RUN sudo apt-key adv --keyserver keys.gnupg.net --recv-key F6E65AC044F831AC80A06380C8B3A55A6F3EFCDE || sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-key F6E65AC044F831AC80A06380C8B3A55A6F3EFCDE &&\
    sudo add-apt-repository "deb https://librealsense.intel.com/Debian/apt-repo $(lsb_release -cs) main" -u &&\
    sudo apt update &&\
    sudo apt-get install -y \
        librealsense2-dkms \
        librealsense2-utils \
        librealsense2-dev \
        ros-humble-realsense2-camera-dbgsym \
        ros-humble-realsense2-camera-msgs \
        ros-humble-realsense2-camera-msgs-dbgsym \
        ros-humble-realsense2-description \
        ros-humble-ros2bag \
        ros-humble-cv-bridge

# upgrade cmake 3.16 to 3.28
RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | sudo tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null &&\
    sudo apt-add-repository "deb https://apt.kitware.com/ubuntu/ $(lsb_release -cs) main" &&\
    sudo apt update &&\
    sudo apt install cmake -y && \
    echo "================= cmake-3.28.1 installed !!! ==================="

# cuda-toolkit
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin &&\
    sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600 &&\
    sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub &&\
    sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /" &&\
    wget https://developer.download.nvidia.com/compute/cuda/12.4.1/local_installers/cuda-repo-ubuntu2204-12-4-local_12.4.1-550.54.15-1_amd64.deb &&\
    sudo dpkg -i cuda-repo-ubuntu2204-12-4-local_12.4.1-550.54.15-1_amd64.deb &&\
    rm cuda-repo-ubuntu2204-12-4-local_12.4.1-550.54.15-1_amd64.deb &&\
    sudo cp /var/cuda-repo-ubuntu2204-12-4-local/cuda-*-keyring.gpg /usr/share/keyrings/ &&\
    sudo apt-get update &&\
    sudo apt-get -y install\
 cuda-toolkit-12-4\
 ros-humble-rosbag2-storage-mcap\
 libstdc++-13-dev\
 clinfo ocl-icd-opencl-dev\
 libcudnn8-dev\
 libcudnn8

WORKDIR /home/jetbot

# this value will be overwritten with that one from devcontainer.json
ARG DOMAIN_ID=17

# add environmental variables and aliases
RUN echo "export MESA_GL_VERSION_OVERRIDE=3.3" >> /home/jetbot/.bashrc &&\
    echo "export ROS_DOMAIN_ID=${DOMAIN_ID}" >> /home/jetbot/.bashrc &&\
    echo -e "alias runColcon=\"colcon build --cmake-args -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_EXPORT_COMPILE_COMMANDS=ON\"" >> /home/jetbot/.bash_aliases &&\
    echo -e "alias runColconVerbose=\"colcon build --event-handlers console_direct+ --cmake-args -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_EXPORT_COMPILE_COMMANDS=ON\"" >> /home/jetbot/.bash_aliases &&\
    echo -e "alias runColconPkgs=\"colcon build --cmake-args -DCMAKE_EXPORT_COMPILE_COMMANDS=ON --packages-up-to\"" >> /home/jetbot/.bash_aliases &&\
    echo -e "alias runColconPkgsVerbose=\"colcon build --event-handlers console_direct+ --cmake-args -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_EXPORT_COMPILE_COMMANDS=ON --packages-up-to\"" >> /home/jetbot/.bash_aliases &&\
    echo -e "alias runColconPkgsDebug=\"colcon build --event-handlers console_direct+ --cmake-args -DCMAKE_BUILD_TYPE=Debug -DCMAKE_EXPORT_COMPILE_COMMANDS=ON --packages-up-to\"" >> /home/jetbot/.bash_aliases &&\
    mkdir Develop
