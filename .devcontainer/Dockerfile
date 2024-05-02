FROM nvidia/cuda:12.2.2-devel-ubuntu22.04

LABEL maintainer="Kornel Mrozowski kpmrozowski@gmail.com"

RUN apt update && \
    apt upgrade -y && \
# tzdata
    ln -fs /usr/share/zoneinfo/Europe/Warsaw /etc/localtime &&\
    apt-get install -y tzdata && \
    dpkg-reconfigure --frontend noninteractive tzdata &&\
    apt install -y curl gnupg lsb-release sudo vim ninja-build wget build-essential ffmpeg libeigen3-dev libglew-dev\   
 libfmt-dev bash-completion dirmngr htop gdb git git-lfs x11-apps libgmp3-dev pybind11-dev gitk meld terminator &&\
# adding user turtle
    adduser --disabled-password --gecos '' turtle &&\
    adduser turtle sudo &&\
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
WORKDIR /home/turtle
USER turtle

RUN mkdir /home/turtle/sdk &&\
    sudo apt update && sudo apt install -y locales &&\
    sudo locale-gen en_US en_US.UTF-8 &&\
    sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 &&\
    export LANG=en_US.UTF-8 &&\
    echo "export LANG=en_US.UTF-8" >> /home/turtle/.bashrc

WORKDIR /home/turtle/sdk

# install older compilers, because duda doesn't like anythng newer than gcc-10
RUN sudo echo "deb http://archive.ubuntu.com/ubuntu focal main universe"  | sudo tee -a /etc/apt/sources.list &&\
    sudo rm /etc/apt/sources.list.d/* &&\
    sudo apt-get update -q &&\
    sudo apt-get -y install gcc-8 g++-8 gcc-9 g++-9 gcc-10 g++-10 &&\
    sudo sed -i '$ d' /etc/apt/sources.list &&\
    sudo apt update

RUN sudo apt install -y software-properties-common &&\
    sudo add-apt-repository --update -y ppa:ubuntu-toolchain-r/test &&\
    wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | sudo apt-key add - &&\
    sudo add-apt-repository --update -y "deb http://apt.llvm.org/jammy/ llvm-toolchain-jammy-18 main" &&\
    sudo apt install -y\
 clang-18\
 lldb-18\
 lld-18\
 clangd-18\
 clang-tidy-18\
 clang-format-18\
 clang-tools-18\
 llvm-18-dev\
 llvm-18-tools\
 libc++-18-dev\
 libc++abi-18-dev\
 libclang-common-18-dev\
 libclang-18-dev\
 libclang-cpp15-dev\
 libunwind-18-dev\
 gcc-11\
 g++-11\
 gcc-12\
 g++-12 &&\
    sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 90 &&\
    sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 70 &&\
    sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 50 &&\
    sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-10 30 &&\
    sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-8 10 &&\
    sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-9 90 &&\
    sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-12 70 &&\
    sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 50 &&\
    sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-10 30 &&\
    sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-8 10 &&\
    sudo rm -rf /var/lib/apt/lists/* /var/lib/apt/sources.list.d/*

# upgrade cmake 3.16 to 3.28
RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | sudo tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null &&\
    sudo apt-add-repository "deb https://apt.kitware.com/ubuntu/ $(lsb_release -cs) main" &&\
    sudo apt update &&\
    sudo apt install cmake -y && \
    sudo rm -rf /var/lib/apt/lists/* /var/lib/apt/sources.list.d/* &&\
    echo "================= cmake-3.28.1 installed !!! ==================="

# install ROS 2 Humble
RUN sudo apt update &&\
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
    echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc &&\
    sudo rm -rf /var/lib/apt/lists/* /var/lib/apt/sources.list.d/*

RUN sudo apt update &&\
    sudo apt install -y \
 python3-dev\
 python3-colcon-common-extensions\
 python3-rosdep\
 python3-argcomplete\
 python3-pip\
 python3-numpy\
 python-setuptools\
 ros-humble-xacro\
 ros-humble-angles\
 ros-humble-ros2-control\
 ros-humble-realtime-tools\
 ros-humble-control-toolbox\
 ros-humble-ros2-controllers\
 ros-humble-joint-state-publisher\
 ros-humble-joint-state-publisher-gui\
 ros-humble-ament-cmake-clang-format\
 ros-humble-dynamic-edt-3d\
 ros-humble-rviz-visual-tools\
 ros-humble-control-msgs\
 ros-humble-sensor-msgs\
 ros-humble-vision-msgs\
 ros-humble-rosbridge-msgs\
 ros-humble-ros2bag\
 ros-humble-cv-bridge

# cuda-toolkit
RUN sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub &&\
    sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /" &&\
    sudo apt-get update &&\
    sudo apt-get -y install\
 cuda-toolkit-12-2\
 ros-humble-rosbag2-storage-mcap\
 libstdc++-13-dev\
 clinfo\
 ocl-icd-opencl-dev\
 libcudnn8-dev\
 libcudnn8\
 cmake-curses-gui\
 g++-12\
 gcc-12

# install opencv
RUN cd /home/turtle/sdk &&\
    git clone --depth 1 --branch 4.7.0 https://github.com/opencv/opencv.git &&\
    git clone --depth 1 --branch 4.7.0 https://github.com/opencv/opencv_contrib.git &&\
    CUDA_HOME=/usr/local/cuda-12.2 \
    CXX=/usr/bin/g++-8 \
    CC=/usr/bin/gcc-8 \
    cmake \
        -D CMAKE_C_COMPILER=/usr/bin/gcc-8 \
        -D CMAKE_CXX_COMPILER=/usr/bin/g++-8 \
        -D CMAKE_BUILD_TYPE=Release \
        -D CMAKE_INSTALL_PREFIX=/usr/local \
        -D OPENCV_EXTRA_MODULES_PATH=/home/turtle/sdk/opencv_contrib/modules \
        -D OPENCV_ENABLE_NONFREE=ON \
        -D OPENCV_GENERATE_PKGCONFIG=ON \
        -D BUILD_opencv_calib3d=ON \
        -D BUILD_opencv_core=ON \
        -D BUILD_opencv_features2d=ON \
        -D BUILD_opencv_flann=ON \
        -D BUILD_opencv_highgui=ON \
        -D BUILD_opencv_imgcodecs=ON \
        -D BUILD_opencv_imgproc=ON \
        -D BUILD_opencv_objdetect=ON \
        -D BUILD_opencv_photo=ON \
        -D BUILD_opencv_stitching=ON \
        -D BUILD_opencv_video=ON \
        -D BUILD_opencv_videoio=ON \
        -D BUILD_opencv_ximgproc=ON \
        -D BUILD_opencv_cudaarithm=ON \
        -D BUILD_opencv_cudabgsegm=ON \
        -D BUILD_opencv_cudacodec=ON \
        -D BUILD_opencv_cudafeatures2d=ON \
        -D BUILD_opencv_cudafilters=ON \
        -D BUILD_opencv_cudaimgproc=ON \
        -D BUILD_opencv_cudaobjdetect=ON \
        -D BUILD_opencv_cudaoptflow=ON \
        -D BUILD_opencv_cudastereo=ON \
        -D BUILD_opencv_cudawarping=ON \
        -D INSTALL_C_EXAMPLES=OFF \
        -D INSTALL_PYTHON_EXAMPLES=OFF \
        -D EIGEN_INCLUDE_PATH=/usr/include/eigen3 \
        -D CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-12.2 \
        -D ENABLE_FAST_MATH=ON \
        # -D ENABLE_NEON=ON \
        -D CUDA_FAST_MATH=ON \
        -D CUDA_ARCH_BIN=8.6 \
        -D OPENCV_DNN_CUDA=ON \
        -D WITH_CUDNN=ON \
        -D WITH_CUDA=ON \
        -D CUDNN_LIBRARY=/usr/lib/x86_64-linux-gnu/libcudnn.so.8.9.7 \
        -D CUDNN_INCLUDE_DIR=/usr/include \
        -D BUILD_NEW_PYTHON_SUPPORT=ON \
        -D BUILD_TESTS=OFF \
        -D BUILD_TIFF=ON \
        -D BUILD_TBB=ON \
        -D BUILD_EXAMPLES=OFF \
        -D BUILD_opencv_python2=OFF \
        -D BUILD_opencv_python3=ON \
        -D BUILD_LIST=\
core,\
highgui,\
calib3d,\
imgproc,\
imgcodecs,\
cudaimgproc,\
stereo,\
cudastereo,\
cudev,\
ximgproc,\
aruco\
# flann,\
# features2d,\
# xfeatures2d,\
# cudafeatures2d,\
# viz,\
# dnn\
# gapi
# ml
# objdetect
# photo
# stitching
# video
# videoio
# alphamat
# barcode
# bgsegm
# bioinspired
# ccalib
# cudaarithm
# cudabgsegm
# cudacodec
# cudafilters
# cudalegacy
# cudaobjdetect
# cudaoptflow
# cudawarping
# datasets
# dnn_objdetect
# dnn_superres
# dpm
# face
# freetype
# fuzzy
# hdf
# hfs
# img_hash
# intensity_transform
# line_descriptor
# mcc
# optflow
# phase_unwrapping
# plot
# quality
# rapid
# reg
# rgbd
# saliency
# sfm
# shape
# structured_light
# superres
# surface_matching
# text
# tracking
# videostab
# wechat_qrcode
# xobjdetect
# xphoto
        -G Ninja \
        -S /home/turtle/sdk/opencv \
        -B /home/turtle/sdk/opencv/build && \
    ninja -j$(nproc) -l$(nproc) -C /home/turtle/sdk/opencv/build && \
    sudo ninja -C /home/turtle/sdk/opencv/build install && \
    sudo ldconfig && \
    sudo apt-get update && \
    cd /home/turtle/sdk/ && \
    rm -r /home/turtle/sdk/* && \
    echo "================= opencv installed !!! ===================" && \
    echo "================= opencv installed !!! ===================" && \
    echo "================= opencv installed !!! ==================="

RUN sudo apt update &&\
    sudo apt install -y\
 cuda-compat-12-3=545.23.08-1\
 ros-humble-image-transport-plugins\
 ros-humble-ffmpeg-image-transport\
 libgtk2.0-dev

WORKDIR /home/turtle

# this value will be overwritten with that one from devcontainer.json
ARG DOMAIN_ID=17

# add environmental variables and aliases
RUN echo "export MESA_GL_VERSION_OVERRIDE=3.3" >> /home/turtle/.bashrc &&\
    echo "export CUDACXX=/usr/local/cuda-12.2/bin/nvcc" >> /home/turtle/.bashrc &&\
    echo "export ROS_DOMAIN_ID=${DOMAIN_ID}" >> /home/turtle/.bashrc &&\
    CMAKE_ARGUMENTS_COMMON="-DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DTCNN_CUDA_ARCHITECTURES=86 -DCMAKE_CUDA_ARCHITECTURES=86" &&\
    echo -e "alias runColcon=\"colcon build --cmake-args -DCMAKE_BUILD_TYPE=RelWithDebInfo $CMAKE_ARGUMENTS_COMMON\"" >> /home/turtle/.bash_aliases &&\
    echo -e "alias runColconVerbose=\"colcon build --event-handlers console_direct+ --cmake-args -DCMAKE_BUILD_TYPE=RelWithDebInfo $CMAKE_ARGUMENTS_COMMON\"" >> /home/turtle/.bash_aliases &&\
    echo -e "alias runColconPkgs=\"colcon build --cmake-args $CMAKE_ARGUMENTS_COMMON --packages-up-to\"" >> /home/turtle/.bash_aliases &&\
    echo -e "alias runColconPkgsVerbose=\"colcon build --event-handlers console_direct+ --cmake-args -DCMAKE_BUILD_TYPE=RelWithDebInfo $CMAKE_ARGUMENTS_COMMON --packages-up-to\"" >> /home/turtle/.bash_aliases &&\
    echo -e "alias runColconPkgsDebug=\"colcon build --event-handlers console_direct+ --cmake-args -DCMAKE_BUILD_TYPE=Debug $CMAKE_ARGUMENTS_COMMON --packages-up-to\"" >> /home/turtle/.bash_aliases &&\
    mkdir Develop
