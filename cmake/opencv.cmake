include_guard()
include(${CMAKE_DIR}/fetchcontent_declare.cmake)

function(add_opencv_modules MODULES)
    foreach(the_module ${MODULES})
        if(TARGET ${the_module})
            set_target_properties(
                ${the_module}
                PROPERTIES
                    INTERFACE_INCLUDE_DIRECTORIES
                        $<TARGET_PROPERTY:${the_module},INCLUDE_DIRECTORIES>
            )
            string(REPLACE "opencv_" "" module_name ${the_module})
            add_library(OpenCV::${module_name} ALIAS ${the_module})
        endif()
    endforeach()
endfunction(add_opencv_modules)

function(print_recognized_opencv_modules MODULES)
    foreach(the_module ${MODULES})
        Message(${the_module})      
    endforeach()
endfunction(print_recognized_opencv_modules)

set(VERSION 4.7.0)

if(USE_SYSTEM_LIBS)
    find_package(OpenCV ${VERSION})

    if(OpenCV_FOUND)
        add_opencv_modules("${OpenCV_LIB_COMPONENTS}")
        include_directories(${OpenCV_INCLUDE_DIRS})
        return()
    endif()
endif()

FetchContent_Declare(
    opencv
    GIT_REPOSITORY https://github.com/opencv/opencv.git
    GIT_TAG ${VERSION}
)
FetchContent_Declare_GH(opencv_contrib opencv/opencv_contrib 4.9.0)

FetchContent_GetProperties(opencv)
if(NOT opencv_POPULATED)
    FetchContent_Populate(opencv_contrib)
    FetchContent_Populate(opencv)

    LIST(APPEND CMAKE_PROGRAM_PATH  "/usr/local/cuda-12.4/bin/")
    set(CUDA_HOME/usr/local/cuda-12.4)
    set(OPENCV_ENABLE_NONFREE ON)
    set(OPENCV_GENERATE_PKGCONFIG ON)
    set(INSTALL_C_EXAMPLES OFF)
    set(INSTALL_PYTHON_EXAMPLES OFF)
    set(EIGEN_INCLUDE_PATH /usr/include/eigen3)
    set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-12.4)
    set(ENABLE_FAST_MATH ON)
    set(CUDA_FAST_MATH ON)
    set(CUDA_ARCH_BIN 8.6)
    set(OPENCV_DNN_CUDA ON)
    set(WITH_CUDNN ON)
    set(WITH_CUDA ON)
    set(CUDNN_LIBRARY /usr/lib/x86_64-linux-gnu/libcudnn.so.8.9.7)
    set(CUDNN_INCLUDE_DIR /usr/include)
    set(BUILD_NEW_PYTHON_SUPPORT ON)
    set(BUILD_TESTS OFF)
    set(BUILD_TIFF ON)
    set(BUILD_TBB ON)
    set(BUILD_EXAMPLES OFF)
    set(BUILD_opencv_python2 OFF)
    set(BUILD_opencv_python3 ON)
    set(WITH_EIGEN ON)
    set(WITH_OPENEXR OFF)
    set(WITH_ITT OFF)
    set(BUILD_PERF_TESTS OFF)
    set(BUILD_WITH_STATIC_CRT ${STATIC_CRT})
    set(BUILD_SHARED_LIBS OFF)
    set(WITH_FFMPEG OFF)
    set(WITH_GSTREAMER OFF)
    set(WITH_GTK OFF)
    set(WITH_QUIRC OFF)
    set(OPENCV_EXTRA_MODULES_PATH ${opencv_contrib_SOURCE_DIR}/modules CACHE PATH "Where to look for additional OpenCV modules (can be ;-separated list of paths)" FORCE)
    message("OPENCV_EXTRA_MODULES_PATH=${OPENCV_EXTRA_MODULES_PATH}")
#     # we whitelist only modules that we use, to check list of all modules that are present call print_recognized_opencv_modules 
#     # print_recognized_opencv_modules("${OPENCV_MODULES_BUILD}")
    set(BUILD_LIST 
core
# aruco
calib3d
# features2d
# flann
imgproc
imgcodecs
# dnn
# gapi
# highgui
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
# cudafeatures2d
# cudafilters
cudaimgproc
# cudalegacy
# cudaobjdetect
# cudaoptflow
cudastereo
# cudawarping
cudev
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
# stereo
# structured_light
# superres
# surface_matching
# text
# tracking
# videostab
# viz
# wechat_qrcode
# xfeatures2d
ximgproc
# xobjdetect
# xphoto
)

    add_subdirectory(${opencv_SOURCE_DIR} ${opencv_BINARY_DIR} EXCLUDE_FROM_ALL)

    add_opencv_modules("${OPENCV_MODULES_BUILD}")
    add_opencv_modules("${OPENCV_EXTRA_MODULES_PATH}")

    get_directory_property(JPEG_LIBRARY DIRECTORY ${opencv_SOURCE_DIR} DEFINITION JPEG_LIBRARY)
    if(TARGET ${JPEG_LIBRARY})
        set_target_properties(${JPEG_LIBRARY}
            PROPERTIES
                INTERFACE_INCLUDE_DIRECTORIES
                    $<TARGET_PROPERTY:${JPEG_LIBRARY},INCLUDE_DIRECTORIES>
        )
        set(CMAKE_DISABLE_FIND_PACKAGE_JPEG TRUE)
        set(JPEG_FOUND TRUE)
        add_library(JPEG::JPEG ALIAS ${JPEG_LIBRARY})
    endif()

    get_directory_property(PNG_LIBRARY DIRECTORY ${opencv_SOURCE_DIR} DEFINITION PNG_LIBRARY)
    if(TARGET ${PNG_LIBRARY})
        set_target_properties(${PNG_LIBRARY}
            PROPERTIES
                INTERFACE_INCLUDE_DIRECTORIES
                    $<TARGET_PROPERTY:${PNG_LIBRARY},INCLUDE_DIRECTORIES>
        )
        set(CMAKE_DISABLE_FIND_PACKAGE_PNG TRUE)
        set(PNG_FOUND TRUE)
        add_library(PNG::PNG ALIAS ${PNG_LIBRARY})
    endif()
endif()
