# Changes done git repo for 1M Robotics 

## Changes
* Changed repo from app to library (i.e. passing mats rather than saving files). 
* Added python bindings using pybind11. 
* Moved the optimization code from main.cpp to mrf_optimization.cpp

## Install 
* Added -fPIC to MRF2's CPPFLAGS in the Makefile and rebuild using 'make -B'
* Added CMAKE_CXX_STANDARD 17 to CMakeLists.txt 
* Install eigen from source 
* Install OpenCV3 from source using cmake command:
    * cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local -DOPENCV_EXTRA_MODULES_PATH=/home/onem/repos/opencv_install/opencv_contrib/modules/ -DWITH_QT=ON -DWITH_OPENGL=ON  -DBUILD_SHARED_LIBS=ON .. 


## Use
* The size of the expected input images are declared in kDepthWidth and kDepthHeight in plane_detector.h
    * If the image size is changed than the camera parameters, i.e. the other k__ values, should be changed accordingly. 
     