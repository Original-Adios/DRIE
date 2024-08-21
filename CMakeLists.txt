set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -m64")#强制使用64位编译
cmake_minimum_required(VERSION 2.8)

project(image_contrast_enhancement)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


find_package(OpenCV REQUIRED)
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    include
)

add_executable(main 
    src/SEF.cpp
    src/main.cpp
)
target_link_libraries(main  
    ${OpenCV_LIBS} 
)
target_link_libraries(main stdc++fs)

