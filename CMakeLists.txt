project(kslam)

find_package(OpenCV REQUIRED)
find_package(Pangolin 0.8 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${Pangolin_INCLUDE_DIRS})

add_executable(kslam kslam.cpp features.cpp frame.cpp viewer.cpp)
target_link_libraries(kslam ${OpenCV_LIBS} pango_display pango_scene)