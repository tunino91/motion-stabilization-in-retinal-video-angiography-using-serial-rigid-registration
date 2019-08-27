PROJECT(registerAllFrames)



CMAKE_MINIMUM_REQUIRED(VERSION 2.8) 



FIND_PACKAGE(ITK)


IF(ITK_FOUND)
  
	INCLUDE(${ITK_USE_FILE})

ENDIF(ITK_FOUND)

FIND_PACKAGE(OpenCV)


if(WIN32)
  set(OpenCV_DIR "d:/libs/opencv-2.4.8/build")
else()
  set(OpenCV_DIR "/usr/lib/opencv")
endif()
find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui)

include_directories(${OpenCV_INCLUDE_DIRS})



ADD_EXECUTABLE(registerAllFrames RegisterAll.cxx)


TARGET_LINK_LIBRARIES(registerAllFrames ${ITK_LIBRARIES})
TARGET_LINK_LIBRARIES(registerAllFrames ${OpenCV_LIBRARIES})


project(registerAllFrames)


