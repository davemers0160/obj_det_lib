# ----------------------------------------------------------------------------
#  Root CMake file for DfD Graph Cuts
# ----------------------------------------------------------------------------
#
# To perform an out of source build do the following on the command line:
# 1. navigate to the root project directory (in this case DfD2)
# 2. mkdir build
# 3. cd build
# 4. depending on your system (windows/linux) type the following
# Example CMake build line for windows & VS2015
# windows: cmake -G "Visual Studio 14 2015 Win64" -DUSE_AVX_INSTRUCTIONS=ON ..
# linux: cmake -DUSE_AVX_INSTRUCTIONS=ON ..
#
# 5. To compile:
# windows: cmake --build . --config Release
# linux: cmake --build . --config Release -- -j10
#

cmake_minimum_required(VERSION 3.9)

set(CMAKE_CXX_STANDARD 11)

set(BUILD_DLL FALSE)
set(LIB_NAME "obj_det_lib")

if(BUILD_DLL)
    project(obj_det_lib VERSION 1.0 DESCRIPTION "Shared library for an object detection network with internal taps to view network during operation")

    message("Building DLL ...")

    if(WIN32)
        # Change these to point to the right location for a particular machine
        set(DLIB_ROOT "D:/dlib-19.17")
        set(COMMON_ROOT "D:/Common")
        add_definitions(-DOBJ_DLL_EXPORTS)
	    #add_subdirectory(${DLIB_ROOT}/dlib dlib_build)
    
    elseif(UNIX)
        # Change these to point to the right location for a particular machine
        set(HOME "$ENV{HOME}" CACHE INTERNAL "Copied from environment variable")
        set(DLIB_ROOT "${HOME}/dlib-19.18")
        set(COMMON_ROOT "${HOME}/Projects/Common")
        #add_subdirectory(${DLIB_ROOT}/dlib dlib_build)
    
    elseif(APPLE)
        message(FATAL_ERROR "Apple -> Not Even!")
    
    endif()    

    include("${DLIB_ROOT}/dlib/cmake")

    # get the source files for the Library
    set(SOURCES "src/obj_det_lib_v1.cpp" 
                #"${DLIB_ROOT}/dlib/all/source.cpp"
                )

    set(project_includes "include" 
                        "${COMMON_ROOT}/include" 
                        ${DLIB_ROOT}
                        )
                    
    # find OpenCV
    find_package(OpenCV REQUIRED)
    if (OpenCV_FOUND)
        set(project_includes ${project_includes} ${OpenCV_INCLUDE_DIRS})
    else()
        message("OpenCV not found, so we won't build the project.")
    endif()


    message(" ")
    message("--------------------------------------------------------")
    message("Source Files: ")
    foreach(S ${SOURCES})
        message("  " ${S})
    endforeach()
    message("--------------------------------------------------------")
    message(" ")

    message(" ")
    message("--------------------------------------------------------")
    message("Project Includes: ")
    foreach(S ${project_includes})
        message("  " ${S})
    endforeach()
    message("--------------------------------------------------------")
    message(" ")


    if(WIN32)
        include_directories(${project_includes})
        add_library(${LIB_NAME} SHARED ${SOURCES})
        set_target_properties(${LIB_NAME} PROPERTIES VERSION ${PROJECT_VERSION})
        target_link_libraries(${LIB_NAME} dlib::dlib ${OpenCV_LIBS})
        add_definitions("-fPIC")
    elseif(UNIX)
        find_package(X11 REQUIRED)
        include_directories(${project_includes})
        add_library(${LIB_NAME} SHARED ${SOURCES})
        set_target_properties(${LIB_NAME} PROPERTIES VERSION ${PROJECT_VERSION})
        #target_link_libraries(${LIB_NAME} "${DLIB_ROOT}/dlib/build/libdlib.so" ${X11_LIBRARIES})
        target_link_libraries(${LIB_NAME} ${X11_LIBRARIES})
        target_link_libraries(${LIB_NAME} ${OpenCV_LIBS})

        #target_link_libraries(${LIB_NAME} dlib::dlib)
        add_definitions("-fPIC")
    endif()


    set(default_build_type "Release")
    if(MSVC AND (${CMAKE_VERSION} VERSION_GREATER "3.6.0"))
        message("MSVC Version: " ${MSVC_VERSION})
        target_compile_options(${PROJECT_NAME} PUBLIC "/MP" "/bigobj")
        set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ${PROJECT_NAME})
    endif()
    



else()
    project(obj_det_test VERSION 1.0 DESCRIPTION "Test Project for the shared library for an object detection network")
    message("Building Project ...")
    set(exe_name "obj_det_ex")


    set(PLATFORM "$ENV{PLATFORM}" CACHE INTERNAL "Copied from environment variable")
    message("Platform: " ${PLATFORM})

    if(WIN32)
        #set(DLIB_ROOT "D:/dlib-19.17")
        set(COMMON_ROOT "D:/Common")
        #add_definitions(-DOBJ_DLL_EXPORTS)
        #set(OBJ_LIB "../build/Release/${LIB_NAME}.dll")
        set(OBJ_LIB "../build/Release/${LIB_NAME}")
    else()
        #set(DLIB_ROOT ~/dlib/dlib/cmake)
        set(COMMON_ROOT "~/Common")
    endif()
    
    message("Object Detector Lib: " ${OBJ_LIB})

    #include(${DLIB_ROOT}/dlib/cmake)

    #file(GLOB SOURCES "src/*.cpp")
    set(SOURCES "src/obj_det_lib_test_v1.cpp")

    set(project_includes "include" "${COMMON_ROOT}/include")

    # find OpenCV
    find_package(OpenCV REQUIRED)
    if (OpenCV_FOUND)
        set(project_includes ${project_includes} ${OpenCV_INCLUDE_DIRS})
    else()
        message("OpenCV not found, so we won't build the project.")
    endif()
    
    
    message(" ")
    message("--------------------------------------------------------")
    message("Source Files: ")
    foreach(S ${SOURCES})
        message("  " ${S})
    endforeach()
    message("--------------------------------------------------------")
    message(" ")

    message(" ")
    message("--------------------------------------------------------")
    message("Project Includes: ")
    foreach(S ${project_includes})
        message("  " ${S})
    endforeach()
    message("--------------------------------------------------------")
    message(" ")


    include_directories(${project_includes})
    add_executable(${exe_name} ${SOURCES})
    #target_link_libraries(${exe_name} dlib::dlib ${OpenCV_LIBS} ${OBJ_LIB} )
    target_link_libraries(${exe_name} ${OpenCV_LIBS} ${OBJ_LIB} )

    if(WIN32)
        message("MSVC Version: " ${MSVC_VERSION})
        # uncomment for MSVC and cmake greater than 3.6.0
        set(default_build_type "Release")
        if(MSVC AND (${CMAKE_VERSION} VERSION_GREATER "3.6.0"))
            target_compile_options(${exe_name} PRIVATE "/MP" "/bigobj")
            set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ${exe_name})
        endif()

    endif()

endif()




message(" ")
message(STATUS "Build Complete!")
message(" ")



