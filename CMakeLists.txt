cmake_minimum_required(VERSION 3.21)

# Set default build type to release
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "" FORCE)
endif()

# Generate compile_commands.json by default
if(NOT CMAKE_EXPORT_COMPILE_COMMANDS)
    set(CMAKE_EXPORT_COMPILE_COMMANDS "YES" CACHE STRING "" FORCE)
endif()

project(visionrubbish)

include(FetchContent)

# Sleipnir dependency
fetchcontent_declare(
    Sleipnir
    GIT_REPOSITORY https://github.com/SleipnirGroup/Sleipnir.git
    # master on 24/3/25
    GIT_TAG 189c07282bf4604db9babbcd395cb6a8cf110f0d
)
fetchcontent_makeavailable(Sleipnir)

set(WITH_PROTOBUF OFF)

# TODO allow system
fetchcontent_declare(
    wpilib
    GIT_REPOSITORY https://github.com/wpilibsuite/allwpilib.git
    GIT_TAG 02de5f710e2579f2079a28944e90b48b34d91cac
)
fetchcontent_makeavailable(wpilib)

set(USE_SYSTEM_EIGEN ON)
set(BUILD_TESTING OFF)

# OpenCV dependency
# TODO fetchcontent
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Eigen dependency
if(NOT USE_SYSTEM_EIGEN)
    set(EIGEN_BUILD_CMAKE_PACKAGE TRUE)
    fetchcontent_declare(
        Eigen3
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        # master on 22/05/2024
        GIT_TAG c4d84dfddc9f9edef0fdbe7cf9966d2f4a303198
    )
    fetchcontent_makeavailable(Eigen3)
else()
    find_package(Eigen3 CONFIG REQUIRED)
endif()

if(BUILD_TESTING)
    # Catch2 dependency
    # TODO system option?
    fetchcontent_declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG v3.7.0
    )
    fetchcontent_makeavailable(Catch2)
endif()

file(GLOB_RECURSE src src/*.cpp)
set_property(SOURCE ${src} PROPERTY COMPILE_FLAGS "-Wno-unused")
add_executable(visionrubbish ${src})

target_include_directories(visionrubbish PRIVATE src)
target_compile_options(
    visionrubbish
    PRIVATE -Wall -Wextra -Werror -pedantic -fsanitize=address,undefined,leak
)
target_link_libraries(
    visionrubbish
    PUBLIC
        Eigen3::Eigen
        Sleipnir
        ${OpenCV_LIBS}
        apriltag
        wpigui
        cscore
        ntcore
        wpimath
        apriltag
        -fsanitize=address,undefined,leak
)
target_compile_features(visionrubbish PUBLIC cxx_std_23)
