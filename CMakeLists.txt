cmake_minimum_required(VERSION 3.16)
project(parallel_sgd_matrix_factorization LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenMP REQUIRED)

add_executable(mf_train
    src/main.cpp
    src/mf.cpp
    src/sgd.cpp
    src/hogwild.cpp
    src/locked_sgd.cpp
)

target_include_directories(mf_train PRIVATE include)
target_link_libraries(mf_train PRIVATE OpenMP::OpenMP_CXX)

if(MSVC)
    target_compile_options(mf_train PRIVATE /O2 /W4)
else()
    target_compile_options(mf_train PRIVATE -O3 -Wall -Wextra -Wpedantic)
endif()
