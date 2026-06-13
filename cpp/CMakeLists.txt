cmake_minimum_required(VERSION 3.15)
project(alphalineage_evaluator LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_evaluator evaluator.cpp)
target_compile_options(_evaluator PRIVATE -O3)

# Statically link the MinGW runtime so the .pyd is self-contained (Python 3.8+ does not
# search PATH for an extension's dependent DLLs). No-op on MSVC/clang.
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  target_link_options(_evaluator PRIVATE
    -static-libgcc -static-libstdc++
    -Wl,-Bstatic,--whole-archive -lwinpthread -Wl,--no-whole-archive)
endif()
