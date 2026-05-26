cmake_minimum_required(VERSION 3.16)
project(LKTorch LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

# --- FORCE OUTPUT TO "build" FOLDER ---

# 1. Define the output path
set(OUTPUT_DIR "${CMAKE_SOURCE_DIR}/build")

# 2. Force ALL configurations (Debug, Release) to go to the same place
# If you don't do this, VS creates "bin/Debug/Visualizer.exe"
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${OUTPUT_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${OUTPUT_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${OUTPUT_DIR})

# 3. (Optional) Also put DLLs (.lib files) there if needed
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${OUTPUT_DIR})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG ${OUTPUT_DIR})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${OUTPUT_DIR})

#Main execcutable file
add_executable(LKTorch "src/main.cpp")

target_include_directories(LKTorch PRIVATE
	${CMAKE_SOURCE_DIR}/include
)
target_sources(LKTorch PRIVATE "src/LinearAlg.cpp" "src/DebugAssist.cpp" "src/Layer.cpp" "src/Tensor.cpp")


#Inputing the folder for accessing assets
target_compile_definitions(LKTorch PRIVATE PROJECT_DIR="${CMAKE_SOURCE_DIR}/")