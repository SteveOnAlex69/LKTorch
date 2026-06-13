@echo off

echo Step 1: CREATING the folder
mkdir build
echo Step 2: opening the folder
cd build
echo Step 3: Cmake gaming
cmake -DBUILD_PYTHON=ON ..
echo Step 4: Cmake gaming
cmake --build . --config Release
echo Step 5: Copy the file
copy "*.pyd" "..\test"

pause