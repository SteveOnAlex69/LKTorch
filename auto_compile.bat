@echo off

echo Step 0: Cleaning up my shit
del /f /q "lib\*.pyd"
del /f /q "lib\*.pyi"

echo Step 1: CREATING the folder
mkdir build
echo Step 2: opening the folder
cd build
echo Step 3: Cmake gaming
cmake -DBUILD_PYTHON=ON ..
echo Step 4: Cmake gaming
cmake --build . --config Release
echo Step 5: Copy the file
copy "*.pyd" "..\lib"
echo Step 6: Creating python auto-suggestion
cd ..\
cd lib
set PYTHONPATH=. && pybind11-stubgen lktorch
copy "stubs\*.pyi" "."

rd /s /q "stubs"

ren "lktorch.cp310-win_amd64.pyd" "lktorch.pyd"


pause