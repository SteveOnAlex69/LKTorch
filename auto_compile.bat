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
copy "*.pyd" "..\examples"
echo Step 6: Creating python auto-suggestion
cd ..\
cd examples
set PYTHONPATH=. && pybind11-stubgen lktorch
copy "stubs\*.pyi" "."

rd /s /q "stubs"

for /d %%i in (*) do (
	copy "*.pyd" "%%i"
	copy "*.pyi" "%%i"
)


pause