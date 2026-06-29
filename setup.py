# Idea: our auto_compile.bat is already working like a charm, so our hack is just to copy the output of that program
# And slap that bitch into pip

import os
import subprocess
import shutil
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

class HackyBuild(build_ext):
    def run(self):
        # Run custom CMake batch script
        #print("Running auto_compile.bat...")
        #try:
        #    subprocess.check_call(["auto_compile.bat"], shell=True)
        #except subprocess.CalledProcessError:
        #    print("Error: auto_compile.bat failed!")
        #    print("Dont care hehe")
            #raise

        # Find out exactly where pip wants to install the package
        ext = self.extensions[0]
        ext_path = self.get_ext_fullpath(ext.name)
        ext_dir = os.path.dirname(ext_path)
        os.makedirs(ext_dir, exist_ok=True)

        # Copy the files from your /lib folder into the pip installation folder!
        print(f"Copying compiled files to {ext_dir}...")
        
        pyd_src = os.path.join("lib", "lktorch.pyd")
        pyi_src = os.path.join("lib", "lktorch.pyi")
        
        if os.path.exists(pyd_src):
            shutil.copy(pyd_src, ext_path)
        else:
            raise FileNotFoundError(f"Could not find {pyd_src}. Did the batch script fail?")
            
        if os.path.exists(pyi_src):
            shutil.copy(pyi_src, os.path.join(ext_dir, "lktorch.pyi"))

setup(
    name="lktorch",
    version="0.1.0",
    description="A PyTorch-like framework built from scratch",
    
    ext_modules=[Extension("lktorch", sources=[])],
    
    cmdclass={"build_ext": HackyBuild},
    
    install_requires=["numpy"],
    python_requires=">=3.7",
)