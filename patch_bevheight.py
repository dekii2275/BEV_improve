import os
import re

def patch_file(path, search_pattern, replacement):
    if not os.path.exists(path):
        print(f"Skipping {path}: File not found")
        return
    with open(path, 'r') as f:
        content = f.read()
    
    new_content = re.sub(search_pattern, replacement, content)
    
    if new_content != content:
        with open(path, 'w') as f:
            f.write(new_content)
        print(f"Successfully patched {path}")
    else:
        print(f"No changes needed for {path}")

def main():
    print("Patching BEVHeight for PyTorch 2.x compatibility...")
    
    # 1. Remove THC/THC.h which is removed in modern PyTorch
    thc_pattern = r'#include <THC/THC\.h>'
    patch_file("ops/voxel_pooling/src/voxel_pooling_forward_cuda.cu", thc_pattern, "// #include <THC/THC.h>")
    patch_file("ops/voxel_pooling/src/voxel_pooling_forward.cpp", thc_pattern, "// #include <THC/THC.h>")

    # 2. Fix potential error in setup.py regarding setuptools versions
    setup_pattern = r"from torch\.utils\.cpp_extension import \(BuildExtension, CppExtension,"
    setup_replacement = "import setuptools\nfrom torch.utils.cpp_extension import (BuildExtension, CppExtension,"
    patch_file("setup.py", setup_pattern, setup_replacement)

    print("Patching complete.")

if __name__ == "__main__":
    main()
