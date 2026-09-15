from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, library_paths

ext = CppExtension(
    "octrans",
    sources=["src/octrans.cpp"],
    include_dirs=["src"],
    runtime_library_dirs=library_paths(),
    extra_compile_args=["-O3", "-march=native", "-Wall"],
)

setup(
    name="octrans",
    version="1.0.0",
    description="Fast rANS coder for octree point cloud compression",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    ext_modules=[ext],
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.8",
    install_requires=["torch"],
)
