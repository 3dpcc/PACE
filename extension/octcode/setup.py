from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import pybind11


class get_pybind_include:
    def __str__(self):
        return pybind11.get_include()


ext_modules = [
    Extension(
        "octcode",
        sources=[
            "src/octree.cpp",
            "src/pybind.cpp",
        ],
        include_dirs=[
            "src/include",
            str(get_pybind_include()),
        ],
        language="c++",
        extra_compile_args=[
            "-std=c++20", "-O3", "-Wall", "-fPIC",
            "-fopenmp", "-mbmi2", "-mavx2",
            "-ffast-math", "-funroll-loops",
        ],
        extra_link_args=["-shared", "-fopenmp"],
    ),
]


class BuildExt(build_ext):
    def build_extensions(self):
        version = self.distribution.get_version()
        for ext in self.extensions:
            ext.extra_compile_args.append(f'-DVERSION_INFO="{version}"')

        build_ext.build_extensions(self)


setup(
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExt},
    zip_safe=False,
)
