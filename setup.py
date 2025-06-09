from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "Submission", 
        ["src/pybind_wrapper.cpp"],
        include_dirs=["src"],
        extra_compile_args=["-O3", "-fopenmp", "-march=native", "--fast-math"],
        extra_link_args=["-fopenmp"],
    ),
]

setup(
    name="Submission",
    version="0.0.1",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)