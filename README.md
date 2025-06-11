# Submission to the SISAP 2025 Indexing Challenge


### Python Bindings
The code is written in C++ and has bindings to interact with Python via Pybind11. To install the python bindings, run the command `pip install .` in the base directory. 

During development (editing the C++ files), you can run `pip install -e .` once to create an editable library (egg-info). Then, when changes are made to the C++ code, you can recompile with `python setup.py build_ext --inplace` instead of running the original command. It's quicker. 


