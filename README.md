# Local DnD LLM Assistant

- TBD


## Installation

When installing llama-cpp-python (in `./requirements.txt`) make sure to have `cuda` installed on your system.
Then `export CMAKE_ARGS="-DGGML_CUDA=on"` (or `set` on Windows) to enable usage of your GPU by the respective llama model.
If you are not using cuda but still want to run on your GPU refer to the llama-cpp-python docs.