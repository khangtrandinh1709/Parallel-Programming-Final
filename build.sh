#!/bin/bash

# Set the name of the executable
TARGET="ANN"

# CUDA Compiler (nvcc)
CC="nvcc"

# Source files
SRCS="main.cu ANN.cu utility_functions.cu"

# Compile directly into an executable
echo "Compiling and linking the CUDA source files into $TARGET..."
$CC $SRCS -o $TARGET

# Check if the compilation was successful
if [ $? -eq 0 ]; then
    echo "Build successful"
else
    echo "Build failed. Please check the errors above."
fi