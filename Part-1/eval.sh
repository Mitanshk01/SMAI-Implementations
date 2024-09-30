#!/bin/bash

# This script has been created by ChatGPT
# ChatGPT prompt: Write bash script which take dataset location as input from terminal and then runs a python file giving the input of dataset file name so that it can be opened in the python file. The name of python file is already known.

if [ $# -eq 0 ]; then
    echo "Inavlid Input file"
    exit 1
fi

dataset_location="$1"
dataset_filename=$(basename "$dataset_location")

python3 "test.py" "$dataset_filename"