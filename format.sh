#! /bin/bash
# Formats all relevant source code files.
# Accepts an extra input argument that is passed to black.
# Useful for --check.

extra_arg=${1}

return_code=0

# find all Python scripts but exclude those in the observational_data
# submodule
scripts=$(find . -name "*.py" -not -path "./observational_data/*")
black $scripts $extra_arg
