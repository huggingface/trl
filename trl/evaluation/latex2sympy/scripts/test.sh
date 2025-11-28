#!/bin/sh

# Get relative path of the root directory of the project
rdir=`git rev-parse --git-dir`
rel_path="$(dirname "$rdir")"
# Change to that path and run the file
cd $rel_path

# Activate virtual environment
echo "activating venv..."
if test -f .env/bin/activate
then source .env/bin/activate && echo "venv activate (bin)"
elif test -f .env/Scripts/activate
then source .env/Scripts/activate && echo "venv activated (Scripts)"
else exit 1
fi

echo ''
echo "compiling parser..."
sh scripts/compile.sh
echo "parser compiled"

echo ''
# Run unit tests
echo "starting tests..."
if pytest tests
then echo "tests finished"
else exit 1
fi

exit 0
