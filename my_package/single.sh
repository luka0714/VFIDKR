#!/usr/bin/env bash

echo "Need pytorch>=1.0.0"
source activate pytorch1.0.0

export PYTHONPATH=$PYTHONPATH:$(pwd)

cd FilterInterpolation
rm -rf build *.egg-info dist
python setup.py install
cd ..

