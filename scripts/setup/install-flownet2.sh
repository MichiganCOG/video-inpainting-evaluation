#!/bin/bash
set -e

SCRIPT_DIR="$(cd $(dirname $0); pwd)"
PROJ_DIR="$(cd $SCRIPT_DIR/../..; pwd)"

cd "$PROJ_DIR/src/models/flownet2/networks/correlation_package"
rm -rf *_cuda.egg-info build dist __pycache__
python setup.py install

cd "$PROJ_DIR/src/models/flownet2/networks/resample2d_package"
rm -rf *_cuda.egg-info build dist __pycache__
python setup.py install

cd "$PROJ_DIR/src/models/flownet2/networks/channelnorm_package"
rm -rf *_cuda.egg-info build dist __pycache__
python setup.py install
