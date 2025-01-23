#!/bin/bash
set -e
set -u

docker run -it --rm -v ${PWD}:/playground alma-dnn
