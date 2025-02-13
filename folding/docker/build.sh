#!/bin/bash
set -e
set -u

docker build -t alma-dnn -f ./Dockerfile .
