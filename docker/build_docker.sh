#!/bin/bash

# Create a temporary directory for the build context
BUILD_CONTEXT=$(mktemp -d)

# Copy necessary files to the build context
cp -r detectron2 $BUILD_CONTEXT/detectron2
cp -r tools $BUILD_CONTEXT/tools
cp -r configs $BUILD_CONTEXT/configs
cp setup.py $BUILD_CONTEXT/
cp setup.cfg $BUILD_CONTEXT/
cp requirements.txt $BUILD_CONTEXT/
cp docker/Dockerfile $BUILD_CONTEXT/
cp docker/.dockerignore $BUILD_CONTEXT/

# Build the Docker image
docker build -t your_custom_detectron2_image -f $BUILD_CONTEXT/Dockerfile $BUILD_CONTEXT

# Remove the temporary build context directory
rm -rf $BUILD_CONTEXT
