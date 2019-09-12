#!/bin/bash
mkdir -p dataset/camera dataset/log

wget --continue https://archive.org/download/comma-dataset/comma-dataset.zip
mkdir -p dataset
cd dataset
unzip ../comma-dataset.zip

mkdir -p /storage/commai/
cp -r dataset /storage/commai/