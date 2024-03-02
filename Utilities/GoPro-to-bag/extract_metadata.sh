#!/usr/bin/bash

cd $1/GoPro-data-extraction

make accl
./gopro_data_extractor $2 $3

make gyro
./gopro_data_extractor $2 $3
