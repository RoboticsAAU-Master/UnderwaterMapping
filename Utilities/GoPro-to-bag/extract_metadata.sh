#!/usr/bin/bash

cd /RUD-PT/rudpt_ws/src/UnderwaterMapping/Utilities/GoPro-data-extraction

make accl
./gopro_data_extractor $1 $2

make gyro
./gopro_data_extractor $1 $2
