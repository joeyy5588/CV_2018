#!/bin/bash

python3 main.py --maxdisp 192 \
               --model stackhourglass \
               --datapath /data/ \
               --epochs 10 \
               --savemodel /trained/10.tar