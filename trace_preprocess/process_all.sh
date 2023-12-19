#!/bin/bash
BASE_DIR=/home/jl/work/trace_data/trace_preprocess
python ${BASE_DIR}/process_files.py small1 &
python ${BASE_DIR}/process_files.py small2 &
python ${BASE_DIR}/process_files.py small3 &
python ${BASE_DIR}/process_files.py small4 &
python ${BASE_DIR}/process_files.py small5 &