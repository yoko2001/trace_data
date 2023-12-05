#!/bin/bash
GEN_GT_DIR=/home/jl/trace_data/gen_gt
RUN_LOG=/home/jl/trace_data/gen_gt/log.txt

rm ${RUN_LOG}
python ${GEN_GT_DIR}/run_flow.py small1 500 >>${RUN_LOG} 2>&1 
python ${GEN_GT_DIR}/run_flow.py small1 1000 >>${RUN_LOG} 2>&1 
python ${GEN_GT_DIR}/run_flow.py small1 2000 >>${RUN_LOG} 2>&1 

python ${GEN_GT_DIR}/run_flow.py small2 500 >>${RUN_LOG} 2>&1
python ${GEN_GT_DIR}/run_flow.py small2 1000 >>${RUN_LOG} 2>&1
python ${GEN_GT_DIR}/run_flow.py small2 2000 >>${RUN_LOG} 2>&1

python ${GEN_GT_DIR}/run_flow.py small3 500 >>${RUN_LOG} 2>&1
python ${GEN_GT_DIR}/run_flow.py small3 1000 >>${RUN_LOG} 2>&1
python ${GEN_GT_DIR}/run_flow.py small3 2000 >>${RUN_LOG} 2>&1

python ${GEN_GT_DIR}/run_flow.py small4 500 >>${RUN_LOG} 2>&1
python ${GEN_GT_DIR}/run_flow.py small4 1000 >>${RUN_LOG} 2>&1
python ${GEN_GT_DIR}/run_flow.py small4 2000 >>${RUN_LOG} 2>&1

python ${GEN_GT_DIR}/run_flow.py small5 500 >>${RUN_LOG} 2>&1
python ${GEN_GT_DIR}/run_flow.py small5 1000 >>${RUN_LOG} 2>&1
python ${GEN_GT_DIR}/run_flow.py small5 2000 >>${RUN_LOG} 2>&1
