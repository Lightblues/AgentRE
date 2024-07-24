#!/usr/bin/env bash

PROJECT_DIR=./src
cd ${PROJECT_DIR}

MODEL=REACT_FSL
DATA=SciERC           # DuIE2.0
EXP_NAME=REACL_FSL
# CONFIG_LIST=react_woM

# set default values
if [ -z ${EXP_NAME} ]; then
    EXP_NAME=${MODEL}
fi
if [ -z ${CONFIG_LIST} ]; then
    CONFIG_LIST=""
fi

CMD="python main.py --logname=${EXP_NAME} --model=${MODEL} --data=${DATA} --config_list=${CONFIG_LIST}"
echo ${CMD}
${CMD}
