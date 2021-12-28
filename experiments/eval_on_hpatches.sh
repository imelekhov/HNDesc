#!/usr/bin/bash

DATASETS_PATH=/ssd/data_tmp/
TASK=hpatches
DESCRIPTOR=hndesc

wget http://icvl.ee.ic.ac.uk/vbalnt/hpatches/hpatches-sequences-release.tar.gz -c -P ${DATASETS_PATH}/
tar -zxvf ${DATASETS_PATH}/hpatches-sequences-release.tar.gz -C ${DATASETS_PATH}/

python main.py task=${TASK} descriptor=${DESCRIPTOR} paths.datasets_home_dir=${DATASETS_PATH}