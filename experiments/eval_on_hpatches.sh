#!/usr/bin/bash

DATASETS_PATH=/ssd/data_tmp/
TASK=hpatches
DESCRIPTOR=hndesc
BACKBONE=caps
SNAPSHOT_NAME=${DESCRIPTOR}_${BACKBONE}_MP_st

wget http://icvl.ee.ic.ac.uk/vbalnt/hpatches/hpatches-sequences-release.tar.gz -c -P ${DATASETS_PATH}/
tar -zxvf ${DATASETS_PATH}/hpatches-sequences-release.tar.gz -C ${DATASETS_PATH}/

export PYTHONPATH=$PWD/../:$PYTHONPATH

python main.py task=${TASK} \
               paths.datasets_home_dir=${DATASETS_PATH} \
               descriptor=${DESCRIPTOR} \
               descriptor.descriptor_params.backbone=${BACKBONE} \
               descriptor.descriptor_params.exper_settings_name=${SNAPSHOT_NAME}
