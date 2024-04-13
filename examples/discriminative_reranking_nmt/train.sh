#!/bin/bash

SELFMEM=${HOME}/Documents/eecs598-genai/SelfMemory
FAIRSEQ=${SELFMEM}/fairseq
DRNMT_DIR=${FAIRSEQ}/examples/discriminative_reranking_nmt

SOURCE_FILE=${SELFMEM}/data/ende/test_small_src.txt
TARGET_FILE=${SELFMEM}/data/ende/test_small_trg.txt
HYPO_FILE=${SELFMEM}/data/ende/test_small_hyp.txt
XLMR_DIR=${FAIRSEQ}/examples/xlmr/xlmr.base
OUTPUT_DIR=${DRNMT_DIR}/output

N=2
SPLIT=train
NUM_SHARDS=1
METRIC=bleu

# Training
EXP_DIR=${DRNMT_DIR}/exp
# An example of training the model with the config for De-En experiment in the paper.
# The config uses 16 GPUs and 50 hypotheses.
# For training with fewer number of GPUs, set
# distributed_training.distributed_world_size=k +optimization.update_freq='[x]' where x = 16/k
# For training with fewer number of hypotheses, set
# task.mt_beam=N dataset.batch_size=N dataset.required_batch_size_multiple=N
fairseq-hydra-train -m \
    --config-dir ${DRNMT_DIR}/config/ --config-name ende \
    task.data=${OUTPUT_DIR}/$METRIC/split1/ \
    task.num_data_splits=${NUM_SHARDS} \
    model.pretrained_model=${XLMR_DIR}/model.pt \
    common.user_dir=${DRNMT_DIR} \
    checkpoint.save_dir=${EXP_DIR}
