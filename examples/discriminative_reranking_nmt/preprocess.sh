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

# 3. Prepare scores and BPE data.
python scripts/prep_data.py \
    --input-source ${SOURCE_FILE} \
    --input-target ${TARGET_FILE} \
    --input-hypo ${HYPO_FILE} \
    --output-dir ${OUTPUT_DIR} \
    --split $SPLIT \
    --beam $N \
    --sentencepiece-model ${XLMR_DIR}/sentencepiece.bpe.model \
    --metric $METRIC \
    --num-shards ${NUM_SHARDS}

# 4. Pre-process the data into fairseq format.
# use comma to separate if there are more than one train or valid set
for suffix in src tgt; do
    fairseq-preprocess --only-source \
        --trainpref ${OUTPUT_DIR}/$METRIC/split1/input_${suffix}/train.bpe \
        --validpref ${OUTPUT_DIR}/$METRIC/split1/input_${suffix}/valid.bpe \
        --destdir ${OUTPUT_DIR}/$METRIC/split1/input_${suffix} \
        --workers 60 \
        --srcdict ${XLMR_DIR}/dict.txt
done

for i in $(seq 2 ${NUM_SHARDS}); do
    for suffix in src tgt; do
        fairseq-preprocess --only-source \
            --trainpref ${OUTPUT_DIR}/$METRIC/split${i}/input_${suffix}/train.bpe \
            --destdir ${OUTPUT_DIR}/$METRIC/split${i}/input_${suffix} \
            --workers 60 \
            --srcdict ${XLMR_DIR}/dict.txt

        ln -s ${OUTPUT_DIR}/$METRIC/split1/input_${suffix}/valid* ${OUTPUT_DIR}/$METRIC/split${i}/input_${suffix}/.
    done

    ln -s ${OUTPUT_DIR}/$METRIC/split1/$METRIC/valid* ${OUTPUT_DIR}/$METRIC/split${i}/$METRIC/.
done
