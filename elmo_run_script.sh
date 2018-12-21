#!/usr/bin/env bash

# max char len for english: 91

python -m elmoformanylangs test \
    --gpu 0 \
    --input_format plain \
    --input ./raw_data/text_only/$1 \
    --model ./model/ \
    --output_prefix ./data/$2 \
    --output_format txt \
    --output_layer -1
