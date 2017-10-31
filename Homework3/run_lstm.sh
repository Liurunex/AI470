#! /bin/bash

OMP_NUM_THREADS=1 THEANO_FLAGS=blas.ldflags=-lopenblas python src/py/main.py \
                  -d 200 \
                  -i 100 \
                  -o 100 \
                  -p none \
                  -u 1 \
                  -t 15,5,5,5 \
                  -c lstm \
                  -m encoderdecoder \
                  --stats-file result/stats_lstm.json \
                  --domain geoquery \
                  -k 5 \
                  --dev-seed 0 \
                  --model-seed 0 \
                  --train-data data/geo880/geo880_train600.tsv \
                  --dev-data data/geo880/geo880_test280.tsv \
                  --save-file result/params_lstm \
                  --sample-file result/sample_lstm
