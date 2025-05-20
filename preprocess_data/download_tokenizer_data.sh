#!/bin/bash

modelscope download --dataset="wangkunqing/mini_llm_pretrain_data" --local_dir="./data/pretrain_data" wikipedia.zip
unzip ./data/pretrain_data/wikipedia.zip -d ./data/pretrain_data
rm -rf ./data/pretrain_data/wikipedia.zip