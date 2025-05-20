#!/bin/bash

modelscope download --dataset="wangkunqing/mini_llm_pretrain_data" --local_dir="./data/sft_data" sft_data_zh.zip
unzip ./data/sft_data/sft_data_zh.zip -d ./data/sft_data
rm -rf ./data/sft_data/sft_data_zh.zip