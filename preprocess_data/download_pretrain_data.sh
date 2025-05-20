#!/bin/bash

modelscope download --dataset="wangkunqing/mini_llm_pretrain_data" --local_dir="./data/pretrain_data" pretrain_data.zip
unzip ./data/pretrain_data/pretrain_data.zip -d ./data/pretrain_data
rm -rf ./data/pretrain_data/pretrain_data.zip