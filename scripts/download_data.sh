#!/bin/bash

# 获取脚本所在目录的上级目录作为项目根路径
SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_PATH="$(dirname "$SCRIPT_PATH")"
PRETRAIN_DATA_PATH="$ROOT_PATH/data/pretrain_data"
SFT_DATA_PATH="$ROOT_PATH/data/sft_data"
TOKENIZER_DATA_PATH="$ROOT_PATH/data/tokenizer_data"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# 清理临时文件的通用函数
cleanup_temp_files() {
    local dir="$1"
    
    # 删除 ._____temp 文件夹
    if [ -d "$dir/._____temp" ]; then
        rm -rf "$dir/._____temp"
    fi
    
    # 删除 .msc 文件
    if [ -f "$dir/.msc" ]; then
        rm -f "$dir/.msc"
    fi
    
    # 删除 .mv 文件
    if [ -f "$dir/.mv" ]; then
        rm -f "$dir/.mv"
    fi
}


# ========== Pretrain Data Functions ==========

# 下载 OpenCSG Fineweb-Edu-Chinese-V2.1 数据集中的 4-5 得分的部分，有 9745 个 parquet 文件，可能等待时间比较久
download_pretrain_raw() {
    echo -e "${GREEN}Downloading OpenCSG Fineweb-Edu-Chinese-V2.1 dataset (4-5 score, raw data)...${NC}"
    echo -e "${YELLOW}Note: This dataset contains 9745 parquet files, download may take a long time.${NC}"
    local download_dir="$PRETRAIN_DATA_PATH/fineweb_edu"
    
    modelscope download \
        --dataset 'opencsg/Fineweb-Edu-Chinese-V2.1' \
        --include '4_5/*.parquet' \
        --local_dir "$download_dir"
    
    if [ $? -eq 0 ]; then
        # 移动所有下载的parquet文件
        echo -e "${YELLOW}Moving data to fineweb_edu directory...${NC}"
        for file in "$download_dir/4_5"/*.parquet; do
            if [ -f "$file" ]; then
                filename=$(basename "$file")
                mv "$file" "$download_dir/$filename"
            fi
        done
        
        # 删除 4_5 文件夹（现在应该为空）
        if [ -d "$download_dir/4_5" ]; then
            rm -rf "$download_dir/4_5"
        fi
        
        cleanup_temp_files "$download_dir"
        
        echo -e "${GREEN}OpenCSG Fineweb-Edu-Chinese-V2.1 dataset download completed.${NC}"
        return 0
    else
        echo -e "${RED}Error: Download command failed with exit code $?${NC}"
        return 1
    fi
}


# 下载 OpenCSG Fineweb-Edu-Chinese-V2.1 数据集的 20% 采样子集
download_pretrain_sampled() {
    echo -e "${GREEN}Downloading wangkunqing/mini_llm_dataset dataset (20% sampled)...${NC}"
    local download_dir="$PRETRAIN_DATA_PATH/fineweb_edu_sampled_20_percent"
    
    modelscope download \
        --dataset 'wangkunqing/mini_llm_dataset' \
        --include 'fineweb_edu_sampled_20_percent.zip' \
        --local_dir "$download_dir"
    
    if [ $? -eq 0 ]; then
        cleanup_temp_files "$download_dir"
        
        # 解压文件
        unzip "$download_dir/fineweb_edu_sampled_20_percent.zip" -d "$download_dir/" 2>/dev/null
        rm -f "$download_dir/fineweb_edu_sampled_20_percent.zip"
        
        # 将子目录中的所有内容上移一层
        local subdir="$download_dir/fineweb_edu_sampled_20_percent"
        if [ -d "$subdir" ]; then
            echo -e "${YELLOW}Moving contents from subdirectory to parent directory...${NC}"
            shopt -s dotglob
            mv "$subdir"/* "$download_dir/" 2>/dev/null
            shopt -u dotglob
            rmdir "$subdir" 2>/dev/null || rm -rf "$subdir"
            echo -e "${GREEN}Contents moved successfully.${NC}"
        fi
        
        echo -e "${GREEN}wangkunqing/mini_llm_dataset dataset download completed.${NC}"
        return 0
    else
        echo -e "${RED}Error: Download command failed with exit code $?${NC}"
        return 1
    fi
}


# 下载经过 mini_tokenizer 进行分词处理的 20% 采样子集
download_pretrain_sampled_tokenized() {
    echo -e "${GREEN}Downloading wangkunqing/mini_llm_dataset dataset (20% sampled, tokenized by mini_tokenizer)...${NC}"
    local download_dir="$PRETRAIN_DATA_PATH/bin"
    
    modelscope download \
        --dataset 'wangkunqing/mini_llm_dataset' \
        --include 'fineweb_edu_sampled_20_percent.bin' \
        --local_dir "$download_dir"
    
    if [ $? -eq 0 ]; then
        cleanup_temp_files "$download_dir"
        
        echo -e "${GREEN}wangkunqing/mini_llm_dataset dataset download completed.${NC}"
        return 0
    else
        echo -e "${RED}Error: Download command failed with exit code $?${NC}"
        return 1
    fi
}


# 下载经过 mini_tokenizer 进行分词处理的 DeepCtrl 数据集
download_pretrain_deepctrl_tokenized() {
    echo -e "${GREEN}Downloading wangkunqing/mini_llm_dataset dataset (DeepCtrl, tokenized by mini_tokenizer)...${NC}"
    local download_dir="$PRETRAIN_DATA_PATH/bin"
    
    modelscope download \
        --dataset 'wangkunqing/mini_llm_dataset' \
        --include 'deepctrl.bin' \
        --local_dir "$download_dir"
    
    if [ $? -eq 0 ]; then
        cleanup_temp_files "$download_dir"
        
        echo -e "${GREEN}wangkunqing/mini_llm_dataset dataset download completed.${NC}"
        return 0
    else
        echo -e "${RED}Error: Download command failed with exit code $?${NC}"
        return 1
    fi
}


# ========== SFT Data Functions ==========

# 下载 SFT 数据集
download_sft_data() {
    echo -e "${GREEN}Downloading deepctrl/deepctrl-sft-data dataset...${NC}"
    local download_dir="$SFT_DATA_PATH/deepctrl"
    
    modelscope download \
        --dataset 'deepctrl/deepctrl-sft-data' \
        --include 'sft_data_zh.jsonl' \
        --local_dir "$download_dir"
    
    if [ $? -eq 0 ]; then
        cleanup_temp_files "$download_dir"
        
        echo -e "${GREEN}deepctrl/deepctrl-sft-data dataset download completed.${NC}"
        return 0
    else
        echo -e "${RED}Error: Download command failed with exit code $?${NC}"
        return 1
    fi
}


# 下载处理过的 SFT Parquet 数据集
download_sft_parquet() {
    echo -e "${GREEN}Downloading wangkunqing/mini_llm_dataset dataset (processed parquet SFT dataset)...${NC}"
    local download_dir="$SFT_DATA_PATH"

    modelscope download \
        --dataset 'wangkunqing/mini_llm_dataset' \
        --include 'sft_parquet.zip' \
        --local_dir "$download_dir"
    
    if [ $? -eq 0 ]; then
        cleanup_temp_files "$download_dir"
        
        # 解压文件到目标目录
        if [ -f "$download_dir/sft_parquet.zip" ]; then
            echo -e "${YELLOW}Extracting sft_parquet.zip to parquet directory...${NC}"
            unzip -o "$download_dir/sft_parquet.zip" -d "$download_dir/" 2>/dev/null
            
            if [ $? -eq 0 ]; then
                # 删除 zip 文件
                rm -f "$download_dir/sft_parquet.zip"
                
                echo -e "${GREEN}wangkunqing/mini_llm_dataset dataset download and extraction completed.${NC}"
                return 0
            else
                echo -e "${RED}Error: Failed to extract sft_parquet.zip${NC}"
                return 1
            fi
        else
            echo -e "${RED}Error: sft_parquet.zip not found after download${NC}"
            return 1
        fi
    else
        echo -e "${RED}Error: Download command failed with exit code $?${NC}"
        return 1
    fi
}


# ========== Tokenizer Data Functions ==========

# 下载 Tokenizer 训练数据
download_tokenizer_data() {
    echo -e "${GREEN}Downloading wangkunqing/mini_llm_dataset dataset (5% sampled for tokenizer)...${NC}"
    local download_dir="$TOKENIZER_DATA_PATH/fineweb_edu_sampled_5_percent"
    
    modelscope download \
        --dataset 'wangkunqing/mini_llm_dataset' \
        --include 'fineweb_edu_sampled_5_percent.zip' \
        --local_dir "$download_dir"
    
    if [ $? -eq 0 ]; then
        cleanup_temp_files "$download_dir"
        
        # 解压文件
        unzip "$download_dir/fineweb_edu_sampled_5_percent.zip" -d "$download_dir/" 2>/dev/null
        rm -f "$download_dir/fineweb_edu_sampled_5_percent.zip"
        
        # 将子目录中的所有内容上移一层
        local subdir="$download_dir/fineweb_edu_sampled_5_percent"
        if [ -d "$subdir" ]; then
            echo -e "${YELLOW}Moving contents from subdirectory to parent directory...${NC}"
            shopt -s dotglob
            mv "$subdir"/* "$download_dir/" 2>/dev/null
            shopt -u dotglob
            rmdir "$subdir" 2>/dev/null || rm -rf "$subdir"
            echo -e "${GREEN}Contents moved successfully.${NC}"
        fi
        
        echo -e "${GREEN}wangkunqing/mini_llm_dataset dataset download completed.${NC}"
        return 0
    else
        echo -e "${RED}Error: Download command failed with exit code $?${NC}"
        return 1
    fi
}


# ========== Dataset Configuration ==========

# 数据集配置数组
# 格式: "ID|Name|Description|Function"
declare -a DATASETS=(
    "1|Tokenizer: 5% Sampled Dataset|Download 5% sampled Fineweb-Edu-Chinese-V2.1 dataset (~3.4 GB for tokenizer training)|download_tokenizer_data"
    "2|Pretrain: Original Dataset|Download original Fineweb-Edu-Chinese-V2.1 dataset (the subset with scores 4-5, 9745 parquet files, ~70 GB)|download_pretrain_raw"
    "3|Pretrain: 20% Sampled Dataset|Download 20% sampled Fineweb-Edu-Chinese-V2.1 dataset (~14 GB for faster pretraining)|download_pretrain_sampled"
    "4|Pretrain: Tokenized 20% Sampled Dataset|Download tokenized 20% sampled Fineweb-Edu-Chinese-V2.1 dataset (~10 GB for faster pretraining, tokenized by mini_tokenizer)|download_pretrain_sampled_tokenized"
    "5|Pretrain: Tokenized DeepCtrl Dataset|Download tokenized DeepCtrl dataset (~4 GB for pretraining, tokenized by mini_tokenizer)|download_pretrain_deepctrl_tokenized"
    "6|SFT: DeepCtrl Dataset|Download original DeepCtrl dataset (~16 GB for SFT)|download_sft_data"
    "7|SFT: Parquet Dataset|Download processed parquet SFT dataset (~3.7 GB for SFT)|download_sft_parquet"
)


# ========== Menu Functions ==========

# 显示菜单
show_menu() {
    clear
    echo -e "${CYAN}${BOLD}========================================${NC}"
    echo -e "${CYAN}${BOLD}  Data Download Menu${NC}"
    echo -e "${CYAN}${BOLD}========================================${NC}"
    echo ""
    
    for dataset in "${DATASETS[@]}"; do
        IFS='|' read -r id name desc func <<< "$dataset"
        echo -e "${GREEN}[$id]${NC} ${BOLD}$name${NC}"
        echo -e "    $desc"
        echo ""
    done
    
    echo -e "${YELLOW}[q/Q]${NC} Quit"
    echo ""
    echo -e "${CYAN}Please select an option:${NC} "
}


# ========== Main Function ==========

# 主函数
main() {
    while true; do
        show_menu
        read -r choice
        
        # 处理退出
        if [[ "$choice" =~ ^[qQ]$ ]] || [[ "$choice" == "" ]]; then
            echo -e "${YELLOW}Exiting...${NC}"
            exit 0
        fi
        
        # 查找匹配的数据集
        local found=false
        for dataset in "${DATASETS[@]}"; do
            IFS='|' read -r id name desc func <<< "$dataset"
            if [ "$choice" == "$id" ]; then
                found=true
                echo ""
                echo -e "${BLUE}Selected: $name${NC}"
                echo -e "${BLUE}Description: $desc${NC}"
                echo ""
                
                # 执行下载函数
                if $func; then
                    echo ""
                    echo -e "${GREEN}${BOLD}Download completed successfully!${NC}"
                else
                    echo ""
                    echo -e "${RED}${BOLD}Download failed!${NC}"
                fi
                
                echo ""
                echo -e "${YELLOW}Press Enter to continue...${NC}"
                read -r
                break
            fi
        done
        
        if [ "$found" = false ]; then
            echo -e "${RED}Invalid option. Please try again.${NC}"
            sleep 1
        fi
    done
}


# 运行主函数
main

