#!/usr/bin/env pwsh

# 获取脚本所在目录的上级目录作为项目根路径
$SCRIPT_PATH = $PSScriptRoot
$ROOT_PATH = Split-Path -Parent $SCRIPT_PATH
$PRETRAIN_DATA_PATH = Join-Path (Join-Path $ROOT_PATH "data") "pretrain_data"
$SFT_DATA_PATH = Join-Path (Join-Path $ROOT_PATH "data") "sft_data"
$TOKENIZER_DATA_PATH = Join-Path (Join-Path $ROOT_PATH "data") "tokenizer_data"

# 清理临时文件的通用函数
function Cleanup-TempFiles {
    param([string]$Dir)
    
    # 删除 ._____temp 文件夹
    $tempDir = Join-Path $Dir "._____temp"
    if (Test-Path -PathType Container $tempDir) {
        Remove-Item -Path $tempDir -Recurse -Force -ErrorAction SilentlyContinue
    }
    
    # 删除 .msc 文件
    $mscFile = Join-Path $Dir ".msc"
    if (Test-Path -PathType Leaf $mscFile) {
        Remove-Item -Path $mscFile -Force -ErrorAction SilentlyContinue
    }
    
    # 删除 .mv 文件
    $mvFile = Join-Path $Dir ".mv"
    if (Test-Path -PathType Leaf $mvFile) {
        Remove-Item -Path $mvFile -Force -ErrorAction SilentlyContinue
    }
}


# ========== Pretrain Data Functions ==========

# 下载 OpenCSG Fineweb-Edu-Chinese-V2.1 数据集中的 4-5 得分的部分，有 9745 个 parquet 文件，可能等待时间比较久
function Download-PretrainRaw {
    Write-Host "Downloading OpenCSG Fineweb-Edu-Chinese-V2.1 dataset (4-5 score, raw data)..." -ForegroundColor Green
    Write-Host "Note: This dataset contains 9745 parquet files, download may take a long time." -ForegroundColor Yellow
    $downloadDir = Join-Path $PRETRAIN_DATA_PATH "fineweb_edu"
    
    try {
        # 直接调用命令，输出到控制台显示进度
        & modelscope download --dataset 'opencsg/Fineweb-Edu-Chinese-V2.1' --include '4_5/*.parquet' --local_dir $downloadDir

        if ($LASTEXITCODE -eq 0) {
            # 移动所有下载的parquet文件
            Write-Host "Moving data to fineweb_edu directory..." -ForegroundColor Yellow
            $parquetFiles = Get-ChildItem -Path (Join-Path $downloadDir "4_5") -Filter "*.parquet" -ErrorAction SilentlyContinue
            foreach ($file in $parquetFiles) {
                $destination = Join-Path $downloadDir $file.Name
                Move-Item -Path $file.FullName -Destination $destination -Force -ErrorAction SilentlyContinue
            }
            
            # 删除 4_5 文件夹（现在应该为空）
            $subDir = Join-Path $downloadDir "4_5"
            if (Test-Path -PathType Container $subDir) {
                Remove-Item -Path $subDir -Recurse -Force -ErrorAction SilentlyContinue
            }
            
            Cleanup-TempFiles -Dir $downloadDir
            
            Write-Host "OpenCSG Fineweb-Edu-Chinese-V2.1 dataset download completed." -ForegroundColor Green
            return $true
        } else {
            Write-Host "Error: Download command failed with exit code $LASTEXITCODE" -ForegroundColor Red
            return $false
        }
    } catch {
        Write-Host "Error: $_" -ForegroundColor Red
        return $false
    }
}


# 下载 OpenCSG Fineweb-Edu-Chinese-V2.1 数据集的 20% 采样子集
function Download-PretrainSampled {
    Write-Host "Downloading wangkunqing/mini_llm_dataset dataset (20% sampled)..." -ForegroundColor Green
    $downloadDir = Join-Path $PRETRAIN_DATA_PATH "fineweb_edu_sampled_20_percent"
    
    try {
        # 直接调用命令，输出到控制台显示进度
        & modelscope download --dataset 'wangkunqing/mini_llm_dataset' --include 'fineweb_edu_sampled_20_percent.zip' --local_dir $downloadDir
        
        if ($LASTEXITCODE -eq 0) {
            Cleanup-TempFiles -Dir $downloadDir
            
            # 解压文件
            $zipFile = Join-Path $downloadDir "fineweb_edu_sampled_20_percent.zip"
            if (Test-Path -PathType Leaf $zipFile) {
                Expand-Archive -Path $zipFile -DestinationPath $downloadDir -Force
                Remove-Item -Path $zipFile -Force -ErrorAction SilentlyContinue
            }
            
            # 将子目录中的所有内容上移一层
            $subDir = Join-Path $downloadDir "fineweb_edu_sampled_20_percent"
            if (Test-Path -PathType Container $subDir) {
                Write-Host "Moving contents from subdirectory to parent directory..." -ForegroundColor Yellow
                
                Get-ChildItem -Path $subDir -Force | ForEach-Object {
                    $destination = Join-Path $downloadDir $_.Name
                    Move-Item -Path $_.FullName -Destination $destination -Force -ErrorAction SilentlyContinue
                }
                
                if (Test-Path -PathType Container $subDir) {
                    Remove-Item -Path $subDir -Recurse -Force -ErrorAction SilentlyContinue
                }
                
                Write-Host "Contents moved successfully." -ForegroundColor Green
            }
            
            Write-Host "wangkunqing/mini_llm_dataset dataset download completed." -ForegroundColor Green
            return $true
        } else {
            Write-Host "Error: Download command failed with exit code $LASTEXITCODE" -ForegroundColor Red
            return $false
        }
    } catch {
        Write-Host "Error: $_" -ForegroundColor Red
        return $false
    }
}


# 下载经过 mini_tokenizer 进行分词处理的 20% 采样子集
function Download-PretrainSampledTokenized {
    Write-Host "Downloading wangkunqing/mini_llm_dataset dataset (20% sampled, tokenized by mini_tokenizer)..." -ForegroundColor Green
    $downloadDir = Join-Path $PRETRAIN_DATA_PATH "bin"
    
    try {
        # 直接调用命令，输出到控制台显示进度
        & modelscope download --dataset 'wangkunqing/mini_llm_dataset' --include 'fineweb_edu_sampled_20_percent.bin' --local_dir $downloadDir
        
        if ($LASTEXITCODE -eq 0) {
            Cleanup-TempFiles -Dir $downloadDir
            
            Write-Host "wangkunqing/mini_llm_dataset dataset download completed." -ForegroundColor Green
            return $true
        } else {
            Write-Host "Error: Download command failed with exit code $LASTEXITCODE" -ForegroundColor Red
            return $false
        }
    } catch {
        Write-Host "Error: $_" -ForegroundColor Red
        return $false
    }
}


# 下载经过 mini_tokenizer 进行分词处理的 DeepCtrl 数据集
function Download-PretrainDeepctrlTokenized {
    Write-Host "Downloading wangkunqing/mini_llm_dataset dataset (DeepCtrl, tokenized by mini_tokenizer)..." -ForegroundColor Green
    $downloadDir = Join-Path $PRETRAIN_DATA_PATH "bin"
    
    try {
        # 直接调用命令，输出到控制台显示进度
        & modelscope download --dataset 'wangkunqing/mini_llm_dataset' --include 'deepctrl.bin' --local_dir $downloadDir
        
        if ($LASTEXITCODE -eq 0) {
            Cleanup-TempFiles -Dir $downloadDir
            
            Write-Host "wangkunqing/mini_llm_dataset dataset download completed." -ForegroundColor Green
            return $true
        } else {
            Write-Host "Error: Download command failed with exit code $LASTEXITCODE" -ForegroundColor Red
            return $false
        }
    } catch {
        Write-Host "Error: $_" -ForegroundColor Red
        return $false
    }
}


# ========== SFT Data Functions ==========

# 下载 SFT 数据集
function Download-SftData {
    Write-Host "Downloading deepctrl/deepctrl-sft-data dataset..." -ForegroundColor Green
    $downloadDir = Join-Path $SFT_DATA_PATH "deepctrl"
    
    try {
        # 直接调用命令，输出到控制台显示进度
        & modelscope download --dataset 'deepctrl/deepctrl-sft-data' --include 'sft_data_zh.jsonl' --local_dir $downloadDir

        if ($LASTEXITCODE -eq 0) {
            Cleanup-TempFiles -Dir $downloadDir
            
            Write-Host "deepctrl/deepctrl-sft-data dataset download completed." -ForegroundColor Green
            return $true
        } else {
            Write-Host "Error: Download command failed with exit code $LASTEXITCODE" -ForegroundColor Red
            return $false
        }
    } catch {
        Write-Host "Error: $_" -ForegroundColor Red
        return $false
    }
}


# 下载处理过的 SFT Parquet 数据集
function Download-SftParquet {
    Write-Host "Downloading wangkunqing/mini_llm_dataset dataset (processed parquet SFT dataset)..." -ForegroundColor Green
    $downloadDir = $SFT_DATA_PATH
    
    try {
        # 直接调用命令，输出到控制台显示进度
        & modelscope download --dataset 'wangkunqing/mini_llm_dataset' --include 'sft_parquet.zip' --local_dir $downloadDir
        
        if ($LASTEXITCODE -eq 0) {
            Cleanup-TempFiles -Dir $downloadDir
            
            # 解压文件到目标目录
            $zipFile = Join-Path $downloadDir "sft_parquet.zip"
            if (Test-Path -PathType Leaf $zipFile) {
                Write-Host "Extracting sft_parquet.zip to parquet directory..." -ForegroundColor Yellow
                Expand-Archive -Path $zipFile -DestinationPath $downloadDir -Force -ErrorAction Stop
                Remove-Item -Path $zipFile -Force -ErrorAction SilentlyContinue
                Write-Host "wangkunqing/mini_llm_dataset dataset download and extraction completed." -ForegroundColor Green
                return $true
            } else {
                Write-Host "Error: sft_parquet.zip not found after download" -ForegroundColor Red
                return $false
            }
        } else {
            Write-Host "Error: Download command failed with exit code $LASTEXITCODE" -ForegroundColor Red
            return $false
        }
    } catch {
        Write-Host "Error: $_" -ForegroundColor Red
        return $false
    }
}


# ========== Tokenizer Data Functions ==========

# 下载 Tokenizer 训练数据
function Download-TokenizerData {
    Write-Host "Downloading wangkunqing/mini_llm_dataset dataset (5% sampled for tokenizer)..." -ForegroundColor Green
    $downloadDir = Join-Path $TOKENIZER_DATA_PATH "fineweb_edu_sampled_5_percent"
    
    try {
        # 直接调用命令，输出到控制台显示进度
        & modelscope download --dataset 'wangkunqing/mini_llm_dataset' --include 'fineweb_edu_sampled_5_percent.zip' --local_dir $downloadDir
        
        if ($LASTEXITCODE -eq 0) {
            Cleanup-TempFiles -Dir $downloadDir
            
            # 解压文件
            $zipFile = Join-Path $downloadDir "fineweb_edu_sampled_5_percent.zip"
            if (Test-Path -PathType Leaf $zipFile) {
                Expand-Archive -Path $zipFile -DestinationPath $downloadDir -Force
                Remove-Item -Path $zipFile -Force -ErrorAction SilentlyContinue
            }
            
            # 将子目录中的所有内容上移一层
            $subDir = Join-Path $downloadDir "fineweb_edu_sampled_5_percent"
            if (Test-Path -PathType Container $subDir) {
                Write-Host "Moving contents from subdirectory to parent directory..." -ForegroundColor Yellow
                
                Get-ChildItem -Path $subDir -Force | ForEach-Object {
                    $destination = Join-Path $downloadDir $_.Name
                    Move-Item -Path $_.FullName -Destination $destination -Force -ErrorAction SilentlyContinue
                }
                
                if (Test-Path -PathType Container $subDir) {
                    Remove-Item -Path $subDir -Recurse -Force -ErrorAction SilentlyContinue
                }
                
                Write-Host "Contents moved successfully." -ForegroundColor Green
            }
            
            Write-Host "wangkunqing/mini_llm_dataset dataset download completed." -ForegroundColor Green
            return $true
        } else {
            Write-Host "Error: Download command failed with exit code $LASTEXITCODE" -ForegroundColor Red
            return $false
        }
    } catch {
        Write-Host "Error: $_" -ForegroundColor Red
        return $false
    }
}


# ========== Dataset Configuration ==========

# 数据集配置数组
# 格式: @{Id="1"; Name="..."; Description="..."; Function={...}}
$Datasets = @(
    @{
        Id = "1"
        Name = "Tokenizer: 5% Sampled Dataset"
        Description = "Download 5% sampled Fineweb-Edu-Chinese-V2.1 dataset (~3.4 GB for tokenizer training)"
        Function = { Download-TokenizerData }
    },
    @{
        Id = "2"
        Name = "Pretrain: Original Dataset"
        Description = "Download original Fineweb-Edu-Chinese-V2.1 dataset (the subset with scores 4-5, 9745 parquet files, ~70 GB)"
        Function = { Download-PretrainRaw }
    },
    @{
        Id = "3"
        Name = "Pretrain: 20% Sampled Dataset"
        Description = "Download 20% sampled Fineweb-Edu-Chinese-V2.1 dataset (~14 GB for faster pretraining)"
        Function = { Download-PretrainSampled }
    },
    @{
        Id = "4"
        Name = "Pretrain: Tokenized 20% Sampled Dataset"
        Description = "Download tokenized 20% sampled Fineweb-Edu-Chinese-V2.1 dataset (~10 GB for faster pretraining, tokenized by mini_tokenizer)"
        Function = { Download-PretrainSampledTokenized }
    },
    @{
        Id = "5"
        Name = "Pretrain: Tokenized DeepCtrl Dataset"
        Description = "Download tokenized DeepCtrl dataset (~4 GB for pretraining, tokenized by mini_tokenizer)"
        Function = { Download-PretrainDeepctrlTokenized }
    },
    @{
        Id = "6"
        Name = "SFT: DeepCtrl Dataset"
        Description = "Download original DeepCtrl dataset (~16 GB for SFT)"
        Function = { Download-SftData }
    },
    @{
        Id = "7"
        Name = "SFT: Parquet Dataset"
        Description = "Download processed parquet SFT dataset (~3.7 GB for SFT)"
        Function = { Download-SftParquet }
    }
)


# ========== Menu Functions ==========

# 显示菜单
function Show-Menu {
    Clear-Host
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "  Data Download Menu" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host ""
    
    foreach ($dataset in $Datasets) {
        Write-Host "[$($dataset.Id)] " -NoNewline -ForegroundColor Green
        Write-Host "$($dataset.Name)" -NoNewline
        Write-Host ""
        Write-Host "    $($dataset.Description)" -ForegroundColor Gray
        Write-Host ""
    }
    
    Write-Host "[q/Q] " -NoNewline -ForegroundColor Yellow
    Write-Host "Quit"
    Write-Host ""
    Write-Host "Please select an option: " -NoNewline -ForegroundColor Cyan
}


# ========== Main Function ==========

# 主函数
function Main {
    while ($true) {
        Show-Menu
        $choice = Read-Host
        
        # 处理退出
        if ([string]::IsNullOrWhiteSpace($choice) -or $choice -match "^[qQ]$") {
            Write-Host "Exiting..." -ForegroundColor Yellow
            exit 0
        }
        
        # 查找匹配的数据集
        $found = $false
        foreach ($dataset in $Datasets) {
            if ($choice -eq $dataset.Id) {
                $found = $true
                Write-Host ""
                Write-Host "Selected: $($dataset.Name)" -ForegroundColor Blue
                Write-Host "Description: $($dataset.Description)" -ForegroundColor Blue
                Write-Host ""
                
                # 执行下载函数
                try {
                    # 直接调用函数，不捕获输出，避免解析错误并保留下载进度显示
                    $result = & $dataset.Function
                    
                    Write-Host ""
                    # 检查函数返回值（布尔值）
                    if ($result -eq $true) {
                        Write-Host "Download completed successfully!" -ForegroundColor Green
                    } else {
                        Write-Host "Download failed!" -ForegroundColor Red
                    }
                } catch {
                    Write-Host "Error during download: $_" -ForegroundColor Red
                }
                
                Write-Host ""
                Write-Host "Press Enter to continue..." -ForegroundColor Yellow
                Read-Host | Out-Null
                break
            }
        }
        
        if (-not $found) {
            Write-Host "Invalid option. Please try again." -ForegroundColor Red
            Start-Sleep -Seconds 1
        }
    }
}


# 运行主函数
Main

