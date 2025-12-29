#!/bin/bash

# 此脚本用于程序化的一件配置，可能存在一定逻辑疏漏，应该能适配大多数使用cuda的情况
# 前三步分别为检测系统信息、安装uv、通过uv同步环境，尤其是特定的torch版本
# 之后的环境配置根据项目的不同进行相应的配置即可

set -e  # 遇到错误时退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'
BOLD='\033[1m'
NC='\033[0m'  # No Color

# 符号定义
CHECK_MARK="✅"
CROSS_MARK="❌"
INFO_MARK="ℹ️"
WARNING_MARK="⚠️"
RUNNING_MARK="⚡"
SYSTEM_MARK="🖥️"
GAME_MARK="🎮"
MEMORY_MARK="💾"
PACKAGE_MARK="📦"
ROCKET_MARK="🚀"
CHART_MARK="📊"
TOOL_MARK="🛠️"
FIRE_MARK="🔥"
FROZZEN_MARK="❄️"
SETTING_MARK="⚙️"
LIGHT_BULB_MARK="💡"
BOOK_MARK="📚"
LINK_MARK="🔗"
SUCCESS_MARK="🎉"

# 输出函数定义
print_phase() {
    echo -e "${MAGENTA}${BOLD}$1${NC}"
}

print_info() {
    echo -e "${BLUE}${INFO_MARK}   $1${NC}"
}

print_success() {
    echo -e "${GREEN}${CHECK_MARK}  $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}${WARNING_MARK}  $1${NC}"
}

print_error() {
    echo -e "${RED}${CROSS_MARK}  $1${NC}"
}

print_running() {
    echo -e "${WHITE}$1${NC}"
}

# 展示系统信息
show_detection_summary() {
    print_info "System Information:"
    echo -e "${GREEN}Operating System: $DETECTED_OS${NC}"
    echo -e "${GREEN}Kernel Version: $DETECTED_KERNEL${NC}"
    echo -e "${GREEN}System Architecture: $DETECTED_ARCH${NC}"
    
    print_info "CUDA Information:"
    if [ "$CUDA_AVAILABLE" = "true" ]; then
        echo -e "${GREEN}CUDA Version: $DETECTED_CUDA${NC}"
        if [ ! -z "$DETECTED_CUDA_RUNTIME" ]; then
            echo -e "${GREEN}Runtime Version: $DETECTED_CUDA_RUNTIME${NC}"
        fi
    else
        echo -e "${RED}CUDA not installed or unavailable${NC}"
    fi

    print_info "GPU Information:"
    if [ "$GPU_AVAILABLE" = "true" ]; then
        echo -e "${GREEN}GPU Count: $DETECTED_GPU_COUNT${NC}"
        echo -e "${GREEN}GPU Details:${NC}"
        echo "$GPU_DETAILS" | while IFS= read -r line; do
            if [ ! -z "$line" ]; then
                echo -e "${GREEN}  - $line${NC}"
            fi
        done
    else
        echo -e "${RED}No NVIDIA GPU detected${NC}"
        if [ ! -z "$OTHER_GPU_INFO" ]; then
            echo -e "${YELLOW}Other Display Devices:${NC}"
            echo -e "${YELLOW}  $OTHER_GPU_INFO${NC}"
        fi
    fi
}

# 检测系统版本
detect_system_info() {
    # Detect operating system
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        DETECTED_OS="$NAME $VERSION"
    else
        DETECTED_OS=$(uname -s -r)
    fi
    
    # Detect kernel version
    DETECTED_KERNEL=$(uname -r)
    
    # Detect architecture
    DETECTED_ARCH=$(uname -m)
}

# 检测cuda版本
# 优先使用 nvidia-smi 检测 CUDA 运行时版本，因为安装 PyTorch 主要需要运行时版本
detect_cuda_info() {
    CUDA_AVAILABLE="false"
    DETECTED_CUDA=""
    DETECTED_CUDA_RUNTIME=""
    
    # 优先使用 nvidia-smi 检测 CUDA 运行时版本（这是安装 PyTorch 最需要的）
    if command -v nvidia-smi &> /dev/null; then
        DETECTED_CUDA_RUNTIME=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')
        if [ ! -z "$DETECTED_CUDA_RUNTIME" ]; then
            DETECTED_CUDA="$DETECTED_CUDA_RUNTIME"
            CUDA_AVAILABLE="true"
        fi
    fi
    
    # 如果没有通过 nvidia-smi 检测到，尝试其他方法作为备选
    if [ "$CUDA_AVAILABLE" = "false" ]; then
        if command -v nvcc &> /dev/null; then
            DETECTED_CUDA=$(nvcc --version | grep "release" | sed 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/')
            if [ ! -z "$DETECTED_CUDA" ]; then
                CUDA_AVAILABLE="true"
            fi
        elif [ -f /usr/local/cuda/version.txt ]; then
            DETECTED_CUDA=$(cat /usr/local/cuda/version.txt | grep "CUDA Version" | awk '{print $3}')
            if [ ! -z "$DETECTED_CUDA" ]; then
                CUDA_AVAILABLE="true"
            fi
        fi
    fi
}

# 检测GPU信息
detect_gpu_info() {
    GPU_AVAILABLE="false"
    DETECTED_GPU_COUNT="0"
    GPU_DETAILS=""
    OTHER_GPU_INFO=""
    
    if command -v nvidia-smi &> /dev/null; then
        # Get GPU count
        DETECTED_GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
        GPU_AVAILABLE="true"
        
        # Get GPU detailed information
        GPU_DETAILS=$(nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader,nounits | while IFS=',' read -r index name memory; do
            index=$(echo $index | xargs)
            name=$(echo $name | xargs)
            memory=$(echo $memory | xargs)
            echo "GPU $index: $name (${memory}MB)"
        done)
    else
        # Try to detect other GPUs
        if command -v lspci &> /dev/null; then
            OTHER_GPU_INFO=$(lspci | grep -i "vga\|3d\|display" | head -1)
        fi
    fi
}

# 检测uv是否安装
detect_uv_installation() {
    UV_AVAILABLE="false"
    DETECTED_UV_VERSION=""
    
    if command -v uv &> /dev/null; then
        DETECTED_UV_VERSION=$(uv --version | awk '{print $2}')
        UV_AVAILABLE="true"
    fi
}

# 安装uv
install_uv() {
    print_running "Starting uv installation..."
    
    # Check if pip is available
    if ! command -v pip &> /dev/null && ! command -v pip3 &> /dev/null; then
        print_error "pip not installed, attempting to install Python and pip..."
        if command -v apt &> /dev/null; then
            sudo apt update && sudo apt install -y python3 python3-pip
        elif command -v yum &> /dev/null; then
            sudo yum install -y python3 python3-pip
        elif command -v dnf &> /dev/null; then
            sudo dnf install -y python3 python3-pip
        else
            print_error "Cannot automatically install pip, please install Python and pip manually then retry again."
            exit 1
        fi
    fi
    
    # Determine which pip command to use
    PIP_CMD="pip"
    if command -v pip3 &> /dev/null; then
        PIP_CMD="pip3"
    fi
    
    # Install uv using pip
    print_running "Installing uv using $PIP_CMD..."
    if $PIP_CMD install uv; then
        # Check if uv is in PATH
        if command -v uv &> /dev/null; then
            UV_VERSION=$(uv --version | awk '{print $2}')
            print_success "uv installation successful, version: $UV_VERSION"
        else
            print_error "uv is not in PATH, please add it to PATH and retry again."
            exit 1
        fi
    else
        print_error "uv installation failed"
        exit 1
    fi
}

# 使用uv同步环境，根据cuda版本选择不同的extra，从而安装相应的torch
sync_env() {
    print_running "Determining appropriate torch installation..."
    # 确定要安装的extra
    EXTRA_TO_INSTALL="cpu"  # 默认使用CPU版本
    
    if [ "$CUDA_AVAILABLE" = "true" ] && [ ! -z "$DETECTED_CUDA" ]; then
        # 提取CUDA主版本号（如11.8 -> 118）
        CUDA_MAJOR=$(echo $DETECTED_CUDA | sed 's/\.//g' | cut -c1-3)
        
        print_info "Detected CUDA version: $DETECTED_CUDA (version code: $CUDA_MAJOR)"
        
        # 根据CUDA版本选择兼容的最高版本
        if [ "$CUDA_MAJOR" -ge "128" ]; then
            EXTRA_TO_INSTALL="cu128"
        elif [ "$CUDA_MAJOR" -ge "126" ]; then
            EXTRA_TO_INSTALL="cu126"
        elif [ "$CUDA_MAJOR" -ge "124" ]; then
            EXTRA_TO_INSTALL="cu124"
        elif [ "$CUDA_MAJOR" -ge "121" ]; then
            EXTRA_TO_INSTALL="cu121"
        elif [ "$CUDA_MAJOR" -ge "118" ]; then
            EXTRA_TO_INSTALL="cu118"
        else
            print_warning "CUDA version $DETECTED_CUDA is not supported, falling back to CPU version"
            EXTRA_TO_INSTALL="cpu"
        fi
    else
        print_info "CUDA not available, using CPU version"
    fi
    
    print_info "Selected installation target: $EXTRA_TO_INSTALL"
    
    # 执行uv sync
    print_running "Running: uv sync --extra $EXTRA_TO_INSTALL"
    if uv sync --extra "$EXTRA_TO_INSTALL"; then
        print_success "Environment synchronization completed with $EXTRA_TO_INSTALL support"
    else
        print_error "Environment synchronization failed"
        exit 1
    fi
}

# 主函数
main() {

    echo -e "${MAGENTA}${BOLD}"
    echo "  ███╗   ███╗ ██╗ ███╗   ██╗ ██╗        ██╗      ██╗      ███╗   ███╗"
    echo "  ████╗ ████║ ██║ ████╗  ██║ ██║        ██║      ██║      ████╗ ████║"
    echo "  ██╔████╔██║ ██║ ██╔██╗ ██║ ██║ █████╗ ██║      ██║      ██╔████╔██║"
    echo "  ██║╚██╔╝██║ ██║ ██║╚██╗██║ ██║ ╚════╝ ██║      ██║      ██║╚██╔╝██║"
    echo "  ██║ ╚═╝ ██║ ██║ ██║ ╚████║ ██║        ███████╗ ███████╗ ██║ ╚═╝ ██║"
    echo "  ╚═╝     ╚═╝ ╚═╝ ╚═╝  ╚═══╝ ╚═╝        ╚══════╝ ╚══════╝ ╚═╝     ╚═╝"
    echo -e "${NC}"
    echo -e "${MAGENTA}${BOLD}"
    echo "======================= Running Setup Script ======================="
    echo -e "${NC}"
    
    sleep 1
    
    # 1. 检测系统环境
    print_phase "1. Detecting System Environment..."
    detect_system_info
    detect_cuda_info
    detect_gpu_info
    show_detection_summary
    
    # 2. 安装uv
    print_phase "2. Installing uv..."
    detect_uv_installation
    if [ "$UV_AVAILABLE" = "false" ]; then
        install_uv
    else
        print_success "uv is already installed, version: $DETECTED_UV_VERSION"
    fi

    # 3. 使用uv同步环境
    print_phase "3. Synchronizing Environment..."
    sync_env
    
    # 4. 额外配置
    print_phase "4. Additional Configuration..."

    # 配置结束
    echo ""
    print_info "You can use 'uv cache clean' to clean the uv cache if needed."
    print_success "Environment setup complete! 🎉"
}

# Run main function
main "$@"