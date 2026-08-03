#!/usr/bin/env bash

set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
FRONTEND_PATH="$PROJECT_ROOT/architecture_lab/frontend"
VENV_PYTHON="$PROJECT_ROOT/.venv/bin/python"
SETUP_HELPER="$SCRIPT_DIR/setup_helper.py"
PYTHON_VERSION="3.12"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BRIGHT_YELLOW='\033[1;93m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'
GRAY='\033[0;90m'
BOLD='\033[1m'
NC='\033[0m'

CHECK_MARK="✅"
CROSS_MARK="❌"
INFO_MARK="ℹ️"
WARNING_MARK="⚠️"
RUNNING_MARK="⏳"

NVIDIA_AVAILABLE="false"
DRIVER_CUDA=""
DETECTED_CUDA_RUNTIME=""
GPU_DETAILS=""
GPU_COUNT=0
DETECTED_OS="$(uname -s)"
DETECTED_ARCH="$(uname -m)"
DETECTED_SHELL="Bash $BASH_VERSION"
SYSTEM_INFO_INITIALIZED="false"
COMPONENT_STATUS_INITIALIZED="false"
ACTION_SUCCEEDED="false"


print_section() {
    echo ""
    printf "%b%s  %s%b\n" "$MAGENTA$BOLD" "$RUNNING_MARK" "$1" "$NC"
}


print_info() {
    printf "%s   %b%s%b\n" "$INFO_MARK" "$BLUE" "$1" "$NC"
}


print_success() {
    printf "%s  %b%s%b\n" "$CHECK_MARK" "$GREEN" "$1" "$NC"
}


print_warning() {
    printf "%s  %b%s%b\n" "$WARNING_MARK" "$YELLOW" "$1" "$NC"
}


print_failure() {
    printf "%s  %b%s%b\n" "$CROSS_MARK" "$RED" "$1" "$NC"
}


show_banner() {
    local logo_lines=(
        "  ███╗   ███╗ ██╗ ███╗   ██╗ ██╗        ██╗      ██╗      ███╗   ███╗"
        "  ████╗ ████║ ██║ ████╗  ██║ ██║        ██║      ██║      ████╗ ████║"
        "  ██╔████╔██║ ██║ ██╔██╗ ██║ ██║ █████╗ ██║      ██║      ██╔████╔██║"
        "  ██║╚██╔╝██║ ██║ ██║╚██╗██║ ██║ ╚════╝ ██║      ██║      ██║╚██╔╝██║"
        "  ██║ ╚═╝ ██║ ██║ ██║ ╚████║ ██║        ███████╗ ███████╗ ██║ ╚═╝ ██║"
        "  ╚═╝     ╚═╝ ╚═╝ ╚═╝  ╚═══╝ ╚═╝        ╚══════╝ ╚══════╝ ╚═╝     ╚═╝"
    )
    local mini_section_width=40
    local line

    echo ""
    for line in "${logo_lines[@]}"; do
        printf "%b%s%b%s%b\n" "$BLUE" "${line:0:mini_section_width}" "$BRIGHT_YELLOW" "${line:mini_section_width}" "$NC"
    done
    echo ""
    printf "%b  Interactive Environment Setup%b\n" "$WHITE" "$NC"
    printf "%b  %s%b\n" "$GRAY" "$PROJECT_ROOT" "$NC"
}


confirm_action() {
    local message="$1"
    local default_yes="${2:-false}"
    local suffix="[y/N]"
    if [ "$default_yes" = "true" ]; then
        suffix="[Y/n]"
    fi
    read -r -p "  $message $suffix " answer
    if [ -z "$answer" ]; then
        [ "$default_yes" = "true" ]
        return
    fi
    [[ "$answer" =~ ^[yY]$ ]]
}


pause_menu() {
    echo ""
    read -r -p "  Press Enter to return to the menu" _
}


node_version_supported() {
    local version="$1"
    local major minor patch
    version="${version#v}"
    version="${version%%-*}"
    IFS='.' read -r major minor patch <<< "$version"
    major="${major:-0}"
    minor="${minor:-0}"
    if [ "$major" -eq 20 ]; then
        [ "$minor" -ge 19 ]
        return
    fi
    [ "$major" -gt 22 ] || { [ "$major" -eq 22 ] && [ "$minor" -ge 12 ]; }
}


venv_module_available() {
    [ -x "$VENV_PYTHON" ] && [ -f "$SETUP_HELPER" ] && \
        "$VENV_PYTHON" "$SETUP_HELPER" module-available "$1" >/dev/null 2>&1
}


detect_system() {
    DETECTED_OS="$(uname -s)"
    DETECTED_ARCH="$(uname -m)"
    DETECTED_SHELL="Bash $BASH_VERSION"
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        DETECTED_OS="$NAME ${VERSION:-}"
    fi

    NVIDIA_AVAILABLE="false"
    DRIVER_CUDA=""
    DETECTED_CUDA_RUNTIME=""
    GPU_DETAILS=""
    GPU_COUNT=0

    if command -v nvidia-smi >/dev/null 2>&1; then
        local smi_output raw_gpu_details
        smi_output="$(nvidia-smi 2>/dev/null)"
        if [ $? -eq 0 ]; then
            DRIVER_CUDA="$(printf '%s\n' "$smi_output" | sed -n 's/.*CUDA Version:[[:space:]]*\([0-9][0-9.]*\).*/\1/p' | head -n 1)"
            DETECTED_CUDA_RUNTIME="$DRIVER_CUDA"
            raw_gpu_details="$(nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader,nounits 2>/dev/null)"
            if [ -n "$raw_gpu_details" ]; then
                GPU_DETAILS="$(while IFS=',' read -r index name memory; do
                    index="$(echo "$index" | xargs)"
                    name="$(echo "$name" | xargs)"
                    memory="$(echo "$memory" | xargs)"
                    echo "GPU $index : $name (${memory}MB)"
                done <<< "$raw_gpu_details")"
                GPU_COUNT="$(printf '%s\n' "$GPU_DETAILS" | sed '/^$/d' | wc -l | xargs)"
                NVIDIA_AVAILABLE="true"
            fi
        fi
    fi

    if [ -z "$DRIVER_CUDA" ] && command -v nvcc >/dev/null 2>&1; then
        DRIVER_CUDA="$(nvcc --version | sed -n 's/.*release \([0-9][0-9.]*\).*/\1/p' | head -n 1)"
    fi
}


show_detection_summary() {
    print_info "System Information:"
    printf "Operating System: %b%s%b\n" "$GREEN" "$DETECTED_OS" "$NC"
    printf "System Architecture: %b%s%b\n" "$GREEN" "$DETECTED_ARCH" "$NC"
    printf "Shell Version: %b%s%b\n" "$GREEN" "$DETECTED_SHELL" "$NC"

    print_info "CUDA Information:"
    if [ -n "$DRIVER_CUDA" ]; then
        printf "CUDA Version: %b%s%b\n" "$GREEN" "$DRIVER_CUDA" "$NC"
        if [ -n "$DETECTED_CUDA_RUNTIME" ]; then
            printf "Runtime Version: %b%s%b\n" "$GREEN" "$DETECTED_CUDA_RUNTIME" "$NC"
        fi
    else
        printf "%bCUDA not installed or unavailable%b\n" "$RED" "$NC"
    fi

    print_info "GPU Information:"
    if [ "$NVIDIA_AVAILABLE" = "true" ]; then
        printf "GPU Count: %b%s%b\n" "$GREEN" "$GPU_COUNT" "$NC"
        printf "%bGPU Details:%b\n" "$GREEN" "$NC"
        while IFS= read -r gpu; do
            [ -n "$gpu" ] && printf "%b  - %s%b\n" "$GREEN" "$gpu" "$NC"
        done <<< "$GPU_DETAILS"
    else
        printf "%bNo NVIDIA GPU detected%b\n" "$RED" "$NC"
    fi
}


get_component_status() {
    UV_READY="false"
    UV_STATUS="not installed"
    if command -v uv >/dev/null 2>&1; then
        UV_STATUS="$(uv --version 2>/dev/null)"
        [ $? -eq 0 ] && UV_READY="true"
    fi

    NODE_READY="false"
    NODE_STATUS="not installed"
    if command -v node >/dev/null 2>&1; then
        local node_version
        node_version="$(node --version 2>/dev/null)"
        if node_version_supported "$node_version"; then
            NODE_READY="true"
            NODE_STATUS="$node_version"
        else
            NODE_STATUS="$node_version (upgrade required)"
        fi
    fi

    FRONTEND_READY="false"
    FRONTEND_STATUS="not installed"
    if [ -d "$FRONTEND_PATH/dist" ]; then
        FRONTEND_READY="true"
        FRONTEND_STATUS="ready"
    fi

    local python_version=""
    local torch_installed=0 torch_available=0 torch_version=""
    local triton_installed=0 triton_available=0 triton_version=""
    local flash_installed=0 flash_available=0 flash_version=""
    if [ -x "$VENV_PYTHON" ] && [ -f "$SETUP_HELPER" ]; then
        while IFS=$'\t' read -r key installed available version; do
            case "$key" in
                python) python_version="$installed" ;;
                torch)
                    torch_installed="$installed"
                    torch_available="$available"
                    torch_version="$version"
                    ;;
                triton)
                    triton_installed="$installed"
                    triton_available="$available"
                    triton_version="$version"
                    ;;
                flash_attn)
                    flash_installed="$installed"
                    flash_available="$available"
                    flash_version="$version"
                    ;;
            esac
        done < <("$VENV_PYTHON" "$SETUP_HELPER" status --format shell 2>/dev/null)
    fi

    MAIN_READY="false"
    if [ "$torch_available" = "1" ]; then
        MAIN_READY="true"
        MAIN_STATUS="ready (Python $python_version, PyTorch $torch_version)"
    elif [ -x "$VENV_PYTHON" ] && [ "$torch_installed" = "1" ]; then
        MAIN_STATUS="incomplete (PyTorch import failed)"
    elif [ -x "$VENV_PYTHON" ]; then
        MAIN_STATUS="incomplete (Python ${python_version:-unknown})"
    else
        MAIN_STATUS="not installed"
    fi

    TRITON_READY="false"
    if [ "$triton_available" = "1" ]; then
        TRITON_READY="true"
        TRITON_STATUS="ready ($triton_version)"
    elif [ "$triton_installed" = "1" ]; then
        TRITON_STATUS="installed but unavailable"
    else
        TRITON_STATUS="not installed"
    fi

    FLASH_READY="false"
    if [ "$flash_available" = "1" ]; then
        FLASH_READY="true"
        FLASH_STATUS_DISPLAY="ready ($flash_version)"
    elif [ "$flash_installed" = "1" ]; then
        FLASH_STATUS_DISPLAY="installed but unavailable"
    else
        FLASH_STATUS_DISPLAY="not installed"
    fi
    COMPONENT_STATUS_INITIALIZED="true"
}


status_line() {
    local name="$1"
    local value="$2"
    local ready="$3"
    local color="$YELLOW"
    [ "$ready" = "true" ] && color="$GREEN"
    printf "%s: %b%s%b\n" "$name" "$color" "$value" "$NC"
}


show_installed_status() {
    print_info "Installed Status:"
    status_line "uv" "$UV_STATUS" "$UV_READY"
    status_line "Node.js" "$NODE_STATUS" "$NODE_READY"
    status_line "Main environment" "$MAIN_STATUS" "$MAIN_READY"
    status_line "Triton" "$TRITON_STATUS" "$TRITON_READY"
    status_line "Architecture Lab" "$FRONTEND_STATUS" "$FRONTEND_READY"
    status_line "flash-attn" "$FLASH_STATUS_DISPLAY" "$FLASH_READY"
}


show_menu() {
    clear
    show_banner

    if [ "$SYSTEM_INFO_INITIALIZED" != "true" ]; then
        sleep 0.5
        print_section "Detecting system devices and CUDA..."
        detect_system
        SYSTEM_INFO_INITIALIZED="true"
    fi
    show_detection_summary

    if [ "$COMPONENT_STATUS_INITIALIZED" != "true" ]; then
        print_section "Checking installed components..."
        get_component_status
    fi
    show_installed_status

    echo ""
    printf "%b========================================%b\n" "$CYAN" "$NC"
    printf "%b  Environment Setup Menu%b\n" "$CYAN" "$NC"
    printf "%b========================================%b\n\n" "$CYAN" "$NC"
    printf "%b[1]%b Install main environment\n" "$GREEN" "$NC"
    printf "%b    Detect CUDA and install PyTorch, Triton and project dependencies%b\n\n" "$GRAY" "$NC"
    printf "%b[2]%b Install frontend\n" "$GREEN" "$NC"
    printf "%b    Check Node.js and build the Architecture Lab frontend%b\n\n" "$GRAY" "$NC"
    printf "%b[3]%b Install flash-attn\n" "$GREEN" "$NC"
    printf "%b    Prefer a matching wheel; ask before compiling from source%b\n\n" "$GRAY" "$NC"
    printf "%b[q/Q]%b Quit\n\n" "$YELLOW" "$NC"
    printf "%bPlease select an option: %b" "$CYAN" "$NC"
}


get_torch_extra() {
    if [ "$NVIDIA_AVAILABLE" != "true" ] || [ -z "$DRIVER_CUDA" ]; then
        echo "cpu"
        return
    fi
    local major minor code
    IFS='.' read -r major minor _ <<< "$DRIVER_CUDA"
    major="${major:-0}"
    minor="${minor:-0}"
    code=$((10#$major * 100 + 10#$minor))
    if [ "$code" -ge 1300 ]; then echo "cu130"
    elif [ "$code" -ge 1208 ]; then echo "cu128"
    elif [ "$code" -ge 1206 ]; then echo "cu126"
    elif [ "$code" -ge 1204 ]; then echo "cu124"
    elif [ "$code" -ge 1201 ]; then echo "cu121"
    elif [ "$code" -ge 1108 ]; then echo "cu118"
    else echo "unsupported"
    fi
}


ensure_uv() {
    if command -v uv >/dev/null 2>&1; then
        print_success "Found $(uv --version)"
        return 0
    fi
    print_info "uv was not found. Installing it with the official installer..."
    if command -v curl >/dev/null 2>&1; then
        curl -LsSf https://astral.sh/uv/install.sh | sh
    elif command -v wget >/dev/null 2>&1; then
        wget -qO- https://astral.sh/uv/install.sh | sh
    else
        print_failure "curl or wget is required to install uv."
        return 1
    fi
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
    if ! command -v uv >/dev/null 2>&1; then
        print_failure "uv was installed but is not available in PATH. Restart the shell and retry."
        return 1
    fi
    print_success "Installed $(uv --version)"
}


install_main_environment() {
    print_section "Main environment"
    print_info "System: $DETECTED_OS ($DETECTED_ARCH)"
    if [ "$NVIDIA_AVAILABLE" = "true" ]; then
        print_info "Driver CUDA capability: $DRIVER_CUDA"
        while IFS= read -r gpu; do
            [ -n "$gpu" ] && print_info "GPU: $gpu"
        done <<< "$GPU_DETAILS"
    else
        print_warning "No usable NVIDIA GPU was detected. The CPU environment will be selected."
    fi

    local torch_extra
    torch_extra="$(get_torch_extra)"
    if [ "$torch_extra" = "unsupported" ]; then
        print_warning "The detected CUDA capability is older than the supported PyTorch profiles."
        if ! confirm_action "Continue with the CPU environment?"; then
            return
        fi
        torch_extra="cpu"
    fi
    if venv_module_available "flash_attn"; then
        print_warning "Exact environment synchronization may remove the existing flash-attn build."
        print_info "Install flash-attn again from menu option 3 after the main environment is finalized."
    fi

    ensure_uv || return
    print_info "Preparing Python $PYTHON_VERSION with uv..."
    if ! uv python install "$PYTHON_VERSION"; then
        print_failure "Python $PYTHON_VERSION installation failed."
        return
    fi

    local sync_args=(sync --python "$PYTHON_VERSION" --extra "$torch_extra")
    local profile="$torch_extra"
    if [ "$torch_extra" != "cpu" ]; then
        sync_args+=(--extra flash-linux)
        profile="$profile + triton"
    fi
    print_info "Installation profile: $profile"
    printf "%b  > uv %s%b\n" "$GRAY" "${sync_args[*]}" "$NC"
    if ! (cd "$PROJECT_ROOT" && uv "${sync_args[@]}"); then
        print_failure "Environment synchronization failed."
        return
    fi

    print_info "Verifying the installed Python environment..."
    if ! "$VENV_PYTHON" "$SETUP_HELPER" verify-main; then
        print_failure "The environment was installed, but the import verification failed."
        return
    fi
    print_success "Main environment installation completed."
    ACTION_SUCCEEDED="true"
}


install_frontend() {
    print_section "Frontend"
    if [ ! -f "$FRONTEND_PATH/package.json" ]; then
        print_failure "Architecture Lab frontend was not found at $FRONTEND_PATH"
        return
    fi
    if ! confirm_action "Install and build the Architecture Lab frontend?" true; then
        print_info "Frontend installation cancelled."
        return
    fi
    if ! command -v node >/dev/null 2>&1 || ! command -v npm >/dev/null 2>&1; then
        print_failure "Node.js or npm is not available."
        print_info "Install Node.js 20.19+ or 22.12+, then run this option again."
        return
    fi
    local node_version
    node_version="$(node --version)"
    if ! node_version_supported "$node_version"; then
        print_failure "Node.js $node_version does not satisfy Vite's requirement."
        print_info "Install Node.js 20.19+ or 22.12+. Node.js is not installed automatically."
        return
    fi
    print_info "Node.js: $node_version"
    print_info "npm: $(npm --version)"
    printf "%b  > npm ci%b\n" "$GRAY" "$NC"
    if ! (cd "$FRONTEND_PATH" && npm ci); then
        print_failure "npm ci failed."
        return
    fi
    printf "%b  > npm run build%b\n" "$GRAY" "$NC"
    if ! (cd "$FRONTEND_PATH" && npm run build); then
        print_failure "Frontend build failed."
        return
    fi
    print_success "Architecture Lab frontend installation completed."
    ACTION_SUCCEEDED="true"
}


get_flash_environment() {
    FLASH_STATUS="MISSING"
    FLASH_REASON="the project virtual environment does not exist"
    if [ ! -x "$VENV_PYTHON" ] || [ ! -f "$SETUP_HELPER" ]; then
        return
    fi
    local line field1 field2 field3 field4 field5 field6 field7 field8 field9 field10 field11
    line="$("$VENV_PYTHON" "$SETUP_HELPER" flash-environment --format shell 2>/dev/null)"
    IFS=$'\t' read -r FLASH_STATUS field1 field2 field3 field4 field5 field6 field7 field8 field9 field10 field11 <<< "$line"
    if [ "$FLASH_STATUS" = "READY" ]; then
        FLASH_PYTHON="$field1"
        FLASH_PYTHON_TAG="$field2"
        FLASH_TORCH="$field3"
        FLASH_TORCH_MM="$field4"
        FLASH_CUDA="$field5"
        FLASH_CUDA_MAJOR="$field6"
        FLASH_ABI="$field7"
        FLASH_GPU="$field8"
        FLASH_CAPABILITY="$field9"
        FLASH_PLATFORM="$field10"
        FLASH_MACHINE="$field11"
        FLASH_REASON=""
    else
        FLASH_REASON="${field1:-unable to inspect the environment}"
    fi
}


find_flash_wheel() {
    local line field1 field2 field3
    line="$("$VENV_PYTHON" "$SETUP_HELPER" find-flash-wheel --format shell 2>/dev/null)"
    IFS=$'\t' read -r FLASH_WHEEL_STATUS field1 field2 field3 <<< "$line"
    FLASH_WHEEL_NAME="$field1"
    FLASH_WHEEL_URL="$field2"
    FLASH_WHEEL_RELEASE="$field3"
}


install_flash_dependencies() {
    print_info "Installing flash-attn Python build dependencies..."
    uv pip install --python "$VENV_PYTHON" --index-url https://pypi.org/simple einops packaging psutil ninja setuptools wheel
}


test_flash_attention() {
    "$VENV_PYTHON" "$SETUP_HELPER" test-flash-attention
}


install_flash_from_source() {
    if ! command -v nvcc >/dev/null 2>&1; then
        print_failure "nvcc was not found. Install a CUDA Toolkit matching PyTorch CUDA $FLASH_CUDA."
        return
    fi
    if ! command -v c++ >/dev/null 2>&1; then
        print_failure "A C++ compiler was not found. Install g++ or clang++ and retry."
        return
    fi
    local nvcc_version nvcc_major
    nvcc_version="$(nvcc --version | sed -n 's/.*release \([0-9][0-9.]*\).*/\1/p' | head -n 1)"
    nvcc_major="${nvcc_version%%.*}"
    if [ -z "$nvcc_version" ]; then
        print_failure "Unable to determine the local CUDA Toolkit version from nvcc."
        return
    fi
    if [ "$nvcc_major" != "$FLASH_CUDA_MAJOR" ]; then
        print_failure "nvcc CUDA $nvcc_version does not match PyTorch CUDA $FLASH_CUDA."
        return
    fi
    if ! install_flash_dependencies; then
        print_failure "Failed to install flash-attn build dependencies."
        return
    fi

    local cuda_home
    cuda_home="${CUDA_HOME:-$(dirname "$(dirname "$(command -v nvcc)")")}"
    print_info "Compiling flash-attn with MAX_JOBS=2 and NVCC_THREADS=1. This may take a long time."
    if ! CUDA_HOME="$cuda_home" MAX_JOBS=2 NVCC_THREADS=1 uv pip install \
        --python "$VENV_PYTHON" \
        --index-url https://pypi.org/simple \
        --reinstall \
        --no-deps \
        --no-build-isolation \
        --no-binary flash-attn \
        "flash-attn>=2,<3"; then
        print_failure "flash-attn source compilation failed."
        return
    fi
    if test_flash_attention; then
        print_success "flash-attn source build completed."
        ACTION_SUCCEEDED="true"
    else
        print_failure "flash-attn was built, but runtime verification failed."
    fi
}


install_flash_attention() {
    print_section "flash-attn"
    ensure_uv || return
    get_flash_environment
    if [ "$FLASH_STATUS" = "MISSING" ]; then
        print_failure "The main environment is missing."
        print_info "Run menu option 1 before installing flash-attn."
        return
    fi
    if [ "$FLASH_STATUS" != "READY" ]; then
        print_failure "flash-attn is not supported by the current environment: $FLASH_REASON"
        return
    fi

    print_info "GPU: $FLASH_GPU (SM $FLASH_CAPABILITY)"
    print_info "Python: $FLASH_PYTHON | Torch: $FLASH_TORCH | CUDA: $FLASH_CUDA | CXX11 ABI: $FLASH_ABI"
    print_info "Searching official flash-attention release assets..."
    find_flash_wheel
    if [ "$FLASH_WHEEL_STATUS" = "FOUND" ]; then
        print_success "Found $FLASH_WHEEL_NAME"
        if ! install_flash_dependencies; then
            print_failure "Failed to install flash-attn runtime dependencies."
            return
        fi
        printf "%b  > uv pip install --reinstall --no-deps <wheel>%b\n" "$GRAY" "$NC"
        if uv pip install --python "$VENV_PYTHON" --reinstall --no-deps "$FLASH_WHEEL_URL" && test_flash_attention; then
            print_success "flash-attn wheel installation completed."
            ACTION_SUCCEEDED="true"
            return
        fi
        print_warning "The matching wheel could not be installed or loaded."
    elif [ "$FLASH_WHEEL_STATUS" = "ERROR" ]; then
        print_warning "Unable to query GitHub releases: $FLASH_WHEEL_NAME"
    else
        print_warning "No compatible prebuilt wheel was found."
        print_info "$FLASH_WHEEL_NAME"
    fi

    print_warning "Source compilation can take a long time and use substantial CPU and memory."
    if confirm_action "Compile flash-attn from source now?"; then
        install_flash_from_source
    else
        print_info "flash-attn installation cancelled. The Triton backend remains available."
    fi
}


run_action() {
    ACTION_SUCCEEDED="false"
    "$@"
    if [ "$ACTION_SUCCEEDED" = "true" ]; then
        get_component_status
    fi
    pause_menu
}


main() {
    while true; do
        show_menu
        read -r choice
        case "${choice,,}" in
            1) run_action install_main_environment ;;
            2) run_action install_frontend ;;
            3) run_action install_flash_attention ;;
            q|"") printf "%bExiting...%b\n" "$YELLOW" "$NC"; exit 0 ;;
            *) print_warning "Invalid option: $choice"; sleep 1 ;;
        esac
    done
}


main "$@"
