#!/bin/bash

# ============================================================================
# Alignment Methods Assignment - Comprehensive Setup Script
# ============================================================================
# This script installs all required dependencies and sets up the environment
# for running the alignment methods experiments.
#
# Usage:
#   bash setup.sh                    # Install everything
#   bash setup.sh --quick            # Skip optional packages
#   bash setup.sh --cpu-only         # Install CPU-only versions
#   bash setup.sh --verify-only      # Only verify installation
# ============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Parse arguments
QUICK_INSTALL=false
CPU_ONLY=false
VERIFY_ONLY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            QUICK_INSTALL=true
            shift
            ;;
        --cpu-only)
            CPU_ONLY=true
            shift
            ;;
        --verify-only)
            VERIFY_ONLY=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: bash setup.sh [--quick] [--cpu-only] [--verify-only]"
            exit 1
            ;;
    esac
done

# ============================================================================
# Helper Functions
# ============================================================================

print_header() {
    echo ""
    echo -e "${BLUE}============================================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}============================================================================${NC}"
    echo ""
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

# ============================================================================
# System Check
# ============================================================================

check_system() {
    print_header "System Check"
    
    # Check Python version
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
        print_success "Python found: $PYTHON_VERSION"
        
        # Check if version is 3.8+
        PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
        PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
        
        if [ "$PYTHON_MAJOR" -ge 3 ] && [ "$PYTHON_MINOR" -ge 8 ]; then
            print_success "Python version is compatible (3.8+)"
        else
            print_error "Python 3.8+ required, found $PYTHON_VERSION"
            exit 1
        fi
    else
        print_error "Python3 not found. Please install Python 3.8+"
        exit 1
    fi
    
    # Check pip
    if command -v pip3 &> /dev/null; then
        print_success "pip3 found"
    else
        print_error "pip3 not found. Please install pip"
        exit 1
    fi
    
    # Check for CUDA (optional but recommended)
    if command -v nvidia-smi &> /dev/null; then
        CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')
        print_success "CUDA found: Version $CUDA_VERSION"
        GPU_AVAILABLE=true
    else
        print_warning "CUDA not found. Will install CPU-only versions"
        GPU_AVAILABLE=false
        CPU_ONLY=true
    fi
    
    # Check available disk space
    AVAILABLE_SPACE=$(df -h . | awk 'NR==2 {print $4}')
    print_info "Available disk space: $AVAILABLE_SPACE"
    
    # Check RAM
    if command -v free &> /dev/null; then
        TOTAL_RAM=$(free -h | awk 'NR==2 {print $2}')
        print_info "Total RAM: $TOTAL_RAM"
    fi
}

# ============================================================================
# Virtual Environment Setup
# ============================================================================

setup_venv() {
    print_header "Virtual Environment Setup"
    
    VENV_NAME="alignment_env"
    
    if [ -d "$VENV_NAME" ]; then
        print_warning "Virtual environment '$VENV_NAME' already exists"
        read -p "Do you want to recreate it? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf "$VENV_NAME"
            print_info "Removed existing environment"
        else
            print_info "Using existing environment"
            source "$VENV_NAME/bin/activate"
            return
        fi
    fi
    
    print_info "Creating virtual environment: $VENV_NAME"
    python3 -m venv "$VENV_NAME"
    
    print_info "Activating virtual environment"
    source "$VENV_NAME/bin/activate"
    
    print_info "Upgrading pip, setuptools, wheel"
    pip install --upgrade pip setuptools wheel
    
    print_success "Virtual environment created and activated"
    print_info "To activate in future: source $VENV_NAME/bin/activate"
}

# ============================================================================
# Install Core Dependencies
# ============================================================================

install_pytorch() {
    print_header "Installing PyTorch"
    
    if [ "$CPU_ONLY" = true ]; then
        print_info "Installing PyTorch (CPU-only)"
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    else
        print_info "Installing PyTorch (with CUDA support)"
        # Install latest PyTorch with CUDA
        pip install torch torchvision torchaudio
    fi
    
    print_success "PyTorch installed"
}

install_transformers() {
    print_header "Installing Transformers & HuggingFace Libraries"
    
    pip install transformers datasets tokenizers accelerate
    pip install huggingface-hub
    
    print_success "Transformers libraries installed"
}

install_rlhf_libraries() {
    print_header "Installing RLHF Libraries"
    
    # PEFT for LoRA
    pip install peft
    
    # TRL for PPO/DPO
    pip install trl
    
    # bitsandbytes for quantization (only if GPU available)
    if [ "$CPU_ONLY" = false ]; then
        print_info "Installing bitsandbytes for 8-bit quantization"
        pip install bitsandbytes
    else
        print_warning "Skipping bitsandbytes (requires CUDA)"
    fi
    
    print_success "RLHF libraries installed"
}

install_scientific() {
    print_header "Installing Scientific Computing Libraries"
    
    pip install numpy scipy pandas
    pip install scikit-learn
    
    print_success "Scientific libraries installed"
}

install_visualization() {
    print_header "Installing Visualization Libraries"
    
    pip install matplotlib seaborn plotly
    pip install tensorboard
    
    print_success "Visualization libraries installed"
}

install_utilities() {
    print_header "Installing Utility Libraries"
    
    pip install tqdm requests aiohttp
    pip install pyarrow
    
    print_success "Utility libraries installed"
}

install_optional() {
    if [ "$QUICK_INSTALL" = true ]; then
        print_warning "Skipping optional packages (--quick mode)"
        return
    fi
    
    print_header "Installing Optional Packages"
    
    # Jupyter for notebooks
    read -p "Install Jupyter notebook? (y/n) " -n 1 -r
    echo
    # if [[ $REPLY =~ ^[Yy]$ ]]; then
    #     pip install jupyter notebook ipywidgets
    #     print_success "Jupyter installed"
    # fi
    
    # Better JSON handling
    pip install orjson
    
    # Configuration management
    pip install omegaconf hydra-core
    
    print_success "Optional packages installed"
}

install_from_requirements() {
    print_header "Installing from requirements.txt"
    
    if [ -f "requirements.txt" ]; then
        print_info "Found requirements.txt, installing..."
        
        if [ "$CPU_ONLY" = true ]; then
            # Skip bitsandbytes for CPU
            grep -v "bitsandbytes" requirements.txt | grep -v "flash-attn" > requirements_cpu.txt
            pip install -r requirements_cpu.txt
            rm requirements_cpu.txt
        else
            pip install -r requirements.txt
        fi
        
        print_success "All requirements installed"
    else
        print_warning "requirements.txt not found, installing packages individually"
        install_pytorch
        install_transformers
        install_rlhf_libraries
        install_scientific
        install_visualization
        install_utilities
        install_optional
    fi
}

# ============================================================================
# Verification
# ============================================================================

verify_installation() {
    print_header "Verifying Installation"
    
    echo "Running verification tests..."
    echo ""
    
    # Test Python imports
    python3 << 'EOF'
import sys
import importlib

def check_package(package_name, import_name=None):
    if import_name is None:
        import_name = package_name
    try:
        mod = importlib.import_module(import_name)
        version = getattr(mod, '__version__', 'unknown')
        print(f"✓ {package_name:20s} {version}")
        return True
    except ImportError:
        print(f"✗ {package_name:20s} NOT FOUND")
        return False

print("Core Libraries:")
print("-" * 50)
all_ok = True
all_ok &= check_package("torch")
all_ok &= check_package("transformers")
all_ok &= check_package("datasets")
all_ok &= check_package("peft")
all_ok &= check_package("trl")

print("\nScientific Computing:")
print("-" * 50)
all_ok &= check_package("numpy")
all_ok &= check_package("pandas")
all_ok &= check_package("scipy")
all_ok &= check_package("sklearn", "sklearn")

print("\nVisualization:")
print("-" * 50)
all_ok &= check_package("matplotlib")
all_ok &= check_package("seaborn")
all_ok &= check_package("tensorboard")

print("\nUtilities:")
print("-" * 50)
all_ok &= check_package("tqdm")
all_ok &= check_package("requests")

print("\n" + "=" * 50)

# Test PyTorch CUDA
import torch
if torch.cuda.is_available():
    print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
    print(f"  CUDA version: {torch.version.cuda}")
    print(f"  GPU count: {torch.cuda.device_count()}")
else:
    print("ℹ CUDA not available (CPU-only mode)")

if all_ok:
    print("\n✓ ALL REQUIRED PACKAGES INSTALLED SUCCESSFULLY!")
    sys.exit(0)
else:
    print("\n✗ SOME PACKAGES MISSING")
    sys.exit(1)
EOF

    if [ $? -eq 0 ]; then
        print_success "Installation verified successfully!"
    else
        print_error "Some packages failed to install"
        return 1
    fi
}

# ============================================================================
# Create Project Structure
# ============================================================================

create_structure() {
    print_header "Creating Project Structure"
    
    # Create directories
    mkdir -p data/raw data/processed
    mkdir -p models checkpoints
    mkdir -p eval runs notebooks
    mkdir -p logs
    
    print_success "Project directories created"
    
    # Create .gitignore
    cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
alignment_env/
*.egg-info/
dist/
build/

# Data
data/raw/
*.jsonl
*.csv
*.parquet

# Models
models/
checkpoints/
*.bin
*.safetensors
*.pt
*.pth

# Logs
logs/
runs/
*.log
wandb/

# Jupyter
.ipynb_checkpoints/
*.ipynb

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db
EOF
    
    print_success ".gitignore created"
}

# ============================================================================
# Download Test
# ============================================================================

test_download() {
    print_header "Testing HuggingFace Download"
    
    python3 << 'EOF'
from transformers import AutoTokenizer
import os

print("Attempting to download a small model to test connectivity...")
try:
    # Download a small tokenizer as test
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    print("✓ Successfully connected to HuggingFace Hub")
    print("✓ Model download works")
    
    # Clean up
    import shutil
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    print(f"\nModels will be cached in: {cache_dir}")
    
except Exception as e:
    print(f"✗ Failed to download: {e}")
    print("\nPlease check your internet connection and try again")
EOF
}

# ============================================================================
# Print Summary
# ============================================================================

print_summary() {
    print_header "Installation Complete!"
    
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "                        ✓ SETUP SUCCESSFUL"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "Next steps:"
    echo ""
    echo "1. Activate the environment:"
    echo "   source alignment_env/bin/activate"
    echo ""
    echo "2. Run quick test:"
    echo "   python main.py --quick_test"
    echo ""
    echo "3. Run full pipeline:"
    echo "   python main.py --full_pipeline"
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "Useful commands:"
    echo "  - Deactivate environment: deactivate"
    echo "  - Check GPU: nvidia-smi"
    echo "  - Monitor training: tail -f main_pipeline.log"
    echo ""
    echo "Documentation:"
    echo "  - Full guide: README.md"
    echo "  - Quick start: QUICKSTART.md"
    echo "  - Troubleshooting: PROJECT_SUMMARY.md"
    echo ""
}

# ============================================================================
# Main Installation Flow
# ============================================================================

main() {
    clear
    echo ""
    echo "╔══════════════════════════════════════════════════════════════════════════╗"
    echo "║           Alignment Methods Assignment - Setup Script                   ║"
    echo "╚══════════════════════════════════════════════════════════════════════════╝"
    echo ""
    
    # if [ "$VERIFY_ONLY" = true ]; then
    #     verify_installation
    #     exit $?
    # fi
    
    # Run setup steps
    check_system
    
    # Ask about virtual environment
    # read -p "Create virtual environment? (recommended) (y/n) " -n 1 -r
    # echo
    # if [[ $REPLY =~ ^[Yy]$ ]]; then
    #     setup_venv
    # fi
    
    # Install packages
    install_from_requirements
    
    # Verify installation
    verify_installation
    
    # Create project structure
    create_structure
    
    # Test HuggingFace download
    test_download
    
    # Print summary
    print_summary
}

# ============================================================================
# Run Main
# ============================================================================

main