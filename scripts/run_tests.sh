#!/bin/bash
# Comprehensive test runner for RoseLLM

set -e  # Exit on error

echo "========================================="
echo "RoseLLM Test Runner"
echo "========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    if [ $1 -eq 0 ]; then
        echo -e "${GREEN}✓ $2 passed${NC}"
    else
        echo -e "${RED}✗ $2 failed${NC}"
        exit 1
    fi
}

# 1. Check Python version
echo -e "\n${YELLOW}1. Checking Python version...${NC}"
python_version=$(python -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "Python version: $python_version"
if [[ $(echo "$python_version >= 3.8" | bc -l) -eq 1 ]]; then
    print_status 0 "Python version check"
else
    print_status 1 "Python version check (requires >= 3.8)"
fi

# 2. Install dependencies
echo -e "\n${YELLOW}2. Installing dependencies...${NC}"
pip install -e ".[test]" --quiet
print_status $? "Dependency installation"

# 3. Run syntax checks
echo -e "\n${YELLOW}3. Running syntax checks...${NC}"
python -m py_compile rosellm/**/*.py 2>/dev/null
print_status $? "Syntax validation"

# 4. Run import tests
echo -e "\n${YELLOW}4. Testing imports...${NC}"
python -c "
from rosellm.rosetrainer import RoseTrainer
from rosellm.rosetrainer.parallelism import initialize_model_parallel
print('All imports successful')
" > /dev/null 2>&1
print_status $? "Import tests"

# 5. Run unit tests
echo -e "\n${YELLOW}5. Running unit tests...${NC}"
pytest tests/unit -v --tb=short -q
print_status $? "Unit tests"

# 6. Run integration tests (CPU only)
echo -e "\n${YELLOW}6. Running integration tests (CPU)...${NC}"
CUDA_VISIBLE_DEVICES="" pytest tests/integration -v -m "not gpu" --tb=short -q
print_status $? "Integration tests (CPU)"

# 7. Check GPU availability and run GPU tests if available
echo -e "\n${YELLOW}7. Checking GPU tests...${NC}"
if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo "GPUs detected, running GPU tests..."
    pytest tests/ -v -m "gpu" --tb=short -q
    print_status $? "GPU tests"
else
    echo "No GPUs detected, skipping GPU tests"
fi

# 8. Run distributed tests (2 processes on CPU)
echo -e "\n${YELLOW}8. Running distributed tests...${NC}"
if [ -f "tests/test_distributed.py" ]; then
    torchrun --nproc_per_node=2 tests/test_distributed.py
    print_status $? "Distributed tests"
else
    echo "No distributed tests found, skipping"
fi

# 9. Code coverage report
echo -e "\n${YELLOW}9. Generating coverage report...${NC}"
pytest tests/ --cov=rosellm --cov-report=term-missing --cov-report=html --quiet
print_status $? "Coverage report"
echo "Coverage report saved to htmlcov/index.html"

# 10. Static analysis (optional)
echo -e "\n${YELLOW}10. Running static analysis...${NC}"
if command -v flake8 &> /dev/null; then
    flake8 rosellm/ --count --select=E9,F63,F7,F82 --show-source --statistics
    print_status $? "Static analysis"
else
    echo "flake8 not installed, skipping"
fi

echo -e "\n${GREEN}========================================="
echo "All tests completed successfully!"
echo "=========================================${NC}"