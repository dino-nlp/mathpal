# include .env

# $(eval export $(shell sed -ne 's/ *#.*$$//; /./ s/=.*$$// p' .env))

PYTHONPATH := $(shell pwd)/src

install: # Create a local Poetry virtual environment and install all required Python dependencies.
	poetry env use 3.11
	poetry install --without superlinked_rag
# 	eval $(poetry env activate)

help:
	@grep -E '^[a-zA-Z0-9 -]+:.*#'  Makefile | sort | while read -r l; do printf "\033[1;32m$$(echo $$l | cut -f 1 -d':')\033[00m:$$(echo $$l | cut -f 2- -d'#')\n"; done

# ======================================
# ------- Docker Infrastructure --------
# ======================================

local-start: # Build and start your local Docker infrastructure.
	docker compose -f docker-compose.yml up --build -d

local-stop: # Stop your local Docker infrastructure.
	docker compose -f docker-compose.yml down --remove-orphans

local-test-crawler: # Make a call to your local AWS Lambda (hosted in Docker) to crawl multiple articles.
	curl -X POST "http://localhost:9010/2015-03-31/functions/function/invocations" \
		-d '{"links": ["https://loigiaihay.com/de-thi-vao-lop-6-mon-toan-truong-cau-giay-nam-2023-a142098.html", "https://loigiaihay.com/de-thi-vao-lop-6-mon-toan-truong-luong-the-vinh-2021-co-dap-an-a134641.html", "https://loigiaihay.com/de-thi-vao-lop-6-mon-toan-truong-nguyen-tat-thanh-nam-2025-co-dap-an-a185630.html"], "grade_name": "grade_5"}'

local-ingest-data: # Ingest all links from data/links.txt in a single batch.
	@echo "Preparing to send all links in a single batch..."
	@links_json=$$(jq -R . data/links.txt | jq -s .); \
	curl -X POST "http://localhost:9010/2015-03-31/functions/function/invocations" \
		-d "{\"grade_name\": \"grade_5\", \"links\": $$links_json}"; \
	echo "\nDone."

# ======================================
# -------- RAG Feature Pipeline --------
# ======================================

local-test-retriever: # Test the RAG retriever using your Poetry env
	cd src/feature_pipeline && poetry run python -m retriever


# ======================================
# ----------- Finetuning ---------------
# ======================================
# Environment setup
setup_finetuning_env: ## Setup the training environment
	@echo "🔧 Setting up environment..."
	python scripts/setup_finetuning_environment.py

# Environment
env-check: ## Check environment and dependencies
	@echo "🔍 Checking environment..."
	python -c "import sys; print(f'Python: {sys.version}')"
	python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
	python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
	python -c "import peft; print(f'PEFT: {peft.__version__}')"
	python -c "import trl; print(f'TRL: {trl.__version__}')"


# Training commands (Legacy v1)
train: ## Run training with default config (legacy)
	@echo "🚀 Starting training (legacy)..."
	python -m training_pipeline.cli.train_gemma

train-dev: ## Run development training (legacy)
	@echo "🛠️ Starting development training (legacy)..."
	python -m training_pipeline.cli.train_gemma --config configs/development.yaml

train-prod: ## Run production training (legacy)
	@echo "🏭 Starting production training (legacy)..."
	python -m training_pipeline.cli.train_gemma --config configs/production.yaml

# Training commands (New v2 Architecture)
train-v2: ## Run training with new architecture and comprehensive config
	@echo "🚀 Starting training (v2 - new architecture)..."
	python -m training_pipeline.cli.train_gemma_v2 --config configs/complete_training_config.yaml

train-dev-v2: ## Run development training with new architecture
	@echo "🛠️ Starting development training (v2)..."
	python -m training_pipeline.cli.train_gemma_v2 --config configs/development.yaml

train-prod-v2: ## Run production training with new architecture
	@echo "🏭 Starting production training (v2)..."
	python -m training_pipeline.cli.train_gemma_v2 --config configs/production.yaml

train-quick: ## Run quick test training (v2)
	@echo "⚡ Starting quick test training..."
	python -m training_pipeline.cli.train_gemma_v2 --config configs/development.yaml --quick-test

train-dry-run: ## Validate config and estimate resources without training
	@echo "🔍 Running training dry run..."
	python -m training_pipeline.cli.train_gemma_v2 --config configs/complete_training_config.yaml --dry-run

train-custom: ## Run training with custom experiment name (usage: make train-custom EXPERIMENT=my-experiment)
	@echo "🎯 Starting custom training: $(EXPERIMENT)..."
	python -m training_pipeline.cli.train_gemma_v2 --config configs/complete_training_config.yaml --experiment-name $(EXPERIMENT)


# ======================================
# ----------- Inference ---------------
# ======================================


# ======================================
# ----------- Utility cmd --------------
# ======================================
# Utility commands
clean: ## Clean up generated files
	@echo "🧹 Cleaning up..."
	rm -rf outputs/*/
	rm -rf logs/*.log
	rm -rf examples/logs/*.log
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "✅ Cleanup completed"

clean-cache: ## Clear CUDA cache and temp files
	@echo "🧹 Clearing cache..."
	python -c "import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else None; print('✅ CUDA cache cleared')"

device-info: ## Show device information
	@echo "🖥️ Device information..."
	python -c "from training_pipeline.utils import DeviceUtils; DeviceUtils.print_device_info()"

memory-info: ## Show memory information
	@echo "💾 Memory information..."
	python -c "from training_pipeline.utils import DeviceUtils; print(DeviceUtils.get_cuda_memory_info())"

# Configuration validation
validate-config: ## Validate training configuration file (usage: make validate-config CONFIG=configs/development.yaml)
	@echo "🔍 Validating configuration: $(CONFIG)..."
	python -m training_pipeline.cli.train_gemma_v2 --config $(CONFIG) --dry-run

# Compare architectures
compare-configs: ## Compare old vs new configuration approaches
	@echo "📊 Configuration Comparison:"
	@echo "Old approach: 50+ CLI arguments, hardcoded defaults"
	@echo "New approach: 5-7 CLI arguments, YAML-driven config"
	@echo ""
	@echo "📁 Available configurations:"
	@ls -la configs/*.yaml 2>/dev/null || echo "No config files found in configs/"

# Environment check for new architecture
env-check-v2: ## Check environment for new architecture dependencies
	@echo "🔍 Checking environment for v2 architecture..."
	python -c "import sys; print(f'Python: {sys.version}')"
	python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
	python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
	python -c "import peft; print(f'PEFT: {peft.__version__}')"
	python -c "import trl; print(f'TRL: {trl.__version__}')"
	python -c "import datasets; print(f'Datasets: {datasets.__version__}')"
	python -c "import yaml; print(f'PyYAML: {yaml.__version__}')"
	@echo "🧪 Testing new architecture imports..."
	python -c "from training_pipeline.core.enhanced_config import ConfigLoader; print('✅ Enhanced config loader')"
	python -c "from training_pipeline.core.training_manager import TrainingManager; print('✅ Training manager')"
	python -c "from training_pipeline.factories import ModelFactory, DatasetFactory, TrainerFactory; print('✅ Factories')"
	@echo "✅ Environment check completed"

# Show architecture comparison
show-architecture: ## Show architecture comparison
	@echo "🏗️ MathPal Training Pipeline Architecture Comparison"
	@echo "======================================================"
	@echo ""
	@echo "📊 OLD ARCHITECTURE (train_gemma.py):"
	@echo "   ❌ Monolithic: 454 lines in single file"
	@echo "   ❌ Complex CLI: 50+ command line arguments"  
	@echo "   ❌ Hardcoded: Default values scattered throughout code"
	@echo "   ❌ Hard to test: Tight coupling between components"
	@echo "   ❌ Poor error handling: Basic try-catch only"
	@echo ""
	@echo "✅ NEW ARCHITECTURE (train_gemma_v2.py):"
	@echo "   ✅ Modular: ~100 lines CLI + separate factories/managers"
	@echo "   ✅ Config-driven: 5-7 CLI args + comprehensive YAML config"
	@echo "   ✅ Centralized: All settings in documented YAML files"
	@echo "   ✅ Testable: Dependency injection + factory pattern"
	@echo "   ✅ Robust: Comprehensive validation + error handling"
	@echo ""
	@echo "📁 Usage Examples:"
	@echo "   Legacy:  make train-dev"
	@echo "   New v2:  make train-dev-v2"
	@echo "   Quick:   make train-quick"
	@echo "   Dry run: make train-dry-run"
	@echo ""

# Testing new architecture
test-architecture: ## Test and validate the new training pipeline architecture
	@echo "🧪 Testing new architecture..."
	python test_new_architecture.py

