# include .env

# $(eval export $(shell sed -ne 's/ *#.*$$//; /./ s/=.*$$// p' .env))

PYTHONPATH := $(shell pwd)/src

help:
	@grep -E '^[a-zA-Z0-9 -]+:.*#'  Makefile | sort | while read -r l; do printf "\033[1;32m$$(echo $$l | cut -f 1 -d':')\033[00m:$$(echo $$l | cut -f 2- -d'#')\n"; done

install: # Create a local Poetry virtual environment and install all required Python dependencies.
	@poetry env use 3.11
	@poetry install --without superlinked_rag
# 	eval $(poetry env activate)

setup-env: ## Setup complete environment for finetuning with GPU support
	@echo "🚀 Setting up complete finetuning environment..."
	@pip install --upgrade pip
	@python3 scripts/setup_finetuning_env.py
	@echo "✅ Environment setup completed"

setup-thuegpu: ## Alias for setup-env (backward compatibility)
	@$(MAKE) setup-env

env-check: ## Check environment for new architecture dependencies
	@echo "🔍 Checking environment architecture..."
	@python3 -c "import sys; print(f'Python: {sys.version}')"
	@python3 -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
	@python3 -c "import transformers; print(f'Transformers: {transformers.__version__}')"
	@python3 -c "import peft; print(f'PEFT: {peft.__version__}')"
	@python3 -c "import trl; print(f'TRL: {trl.__version__}')"
	@python3 -c "import datasets; print(f'Datasets: {datasets.__version__}')"
	@python3 -c "import yaml; print(f'PyYAML: {yaml.__version__}')"
	@echo "🧪 Testing new architecture imports..."
	@PYTHONPATH=$(PYTHONPATH) python3 -c "from training_pipeline.core.enhanced_config import ConfigLoader; print('✅ Enhanced config loader')"
	@PYTHONPATH=$(PYTHONPATH) python3 -c "from training_pipeline.core.training_manager import TrainingManager; print('✅ Training manager')"
	@PYTHONPATH=$(PYTHONPATH) python3 -c "from training_pipeline.factories import ModelFactory, DatasetFactory, TrainerFactory; print('✅ Factories')"
	@echo "✅ Environment check completed"

env-info: ## Show comprehensive environment information
	@echo "🔍 Comprehensive Environment Information"
	@echo "========================================"
	@echo "🐍 Python Environment:"
	@python3 --version
	@which python3
	@echo ""
	@echo "📦 Core ML Libraries:"
	@python3 -c "import torch; print(f'PyTorch: {torch.__version__}')"
	@python3 -c "import transformers; print(f'Transformers: {transformers.__version__}')"
	@python3 -c "import datasets; print(f'Datasets: {datasets.__version__}')"
	@python3 -c "import peft; print(f'PEFT: {peft.__version__}')"
	@python3 -c "import trl; print(f'TRL: {trl.__version__}')"
	@echo ""
	@echo "🔥 GPU Information:"
	@python3 -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'Device Count: {torch.cuda.device_count() if torch.cuda.is_available() else 0}'); print(f'Current Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() and torch.cuda.device_count() > 0 else \"N/A\"}')"
	@echo ""
	@echo "🧪 Training Pipeline:"
	@PYTHONPATH=$(PYTHONPATH) python3 -c "from training_pipeline.core.enhanced_config import ConfigLoader; print('✅ Config Loader')"
	@PYTHONPATH=$(PYTHONPATH) python3 -c "from training_pipeline.core.training_manager import TrainingManager; print('✅ Training Manager')"
	@PYTHONPATH=$(PYTHONPATH) python3 -c "from training_pipeline.factories import ModelFactory, DatasetFactory, TrainerFactory; print('✅ All Factories')"
	@echo "========================================"
# ======================================
# ------- Docker Infrastructure --------
# ======================================

local-start: # Build and start your local Docker infrastructure.
	@docker compose -f docker-compose.yml up --build -d

local-stop: # Stop your local Docker infrastructure.
	@docker compose -f docker-compose.yml down --remove-orphans

local-test-crawler: # Make a call to your local AWS Lambda (hosted in Docker) to crawl multiple articles.
	@curl -X POST "http://localhost:9010/2015-03-31/functions/function/invocations" \
		-d '{"links": ["https://loigiaihay.com/de-thi-vao-lop-6-mon-toan-truong-cau-giay-nam-2023-a142098.html", "https://loigiaihay.com/de-thi-vao-lop-6-mon-toan-truong-luong-the-vinh-2021-co-dap-an-a134641.html", "https://loigiaihay.com/de-thi-vao-lop-6-mon-toan-truong-nguyen-tat-thanh-nam-2025-co-dap-an-a185630.html"], "grade_name": "grade_5"}'

local-ingest-data: # Ingest all links from data/links.txt in a single batch.
	@echo "Preparing to send all links in a single batch..."
	@links_json=$$(jq -R . data/links.txt | jq -s .); \
	curl -X POST "http://localhost:9010/2015-03-31/functions/function/invocations" \
		-d "{\"grade_name\": \"grade_5\", \"links\": $$links_json}"
	@echo "Done."

# ======================================
# -------- RAG Feature Pipeline --------
# ======================================

local-test-retriever: # Test the RAG retriever using your Poetry env
	@cd src/feature_pipeline && poetry run python -m retriever


# ======================================
# ----------- Training Pipeline --------
# ======================================

# Main training commands
train: ## Run training with comprehensive config
	@echo "🚀 Starting training with complete config..."
	@./train.sh

train-dev: ## Run development training (quick config)
	@echo "🛠️ Starting development training..."
	@./train.sh configs/development.yaml

train-prod: ## Run production training
	@echo "🏭 Starting production training..."
	@./train.sh configs/production.yaml

train-quick: ## Run quick test training (20 steps)
	@echo "⚡ Starting quick test training..."
	@./train.sh --quick-test

# Validation and testing
train-dry-run: ## Validate config and estimate resources without training
	@echo "🔍 Running training dry run..."
	@./train.sh --dry-run

train-dry-run-dev: ## Dry run with development config
	@echo "🔍 Running development dry run..."
	@./train.sh configs/development.yaml --dry-run

train-dry-run-prod: ## Dry run with production config
	@echo "🔍 Running production dry run..."
	@./train.sh configs/production.yaml --dry-run

# Custom training
train-custom: ## Run training with custom experiment name (usage: make train-custom EXPERIMENT=my-experiment)
	@echo "🎯 Starting custom training: $(EXPERIMENT)..."
	@./train.sh --experiment-name $(EXPERIMENT)

train-custom-config: ## Run training with custom config (usage: make train-custom-config CONFIG=my-config.yaml)
	@echo "📋 Starting training with custom config: $(CONFIG)..."
	@./train.sh $(CONFIG)

# Advanced training options
train-debug: ## Run training with debug logging
	@echo "🐛 Starting training with debug logging..."
	@./train.sh --debug

train-no-comet: ## Run training without Comet ML tracking
	@echo "🚫 Starting training without Comet ML..."
	@./train.sh --no-comet

# Training with overrides
train-steps: ## Run training with custom max steps (usage: make train-steps STEPS=500)
	@echo "🎯 Starting training with $(STEPS) steps..."
	@./train.sh --max-steps $(STEPS)

train-output: ## Run training with custom output directory (usage: make train-output OUTPUT=my-output)
	@echo "📁 Starting training with output: $(OUTPUT)..."
	@./train.sh --output-dir $(OUTPUT)


# ======================================
# ----------- Inference ---------------
# ======================================


# ======================================
# ----------- Utility cmd --------------
# ======================================
# Utility commands
clean: ## Clean up generated files
	@echo "🧹 Cleaning up..."
	@rm -rf outputs/*/
	@rm -rf logs/*.log
	@rm -rf examples/logs/*.log
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "✅ Cleanup completed"

clean-cache: ## Clear CUDA cache and temp files
	@echo "🧹 Clearing cache..."
	@PYTHONPATH=$(PYTHONPATH) python3 -c "import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else None; print('✅ CUDA cache cleared')"

device-info: ## Show device information
	@echo "🖥️ Device information..."
	@PYTHONPATH=$(PYTHONPATH) python3 -c "from training_pipeline.utils import DeviceUtils; DeviceUtils.print_device_info()"

memory-info: ## Show memory information
	@echo "💾 Memory information..."
	@PYTHONPATH=$(PYTHONPATH) python3 -c "from training_pipeline.utils import DeviceUtils; print(DeviceUtils.get_cuda_memory_info())"

# Configuration validation
validate-config: ## Validate training configuration file (usage: make validate-config CONFIG=configs/development.yaml)
	@echo "🔍 Validating configuration: $(CONFIG)..."
	@PYTHONPATH=$(PYTHONPATH) python3 -m training_pipeline.cli.train_gemma_v2 --config $(CONFIG) --dry-run

# Compare architectures
compare-configs: ## Compare old vs new configuration approaches
	@echo "📊 Configuration Comparison:"
	@echo "Old approach: 50+ CLI arguments, hardcoded defaults"
	@echo "New approach: 5-7 CLI arguments, YAML-driven config"
	@echo ""
	@echo "📁 Available configurations:"
	@ls -la configs/*.yaml 2>/dev/null || echo "No config files found in configs/"

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

# Testing and validation
test-architecture: ## Test and validate the new training pipeline architecture
	@echo "🧪 Testing new architecture..."
	@PYTHONPATH=$(PYTHONPATH) python3 test_new_architecture.py

test-imports: ## Test all training pipeline imports
	@echo "🧪 Testing training pipeline imports..."
	@PYTHONPATH=$(PYTHONPATH) python3 -c "from training_pipeline.core.enhanced_config import ConfigLoader; print('✅ ConfigLoader')"
	@PYTHONPATH=$(PYTHONPATH) python3 -c "from training_pipeline.core.training_manager import TrainingManager; print('✅ TrainingManager')"
	@PYTHONPATH=$(PYTHONPATH) python3 -c "from training_pipeline.factories import ModelFactory, DatasetFactory, TrainerFactory; print('✅ Factories')"
	@PYTHONPATH=$(PYTHONPATH) python3 -c "from training_pipeline.utils import setup_logging, get_logger, DeviceUtils; print('✅ Utils')"
	@echo "✅ All imports successful"

test-configs: ## Test all configuration files
	@echo "🧪 Testing all configuration files..."
	@for config in configs/*.yaml; do \
		echo "Testing $$config..."; \
		./train.sh $$config --dry-run || echo "❌ Failed: $$config"; \
	done
	@echo "✅ Configuration testing completed"

# Performance and monitoring
show-gpu-usage: ## Show current GPU usage
	@echo "🔥 Current GPU Usage:"
	@nvidia-smi 2>/dev/null || echo "nvidia-smi not available"

monitor-training: ## Monitor training logs (requires running training)
	@echo "👀 Monitoring training logs..."
	@tail -f logs/training.log 2>/dev/null || echo "No training logs found"

