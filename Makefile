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

env-check: ## Check environment for training pipeline dependencies
	@echo "🔍 Checking environment architecture..."
	@python3 -c "import sys; print(f'Python: {sys.version}')"
	@python3 -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
	@python3 -c "import transformers; print(f'Transformers: {transformers.__version__}')"
	@python3 -c "import peft; print(f'PEFT: {peft.__version__}')"
	@python3 -c "import trl; print(f'TRL: {trl.__version__}')"
	@python3 -c "import datasets; print(f'Datasets: {datasets.__version__}')"
	@python3 -c "import yaml; print(f'PyYAML: {yaml.__version__}')"
	@echo "🧪 Testing training pipeline imports..."
	@PYTHONPATH=$(PYTHONPATH) python3 -c "from training_pipeline.config.config_manager import ConfigManager; print('✅ ConfigManager')"
	@PYTHONPATH=$(PYTHONPATH) python3 -c "from training_pipeline.utils.training_manager import TrainingManager; print('✅ Training manager')"
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
	@PYTHONPATH=$(PYTHONPATH) python3 -c "from training_pipeline.config.config_manager import ConfigManager; print('✅ ConfigManager')"
	@PYTHONPATH=$(PYTHONPATH) python3 -c "from training_pipeline.managers.training_manager import TrainingManager; print('✅ Training Manager')"
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
train: ## Run training with unified config
	@echo "🚀 Starting training with unified config..."
	@PYTHONPATH=$(PYTHONPATH) python3 -m training_pipeline.cli.train_gemma --config configs/unified_training_config.yaml

train-quick: ## Run quick test training (20 steps)
	@echo "⚡ Starting quick test training..."
	@PYTHONPATH=$(PYTHONPATH) python3 -m training_pipeline.cli.train_gemma --config configs/quick_test.yaml

train-prod: ## Run production training with full features
	@echo "🏭 Starting production training..."
	@PYTHONPATH=$(PYTHONPATH) python3 -m training_pipeline.cli.train_gemma --config configs/production.yaml

# Hardware-optimized training commands
train-tesla-t4: ## Run training optimized for Tesla T4 (16GB VRAM)
	@echo "🔧 Starting Tesla T4 optimized training..."
	@PYTHONPATH=$(PYTHONPATH) python3 -m training_pipeline.cli.train_gemma --config configs/tesla_t4_optimized.yaml

train-a100: ## Run training optimized for A100 (40GB/80GB VRAM)
	@echo "🚀 Starting A100 optimized training..."
	@PYTHONPATH=$(PYTHONPATH) python3 -m training_pipeline.cli.train_gemma --config configs/a100_optimized.yaml

train-tesla-t4-quick: ## Run quick test on Tesla T4 config
	@echo "⚡ Starting Tesla T4 quick test..."
	@PYTHONPATH=$(PYTHONPATH) python3 -m training_pipeline.cli.train_gemma --config configs/tesla_t4_optimized.yaml --quick-test

train-a100-quick: ## Run quick test on A100 config
	@echo "⚡ Starting A100 quick test..."
	@PYTHONPATH=$(PYTHONPATH) python3 -m training_pipeline.cli.train_gemma --config configs/a100_optimized.yaml --quick-test

# Validation and testing
train-dry-run: ## Validate config and estimate resources without training
	@echo "🔍 Running training dry run..."
	@PYTHONPATH=$(PYTHONPATH) python3 -m training_pipeline.cli.train_gemma --config configs/unified_training_config.yaml --dry-run

train-dry-run-quick: ## Dry run with quick test config
	@echo "🔍 Running quick test dry run..."
	@PYTHONPATH=$(PYTHONPATH) python3 -m training_pipeline.cli.train_gemma --config configs/quick_test.yaml --dry-run

train-dry-run-prod: ## Dry run with production config
	@echo "🔍 Running production dry run..."
	@PYTHONPATH=$(PYTHONPATH) python3 -m training_pipeline.cli.train_gemma --config configs/production.yaml --dry-run

# Hardware-optimized dry run commands
train-dry-run-tesla-t4: ## Dry run with Tesla T4 config
	@echo "🔍 Running Tesla T4 dry run..."
	@PYTHONPATH=$(PYTHONPATH) python3 -m training_pipeline.cli.train_gemma --config configs/tesla_t4_optimized.yaml --dry-run

train-dry-run-a100: ## Dry run with A100 config
	@echo "🔍 Running A100 dry run..."
	@PYTHONPATH=$(PYTHONPATH) python3 -m training_pipeline.cli.train_gemma --config configs/a100_optimized.yaml --dry-run

# Custom training with any config
train-custom-config: ## Run training with custom config (usage: make train-custom-config CONFIG=my-config.yaml)
	@echo "📋 Starting training with custom config: $(CONFIG)..."
	@PYTHONPATH=$(PYTHONPATH) python3 -m training_pipeline.cli.train_gemma --config $(CONFIG)

# Quick test mode with any config
train-quick-test: ## Apply quick test overrides to any config (usage: make train-quick-test CONFIG=configs/production.yaml)
	@echo "⚡ Starting quick test with config: $(or $(CONFIG),configs/unified_training_config.yaml)..."
	@PYTHONPATH=$(PYTHONPATH) python3 -m training_pipeline.cli.train_gemma --config $(or $(CONFIG),configs/unified_training_config.yaml) --quick-test

# Training with overrides
train-custom: ## Run training with custom experiment name (usage: make train-custom EXPERIMENT=my-experiment)
	@echo "🎯 Starting custom training: $(EXPERIMENT)..."
	@PYTHONPATH=$(PYTHONPATH) python3 -m training_pipeline.cli.train_gemma --config configs/unified_training_config.yaml --experiment-name $(EXPERIMENT)

train-steps: ## Run training with custom max steps (usage: make train-steps STEPS=500)
	@echo "🎯 Starting training with $(STEPS) steps..."
	@PYTHONPATH=$(PYTHONPATH) python3 -m training_pipeline.cli.train_gemma --config configs/unified_training_config.yaml --max-steps $(STEPS)

train-output: ## Run training with custom output directory (usage: make train-output OUTPUT=my-output)
	@echo "📁 Starting training with output: $(OUTPUT)..."
	@PYTHONPATH=$(PYTHONPATH) python3 -m training_pipeline.cli.train_gemma --config configs/unified_training_config.yaml --output-dir $(OUTPUT)

# Advanced training options
train-debug: ## Run training with debug logging
	@echo "🐛 Starting training with debug logging..."
	@PYTHONPATH=$(PYTHONPATH) python3 -m training_pipeline.cli.train_gemma --config configs/unified_training_config.yaml --debug

train-no-comet: ## Run training without Comet ML tracking
	@echo "🚫 Starting training without Comet ML..."
	@PYTHONPATH=$(PYTHONPATH) python3 -m training_pipeline.cli.train_gemma --config configs/unified_training_config.yaml --no-comet


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
validate-config: ## Validate training configuration file (usage: make validate-config CONFIG=configs/quick_test.yaml)
	@echo "🔍 Validating configuration: $(CONFIG)..."
	@PYTHONPATH=$(PYTHONPATH) python3 -m training_pipeline.cli.train_gemma --config $(CONFIG) --dry-run

# Compare architectures
compare-configs: ## Compare old vs new configuration approaches
	@echo "📊 Configuration Comparison:"
	@echo "Old approach: 50+ CLI arguments, hardcoded defaults"
	@echo "New approach: 5-7 CLI arguments, YAML-driven config"
	@echo ""
	@echo "📁 Available configurations:"
	@ls -la configs/*.yaml 2>/dev/null || echo "No config files found in configs/"

# Hardware optimization guide
show-hardware-guide: ## Show hardware optimization guide
	@echo "🔧 Hardware Optimization Guide"
	@echo "=============================="
	@echo ""
	@echo "🚀 Tesla T4 Optimizations (16GB VRAM):"
	@echo "   • Model: Gemma-3n-E2B (smaller)"
	@echo "   • Mixed Precision: fp16 (required)"
	@echo "   • Quantization: 4-bit"
	@echo "   • Batch Size: 2 + gradient accumulation 8"
	@echo "   • LoRA Rank: 16"
	@echo "   • Usage: make train-tesla-t4"
	@echo ""
	@echo "🔥 A100 Optimizations (40GB/80GB VRAM):"
	@echo "   • Model: Gemma-3n-E4B (full)"
	@echo "   • Mixed Precision: bf16 (optimal)"
	@echo "   • Quantization: 4-bit"
	@echo "   • Batch Size: 8 + gradient accumulation 4"
	@echo "   • LoRA Rank: 32"
	@echo "   • Usage: make train-a100"
	@echo ""
	@echo "📊 Performance Expectations:"
	@echo "   Tesla T4: ~2-3 hours, ~12-14GB VRAM"
	@echo "   A100: ~30-45 minutes, ~25-35GB VRAM"
	@echo ""
	@echo "📖 For detailed info: cat configs/README_OPTIMIZATION.md"

# Show architecture overview
show-architecture: ## Show current training pipeline architecture
	@echo "🏗️ MathPal Training Pipeline - Enhanced Architecture"
	@echo "===================================================="
	@echo ""
	@echo "✅ CURRENT ARCHITECTURE (train_gemma.py):"
	@echo "   ✅ Modular: ConfigManager + Factory + Manager pattern"
	@echo "   ✅ Type-safe: Config sections with validation"
	@echo "   ✅ Clean CLI: 5-7 arguments + comprehensive YAML config"
	@echo "   ✅ Unified config: Handles all legacy formats seamlessly"
	@echo "   ✅ Dependency injection: Managers get only needed configs"
	@echo "   ✅ Robust: Comprehensive validation + error handling"
	@echo ""
	@echo "📁 Available Configurations:"
	@echo "   🚀 unified_training_config.yaml - Full configuration template"
	@echo "   ⚡ quick_test.yaml - Quick development testing (20 steps)"
	@echo "   🏭 production.yaml - Production with Comet ML + Hub push"
	@echo "   🔧 tesla_t4_optimized.yaml - Tesla T4 optimized (16GB VRAM)"
	@echo "   🔥 a100_optimized.yaml - A100 optimized (40GB/80GB VRAM)"
	@echo ""
	@echo "📁 Usage Examples:"
	@echo "   Quick test:  make train-quick"
	@echo "   Production:  make train-prod"
	@echo "   Tesla T4:    make train-tesla-t4"
	@echo "   A100:        make train-a100"
	@echo "   Custom:      make train-custom EXPERIMENT=my-exp"
	@echo "   Dry run:     make train-dry-run"
	@echo "   With config: make train-custom-config CONFIG=my-config.yaml"
	@echo ""

# List all available configurations
list-configs: ## List all available configuration files with descriptions
	@echo "📁 Available Configuration Files:"
	@echo "=================================="
	@echo ""
	@echo "🔧 Hardware Optimized:"
	@echo "   tesla_t4_optimized.yaml - Tesla T4 (16GB VRAM) optimized config"
	@echo "   a100_optimized.yaml     - A100 (40GB/80GB VRAM) optimized config"
	@echo ""
	@echo "⚡ Development:"
	@echo "   quick_test.yaml         - Quick test (20 steps) for development"
	@echo ""
	@echo "🏭 Production:"
	@echo "   production.yaml         - Full production with Comet ML + Hub"
	@echo "   unified_training_config.yaml - Template for custom configs"
	@echo ""
	@echo "📖 Usage:"
	@echo "   make train-tesla-t4     # Train on Tesla T4"
	@echo "   make train-a100         # Train on A100"
	@echo "   make train-quick        # Quick test"
	@echo "   make train-prod         # Production training"
	@echo "   make show-hardware-guide # Detailed optimization guide"

# Testing and validation
test-architecture: ## Test and validate the new training pipeline architecture
	@echo "🧪 Testing new architecture..."
	@PYTHONPATH=$(PYTHONPATH) python3 test_new_architecture.py

test-imports: ## Test all training pipeline imports
	@echo "🧪 Testing training pipeline imports..."
	@PYTHONPATH=$(PYTHONPATH) python3 -c "from training_pipeline.config.config_manager import ConfigManager; print('✅ ConfigManager')"
	@PYTHONPATH=$(PYTHONPATH) python3 -c "from training_pipeline.managers.training_manager import TrainingManager; print('✅ TrainingManager')"
	@PYTHONPATH=$(PYTHONPATH) python3 -c "from training_pipeline.factories import ModelFactory, DatasetFactory, TrainerFactory; print('✅ Factories')"
	@PYTHONPATH=$(PYTHONPATH) python3 -c "from training_pipeline.utils import setup_logging, get_logger, DeviceUtils; print('✅ Utils')"
	@echo "✅ All imports successful"

test-configs: ## Test all configuration files
	@echo "🧪 Testing all configuration files..."
	@for config in configs/*.yaml; do \
		echo "Testing $$config..."; \
		PYTHONPATH=$(PYTHONPATH) python3 -m training_pipeline.cli.train_gemma --config $$config --dry-run || echo "❌ Failed: $$config"; \
	done
	@echo "✅ Configuration testing completed"

test-hardware-configs: ## Test hardware-optimized configurations
	@echo "🧪 Testing hardware-optimized configurations..."
	@echo "Testing Tesla T4 config..."
	@PYTHONPATH=$(PYTHONPATH) python3 -m training_pipeline.cli.train_gemma --config configs/tesla_t4_optimized.yaml --dry-run
	@echo "Testing A100 config..."
	@PYTHONPATH=$(PYTHONPATH) python3 -m training_pipeline.cli.train_gemma --config configs/a100_optimized.yaml --dry-run
	@echo "✅ Hardware configuration testing completed"

test-config-manager: ## Test ConfigManager with demo script
	@echo "🧪 Testing ConfigManager system..."
	@PYTHONPATH=$(PYTHONPATH) python3 examples/config_manager_demo.py

# Performance and monitoring
show-gpu-usage: ## Show current GPU usage
	@echo "🔥 Current GPU Usage:"
	@nvidia-smi 2>/dev/null || echo "nvidia-smi not available"

monitor-training: ## Monitor training logs (requires running training)
	@echo "👀 Monitoring training logs..."
	@tail -f logs/training.log 2>/dev/null || echo "No training logs found"

