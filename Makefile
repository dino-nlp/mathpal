# MathPal - Vietnamese Math Education AI Platform
# Simplified Makefile for core functionality

PYTHONPATH := $(shell pwd)/src

# ======================================
# ----------- Setup & Install ----------
# ======================================

help: ## Show this help message
	@grep -E '^[a-zA-Z0-9 -]+:.*#'  Makefile | sort | while read -r l; do printf "\033[1;32m$$(echo $$l | cut -f 1 -d':')\033[00m:$$(echo $$l | cut -f 2- -d'#')\n"; done

install: ## Install dependencies with Poetry
	@echo "📦 Installing dependencies..."
	@poetry env use 3.11
	@poetry install --without superlinked_rag
	@echo "✅ Installation completed"
## eval $(poetry env activate)

setup-env: ## Setup complete environment for finetuning with GPU support
	@echo "🚀 Setting up complete finetuning environment..."
	@pip install --upgrade pip
	@python3 scripts/setup_finetuning_env.py
	@echo "✅ Environment setup completed"

setup-thuegpu: ## Alias for setup-env (backward compatibility)
	@$(MAKE) setup-en

test-env: ## Test environment setup
	@echo "🧪 Testing environment setup..."
	@echo "🔍 Checking required environment variables..."
	@if [ -z "$$OPENROUTER_API_KEY" ]; then echo "❌ OPENROUTER_API_KEY not set"; exit 1; else echo "✅ OPENROUTER_API_KEY: $$(echo $$OPENROUTER_API_KEY | cut -c1-10)..."; fi
	@if [ -z "$$OPIK_API_KEY" ]; then echo "❌ OPIK_API_KEY not set"; exit 1; else echo "✅ OPIK_API_KEY: $$(echo $$OPIK_API_KEY | cut -c1-10)..."; fi
	@echo "✅ Environment setup is correct"

# ======================================
# ----------- Evaluation Pipeline ------
# ======================================

evaluate-quick: ## Run quick evaluation (3 samples)
	@echo "⚡ Starting quick evaluation..."
	@PYTHONPATH=$(PYTHONPATH) python3 -m src.inference_pipeline.mathpal

evaluate-llm: ## Run evaluation tests on the LLM model's performance using your Poetry env.
	@echo "⚡ Starting evaluation..."
	cd src/inference_pipeline && PYTHONPATH=$(PYTHONPATH) python -m evaluation.evaluate

evaluate-llm-progress: ## Run evaluation with progress tracking and detailed metrics
	@echo "📊 Starting evaluation with progress tracking..."
	cd src/inference_pipeline && PYTHONPATH=$(PYTHONPATH) python -m evaluation.evaluate --use_progress_metrics

evaluate-llm-quick: ## Run quick evaluation (5 samples) with progress tracking
	@echo "⚡ Starting quick evaluation with progress tracking..."
	cd src/inference_pipeline && PYTHONPATH=$(PYTHONPATH) python -m evaluation.evaluate --max_samples 5 --use_progress_metrics

evaluate-llm-fast: ## Run evaluation without progress tracking (faster)
	@echo "🚀 Starting fast evaluation without progress tracking..."
	cd src/inference_pipeline && PYTHONPATH=$(PYTHONPATH) python -m evaluation.evaluate --no_progress_tracking

evaluate-llm-custom: ## Run evaluation with custom parameters (usage: make evaluate-llm-custom SAMPLES=10 EXPERIMENT="My Test")
	@echo "🔧 Starting custom evaluation..."
	cd src/inference_pipeline && PYTHONPATH=$(PYTHONPATH) python -m evaluation.evaluate --max_samples $(SAMPLES) --experiment_name "$(EXPERIMENT)" --use_progress_metrics

check-gpu: ## Check GPU memory and provide recommendations
	@echo "🔍 Checking GPU memory and system resources..."
	@python3 check_gpu_memory.py

git acheck-compatibility: ## Check Torch/Unsloth compatibility and fix issues
	@echo "🔍 Checking Torch/Unsloth compatibility..."
	@python3 fix_torch_compatibility.py

test-model: ## Test model loading safely
	@echo "🔍 Testing model loading..."
	@python3 test_model_loading.py

test-evaluation: ## Test evaluation setup safely
	@echo "🔍 Testing evaluation setup..."
	@python3 test_evaluation.py

evaluate-llm-safe: ## Run evaluation with GPU memory check first
	@echo "🔍 Checking GPU memory before evaluation..."
	@python3 check_gpu_memory.py
	@echo ""
	@echo "🚀 Starting safe evaluation..."
	cd src/inference_pipeline && PYTHONPATH=$(PYTHONPATH) python -m evaluation.evaluate --max_samples 3 --use_progress_metrics

evaluate-llm-compatible: ## Run evaluation with compatibility check first
	@echo "🔍 Checking Torch/Unsloth compatibility..."
	@python3 fix_torch_compatibility.py
	@echo ""
	@echo "🚀 Starting compatible evaluation..."
	cd src/inference_pipeline && PYTHONPATH=$(PYTHONPATH) python -m evaluation.evaluate --max_samples 3 --use_progress_metrics

# ======================================
# ----------- Training Pipeline --------
# ======================================

train: ## Run training with production config
	@echo "🚀 Starting training..."
	@PYTHONPATH=$(PYTHONPATH) python3 -m src.training_pipeline.cli.train_gemma --config configs/production.yaml

train-quick: ## Run quick training test (20 steps)
	@echo "⚡ Starting quick training test..."
	@PYTHONPATH=$(PYTHONPATH) python3 -m src.training_pipeline.cli.train_gemma --config configs/production.yaml --max-steps 20

train-custom: ## Run training with custom config (usage: make train-custom CONFIG=path/to/config.yaml)
	@echo "📋 Starting training with config: $(CONFIG)..."
	@PYTHONPATH=$(PYTHONPATH) python3 -m src.training_pipeline.cli.train_gemma --config $(CONFIG)

# ======================================
# ----------- Data Pipeline ------------
# ======================================

crawl: ## Start data crawling
	@echo "🕷️ Starting data crawling..."
	@PYTHONPATH=$(PYTHONPATH) python3 -m src.data_crawling.main

process: ## Process crawled data
	@echo "⚙️ Processing data..."
	@PYTHONPATH=$(PYTHONPATH) python3 -m src.feature_pipeline.main

# ======================================
# ----------- Docker Commands ----------
# ======================================

docker-start: ## Start Docker infrastructure
	@echo "🐳 Starting Docker infrastructure..."
	@docker compose -f docker-compose.yml up --build -d

docker-stop: ## Stop Docker infrastructure
	@echo "🛑 Stopping Docker infrastructure..."
	@docker compose -f docker-compose.yml down --remove-orphans

docker-logs: ## Show Docker logs
	@echo "📋 Showing Docker logs..."
	@docker compose -f docker-compose.yml logs -f

# ======================================
# ----------- Utility Commands ---------
# ======================================

clean: ## Clean up generated files and cache
	@echo "🧹 Cleaning up..."
	@rm -rf evaluation_outputs/*/
	@rm -rf outputs/*/
	@rm -rf logs/*.log
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@python3 -c "import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else None; print('✅ CUDA cache cleared')"
	@echo "✅ Cleanup completed"

status: ## Show project status
	@echo "📊 MathPal Project Status"
	@echo "========================"
	@echo "🐍 Python: $$(python3 --version)"
	@echo "📦 Poetry: $$(poetry --version)"
	@echo "🔥 CUDA: $$(python3 -c 'import torch; print(torch.cuda.is_available())')"
	@echo "📁 Configs: $$(ls configs/*.yaml | wc -l) files"
	@echo "📊 Evaluation outputs: $$(ls evaluation_outputs/ 2>/dev/null | wc -l) directories"
	@echo "📈 Training outputs: $$(ls outputs/ 2>/dev/null | wc -l) directories"

info: ## Show detailed project information
	@echo "ℹ️ MathPal Project Information"
	@echo "============================="
	@echo "📁 Available configurations:"
	@ls -la configs/*.yaml 2>/dev/null || echo "No config files found"
	@echo ""
	@echo "📊 Available evaluation commands:"
	@echo "  make evaluate-quick              # Quick evaluation (3 samples)"
	@echo "  make evaluate-llm                # Standard evaluation"
	@echo "  make evaluate-llm-progress       # Evaluation with progress tracking"
	@echo "  make evaluate-llm-quick          # Quick evaluation with progress (5 samples)"
	@echo "  make evaluate-llm-fast           # Fast evaluation without progress tracking"
	@echo "  make evaluate-llm-custom         # Custom evaluation (usage: SAMPLES=10 EXPERIMENT=\"My Test\")"
	@echo "  make evaluate-llm-safe           # Safe evaluation with GPU memory check"
	@echo "  make evaluate-llm-compatible     # Evaluation with compatibility check"
	@echo "  make check-gpu                   # Check GPU memory and system resources"
	@echo "  make check-compatibility         # Check Torch/Unsloth compatibility"
	@echo "  make test-model                  # Test model loading safely"
	@echo "  make test-evaluation             # Test evaluation setup safely"
	@echo ""
	@echo "🚀 Available training commands:"
	@echo "  make train                       # Start training"
	@echo "  make train-quick                 # Quick training test"
	@echo "  make train-custom                # Custom training (usage: CONFIG=path/to/config.yaml)"
	@echo ""
	@echo "🛠️  Available utility commands:"
	@echo "  make crawl                       # Start data crawling"
	@echo "  make process                     # Process data"
	@echo "  make clean                       # Clean up files"
	@echo "  make status                      # Show project status"

.PHONY: help install setup-env test-env evaluate-quick evaluate-llm evaluate-llm-progress evaluate-llm-quick evaluate-llm-fast evaluate-llm-custom evaluate-llm-safe evaluate-llm-compatible check-gpu check-compatibility test-model test-evaluation train train-quick train-custom crawl process docker-start docker-stop docker-logs clean status info
