include .env

$(eval export $(shell sed -ne 's/ *#.*$$//; /./ s/=.*$$// p' .env))

PYTHONPATH := $(shell pwd)/src

install: # Create a local Poetry virtual environment and install all required Python dependencies.
	poetry env use 3.11
	poetry install --without superlinked_rag
# 	eval $(poetry env activate)

# Default target
help: ## Show this help message
	@echo "🚀 Gemma3N Training Pipeline Commands"
	@echo "=" * 50
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# # Environment setup
setup_training_env: ## Setup the training environment
	@echo "🔧 Setting up environment..."
	cd scripts && poetry run python setup_environment.py

# install_training_dependenceies: ## Install dependencies manually
# 	@echo "📦 Installing dependencies..."
# 	pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# 	pip install --no-deps xformers==0.0.29.post3 bitsandbytes accelerate peft trl
# 	pip install --no-deps unsloth
# 	pip install transformers datasets tokenizers sentencepiece protobuf
# 	pip install comet-ml wandb pyyaml rich click

# ======================================
# ------- Docker Infrastructure --------
# ======================================

local-start: # Build and start your local Docker infrastructure.
	docker compose -f docker-compose.yml up --build -d

local-stop: # Stop your local Docker infrastructure.
	docker compose -f docker-compose.yml down --remove-orphans

local-test-crawler: # Make a call to your local AWS Lambda (hosted in Docker) to crawl a Medium article.
	curl -X POST "http://localhost:9010/2015-03-31/functions/function/invocations" \
	  	-d '{"grade_name": "grade_5", "link": "https://loigiaihay.com/de-thi-vao-lop-6-mon-toan-truong-cau-giay-nam-2023-a142098.html"}'

local-ingest-data: # Ingest all links from data/links.txt by calling your local AWS Lambda hosted in Docker.
	while IFS= read -r link; do \
		echo "Processing: $$link"; \
		curl -X POST "http://localhost:9010/2015-03-31/functions/function/invocations" \
			-d "{\"grade_name\": \"grade_5\", \"link\": \"$$link\"}"; \
		echo "\n"; \
		sleep 2; \
	done < data/links.txt

# ======================================
# -------- RAG Feature Pipeline --------
# ======================================

local-test-retriever: # Test the RAG retriever using your Poetry env
	cd src/feature_pipeline && poetry run python -m retriever

local-generate-instruct-dataset: # Generate the fine-tuning instruct dataset using your Poetry env.
	cd src/feature_pipeline && poetry run python -m generate_dataset.generate

# Testing
test: ## Run quick test
	@echo "🧪 Running quick test..."
	cd scripts && poetry run python quick_test.py

test-basic: ## Run basic usage example
	@echo "📝 Running basic usage example..."
	python examples/basic_usage.py

test-advanced: ## Run advanced usage example  
	@echo "🔬 Running advanced usage example..."
	python examples/advanced_usage.py

# Evaluation
eval: ## Run evaluation with default settings (Opik required)
	@echo "🧮 Running evaluation..."
	python -m evaluation_pipeline.cli --dataset ngohongthai/exam-sixth_grade-instruct-dataset --split test --max-samples 100 --experiment-name gemma3n-math-eval

eval-model: ## Run evaluation using a specific HF model repo (merged or LoRA)
	@echo "🧮 Running evaluation on model..."
	python -m evaluation_pipeline.cli --dataset ngohongthai/exam-sixth_grade-instruct-dataset --split test --max-samples 100 --experiment-name gemma3n-math-eval --model-repo unsloth/gemma-3n-E4B-it

# Training commands
train: ## Run training with default config
	@echo "🚀 Starting training..."
	python -m training_pipeline.cli.train_gemma

train-dev: ## Run development training
	@echo "🛠️ Starting development training..."
	python -m training_pipeline.cli.train_gemma --config configs/development.yaml

train-prod: ## Run production training
	@echo "🏭 Starting production training..."
	python -m training_pipeline.cli.train_gemma --config configs/production.yaml

train-quick: ## Run quick test training
	@echo "⚡ Starting quick test training..."
	python -m training_pipeline.cli.train_gemma --quick-test

# Configuration
config-validate: ## Validate configuration files
	@echo "✅ Validating configurations..."
	@python -c "from training_pipeline.config import TrainingConfig; TrainingConfig.from_yaml('configs/training_config.yaml').validate(); print('✅ training_config.yaml is valid')"
	@python -c "from training_pipeline.config import TrainingConfig; TrainingConfig.from_yaml('configs/development.yaml').validate(); print('✅ development.yaml is valid')"
	@python -c "from training_pipeline.config import TrainingConfig; TrainingConfig.from_yaml('configs/production.yaml').validate(); print('✅ production.yaml is valid')"

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

# Development
format: ## Format code with black
	@echo "🎨 Formatting code..."
	black src/ examples/ scripts/ --line-length 100

lint: ## Lint code with flake8
	@echo "🔍 Linting code..."
	flake8 src/ examples/ scripts/ --max-line-length 100 --ignore E203,W503

check: format lint ## Format and lint code

# Documentation
docs: ## Generate documentation
	@echo "📚 Generating documentation..."
	@echo "Documentation available in README.md"
	@echo "Example configurations in configs/"
	@echo "Usage examples in examples/"

# Model management
list-models: ## List saved models
	@echo "📦 Saved models:"
	@find outputs/ -name "*.bin" -o -name "*.safetensors" -o -name "*.gguf" 2>/dev/null | head -20 || echo "No models found"

clean-models: ## Remove saved models
	@echo "🗑️ Removing saved models..."
	find outputs/ -name "*.bin" -delete 2>/dev/null || true
	find outputs/ -name "*.safetensors" -delete 2>/dev/null || true
	find outputs/ -name "*.gguf" -delete 2>/dev/null || true
	@echo "✅ Models removed"

# Environment
env-check: ## Check environment and dependencies
	@echo "🔍 Checking environment..."
	python -c "import sys; print(f'Python: {sys.version}')"
	python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
	python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
	python -c "import peft; print(f'PEFT: {peft.__version__}')"
	python -c "import trl; print(f'TRL: {trl.__version__}')"

requirements: ## Generate requirements.txt
	@echo "📋 Generating requirements.txt..."
	pip freeze > requirements.txt
	@echo "✅ requirements.txt generated"

# Monitoring
logs: ## Show recent logs
	@echo "📊 Recent logs:"
	@tail -n 50 logs/*.log 2>/dev/null || echo "No logs found"

logs-follow: ## Follow logs in real-time
	@echo "📊 Following logs..."
	@tail -f logs/*.log 2>/dev/null || echo "No logs to follow"

# Git helpers
git-status: ## Show git status with useful info
	@echo "📂 Git status:"
	git status --short
	@echo "\n📝 Recent commits:"
	git log --oneline -5

commit: ## Quick commit with message
	@read -p "Commit message: " msg; git add -A && git commit -m "$$msg"

# Performance
benchmark: ## Run performance benchmark
	@echo "🏃‍♂️ Running benchmark..."
	python -c "from training_pipeline.utils import DeviceUtils; DeviceUtils.benchmark_device()"

profile: ## Profile training performance
	@echo "📊 Profiling training..."
	python -m cProfile -o profile.stats -m training_pipeline.cli.train_gemma --quick-test
	@echo "Profile saved to profile.stats"

# Jupyter notebooks (if needed)
notebook: ## Start Jupyter notebook server
	@echo "📓 Starting Jupyter notebook..."
	jupyter notebook notebooks/

lab: ## Start JupyterLab server
	@echo "🧪 Starting JupyterLab..."
	jupyter lab notebooks/