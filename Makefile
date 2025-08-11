include .env

$(eval export $(shell sed -ne 's/ *#.*$$//; /./ s/=.*$$// p' .env))

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

