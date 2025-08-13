# 🧮 MathPal - Vietnamese Math Education AI Platform

A comprehensive AI platform for Vietnamese math education, featuring data crawling, processing, model training, and inference capabilities. MathPal is designed to help students learn mathematics through intelligent tutoring and personalized assistance.

## 🌟 Features

### 📊 Data Pipeline
- **Web Crawling**: Automated data collection from Vietnamese math education websites
- **Data Processing**: Intelligent cleaning, chunking, and embedding of math problems
- **Stream Processing**: Real-time data flow with Bytewax integration
- **Quality Control**: Automated validation and filtering of educational content

### 🤖 AI Training Pipeline
- **Modular Architecture**: Clean, maintainable, and extensible training pipeline
- **Hardware Optimization**: Pre-configured for Tesla T4 (16GB) and A100 (40GB/80GB)
- **Unsloth Integration**: Optimized speed and memory usage
- **Multiple Training Methods**: Support for LoRA, QLoRA, and full fine-tuning
- **Experiment Tracking**: Comet ML integration for experiment monitoring
- **Flexible Configuration**: YAML-based configuration management
- **Multiple Save Formats**: Model saving in various formats (LoRA, merged, GGUF)

### 🎯 Inference Engine
- **Real-time Generation**: Fast inference with optimized models
- **Vietnamese Math Focus**: Specialized for Vietnamese mathematics
- **Batch Processing**: Efficient handling of multiple queries
- **Quality Assessment**: Built-in evaluation metrics

## 🏗️ Project Architecture

```
mathpal/
├── src/
│   ├── data_crawling/           # Web crawling and data collection
│   │   ├── crawlers/            # Website-specific crawlers
│   │   ├── dispatcher.py        # Crawling orchestration
│   │   └── main.py             # Crawling entry point
│   │
│   ├── feature_pipeline/        # Data processing and feature engineering
│   │   ├── data_flow/          # Stream processing with Bytewax
│   │   ├── data_logic/         # Data transformation logic
│   │   ├── models/             # Data models and schemas
│   │   ├── utils/              # Processing utilities
│   │   └── main.py             # Pipeline orchestration
│   │
│   ├── training_pipeline/       # Model training and fine-tuning
│   │   ├── config/             # Configuration management
│   │   ├── managers/           # Training orchestration
│   │   ├── factories/          # Object factories
│   │   ├── training/           # Training logic
│   │   ├── inference/          # Inference engine
│   │   ├── experiments/        # Experiment tracking
│   │   ├── cli/               # Command line interface
│   │   └── utils/             # Training utilities
│   │
│   └── core/                   # Shared core components
│       ├── db/                # Database connections
│       ├── rag/               # Retrieval-Augmented Generation
│       ├── mq/                # Message queue integration
│       └── utils/             # Shared utilities
│
├── configs/                    # Configuration files
│   ├── tesla_t4_optimized.yaml # Tesla T4 optimized config
│   ├── a100_optimized.yaml     # A100 optimized config
│   ├── quick_test.yaml         # Quick development testing
│   ├── production.yaml         # Production configuration
│   └── README_OPTIMIZATION.md  # Hardware optimization guide
│
├── data/                       # Data storage
├── outputs/                    # Training outputs
├── notebooks/                  # Jupyter notebooks
└── scripts/                    # Utility scripts
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- CUDA 11.8+ (recommended)
- GPU with at least 8GB VRAM
- Poetry (for dependency management)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd mathpal

# Install dependencies
make install

# Setup environment
make setup-env

# Check environment
make env-check
```

### Data Collection

```bash
# Start data crawling
make local-start

# Crawl specific websites
make local-test-crawler

# Ingest data from links
make local-ingest-data
```

### Model Training

```bash
# Tesla T4 (16GB VRAM)
make train-tesla-t4

# A100 (40GB/80GB VRAM)
make train-a100

# Quick test
make train-quick

# Production training
make train-prod
```

## 🔧 Hardware Optimization

MathPal provides pre-optimized configurations for different hardware:

### Tesla T4 (16GB VRAM)
- **Model**: Gemma-3n-E2B (smaller)
- **Mixed Precision**: fp16 (required)
- **Quantization**: 4-bit
- **Batch Size**: 2 + gradient accumulation 8
- **Training Time**: ~2-3 hours

### A100 (40GB/80GB VRAM)
- **Model**: Gemma-3n-E4B (full)
- **Mixed Precision**: bf16 (optimal)
- **Quantization**: 4-bit
- **Batch Size**: 8 + gradient accumulation 4
- **Training Time**: ~30-45 minutes

```bash
# View hardware optimization guide
make show-hardware-guide

# List all configurations
make list-configs
```

## 📊 Data Pipeline

### Web Crawling

The data crawling system automatically collects Vietnamese math problems from educational websites:

```python
from src.data_crawling.main import start_crawling

# Start crawling with configuration
start_crawling(
    websites=["loigiaihay.com", "vietjack.com"],
    grade_levels=["grade_5", "grade_6"],
    max_pages=1000
)
```

### Data Processing

The feature pipeline processes raw data into training-ready format:

```python
from src.feature_pipeline.main import run_pipeline

# Process data with streaming
run_pipeline(
    input_data="data/raw/",
    output_data="data/processed/",
    chunk_size=512,
    embedding_model="sentence-transformers/all-MiniLM-L6-v2"
)
```

## 🤖 Training Pipeline

### Configuration Management

MathPal uses a unified configuration system:

```yaml
# configs/tesla_t4_optimized.yaml
model:
  name: "unsloth/gemma-3n-E2B-it"
  max_seq_length: 1024
  load_in_4bit: true

training:
  max_steps: 200
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 8
  learning_rate: 3.0e-4

lora:
  r: 16
  alpha: 32
  dropout: 0.0
```

### Training Commands

```bash
# Hardware-specific training
make train-tesla-t4          # Tesla T4 optimized
make train-a100              # A100 optimized

# Development and testing
make train-quick             # Quick test (20 steps)
make train-dry-run-tesla-t4  # Validate Tesla T4 config

# Custom training
make train-custom-config CONFIG=my-config.yaml
make train-custom EXPERIMENT=my-experiment
```

### Experiment Tracking

```python
from src.training_pipeline.experiments import CometTracker

# Setup experiment tracking
tracker = CometTracker(
    workspace="mathpal",
    project="vietnamese-math",
    experiment_name="tesla-t4-training"
)

# Training automatically logs metrics
trainer.train()
```

## 🎯 Inference

### Basic Inference

```python
from src.training_pipeline.inference import InferenceEngine

# Load trained model
engine = InferenceEngine(
    model_path="outputs/mathpal-tesla-t4-optimized/",
    device="cuda"
)

# Generate responses
questions = [
    "Tính tổng của 15 + 27 = ?",
    "Một hình chữ nhật có chiều dài 8cm và chiều rộng 5cm. Tính chu vi.",
    "Lan có 24 cái kẹo, Lan cho bạn 8 cái. Hỏi Lan còn lại bao nhiêu?"
]

for question in questions:
    answer = engine.generate(question)
    print(f"Q: {question}")
    print(f"A: {answer}\n")
```

### Batch Processing

```python
# Process multiple questions efficiently
answers = engine.generate_batch(
    questions=questions,
    batch_size=4,
    temperature=0.7
)
```

### Quality Evaluation

```python
# Evaluate model performance
evaluation_results = engine.evaluate_model(
    test_dataset="data/test/",
    metrics=["accuracy", "bleu", "rouge"]
)

print(f"Accuracy: {evaluation_results['accuracy']:.2f}")
print(f"BLEU Score: {evaluation_results['bleu']:.2f}")
```

## 📈 Monitoring and Debugging

### Environment Information

```bash
# Check system information
make env-info

# Monitor GPU usage
make show-gpu-usage

# Check memory usage
make memory-info
```

### Training Monitoring

```bash
# Monitor training logs
make monitor-training

# Test configurations
make test-configs

# Validate specific config
make validate-config CONFIG=configs/tesla_t4_optimized.yaml
```

## 🧪 Testing

### Quick Testing

```bash
# Quick test on Tesla T4
make train-tesla-t4-quick

# Quick test on A100
make train-a100-quick

# Test all configurations
make test-hardware-configs
```

### Data Pipeline Testing

```bash
# Test crawler
make local-test-crawler

# Test retriever
make local-test-retriever
```

## 📊 Performance Benchmarks

| Hardware | Training Time | VRAM Usage | Throughput | Model Size |
|----------|---------------|------------|------------|------------|
| Tesla T4 | ~2-3 hours | ~12-14GB | ~2-3 samples/sec | 2B parameters |
| A100 | ~30-45 min | ~25-35GB | ~8-12 samples/sec | 4B parameters |

## 🔧 Customization

### Adding New Data Sources

```python
from src.data_crawling.crawlers.base import BaseCrawler

class CustomCrawler(BaseCrawler):
    def extract_math_problems(self, page_content):
        # Custom extraction logic
        return problems
```

### Custom Training Configuration

```python
from src.training_pipeline.config import ConfigManager

# Create custom config
config = ConfigManager()
config.model.name = "custom-model"
config.training.max_steps = 500
config.save_config("configs/custom.yaml")
```

### Custom Evaluation Metrics

```python
from src.training_pipeline.managers import EvaluationManager

class CustomEvaluator(EvaluationManager):
    def custom_metric(self, predictions, targets):
        # Custom evaluation logic
        return score
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Setup development environment
make setup-env

# Run tests
make test-configs
make test-hardware-configs

# Check code quality
make env-check
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Unsloth](https://github.com/unslothai/unsloth) - Fast LLM fine-tuning
- [TRL](https://github.com/huggingface/trl) - Transformer Reinforcement Learning
- [PEFT](https://github.com/huggingface/peft) - Parameter-Efficient Fine-Tuning
- [Comet ML](https://www.comet.ml/) - Experiment tracking
- [Gemma](https://deepmind.google/technologies/gemma/) - Base model
- [Bytewax](https://bytewax.io/) - Stream processing
- [Vietnamese Math Education Community](https://loigiaihay.com/) - Data sources

## 📧 Contact

- **Project Maintainer**: [Your Name]
- **Email**: your.email@example.com
- **Project Link**: [https://github.com/username/mathpal](https://github.com/username/mathpal)

## 📚 Documentation

- [Hardware Optimization Guide](configs/README_OPTIMIZATION.md)
- [Makefile Updates](MAKEFILE_UPDATES.md)
- [Training Pipeline Architecture](src/training_pipeline/README.md)
- [Data Pipeline Documentation](src/data_crawling/README.md)

---

**MathPal** - Empowering Vietnamese students with AI-powered math education! 🧮✨