# MathPal Evaluation Pipeline

A comprehensive evaluation pipeline for Vietnamese math education AI models, designed to assess the quality and effectiveness of AI models in Vietnamese mathematics education.

## 🎯 Overview

The MathPal Evaluation Pipeline provides a complete framework for evaluating Vietnamese math AI models using multiple evaluation approaches:

- **Opik Metrics**: Industry-standard evaluation metrics
- **Vietnamese Math Metrics**: Specialized metrics for Vietnamese mathematics
- **LLM-as-a-Judge**: AI-powered evaluation using OpenRouter
- **Custom Educational Metrics**: Tailored for grade 5-6 transition assessment

## 🏗️ Architecture

```
src/evaluation_pipeline/
├── cli/                    # Command-line interface
│   ├── __init__.py
│   └── main.py            # Main CLI entry point
├── config/                # Configuration management
│   ├── __init__.py
│   ├── config_manager.py  # Configuration manager
│   └── evaluation_config.yaml
├── managers/              # Core evaluation logic
│   ├── __init__.py
│   ├── evaluation_manager.py
│   ├── dataset_manager.py
│   └── metrics_manager.py
├── factories/             # Factory patterns
│   ├── __init__.py
│   ├── model_factory.py
│   ├── evaluator_factory.py
│   └── provider_factory.py
├── evaluators/            # Evaluation engines
│   ├── __init__.py
│   ├── opik_evaluator.py
│   ├── vietnamese_math_metrics.py
│   └── custom_metrics.py
├── providers/             # LLM providers
│   ├── __init__.py
│   ├── openrouter_provider.py
│   └── fallback_provider.py
├── inference/             # Model inference
│   ├── __init__.py
│   ├── matformer_utils.py
│   ├── gemma3n_inference.py
│   └── batch_inference.py
├── utils/                 # Utilities
│   ├── __init__.py
│   ├── exceptions.py
│   ├── logging.py
│   └── helpers.py
└── __init__.py
```

## 🚀 Quick Start

### Installation

The evaluation pipeline is part of the MathPal project. Ensure you have the required dependencies:

```bash
# Install dependencies
pip install torch transformers accelerate bitsandbytes click pydantic structlog pyyaml requests

# Setup environment variables
cp env.example .env
# Edit .env file with your actual API keys
```

### Environment Variables Setup

For security reasons, API keys are read from environment variables instead of config files:

1. **Copy the example environment file:**
   ```bash
   cp env.example .env
   ```

2. **Edit `.env` file with your actual API keys:**
   ```bash
   # OpenRouter API Key for LLM-as-a-judge evaluation
   # Get your API key from: https://openrouter.ai/keys
   OPENROUTER_API_KEY=sk-or-v1-your-actual-api-key-here
   
   # Opik API Key for evaluation metrics
   # Get your API key from: https://opik.ai
   OPIK_API_KEY=your-actual-opik-api-key-here
   
   # OpenAI API Key (optional, for fallback)
   # Get your API key from: https://platform.openai.com/api-keys
   OPENAI_API_KEY=sk-proj-your-actual-openai-api-key-here
   ```

3. **Load environment variables:**
   ```bash
   # Option 1: Source the .env file
   source .env
   
   # Option 2: Use poetry environment
   eval $(poetry env activate)
   ```

### Basic Usage

#### Using Makefile (Recommended)

```bash
# Quick evaluation
make evaluate-quick

# Comprehensive evaluation
make evaluate-comprehensive

# Tesla T4 optimized evaluation
make evaluate-tesla-t4

# A100 optimized evaluation
make evaluate-a100

# Custom model evaluation
make evaluate-custom MODEL=/path/to/your/model
```

#### Using CLI Directly

```bash
# Basic evaluation
python -m src.evaluation_pipeline.cli.main evaluate -m /path/to/model

# With custom config
python -m src.evaluation_pipeline.cli.main -c configs/evaluation_config.yaml evaluate -m /path/to/model

# Quick evaluation with limited samples
python -m src.evaluation_pipeline.cli.main evaluate -m /path/to/model --mode quick --max-samples 10

# Dry run (no actual evaluation)
python -m src.evaluation_pipeline.cli.main evaluate -m /path/to/model --dry-run

# Evaluation with Hugging Face dataset
python -m src.evaluation_pipeline.cli.main evaluate -m /path/to/model -d username/dataset_name

# Evaluation with specific Hugging Face dataset split
python -m src.evaluation_pipeline.cli.main evaluate -m /path/to/model -d username/dataset_name --split test
```

## 📋 Configuration

### Configuration Files

The pipeline uses YAML configuration files for easy customization:

- **`configs/evaluation_config.yaml`**: Default configuration
- **`configs/evaluation_tesla_t4.yaml`**: Tesla T4 (16GB VRAM) optimized
- **`configs/evaluation_a100.yaml`**: A100 (40GB/80GB VRAM) optimized

### Dataset Sources

The evaluation pipeline supports multiple dataset sources:

#### Local JSON Files
```bash
# Local dataset file
python -m src.evaluation_pipeline.cli.main evaluate -m /path/to/model -d /path/to/dataset.json
```

#### Hugging Face Datasets
```bash
# Hugging Face dataset
python -m src.evaluation_pipeline.cli.main evaluate -m /path/to/model -d username/dataset_name

# Example with your dataset
python -m src.evaluation_pipeline.cli.main evaluate -m /path/to/model -d ngohongthai/exam-sixth_grade-instruct-dataset

# Quick evaluation with your dataset (10 samples)
make evaluate-ngohongthai-quick

# Tesla T4 optimized evaluation with your dataset
make evaluate-ngohongthai-tesla

# A100 optimized evaluation with your dataset
make evaluate-ngohongthai-a100
```

The pipeline automatically detects Hugging Face dataset IDs (containing "/") and loads them using the `datasets` library. It maps common field names:
- `question`, `instruction`, `text` → Question field
- `context`, `input` → Context field  
- `answer`, `output`, `response` → Expected answer field
- `grade_level`, `grade` → Grade level
- `subject` → Subject
- `difficulty` → Difficulty level

### Your Dataset: ngohongthai/exam-sixth_grade-instruct-dataset

This evaluation pipeline is specifically tested with your Vietnamese math dataset:

- **Dataset ID**: `ngohongthai/exam-sixth_grade-instruct-dataset`
- **Split**: `test`
- **Samples**: 113 Vietnamese math problems
- **Grade Level**: Grade 5-6 transition
- **Content**: Mathematical problems in Vietnamese language

#### Quick Start with Your Dataset

```bash
# Quick evaluation (10 samples)
make evaluate-ngohongthai-quick

# Full evaluation
make evaluate-ngohongthai

# Tesla T4 optimized
make evaluate-ngohongthai-tesla

# A100 optimized
make evaluate-ngohongthai-a100
```

#### Dataset Information

The dataset contains 113 Vietnamese math problems designed for grade 5-6 students, covering:
- Mathematical operations
- Geometry problems
- Word problems
- Step-by-step reasoning requirements

Sample questions include:
- "Trường THCS Thanh Xuân lập 1 đội 32 học sinh để trồng cây..."
- "Một hình hộp chữ nhật khi tăng chiều rộng lên ba lần..."
- "Tính $2\frac{4}{9} + 6\frac{7}{{11}} + 7\frac{5}{9} + 13\frac{4}{{11}}$..."

### Key Configuration Sections

#### Model Configuration
```yaml
model:
  name: "google/gemma-3n-2b-it"
  max_seq_length: 2048
  load_in_4bit: true
  use_matformer: true
  batch_size: 8
```

#### Evaluation Configuration
```yaml
evaluation:
  mode: "comprehensive"  # quick, comprehensive
  max_samples: 1000
  save_predictions: false
  metrics:
    opik:
      enabled: true
      metrics: ["hallucination", "context_precision", "answer_relevance"]
    vietnamese_math:
      enabled: true
      metrics: ["mathematical_accuracy", "vietnamese_language_quality"]
    llm_as_judge:
      enabled: true
      metrics: ["accuracy", "completeness", "clarity"]
```

#### OpenRouter Configuration
```yaml
openrouter:
  api_key: "${OPENROUTER_API_KEY}"
  models:
    primary: "anthropic/claude-3.5-sonnet"
    fallback: "openai/gpt-4o-mini"
    judge: "openai/gpt-4o"
  rate_limits:
    requests_per_minute: 60
    tokens_per_minute: 10000
```

## 📊 Evaluation Metrics

### 1. Opik Metrics
- **Hallucination**: Detects false or misleading information
- **Context Precision**: Measures relevance of retrieved context
- **Context Recall**: Measures completeness of retrieved context
- **Answer Relevance**: Assesses answer relevance to question
- **Usefulness**: Evaluates overall usefulness of the response

### 2. Vietnamese Math Metrics
- **Mathematical Accuracy**: Correctness of mathematical solutions
- **Vietnamese Language Quality**: Quality of Vietnamese language usage
- **Step-by-Step Reasoning**: Presence and quality of reasoning steps
- **Grade Level Appropriateness**: Suitability for target grade level
- **Problem Solving Approach**: Systematic approach to problem solving

### 3. LLM-as-a-Judge Metrics
- **Accuracy**: Overall accuracy of the response
- **Completeness**: Completeness of the answer
- **Clarity**: Clarity and readability
- **Relevance**: Relevance to the question
- **Helpfulness**: Overall helpfulness for students

### 4. Additional Metrics
- **Answer Completeness**: Completeness of mathematical solutions
- **Response Consistency**: Consistency across different parts
- **Educational Value**: Educational value for students
- **Clarity & Readability**: Overall clarity and readability

## 🔧 Hardware Optimization

### Tesla T4 (16GB VRAM)
```bash
# Use Tesla T4 optimized config
make evaluate-tesla-t4

# Or with CLI
python -m src.evaluation_pipeline.cli.main -c configs/evaluation_tesla_t4.yaml evaluate -m /path/to/model
```

**Optimizations:**
- 4-bit quantization
- Reduced batch size (4)
- Smaller sequence length (1024)
- Memory-efficient settings
- Conservative rate limits

### A100 (40GB/80GB VRAM)
```bash
# Use A100 optimized config
make evaluate-a100

# Or with CLI
python -m src.evaluation_pipeline.cli.main -c configs/evaluation_a100.yaml evaluate -m /path/to/model
```

**Optimizations:**
- Full precision (no quantization)
- Larger batch size (16)
- Full sequence length (4096)
- Performance-optimized settings
- Higher rate limits

## 🧪 Testing

### Run All Tests
```bash
make evaluate-test
```

### Run Specific Phase Tests
```bash
make evaluate-test-phase PHASE=3
```

### Test Configuration
```bash
make evaluate-config-validate
```

## 📈 Usage Examples

### Basic Evaluation
```bash
# Evaluate a model with default settings
make evaluate MODEL=/path/to/gemma3n

# Quick evaluation for testing
make evaluate-quick
```

### Custom Evaluation
```bash
# Evaluate with custom dataset
make evaluate-dataset DATASET=/path/to/dataset.json

# Evaluate with Hugging Face dataset
make evaluate-dataset DATASET=username/dataset_name

# Evaluate with custom batch size
make evaluate-batch-size BATCH=16

# Evaluate with custom sample limit
make evaluate-samples SAMPLES=100

# Save predictions
make evaluate-save-predictions
```

### Configuration Management
```bash
# Show current configuration
make evaluate-config-show

# Create new configuration
make evaluate-config-create CONFIG=my_config.yaml

# Validate configuration
make evaluate-config-validate
```

### Debugging
```bash
# Debug mode with verbose logging
make evaluate-debug

# Dry run without actual evaluation
make evaluate-dry-run
```

## 🔍 CLI Commands

### Main Commands
- `evaluate`: Run model evaluation
- `test`: Run pipeline tests
- `config`: Manage configuration
- `info`: Show pipeline information

### Evaluation Options
- `--model-path, -m`: Path to model (required)
- `--dataset, -d`: Path to evaluation dataset
- `--output, -o`: Output file path
- `--mode`: Evaluation mode (quick/comprehensive)
- `--batch-size`: Batch size for inference
- `--max-samples`: Maximum samples to evaluate
- `--save-predictions`: Save model predictions
- `--dry-run`: Run without actual evaluation

### Global Options
- `--config, -c`: Configuration file path
- `--log-level`: Logging level
- `--log-file`: Log file path
- `--verbose, -v`: Enable verbose output

## 📝 Output Format

Evaluation results are saved in JSON format:

```json
{
  "overall_score": 0.85,
  "metrics": {
    "hallucination": 0.92,
    "mathematical_accuracy": 0.88,
    "vietnamese_language_quality": 0.91,
    "accuracy": 0.87,
    "completeness": 0.84
  },
  "evaluation_info": {
    "model_path": "/path/to/model",
    "dataset_size": 100,
    "evaluation_mode": "comprehensive",
    "timestamp": "2024-01-01T12:00:00Z"
  },
  "detailed_results": {
    "per_sample_scores": [...],
    "predictions": [...],
    "metadata": {...}
  }
}
```

## 🛠️ Development

### Project Structure
The evaluation pipeline follows a modular architecture:

- **Managers**: Core business logic
- **Factories**: Object creation and dependency injection
- **Evaluators**: Specific evaluation engines
- **Providers**: External service integrations
- **Utils**: Common utilities and helpers

### Adding New Metrics
1. Create new evaluator in `evaluators/`
2. Add metric configuration in config files
3. Update `metrics_manager.py` to include new metric
4. Add tests for the new metric

### Adding New Models
1. Create new model factory in `factories/`
2. Add model configuration in config files
3. Update inference engines if needed
4. Add tests for the new model

## 🐛 Troubleshooting

### Common Issues

#### Memory Issues
```bash
# Reduce batch size
make evaluate-batch-size BATCH=2

# Use Tesla T4 config
make evaluate-tesla-t4
```

#### API Rate Limits
```bash
# Check rate limit settings in config
make evaluate-config-show

# Use fallback providers
# (Automatically handled by the pipeline)
```

#### Configuration Errors
```bash
# Validate configuration
make evaluate-config-validate

# Create new default config
make evaluate-config-create CONFIG=new_config.yaml
```

### Debug Mode
```bash
# Enable debug logging
make evaluate-debug

# Check logs
tail -f logs/evaluation.log
```

## 📚 API Reference

### Core Classes

#### EvaluationManager
Main orchestrator for evaluation pipeline.

```python
from src.evaluation_pipeline.managers import EvaluationManager
from src.evaluation_pipeline.config import ConfigManager

config = ConfigManager.create_default("evaluation")
eval_manager = EvaluationManager(config)

results = eval_manager.evaluate_model_on_dataset(
    model_path="/path/to/model",
    dataset=samples
)
```

#### ConfigManager
Configuration management with validation.

```python
from src.evaluation_pipeline.config import ConfigManager

# Load from file
config = ConfigManager.from_file("config.yaml")

# Create default
config = ConfigManager.create_default("evaluation")

# Validate
config.validate()
```

#### OpikEvaluator
Opik-based evaluation engine.

```python
from src.evaluation_pipeline.evaluators import OpikEvaluator

evaluator = OpikEvaluator(config)
metrics = evaluator.evaluate(
    questions=questions,
    contexts=contexts,
    answers=answers
)
```

## 🤝 Contributing

1. Follow the existing code structure
2. Add tests for new features
3. Update documentation
4. Use type hints and docstrings
5. Follow the established naming conventions

## 🔧 Issues & Improvements

For a comprehensive list of known issues and planned improvements, see:
- **[ISSUES_SUMMARY.md](./ISSUES_SUMMARY.md)** - Quick overview and status
- **[ISSUES_AND_IMPROVEMENTS.md](./ISSUES_AND_IMPROVEMENTS.md)** - Detailed analysis and solutions

These documents include:
- 🔴 High priority issues that need immediate attention
- 🟡 Medium priority improvements for better stability
- 🟢 Low priority enhancements for long-term development
- Detailed code examples and solutions
- Implementation roadmap and timeline

## 📄 License

This evaluation pipeline is part of the MathPal project and follows the same license terms.

## 🆘 Support

For issues and questions:
1. Check the troubleshooting section
2. Review configuration examples
3. Run tests to verify setup
4. Check logs for detailed error messages
5. Review the [Issues & Improvements](./ISSUES_AND_IMPROVEMENTS.md) document

---

**MathPal Evaluation Pipeline** - Comprehensive evaluation for Vietnamese math AI models 🧮✨
