# MathPal 🤖

 An AI-powered math buddy to help students transition from 5th to 6th grade.

## GUIDE

- Install libs: `make install`
- Start infra: `make local-start`
- Stop infra: `make local-stop`
- Crawl test data: `make local-test-crawler`
- Crawl full data: `make local-ingest-data`
- Test retriever: `make local-test-retriever`

## **IMPORTANT NOTE:**

- For MongoDB to work with multiple replicas (as we use it in our Docker setup) on macOS or Linux systems, you have to add the following lines of code to `/etc/hosts`:

  - 127.0.0.1       mongo1
  - 127.0.0.1       mongo2 
  - 127.0.0.1       mongo3

-  Qdrant UI: `localhost:6333/dashboard`

## 🚀 Gemma3N Fine-tuning Pipeline

Một pipeline modular và dễ mở rộng để fine-tune mô hình Gemma3N cho bài toán hỗ trợ học toán lớp 6 bằng tiếng Việt. Pipeline được xây dựng với Unsloth, TRL, PEFT và Comet ML.

### ✨ Tính năng

- **Modular Architecture**: Cấu trúc module rõ ràng, dễ bảo trì và mở rộng
- **Unsloth Integration**: Tối ưu hóa tốc độ và memory với Unsloth
- **Multiple Training Methods**: Hỗ trợ LoRA, QLoRA và full fine-tuning
- **Experiment Tracking**: Tích hợp với Comet ML để theo dõi thí nghiệm
- **Flexible Configuration**: Hỗ trợ cấu hình qua YAML/JSON files
- **Multiple Save Formats**: Lưu model ở nhiều định dạng (LoRA, merged, GGUF)
- **Inference Engine**: Engine inference với nhiều tùy chọn generation
- **Command Line Interface**: CLI đơn giản và mạnh mẽ

### 🏗️ Cấu trúc dự án

```
src/training_pipeline/
├── config/                     # Quản lý cấu hình
│   ├── base_config.py          # Base configuration class
│   ├── training_config.py      # Training configuration
│   └── comet_config.py         # Comet ML configuration
├── data/                       # Xử lý dữ liệu
│   ├── dataset_processor.py    # Dataset loading và processing
│   └── chat_formatter.py       # Chat template formatting
├── models/                     # Quản lý model
│   ├── model_loader.py         # Model loading với Unsloth
│   ├── lora_config.py          # LoRA configuration
│   └── model_saver.py          # Model saving utilities
├── training/                   # Training logic
│   ├── trainer_factory.py      # SFTTrainer setup
│   └── training_utils.py       # Training utilities
├── experiments/                # Experiment tracking
│   └── comet_tracker.py        # Comet ML integration
├── inference/                  # Inference engine
│   └── inference_engine.py     # Model inference
├── cli/                        # Command line interface
│   └── train_gemma.py          # Main training script
└── utils/                      # Utilities
    ├── logging.py              # Logging utilities
    └── device_utils.py         # Device management
```

### 🛠️ Cài đặt

#### Yêu cầu hệ thống

- Python 3.8+
- CUDA 11.8+ (khuyến nghị)
- GPU với ít nhất 8GB VRAM

#### Cài đặt dependencies

```bash
# Cài đặt core dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Cài đặt Unsloth và các thư viện liên quan
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps xformers trl peft accelerate bitsandbytes

# Cài đặt experiment tracking (optional)
pip install comet-ml wandb

# Cài đặt các thư viện khác
pip install transformers datasets tokenizers sentencepiece
pip install pyyaml rich click
```

#### Cài đặt từ source

```bash
git clone <repository-url>
cd gemma3n-training-pipeline
pip install -e .
```

### 🚀 Sử dụng

#### 1. Cấu hình environment variables

```bash
# Comet ML (optional)
export COMET_API_KEY="your-api-key"
export COMET_WORKSPACE="your-workspace" 
export COMET_PROJECT="your-project"

# HuggingFace Hub (optional, for model upload)
export HF_TOKEN="your-token"
```

#### 2. Training cơ bản

```bash
# Training với cấu hình mặc định
python -m training_pipeline.cli.train_gemma

# Training với cấu hình tùy chỉnh
python -m training_pipeline.cli.train_gemma \
    --config configs/training_config.yaml \
    --experiment-name my_experiment \
    --max-steps 500
```

#### 3. Training với các cấu hình có sẵn

```bash
# Development (quick test)
python -m training_pipeline.cli.train_gemma \
    --config configs/development.yaml

# Production (full training)
python -m training_pipeline.cli.train_gemma \
    --config configs/production.yaml \
    --push-to-hub \
    --hub-username your-username
```

#### 4. Training với các tùy chọn nâng cao

```bash
python -m training_pipeline.cli.train_gemma \
    --model-name unsloth/gemma-3n-E4B-it \
    --dataset-name your-dataset \
    --max-steps 1000 \
    --batch-size 2 \
    --learning-rate 1e-4 \
    --lora-r 16 \
    --lora-alpha 32 \
    --save-formats lora merged_fp16 gguf_q8 \
    --test-model
```

### ⚙️ Cấu hình

#### Training Configuration

Tạo file cấu hình YAML:

```yaml
# training_config.yaml
model_name: "unsloth/gemma-3n-E4B-it"
max_seq_length: 2048
dataset_name: "ngohongthai/exam-sixth_grade-instruct-dataset"

# Training settings
max_steps: 100
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 0.0002

# LoRA settings
lora_r: 8
lora_alpha: 8
lora_dropout: 0.0

# Output
output_dir: "outputs/my-experiment"
experiment_name: "baseline"
```


#### Comet ML Configuration

```python
from training_pipeline.config import CometConfig

comet_config = CometConfig(
    workspace="your-workspace",
    project="gemma3n-finetuning",
    experiment_name="my_experiment",
    tags=["gemma3n", "math-tutor", "vietnamese"]
)
```

### 📊 Experiment Tracking

Pipeline hỗ trợ tracking với:

- **Comet ML**: Tracking metrics, parameters, models
- **TensorBoard**: Local tracking
- **Weights & Biases**: Community platform

Ví dụ với Comet ML:

```python
from training_pipeline.experiments import CometTracker

tracker = CometTracker(comet_config)
experiment = tracker.setup_experiment(training_config)

# Training sẽ tự động log metrics
# trainer.train()

tracker.log_model("path/to/saved/model")
tracker.end_experiment()
```

### 🎯 Inference

Sử dụng trained model cho inference:

```python
from training_pipeline.inference import InferenceEngine
from training_pipeline.models import ModelLoader

# Load model
model_loader = ModelLoader(config)
model, tokenizer = model_loader.load_complete_model()

# Setup inference
engine = InferenceEngine(model, tokenizer)

# Generate response
question = "Tính tổng của 15 + 27 = ?"
answer = engine.generate(question)
print(f"Q: {question}")
print(f"A: {answer}")

# Test với nhiều câu hỏi
test_results = engine.test_model()
```

### 💾 Model Saving

Pipeline hỗ trợ lưu model ở nhiều định dạng:

```python
from training_pipeline.models import ModelSaver

saver = ModelSaver(model, tokenizer)

# Lưu LoRA adapters
saver.save_lora_adapters("models/lora-adapters")

# Lưu merged model
saver.save_merged_model("models/merged-fp16", precision="fp16")

# Lưu GGUF cho llama.cpp
saver.save_gguf_model("models/gguf", quantization="q8_0")

# Lưu tất cả formats
results = saver.save_all_formats(
    base_save_path="models",
    model_name="gemma3n-math-tutor",
    formats={
        "lora": {},
        "merged_fp16": {"precision": "fp16"},
        "gguf_q8": {"quantization": "q8_0"}
    }
)
```

### 🔧 Customization

#### Thêm dataset processor mới

```python
from training_pipeline.data import DatasetProcessor

class CustomDatasetProcessor(DatasetProcessor):
    def custom_preprocessing(self, sample):
        # Custom logic
        return processed_sample
```

#### Thêm LoRA configuration

```python
from training_pipeline.models import LoRAConfigManager

custom_config = LoRAConfigManager.create_lora_config(
    r=32,
    lora_alpha=64,
    target_modules=["q_proj", "v_proj", "gate_proj"],
    lora_dropout=0.1
)
```

#### Custom training logic

```python
from training_pipeline.training import TrainerFactory

class CustomTrainerFactory(TrainerFactory):
    def create_custom_trainer(self, model, tokenizer, dataset):
        # Custom trainer setup
        return trainer
```

### 📈 Monitoring và Debugging

#### Memory monitoring

```python
from training_pipeline.utils import DeviceUtils

# Print device info
DeviceUtils.print_device_info()

# Monitor memory during training
result, memory_stats = DeviceUtils.monitor_memory_usage(trainer.train)
```

#### Logging

```python
from training_pipeline.utils import setup_logging, TrainingLogger

# Setup logging
setup_logging(log_level="DEBUG", log_file="training.log")

# Use training logger
logger = TrainingLogger()
logger.info("Training started")
logger.metric("loss", 0.5, step=100)
logger.success("Training completed")
```

### 🧪 Testing

#### Quick test

```bash
python -m training_pipeline.cli.train_gemma --quick-test
```

#### Unit tests

```bash
pytest tests/
```

#### Benchmark inference

```python
questions = ["Câu hỏi 1", "Câu hỏi 2", "Câu hỏi 3"]
results = engine.benchmark_inference(questions, num_runs=3)
print(f"Average tokens/second: {results['avg_tokens_per_second']:.1f}")
```

### 🤝 Contributing

1. Fork repository
2. Tạo feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Tạo Pull Request

### 📝 License

Distributed under the MIT License. See `LICENSE` for more information.

### 🙏 Acknowledgments

- [Unsloth](https://github.com/unslothai/unsloth) - Fast LLM fine-tuning
- [TRL](https://github.com/huggingface/trl) - Transformer Reinforcement Learning
- [PEFT](https://github.com/huggingface/peft) - Parameter-Efficient Fine-Tuning
- [Comet ML](https://www.comet.ml/) - Experiment tracking
- [Gemma](https://deepmind.google/technologies/gemma/) - Base model

### 📧 Contact

Dino - ngohongthai.uet@gmail.com

Project Link: [https://github.com/username/gemma3n-training-pipeline](https://github.com/username/gemma3n-training-pipeline)