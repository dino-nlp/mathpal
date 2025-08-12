# MathPal Training Pipeline - User Guide

## 🚀 **Quick Start**

### **1. Quick Test (Development)**
```bash
# Fast 20-step training for testing
make train-quick

# Or directly with CLI
python -m training_pipeline.cli.train_gemma --config configs/quick_test.yaml
```

### **2. Production Training**
```bash
# Full production training with tracking
make train-prod

# Or directly with CLI  
python -m training_pipeline.cli.train_gemma --config configs/production.yaml
```

### **3. Custom Training**
```bash
# Custom experiment name
make train-custom EXPERIMENT=my-experiment

# Custom config file
make train-custom-config CONFIG=my-config.yaml

# Quick test with any config
make train-quick-test CONFIG=configs/production.yaml
```

## 📋 **Configuration Files**

### **Available Configs**
- `configs/quick_test.yaml` - Quick development testing (20 steps)
- `configs/production.yaml` - Production training with full features
- `configs/unified_training_config.yaml` - Template with all options

### **Configuration Structure**
```yaml
model:
  name: "unsloth/gemma-3n-E4B-it"
  max_seq_length: 2048
  load_in_4bit: true

dataset:
  name: "ngohongthai/exam-sixth_grade-instruct-dataset"
  train_split: "train"
  text_field: "text"

training:
  max_steps: 100
  learning_rate: 2.0e-4
  per_device_train_batch_size: 2

lora:
  r: 16
  alpha: 32
  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]

output:
  base_dir: "outputs"
  experiment_name: "my-experiment"
  save_formats: ["lora", "merged_16bit"]

comet:
  enabled: false  # Set to true for tracking

hub:
  push_to_hub: false  # Set to true for Hub upload
```

## 🔧 **CLI Options**

### **Basic Usage**
```bash
python -m training_pipeline.cli.train_gemma --config CONFIG_FILE [OPTIONS]
```

### **Available Options**
- `--config, -c`: Path to configuration file (required)
- `--experiment-name`: Override experiment name
- `--max-steps`: Override training steps
- `--learning-rate`: Override learning rate
- `--model-name`: Override model name
- `--output-dir`: Override output directory
- `--quick-test`: Apply quick test settings (20 steps)
- `--dry-run`: Validate config without training
- `--debug`: Enable debug logging
- `--no-comet`: Disable Comet ML tracking
- `--no-env`: Disable environment variable overrides

### **Examples**
```bash
# Production training with custom name
python -m training_pipeline.cli.train_gemma \
  --config configs/production.yaml \
  --experiment-name my-production-v1

# Quick test any config
python -m training_pipeline.cli.train_gemma \
  --config configs/production.yaml \
  --quick-test

# Custom training with overrides
python -m training_pipeline.cli.train_gemma \
  --config configs/unified_training_config.yaml \
  --max-steps 500 \
  --learning-rate 1e-4 \
  --experiment-name custom-experiment
```

## 🛠️ **Makefile Commands**

### **Main Training Commands**
- `make train` - Training with unified config
- `make train-quick` - Quick test (20 steps)
- `make train-prod` - Production training

### **Validation & Testing**
- `make train-dry-run` - Validate config without training
- `make train-dry-run-quick` - Dry run with quick config
- `make train-dry-run-prod` - Dry run with production config
- `make test-configs` - Test all configuration files
- `make test-config-manager` - Test ConfigManager system

### **Custom Training**
- `make train-custom EXPERIMENT=name` - Custom experiment name
- `make train-custom-config CONFIG=file.yaml` - Custom config
- `make train-steps STEPS=500` - Custom step count
- `make train-output OUTPUT=dir` - Custom output directory

### **Development**
- `make env-check` - Check environment dependencies
- `make test-imports` - Test all imports
- `make show-architecture` - Show architecture overview
- `make device-info` - Show GPU information

## 🏭 **Production Setup**

### **Environment Variables**
For production training, set these environment variables:

```bash
# Comet ML (for experiment tracking)
export COMET_API_KEY="your-api-key"
export COMET_WORKSPACE="your-workspace"

# HuggingFace Hub (for model upload)
export HF_TOKEN="your-hf-token"
export HF_USERNAME="your-username"

# Optional model/training overrides
export MODEL_NAME="custom-model"
export MAX_STEPS=2000
export LEARNING_RATE=1e-4
export OUTPUT_DIR="outputs/custom"
export EXPERIMENT_NAME="production-v1"
```

### **Production Config Features**
```yaml
# configs/production.yaml includes:
training:
  max_steps: 2000           # Full training
  per_device_train_batch_size: 4
  learning_rate: 1.0e-4

lora:
  r: 32                     # Higher rank for better adaptation
  alpha: 64

output:
  save_formats:
    - "lora"               # LoRA adapters
    - "merged_16bit"       # Full model for inference
    - "merged_4bit"        # Quantized for deployment

comet:
  enabled: true            # Full experiment tracking

hub:
  push_to_hub: true        # Automatic model upload
  private: false           # Public model sharing
```

## 🧪 **Development Workflow**

### **1. Quick Testing**
```bash
# Fast iteration cycle
make train-quick              # 20 steps, ~2 minutes
make train-dry-run-quick      # Validate config only
```

### **2. Configuration Testing**
```bash
# Test config validity
make validate-config CONFIG=my-config.yaml

# Test all configs
make test-configs

# Test ConfigManager system
make test-config-manager
```

### **3. Environment Validation**
```bash
# Check dependencies
make env-check

# Test imports
make test-imports

# Show system info
make device-info
make memory-info
```

## 📊 **Monitoring & Outputs**

### **Training Outputs**
```
outputs/
├── {experiment-name}/
│   ├── {experiment-name}-lora/          # LoRA adapters
│   ├── {experiment-name}-merged-16bit/  # Full model (16-bit)
│   ├── {experiment-name}-merged-4bit/   # Quantized model
│   └── training_metadata.json          # Training info
```

### **Comet ML Integration**
- Automatic metric logging
- Parameter tracking  
- Model artifacts
- Training curves
- System monitoring

### **HuggingFace Hub**
- Automatic model upload
- Model cards
- Public/private repositories
- Version management

## ⚡ **Performance Tips**

### **Memory Optimization**
- Use 4-bit quantization: `load_in_4bit: true`
- Enable gradient checkpointing: `use_gradient_checkpointing: "unsloth"`
- Adjust batch size: `per_device_train_batch_size: 1-4`

### **Speed Optimization**
- Use efficient optimizers: `optim: "adamw_8bit"`
- Enable mixed precision: `bf16: true`
- Use Unsloth optimizations (automatic)

### **Quality Optimization**
- Higher LoRA rank: `r: 32` (production)
- More training steps: `max_steps: 2000+`
- Learning rate tuning: `learning_rate: 1e-4`

## 🐛 **Troubleshooting**

### **Common Issues**

**Configuration Errors:**
```bash
# Validate config first
make train-dry-run-CONFIG

# Check specific config
python -m training_pipeline.cli.train_gemma --config CONFIG --dry-run
```

**Memory Issues:**
```bash
# Reduce batch size, enable quantization
training:
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 8
model:
  load_in_4bit: true
```

**Import Errors:**
```bash
# Check environment
make env-check
make test-imports
```

**CUDA Issues:**
```bash
# Check GPU status
make device-info
make show-gpu-usage
```

### **Debug Mode**
```bash
# Enable detailed logging
python -m training_pipeline.cli.train_gemma --config CONFIG --debug
```

## 📚 **Additional Resources**

- **ConfigManager Demo**: `python examples/config_manager_demo.py`
- **Architecture Overview**: `make show-architecture`
- **All Commands**: `make help`
- **Configuration Template**: `configs/unified_training_config.yaml`

## 🎯 **Migration from Old System**

**Old CLI (deprecated):**
```bash
# 50+ arguments, complex
python train_old.py --model-name X --dataset-name Y --max-steps Z ...
```

**New CLI (current):**
```bash
# 5-7 arguments, config-driven
python -m training_pipeline.cli.train_gemma --config configs/my-config.yaml
```

**Benefits:**
- ✅ Type-safe configuration
- ✅ Better error messages
- ✅ Environment variable support
- ✅ Unified config management
- ✅ Proper dependency injection
