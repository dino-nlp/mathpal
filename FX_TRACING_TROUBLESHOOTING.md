# FX Tracing Troubleshooting Guide

## 🚨 Lỗi thường gặp

### Lỗi: "Detected that you are using FX to symbolically trace a dynamo-optimized function"

```
RuntimeError: Detected that you are using FX to symbolically trace a dynamo-optimized function. This is not supported at the moment.
```

**Nguyên nhân:** Xung đột giữa TorchDynamo và FX symbolic tracing trong Unsloth.

## 🔧 Giải pháp

### 1. Kiểm tra Compatibility
```bash
make check-compatibility
```

### 2. Sử dụng Compatible Evaluation
```bash
make evaluate-llm-compatible
```

### 3. Các tùy chọn khác

#### A. Safe Evaluation
```bash
make evaluate-llm-safe
```

#### B. Fast Evaluation (không có progress tracking)
```bash
make evaluate-llm-fast
```

## 💡 Tự động xử lý

Hệ thống đã được cập nhật để tự động xử lý lỗi FX tracing:

1. **Tắt TorchDynamo** - `torch._dynamo.config.disable = True`
2. **Set Environment Variables** - `TORCH_COMPILE_DISABLE=1`
3. **Safe Model Loading** - Sử dụng cấu hình an toàn
4. **Safe Inference Mode** - `FastModel.for_inference(model)`

## 📊 Cấu hình đã được áp dụng

### Environment Variables
```bash
TOKENIZERS_PARALLELISM=false
TORCH_COMPILE_DISABLE=1
TORCH_LOGS=off
CUDA_LAUNCH_BLOCKING=1
```

### TorchDynamo Configuration
```python
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.disable = True
```

### Model Loading
```python
FastModel.from_pretrained(
    # ... other params
    device_map="auto"  # Safe device mapping
)
```

## 🛠️ Commands hữu ích

```bash
# Kiểm tra compatibility
make check-compatibility

# Compatible evaluation
make evaluate-llm-compatible

# Safe evaluation
make evaluate-llm-safe

# Fast evaluation (không progress tracking)
make evaluate-llm-fast

# Quick evaluation (5 samples)
make evaluate-llm-quick
```

## 🔍 Debug Tips

### 1. Check PyTorch Version
```bash
python3 -c "import torch; print(torch.__version__)"
```

### 2. Check Unsloth Version
```bash
python3 -c "import unsloth; print(unsloth.__version__)"
```

### 3. Monitor System Resources
```bash
htop
```

### 4. Check GPU Memory
```bash
nvidia-smi
```

## 📝 Log Messages

Hệ thống sẽ hiển thị các log messages sau:

```
✅ TorchDynamo disabled
✅ FX tracing conflicts prevented
✅ Model loaded successfully with GPU (safe mode)
✅ Model prepared for inference (safe mode)
```

## 🎯 Best Practices

1. **Luôn chạy `make check-compatibility` trước khi evaluation**
2. **Sử dụng `make evaluate-llm-compatible` cho lần đầu**
3. **Monitor logs cho bất kỳ lỗi nào**
4. **Đảm bảo PyTorch và Unsloth versions tương thích**
5. **Sử dụng safe mode khi có vấn đề**

## 🔧 Manual Fixes

### Nếu vẫn gặp lỗi, thử các bước sau:

1. **Restart Python process**
2. **Clear PyTorch cache**
   ```python
   import torch
   torch.cuda.empty_cache()
   ```

3. **Set environment variables manually**
   ```bash
   export TORCH_COMPILE_DISABLE=1
   export TORCH_LOGS=off
   ```

4. **Use CPU-only mode**
   ```python
   device_map="cpu"
   ```

## 🆘 Nếu vẫn gặp lỗi

1. **Check logs chi tiết**
2. **Verify PyTorch/Unsloth versions**
3. **Try different quantization settings**
4. **Consider using standard transformers instead of Unsloth**
5. **Contact support với logs đầy đủ**

## 📋 Version Compatibility

### Recommended Versions
- **PyTorch**: 2.2+ (2.1 may have issues)
- **Unsloth**: Latest version
- **Transformers**: Compatible with Unsloth

### Known Issues
- PyTorch 2.1 + Unsloth: May have FX tracing conflicts
- Older PyTorch versions: Not supported by Unsloth
- Mixed precision: May cause issues with some configurations
