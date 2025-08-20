# GPU Memory Troubleshooting Guide

## 🚨 Lỗi thường gặp

### Lỗi: "Some modules are dispatched on the CPU or the disk"

```
ValueError: Some modules are dispatched on the CPU or the disk. Make sure you have enough GPU RAM to fit the quantized model.
```

**Nguyên nhân:** Không đủ GPU RAM để load model quantized.

## 🔧 Giải pháp

### 1. Kiểm tra GPU Memory
```bash
make check-gpu
```

### 2. Sử dụng Safe Evaluation
```bash
make evaluate-llm-safe
```

### 3. Các tùy chọn khác

#### A. Fast Evaluation (không có progress tracking)
```bash
make evaluate-llm-fast
```

#### B. Quick Evaluation (chỉ 5 samples)
```bash
make evaluate-llm-quick
```

#### C. Custom Evaluation với ít samples
```bash
make evaluate-llm-custom SAMPLES=3 EXPERIMENT="Quick Test"
```

## 💡 Tự động xử lý

Hệ thống đã được cập nhật để tự động xử lý lỗi GPU memory:

1. **Thử GPU first** - Load model với GPU
2. **CPU Offload** - Nếu GPU không đủ, sử dụng CPU offload
3. **8-bit Quantization** - Nếu 4-bit không work, thử 8-bit
4. **No Quantization** - Cuối cùng, load không có quantization

## 📊 Memory Requirements

### Gemma-3N Model
- **4-bit quantization**: ~4-6 GB GPU RAM
- **8-bit quantization**: ~8-12 GB GPU RAM
- **No quantization**: ~16-24 GB GPU RAM

### Recommendations
- **< 8 GB GPU RAM**: Sử dụng CPU offload
- **8-16 GB GPU RAM**: Sử dụng 4-bit quantization
- **> 16 GB GPU RAM**: Standard loading

## 🛠️ Commands hữu ích

```bash
# Kiểm tra GPU memory
make check-gpu

# Safe evaluation với memory check
make evaluate-llm-safe

# Fast evaluation (không progress tracking)
make evaluate-llm-fast

# Quick evaluation (5 samples)
make evaluate-llm-quick

# Custom evaluation
make evaluate-llm-custom SAMPLES=3 EXPERIMENT="Test"
```

## 🔍 Debug Tips

### 1. Monitor GPU Memory
```bash
watch -n 1 nvidia-smi
```

### 2. Clear GPU Cache
```bash
python3 -c "import torch; torch.cuda.empty_cache()"
```

### 3. Check System Resources
```bash
htop
```

## 📝 Log Messages

Hệ thống sẽ hiển thị các log messages sau:

```
🔄 Attempting to load model with GPU...
✅ Model loaded successfully with GPU

# Hoặc nếu GPU không đủ:
⚠️  GPU RAM insufficient, attempting CPU offload...
✅ Model loaded successfully with CPU offload

# Hoặc nếu cần 8-bit:
🔄 Attempting to load with 8-bit quantization...
✅ Model loaded successfully with 8-bit quantization
```

## 🎯 Best Practices

1. **Luôn chạy `make check-gpu` trước khi evaluation**
2. **Sử dụng `make evaluate-llm-safe` cho lần đầu**
3. **Monitor memory usage trong quá trình evaluation**
4. **Đóng các ứng dụng khác sử dụng GPU**
5. **Sử dụng smaller model nếu có thể**

## 🆘 Nếu vẫn gặp lỗi

1. **Kiểm tra logs chi tiết**
2. **Thử với ít samples hơn**
3. **Restart system để clear memory**
4. **Consider using CPU-only mode**
5. **Contact support với logs đầy đủ**
