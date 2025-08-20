# MathPal Evaluation Pipeline

Hệ thống evaluation cho MathPal với progress tracking và các metrics tùy chỉnh.

## 🚀 Tính năng mới

### Progress Tracking
- **Progress Bar**: Hiển thị tiến trình evaluation với tqdm
- **Real-time Updates**: Cập nhật thời gian thực về tiến độ xử lý
- **Detailed Logging**: Log chi tiết cho từng sample và metric
- **Performance Metrics**: Thống kê thời gian xử lý và hiệu suất

### Custom Metrics với Progress Tracking
- **ProgressLevenshteinRatio**: Tính toán độ tương đồng với progress tracking
- **ProgressHallucination**: Phát hiện hallucination với progress tracking  
- **ProgressModeration**: Kiểm duyệt nội dung với progress tracking

## 📊 Cách sử dụng

### 1. Evaluation cơ bản
```bash
# Evaluation tiêu chuẩn
make evaluate-llm

# Evaluation với progress tracking
make evaluate-llm-progress

# Evaluation nhanh (5 samples) với progress tracking
make evaluate-llm-quick

# Evaluation nhanh không có progress tracking
make evaluate-llm-fast
```

### 2. Evaluation tùy chỉnh
```bash
# Evaluation với số lượng samples tùy chỉnh
make evaluate-llm-custom SAMPLES=10 EXPERIMENT="My Custom Test"

# Hoặc chạy trực tiếp
cd src/inference_pipeline && python -m evaluation.evaluate \
    --max_samples 10 \
    --experiment_name "My Custom Test" \
    --use_progress_metrics
```

### 3. Các tùy chọn command line
```bash
python -m evaluation.evaluate --help
```

**Các tùy chọn có sẵn:**
- `--dataset_name`: Tên dataset (mặc định: "mathpal-testset")
- `--max_samples`: Số lượng samples tối đa để evaluate
- `--experiment_name`: Tên experiment
- `--use_progress_metrics`: Sử dụng custom progress metrics
- `--no_progress_tracking`: Tắt progress tracking (nhanh hơn)

## 🔧 Cấu trúc code

### EvaluationProgressCallback
Class callback để theo dõi tiến trình evaluation:

```python
class EvaluationProgressCallback:
    def on_evaluation_start(self):
        # Khởi tạo progress bar
    
    def on_sample_start(self, sample_idx, sample_data):
        # Cập nhật khi bắt đầu xử lý sample
    
    def on_sample_complete(self, sample_idx, result):
        # Cập nhật khi hoàn thành sample
    
    def on_evaluation_complete(self, results):
        # Hiển thị kết quả cuối cùng
```

### ProgressTrackingMetric
Base class cho các metrics có progress tracking:

```python
class ProgressTrackingMetric(base_metric.BaseMetric):
    def _init_progress_bar(self, total, desc):
        # Khởi tạo progress bar cho metric
    
    def _update_progress(self, additional_info):
        # Cập nhật progress
    
    def _close_progress_bar(self):
        # Đóng progress bar
```

## 📈 Metrics có sẵn

### Standard Metrics (Opik)
- `LevenshteinRatio`: Tính độ tương đồng chuỗi
- `Hallucination`: Phát hiện hallucination
- `Moderation`: Kiểm duyệt nội dung
- `Style`: Đánh giá phong cách viết

### Progress Metrics (Custom)
- `ProgressLevenshteinRatio`: Levenshtein với progress tracking
- `ProgressHallucination`: Hallucination detection với progress tracking
- `ProgressModeration`: Content moderation với progress tracking

## 🎯 Ví dụ sử dụng

### 1. Evaluation nhanh với progress tracking
```bash
make evaluate-llm-quick
```

Output:
```
📊 MathPal Evaluation: 100%|██████████| 5/5 [00:30<00:00,  6.12s/sample]
🎉 MathPal Evaluation completed!
⏱️  Total time: 30.61s
📈 Average time per sample: 6.12s
📊 Total samples evaluated: 5
```

### 2. Evaluation tùy chỉnh
```bash
make evaluate-llm-custom SAMPLES=3 EXPERIMENT="Quick Test"
```

### 3. Evaluation không có progress tracking (nhanh)
```bash
make evaluate-llm-fast
```

## 🔍 Debug và Troubleshooting

### 1. Kiểm tra environment
```bash
make test-env
```

### 2. Xem logs chi tiết
```bash
# Set log level to DEBUG
export LOG_LEVEL=DEBUG
make evaluate-llm-progress
```

### 3. Clean up cache
```bash
make clean
```

## 📊 Kết quả evaluation

Kết quả evaluation sẽ được hiển thị trong:
1. **Console**: Progress bar và summary
2. **Opik Dashboard**: Chi tiết metrics và traces
3. **Logs**: Thông tin debug và error

### Ví dụ output:
```
📊 Final Evaluation Results:
   levenshtein_ratio: 0.8234
   hallucination: 0.1567
   moderation: 0.0234
   style: 0.7890
```

## 🚀 Performance Tips

1. **Sử dụng `--no_progress_tracking`** cho evaluation nhanh
2. **Giới hạn số samples** với `--max_samples` cho testing
3. **Sử dụng GPU** để tăng tốc inference
4. **Clean cache** thường xuyên với `make clean`

## 🔗 Links hữu ích

- [Opik Documentation](https://www.comet.com/docs/opik/)
- [TQDM Documentation](https://tqdm.github.io/)
- [MathPal Project](https://github.com/your-repo/mathpal)
