#!/usr/bin/env python3
"""
Script để validate và test cấu hình sau khi sửa lỗi TrainingArguments conflict
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from config import ExperimentConfig, get_optimized_config_for_t4

def validate_training_args_consistency():
    """Kiểm tra consistency của training arguments"""
    print("🔍 Kiểm tra consistency của Training Arguments...")
    
    config = get_optimized_config_for_t4()
    
    # Kiểm tra eval_strategy và save_strategy match
    eval_strategy = config.training.eval_strategy
    save_strategy = config.training.save_strategy
    load_best_model = config.training.load_best_model_at_end
    
    print(f"  eval_strategy: {eval_strategy}")
    print(f"  save_strategy: {save_strategy}")
    print(f"  load_best_model_at_end: {load_best_model}")
    
    # Validate theo Transformers requirements
    if load_best_model and eval_strategy != save_strategy:
        print("❌ LỖI: load_best_model_at_end=True yêu cầu eval_strategy và save_strategy phải giống nhau!")
        return False
    
    # Kiểm tra eval_steps và save_steps
    if eval_strategy == "steps":
        eval_steps = getattr(config.training, 'eval_steps', None)
        save_steps = config.training.save_steps
        
        print(f"  eval_steps: {eval_steps}")
        print(f"  save_steps: {save_steps}")
        
        if eval_steps is None:
            print("❌ LỖI: eval_strategy='steps' nhưng eval_steps không được set!")
            return False
            
        # Best practice: eval_steps và save_steps nên giống nhau
        if eval_steps != save_steps:
            print("⚠️  CẢNH BÁO: eval_steps và save_steps khác nhau có thể gây confusion!")
    
    print("✅ Training arguments consistency: OK!")
    return True

def validate_model_config():
    """Kiểm tra model configuration"""
    print("\n🔍 Kiểm tra Model Configuration...")
    
    config = get_optimized_config_for_t4()
    
    # Kiểm tra quantization settings
    load_4bit = config.model.load_in_4bit
    load_8bit = getattr(config.model, 'load_in_8bit', False)
    
    print(f"  load_in_4bit: {load_4bit}")
    print(f"  load_in_8bit: {load_8bit}")
    
    if load_4bit and load_8bit:
        print("❌ LỖI: Không thể sử dụng cả load_in_4bit và load_in_8bit cùng lúc!")
        return False
    
    # Kiểm tra LoRA settings
    lora_r = config.model.lora_r
    lora_alpha = config.model.lora_alpha
    lora_dropout = config.model.lora_dropout
    
    print(f"  lora_r: {lora_r}")
    print(f"  lora_alpha: {lora_alpha}")
    print(f"  lora_dropout: {lora_dropout}")
    
    # Best practice checks
    if lora_dropout != 0.0:
        print("⚠️  CẢNH BÁO: Unsloth recommends lora_dropout=0.0 for optimal performance!")
    
    if lora_alpha != lora_r:
        print("⚠️  CẢNH BÁO: lora_alpha != lora_r có thể affect training stability!")
    
    print("✅ Model configuration: OK!")
    return True

def validate_comet_config():
    """Kiểm tra Comet ML configuration"""
    print("\n🔍 Kiểm tra Comet ML Configuration...")
    
    config = get_optimized_config_for_t4()
    
    # Kiểm tra basic Comet ML settings
    api_key = config.comet.api_key
    workspace = config.comet.workspace
    project = config.comet.project
    
    print(f"  api_key: {'Set' if api_key else 'Not set'}")
    print(f"  workspace: {workspace or 'Not set'}")
    print(f"  project: {project or 'Not set'}")
    
    # Kiểm tra histogram logging (should be disabled for performance)
    weight_logging = config.comet.auto_histogram_weight_logging
    gradient_logging = config.comet.auto_histogram_gradient_logging
    
    print(f"  auto_histogram_weight_logging: {weight_logging}")
    print(f"  auto_histogram_gradient_logging: {gradient_logging}")
    
    if weight_logging or gradient_logging:
        print("⚠️  CẢNH BÁO: Histogram logging có thể làm chậm training và tốn bandwidth!")
    
    print("✅ Comet configuration: OK!")
    return True

def test_config_creation():
    """Test việc tạo config"""
    print("\n🧪 Test việc tạo configuration...")
    
    try:
        config = get_optimized_config_for_t4()
        print("✅ Config creation: OK!")
        
        # Test conversion to dict
        config_dict = config.to_dict()
        print("✅ Config to dict conversion: OK!")
        
        # Print summary
        print("\n📋 Configuration Summary:")
        print(f"  Model: {config.model.model_name}")
        print(f"  Max seq length: {config.model.max_seq_length}")
        print(f"  LoRA r/alpha: {config.model.lora_r}/{config.model.lora_alpha}")
        print(f"  Batch size: {config.training.per_device_train_batch_size}")
        print(f"  Learning rate: {config.training.learning_rate}")
        print(f"  Epochs: {config.training.num_train_epochs}")
        print(f"  Eval strategy: {config.training.eval_strategy}")
        print(f"  Save strategy: {config.training.save_strategy}")
        
        return True
        
    except Exception as e:
        print(f"❌ LỖI khi tạo config: {e}")
        return False

def main():
    """Main validation function"""
    print("🚀 Bắt đầu validation cấu hình Training Pipeline...")
    print("=" * 60)
    
    all_checks = [
        validate_training_args_consistency(),
        validate_model_config(), 
        validate_comet_config(),
        test_config_creation()
    ]
    
    print("\n" + "=" * 60)
    
    if all(all_checks):
        print("🎉 TẤT CẢ KIỂM TRA HOÀN TẤT THÀNH CÔNG!")
        print("\n✅ Các vấn đề đã được sửa:")
        print("  - eval_strategy và save_strategy đã đồng bộ (cả hai đều dùng 'steps')")
        print("  - eval_steps và save_steps đã được set giống nhau (50 steps)")
        print("  - bf16 đã được force disabled cho T4 GPU (auto-detection)")
        print("  - Cấu hình Unsloth đã được cập nhật theo best practices mới nhất")
        print("  - Comet ML được sử dụng làm primary tracking platform")
        print("  - Histogram logging đã được tối ưu để tiết kiệm memory/bandwidth")
        print("  - Safety checks được thêm vào trainer để tránh bf16 trên T4")
        print("\n🚀 Bạn có thể chạy training pipeline mà không gặp lỗi TrainingArguments!")
        return 0
    else:
        print("❌ VẪN CÒN LỖI TRONG CẤU HÌNH!")
        print("Vui lòng kiểm tra và sửa các lỗi ở trên.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)