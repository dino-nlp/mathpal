# Training Pipeline Issues & Solutions

## 🎯 Executive Summary

Training pipeline for Gemma 3N with Unsloth was successfully fixed by implementing notebook-proven approach. **Key insight**: SFTConfig + pre-formatted data + no data collator = stable training.

## 🔴 Root Causes Analysis

### 1. "Excessive Nesting" Tensor Error
**Symptom**: `Unable to create tensor... excessive nesting (inputs type 'list' where type 'int' is expected)`

**Root Cause**: 
- TrainingArguments + DataCollatorForSeq2Seq creates tensor shape conflicts
- Packing + complex data collator pipeline incompatible with Unsloth optimization

**Solution**: 
- ✅ Use SFTConfig instead of TrainingArguments
- ✅ Remove data_collator completely
- ✅ Disable packing

### 2. Formatting Function Conflicts
**Symptom**: `completion_only_loss=True conflicts with formatting_func`

**Root Cause**:
- SFTTrainer expects EITHER formatting_func OR pre-formatted data, not both
- completion_only_loss incompatible with custom formatting

**Solution**:
- ✅ Pre-format data in dataset preprocessing 
- ✅ Remove formatting_func from SFTTrainer
- ✅ Apply train_on_responses_only AFTER trainer creation

### 3. Multiprocessing Issues
**Symptom**: Silent failures, empty datasets after processing

**Root Cause**:
- tokenizer.apply_chat_template not thread-safe with num_proc > 1
- Multiprocessing filter operations cause silent failures

**Solution**:
- ✅ Force num_proc=1 for tokenizer operations
- ✅ Force num_proc=1 for dataset filtering

### 4. Config Structure Problems
**Symptom**: Config values not loaded correctly

**Root Cause**:
- Flat YAML structure vs nested config object expectations
- Wrong config file used by training script

**Solution**:
- ✅ Use nested YAML structure matching config objects
- ✅ Ensure correct config file path

## ✅ Working Approach (Proven)

### Dataset Processing
```python
# 1. Convert to conversations format
conversations = [
    {"role": "user", "content": [{"type": "text", "text": question}]},
    {"role": "assistant", "content": [{"type": "text", "text": solution}]}
]

# 2. Apply chat template with num_proc=1
text = tokenizer.apply_chat_template(conversations, tokenize=False).removeprefix('<bos>')

# 3. Keep only text field, remove all other columns
```

### Trainer Configuration
```python
# 1. Use SFTConfig instead of TrainingArguments
sft_config = SFTConfig(
    dataset_text_field="text",
    max_length=max_seq_length,
    optim="adamw_torch_fused",
    lr_scheduler_type="cosine",
    # ... other settings
)

# 2. Create SFTTrainer with minimal args
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=sft_config,
    # NO data_collator
    # NO formatting_func
)

# 3. Apply train_on_responses_only AFTER creation
trainer = train_on_responses_only(trainer, ...)
```

## 🚫 Failed Approaches (Avoid)

### ❌ Don't Use
- TrainingArguments + DataCollator combination
- formatting_func in SFTTrainer when data is pre-formatted
- num_proc > 1 for tokenizer operations
- Packing with complex datasets
- prompt/completion format (use text field instead)

### ❌ Problematic Patterns
```python
# WRONG: Complex data collator pipeline
data_collator = DataCollatorForSeq2Seq(...)
trainer = SFTTrainer(..., data_collator=data_collator)

# WRONG: Formatting function + pre-formatted data
trainer = SFTTrainer(..., formatting_func=format_func)  # Data already formatted

# WRONG: Multiprocessing with tokenizer
dataset.map(apply_chat_template, num_proc=4)  # Silent failures
```

## 📋 Best Practices

### 1. Dataset Processing
- ✅ Use conversations format for chat templates
- ✅ Apply chat template in preprocessing, not during training
- ✅ Force num_proc=1 for tokenizer operations
- ✅ Keep only text field in final dataset
- ✅ Filter empty samples with num_proc=1

### 2. Trainer Configuration
- ✅ Use SFTConfig for all scenarios (Unsloth + HuggingFace)
- ✅ Let SFTTrainer handle data collation internally
- ✅ Pre-format data, avoid formatting_func
- ✅ Apply train_on_responses_only after trainer creation

### 3. Config Management
- ✅ Use nested YAML structure matching config objects
- ✅ Disable packing by default for stability
- ✅ Set dataset.num_proc: 1 for tokenizer compatibility
- ✅ Use proven optimizer settings (adamw_torch_fused)

## 🎊 Success Metrics

**Before Fixes**: Training failed with tensor creation errors
**After Fixes**: 
- ✅ Training completed successfully (274.44 seconds)
- ✅ Loss reduction: 6.0116 → 1.3135  
- ✅ Models saved: LoRA + merged 16bit
- ✅ No tensor errors or crashes

## 🔧 Implementation Status

- ✅ Root causes identified and documented
- ✅ Working approach implemented in both Unsloth and HuggingFace paths
- ✅ Default configs updated with proven settings
- ✅ Critical num_proc=1 enforced for tokenizer operations
- ✅ All tensor creation issues resolved

**Result**: Stable, production-ready training pipeline for Gemma 3N with Unsloth.
