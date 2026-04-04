# 🚀 Fine-Tuning LLaMA 3.1 using LoRA + QLoRA (Unsloth)

Welcome! 👋  
This project demonstrates **efficient fine-tuning of a large language model** using **LoRA + Quantization (QLoRA)** with memory optimization techniques.

We fine-tune only a **small fraction (~0.5%) of parameters** while keeping the base model frozen ❄️

---

# 🧠 Base Model

We use:

- 🦙 LLaMA 3.1 8B Instruct (Meta)
- Designed for strong general-purpose instruction following
- Large-scale transformer model with billions of parameters

---

# ⚡ Key Techniques Used

## 🔹 1. LoRA (Low-Rank Adaptation)

👉 Instead of training the full model, we add small trainable matrices called **LoRA adapters**.

### 🧩 Idea:

### 💡 Benefits:
- ⚡ Faster training
- 💾 Less memory usage
- 🔥 Easy adaptation to new tasks

---

## 🔹 2. QLoRA (Quantized LoRA)

👉 Combines:
- 📉 4-bit quantization of base model
- 🧠 LoRA adapters for training

### 💡 Benefits:
- 🚀 Runs on limited GPU (even Colab)
- 🧊 Massive memory reduction
- ⚙️ Still high performance

---

## 🔹 3. Unsloth Optimization

We use **Unsloth** for:

- ⚡ Faster training (2–5× speedup)
- 💾 Lower VRAM usage
- 🧠 Optimized transformer kernels

---

# 🧱 LoRA Configuration

```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,                      # rank of adapters
    lora_alpha=32,            # scaling factor
    target_modules=["q_proj", "v_proj"],  # attention layers
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
