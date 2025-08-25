# DPO

让模型倾向于输出人类偏好的回答 `y⁺`，而不是 `y⁻`，同时不脱离原始参考模型 `π₀` 太远。而不是训练一个 reward model 再优化它（RLHF）
$$
L 
DPO
​
 =−logσ(β⋅[logπ 
θ
​
 (y 
+
 ∣x)−logπ 
θ
​
 (y 
−
 ∣x)]−β⋅[logπ 
0
​
 (y 
+
 ∣x)−logπ 
0
​
 (y 
−
 ∣x)])
$$
$x$：输入 prompt。

$y^+$：人类偏好的回答（preferred/chosen）。

$y^-$：人类不偏好的回答（less preferred/rejected）。

$π_θ$：当前训练的模型（可学习）。

$π_0$：参考模型（通常为 SFT 模型，固定）。

$β$：温度超参数（控制训练幅度）。

$\sigma(\cdot)$：Sigmoid 函数。

## 调参

`beta` 决定了模型在 DPO 中“多激进地学习人类偏好”的程度，调得太小学不到偏好，调得太大容易过拟合。

![image-20250710134503764](DPO.assets/image-20250710134503764.png)



![image-20250710134939457](DPO.assets/image-20250710134939457.png)





## 对于reference adapter和dpo_train adapter的理解

![image-20250710135843769](DPO.assets/image-20250710135843769.png)![image-20250710141322998](DPO.assets/image-20250710141322998.png)



## 参考代码

```python
import click
import torch
import random
from trl import DPOTrainer, DPOConfig
from datasets import load_from_disk, load_dataset
from peft import PeftModel
from transformers import (
    AutoTokenizer,
    MistralForCausalLM,
    BitsAndBytesConfig,
)

# ========= 默认配置 =========
base_model = "mistralai/Mistral-Nemo-Instruct-2407"
max_seq_length = 2048
max_new_tokens = 256
default_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
```



```python
@click.command()
@click.option("--lora_model_dir", default="home/model/dpo/adapter", help="训练用 LoRA adapter 路径")
@click.option("--ref_adapter_dir", default="home/model/sft-mistral-chatml", help="参考策略 adapter 路径")
@click.option("--dataset_path", default="rica40325/9_25_dpo_fd_mistral", help="HF数据集名称或本地路径")
@click.option("--output_dir", default="home/model/dpo/dpo_mistral_output", help="模型保存路径")
@click.option("--data_scale_rate", type=float, default=1.0, help="训练数据采样比例")
# ========= 加载 tokenizer =========
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.truncation_side = "left"
tokenizer.model_max_length = max_seq_length - max_new_tokens

# ========= 配置量化 & 加速 =========
bnb_config = BitsAndBytesConfig(
    load_in_4bit=False,
    llm_int8_threshold=6.0,
    llm_int8_has_fp16_weight=False,
    bnb_4bit_compute_dtype=default_dtype,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

# ========= 加载基座模型 + 训练 adapter =========
model = MistralForCausalLM.from_pretrained(
    base_model,
    quantization_config=bnb_config,
    attn_implementation="flash_attention_2",
    torch_dtype=default_dtype,
    device_map="auto",
)
model.config.use_cache = False

model = PeftModel.from_pretrained(
    model,
    lora_model_dir,
    is_trainable=True,
    adapter_name="dpo_train"
)# 在此处合并基座模型model与lora-adapter，合并指的是将参数融合了
model.load_adapter(ref_adapter_dir, adapter_name="reference")

# ========= 加载数据 =========
try:
    raw_dataset = load_from_disk(dataset_path)
except:
    raw_dataset = load_dataset(dataset_path)["train"]

sample_count = int(len(raw_dataset) * data_scale_rate)
sampled_dataset = raw_dataset.select(random.sample(range(len(raw_dataset)), sample_count))

```



```python
# ========= 训练参数 =========
training_args = DPOConfig(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=20,
    num_train_epochs=2,
    learning_rate=5e-6,
    output_dir=output_dir,
    logging_steps=1,
    save_strategy="steps",
    save_steps=500,
    save_total_limit=2,
    beta=0.05,
    bf16=torch.cuda.is_bf16_supported(),
    fp16=not torch.cuda.is_bf16_supported(),
    gradient_checkpointing=True,
    max_prompt_length=max_seq_length - max_new_tokens,
    max_length=max_seq_length,
    model_adapter_name="dpo_train",
    ref_adapter_name="reference",
    report_to=["wandb", "tensorboard"],
    run_name="dpo-mistral-train",
)

# ========= 初始化 Trainer =========
trainer = DPOTrainer(
    model=model,
    args=training_args,
    train_dataset=sampled_dataset,
    tokenizer=tokenizer,
)

# ========= 开始训练 =========
trainer.train()
```

### 合并模型

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Step 1: 加载原始模型
base_model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-Nemo-Instruct-2407",
    torch_dtype=torch.float16,
    device_map="auto"
)

# Step 2: 加载训练好的 LoRA adapter
peft_model = PeftModel.from_pretrained(base_model, "home/model/dpo/dpo_mistral_output")

# Step 3: 合并权重
merged_model = peft_model.merge_and_unload()

# Step 4: 保存合并后的模型（可上传 HF）
merged_model.save_pretrained("/home/model/dpo/dpo-merged-model")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-Nemo-Instruct-2407")
tokenizer.save_pretrained("/home/model/dpo/dpo-merged-model")

```

