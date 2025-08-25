![image-20250712172554567](unsloth+LoRA微调全流程.assets/image-20250712172554567.png)

[TOC]

# 一、加载预训练模型

## 1. Unsloth介绍

`Unsloth` 是一个开源工具，专门用来加速大语言模型（LLMs）的微调过程。它的主要功能和优势包括：

- 高效微调：Unsloth 的微调速度比传统方法快 2-5 倍，内存占用减少 50%-80%。这意味着你可以用更少的资源完成微调任务。
- 低显存需求：即使是消费级 GPU（如 RTX 3090），也能轻松运行 Unsloth。例如，仅需 7GB 显存就可以训练 1.5B 参数的模型。
- 支持多种模型和量化：Unsloth 支持 Llama、Mistral、Phi、Gemma 等主流模型，并且通过动态 4-bit 量化技术，显著降低显存占用，同时几乎不损失模型精度。
- 开源与免费：Unsloth 提供免费的 Colab Notebook，用户只需添加数据集并运行代码即可完成微调。

官方文档：[[unslothai/unsloth: Fine-tuning & Reinforcement Learning for LLMs. 🦥 Train Qwen3, Llama 4, DeepSeek-R1, Gemma 3, TTS 2x faster with 70% less VRAM.](https://github.com/unslothai/unsloth)](https://github.com/unslothai/unsloth)

```python
#安装unsloth
conda create --name 1 \
    python=3.11 \
    pytorch-cuda=12.1 \
    pytorch cudatoolkit xformers -c pytorch -c nvidia -c xformers \
    -y
conda activate 1

pip install unsloth
#安装vLLM&mistral common
pip install --upgrade vllm
pip install --upgrade mistral_common
```



```python
#利用unsloth加载模型
from unsloth import FastLanguageModel, is_bfloat16_supported

max_seq_length=2048 #设置模型处理文本的最大长度
dtype=None #设置数据类型，让模型自动选择最适合的精度
load_in_4bit=True

model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="mistralai/Mistral-Nemo-Instruct-2407",
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,#4bit量化，通过减少模型的精度来节省内存，为true是qlora，false是lora
    )
```

------



 ## 2. LoRA 的原理

对于某一线性层权重矩阵 $W \in \mathbb{R}^{d \times k}$，LoRA 不直接更新 $W$，而是在训练时添加一个 **低秩矩阵变化项**：

$$
W_{\text{new}} = W + \Delta W, \quad \Delta W = BA
$$
其中：

- $A \in \mathbb{R}^{r \times k}$，$B \in \mathbb{R}^{d \times r}$，$r \ll \min(d,k)$；
- $A, B$ 是训练参数，$W$ 冻结不变；
- 推理时可将 $W + BA$ 合并为单一矩阵，几乎无推理开销。
- 全参数更新需要更新
    $$
    d*k
    $$
    个参数，lora微调只需要更新
    $$
    k*r+r*d=r*（k+d）
    $$
    个参数，大大减少了需要更新的参数

- 训练过程中，LoRA 仅对少量低秩矩阵参数进行调整，相比全量参数微调，计算量和内存占用显著降低，训练速度大幅提升；同时，参数调整量少，极大降低了过拟合风险，使模型泛化能力更强。应用层面，它能灵活适配不同任务，为同一预训练模型构建多个 LoRA 模块，快速切换实现多任务学习；在部署阶段，其注入的线性层可与冻结参数无缝合并，不增加推理延迟。

---

## 3. LoRA 实现细节

### 插入 LoRA 层

通常插入点有：

- Attention：
  - Query（Q）和 Value（V）最常用；
- Linear：
  - Feedforward 网络中的 FC 层；

---

## 4. LoRA微调参数

```python
 # 应用 LoRA 参数
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,#秩
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj","gate_proj", "up_proj", "down_proj",],#应用lora方法的模块名称
        lora_alpha=16,#缩放系数，一般情况与r相同或为r两倍
        lora_dropout=0.05,#大数据集：0.0，小数据集：0.05-0.1
        bias="none",
        use_gradient_checkpointing="unsloth",
        #random_state = 3407,随机数种子
        use_rslora=False,
        loftq_config=None,
    )
```

- LoRA（低秩适应）中的秩（Rank）是决定模型微调时参数更新 “表达能力” 的关键参数。它通过低秩矩阵分解的方式，控制可训练参数的规模与模型调整的灵活程度。秩的数值越小，模型微调时的参数更新越 “保守”；秩的数值越大，模型能捕捉的特征复杂度越高，但也会消耗更多计算资源。
- 秩低（如 4）：相当于用固定的几种 “思维模板” 学习新知识。比如学数学时，只记 3 种解题套路，遇到新题只能用这 3 种方法套。优点是学习过程很稳定，不容易学偏（过拟合风险低），且 “脑子”（显存）不费力；缺点是遇到复杂问题可能束手无策，因为模板太少（表达能力有限）。
- 秩高（如 64）：相当于掌握了 100 种解题思路，遇到新题可以灵活组合方法。优点是能处理更复杂的任务（比如生成风格更细腻的内容），模型微调的 “潜力” 更大；缺点是容易学 “乱”—— 可能把无关的知识强行关联（过拟合），而且太费 “脑子”（显存消耗显著增加）。
- 日常微调建议从 8-16 开始尝试，这是平衡效果与效率的常用区间。一般就是从 8 开始，如果微调完觉得模型没学会就调大。

![image-20250710112907051](unsloth+LoRA微调全流程.assets/image-20250710112907051.png)

---

## 加载预训练模型完整代码

```python
#利用unsloth加载模型
from unsloth import FastLanguageModel, is_bfloat16_supported

max_seq_length=2048 #设置模型处理文本的最大长度
dtype=None #设置数据类型，让模型自动选择最适合的精度

model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="mistralai/Mistral-Nemo-Instruct-2407",
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=True,#4bit量化，通过减少模型的精度来节省内存，为true是qlora，false是lora
    )

# 应用 LoRA 参数
model = FastLanguageModel.get_peft_model(
    model,
    r=32,#秩
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "embed_tokens", "lm_head"],#应用lora方法的模块名称
    lora_alpha=64,#缩放系数，一般情况为r两倍
    lora_dropout=0.05,#大数据集：0.0，小数据集：0.05-0.1
    bias="none",
    use_gradient_checkpointing="unsloth",
    use_rslora=False,
    loftq_config=None,
)
model.model.model.embed_tokens = model.model.model.embed_tokens.to(dtype=torch.bfloat16)
model.model.lm_head = model.model.lm_head.to(dtype=torch.bfloat16)

```

# 二、微调前测试

设置推理问题（按照你想要微调的方向设置）

```python
prompt_style = """以下是描述任务的指令，以及提供进一步上下文的输入。请写出一个适当完成请求的回答。
在回答之前，请仔细思考问题，并创建一个逻辑连贯的思考过程，以确保回答准确无误。

### 指令：
你是一位精通卜卦、星象和运势预测的算命大师。
请回答以下算命问题。

### 问题：
{}

### 回答：
<think>{}</think>""" 
# 定义提示风格的字符串模板，用于格式化问题

question = "1992年闰四月初九巳时生人，女，想了解健康运势" 
# 定义具体的算命问题
```

```python
FastLanguageModel.for_inference(model)
# 准备模型以进行推理

inputs = tokenizer([prompt_style.format(question, "")], return_tensors="pt").to("cuda")
# 使用 tokenizer 对格式化后的问题进行编码，并移动到 GPU

outputs = model.generate(
    input_ids=inputs.input_ids,
    attention_mask=inputs.attention_mask,
    max_new_tokens=1200,
    use_cache=True,
)
# 使用模型生成回答

response = tokenizer.batch_decode(outputs)
# 解码模型生成的输出为可读文本

print(response[0])
# 打印生成的回答部分
```



# 三、加载数据集

## 1. 修改数据集的格式为chatml

在这一步里，每一个数据段末尾加上EOS_TOKEN 

EOS_TOKEN = tokenizer.eos_token

以alpaca为例

```python
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }
pass

from datasets import load_dataset
dataset = load_dataset("yahma/alpaca-cleaned", split = "train")
dataset = dataset.map(formatting_prompts_func, batched = True,)
```



```python
import json
import re
from tqdm import tqdm
from datasets import load_dataset, Dataset

def chatml_turn(role: str, content: str) -> str:
    return f"<|im_start|>{role}\n{content}\n<|im_end|>"

def extract_user_assistant_turns(text):
    """
    从原始字符串中提取多轮 <|user|>...<|assistant|> 对话。
    返回 ChatML 格式的轮次列表。
    """
    turns = re.findall(r"<\|user\|>(.*?)<\|assistant\|>(.*?)(?=(<\|user\|>|$))", text, re.DOTALL)
    formatted = []
    for user_text, assistant_text, _ in turns:
        user_text = user_text.strip()
        assistant_text = assistant_text.strip()
        if user_text and assistant_text:
            formatted.append(chatml_turn("user", user_text))
            formatted.append(chatml_turn("assistant", assistant_text))
    return formatted

def process_roleplay_dataset():
    """
    处理 hieunguyenminh/roleplay 数据集
    """
    dataset = load_dataset("hieunguyenminh/roleplay", split="train")
    formatted = []

    for example in tqdm(dataset, desc="Formatting hieunguyenminh/roleplay"):
        description = example.get("description", "").strip()
        text = example.get("text", "").strip()
        if not description or not text:
            continue

        header = chatml_turn("system", description)
        dialogue_turns = extract_user_assistant_turns(text)
        if not dialogue_turns:
            continue

        full_text = header + "\n" + "\n".join(dialogue_turns)
        formatted.append({"text": full_text})

    return formatted

def process_charcard_dataset():
    """
    处理 Gryphe/Sonnet3.5-Charcard-Roleplay 数据集
    """
    dataset = load_dataset("Gryphe/Sonnet3.5-Charcard-Roleplay", split="train")
    formatted = []

    for example in tqdm(dataset, desc="Formatting Gryphe/Charcard"):
        conversations = example.get("conversations", [])
        if not conversations or conversations[0]["from"] != "system":
            continue
        if not any(msg["from"] == "gpt" for msg in conversations):
            continue  # 必须至少有 assistant 回复

        turns = [chatml_turn("system", conversations[0]["value"].strip())]

        for msg in conversations[1:]:
            role = msg["from"]
            content = msg["value"].strip()
            if role == "human":
                turns.append(chatml_turn("user", content))
            elif role == "gpt":
                turns.append(chatml_turn("assistant", content))

        full_text = "\n".join(turns)
        formatted.append({"text": full_text})

    return formatted

def process_feedback_dataset():
    """
    处理 rica40325/9.8feedback 数据集
    """
    dataset = load_dataset("rica40325/9.8feedback", split="train")
    formatted = []

    for example in tqdm(dataset, desc="Formatting rica40325/9.8feedback"):
        conversations = example.get("conversations", [])
        if not conversations or conversations[0]["from"] != "system":
            continue
        if not any(msg["from"] == "assistant" for msg in conversations):
            continue  # 必须至少有 assistant 回复

        # 第一条 system 消息
        turns = [chatml_turn("system", conversations[0]["value"].strip())]

        # 处理后续轮次
        for msg in conversations[1:]:
            role = msg["from"]
            content = msg["value"].strip()
            if role == "user":
                turns.append(chatml_turn("user", content))
            elif role == "assistant":
                turns.append(chatml_turn("assistant", content))

        # 一整条对话保存一次
        full_text = "\n".join(turns)
        formatted.append({"text": full_text})

    return formatted

def is_valid_chatml(text):
    """确保每个样本都至少包含 user 和 assistant 两种角色"""
    return "<|im_start|>user" in text and "<|im_start|>assistant" in text

def save_as_jsonl(data, path):
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")

if __name__ == "__main__":
    print(" 正在格式化三个角色扮演数据集...")

    data1 = process_roleplay_dataset()
    data2 = process_charcard_dataset()
    data3 = process_feedback_dataset()
    all_data = data1 + data2 + data3

    print(f" 格式化前样本总数: {len(all_data)}")

    # 清洗不符合 ChatML 的样本
    cleaned_data = [item for item in all_data if is_valid_chatml(item["text"])]
    print(f" 格式化后合法样本数: {len(cleaned_data)}")

    # 保存为 jsonl
    save_path = "/home/qwen-sft/dataset-sft/processed_data.jsonl"
    save_as_jsonl(cleaned_data, save_path)
    print(f" 已保存到: {save_path}")

```

## 2. 将处理好格式的数据进行tokenize

```python
# 判断是否已有缓存的 Tokenized 数据集
if os.path.exists(TOKENIZED_PATH):
    print("加载缓存的 tokenized 数据集...")
    tokenized_dataset = load_from_disk(TOKENIZED_PATH)
else:
    print("首次处理数据并保存缓存...")
    raw_dataset = load_dataset("json", data_files="/home/qwen-sft/dataset-sft/processed_data.jsonl", split="train")
    dataset = raw_dataset.filter(lambda x: isinstance(x["text"], str) and len(x["text"].strip()) > 50)

    # Tokenization
    def tokenize(example):
        return tokenizer(example["text"], truncation=True, max_length=2048, padding="max_length")

    tokenized_dataset = dataset.map(tokenize, batched=True, num_proc=14)

    # 保存成 Arrow 格式
    tokenized_dataset.save_to_disk(TOKENIZED_PATH)
    print(f" Tokenized 数据集已保存到 {TOKENIZED_PATH}")
```



# 四、训练超参数设置

**训练轮数（Number of Epochs）** Epoch 是机器学习中用于描述模型训练过程的一个术语，指的是模型完整地遍历一次整个训练数据集的次数。换句话说，一个 `Epoch` 表示模型已经看到了所有训练样本一次。

- 轮数少：比如你只复习一遍，可能对书里的内容还不是很熟悉，考试成绩可能不会太理想。
- 轮数多：比如你复习了 10 遍，对书里的内容就很熟悉了，但可能会出现一个问题——你对书里的内容背得很熟，但遇到新的、类似的问题就不会解答了，简单讲就是 “学傻了“，只记住这本书里的内容了，稍微变一变就不会了（**过拟合**）。
- 一般情况下 3 轮就可以，在实际开始训练后，只要 LOSS 没有趋于平缓，就可以再继续加大训练轮数，反之如果 LOSS 开始提前收敛，就可以减小 Epoch 或者提前结束。注意不要把 LOSS 降的过低（趋近于零），训练轮数尽量不要超过 10 ，这两种情况大概率会导致过拟合（把模型练傻）。一般情况下，数据集越小，需越多 Epoch；数据集越大，需越少 Epoch，LOSS 控制在 0.5 -> 1.5 之间。  

**学习率（Learning Rate）** 决定了模型在每次更新时参数调整的幅度，通常在 (0, 1) 之间。也就是告诉模型在训练过程中 “学习” 的速度有多快。学习率越大，模型每次调整的幅度就越大；学习率越小，调整的幅度就越小。

- 学习率大（比如0.1）：每次做完一道题后，你会对解题方法进行很大的调整。比如，你可能会完全改变解题思路。优点是进步可能很快，因为你每次都在进行较大的调整。缺点就是可能会因为调整幅度过大而“走偏”，比如突然改变了一个已经掌握得很好的方法，导致之前学的东西都忘了。
- 学习率小（比如0.0001）：每次做完一道题后，你只对解题方法进行非常细微的调整。比如，你发现某个步骤有点小错误，就只调整那个小错误。优点是**非常稳定**，不会因为一次错误而“走偏”，适合需要精细调整的场景。缺点就是**进步会很慢**，因为你每次只调整一点点。
- 一般保持在 5e-5（0.00005） 和 4e-5（0.00004）之间，小数据集尽量不要用大学习率，不要担心速度慢，速度快了反而会影响微调效果。

**批量大小（Batch Size）** 是指在模型训练过程中，每次更新模型参数时所使用的样本数量。它是训练数据被分割成的小块，模型每次处理一个小块的数据来更新参数。

通俗来说，批量大小可以用来平衡复习速度和专注度，确保既能快速推进复习进度，又能专注细节。假设你决定每次复习时集中精力做一定数量的题目，而不是一次只做一道题。

- 批量大（比如100）：每次复习时，你集中精力做100道题。**优点是复习速度很快，因为你每次处理很多题目，能快速了解整体情况（训练更稳定，易收敛到全局最优）**。**缺点是可能会因为一次处理太多题目而感到压力过大，甚至错过一些细节（耗显存，泛化能力差，易过拟合）。**
- 批量小（比如1）：每次复习时，你只做一道题，做完后再做下一道。**优点是可以非常专注，能仔细分析每道题的细节，适合需要深入理解的场景（省显存，易捕捉数据细节，泛化能力强）。****缺点就是复习速度很慢，因为每次只处理一道题（训练不稳定，易陷入局部最优）。**
- 一般情况下，大 batch_size 需搭配大学习率，小 batch_size 需搭配小学习率，对于一些小参数模型、小数据集的微调，单 GPU 批处理大小（Per Device Train Batch Size）建议从 2 甚至是 1 开始（要想省显存就直接设置 1），通过调大梯度累积步数（比如设置为 4 或者 8）来加快模型微调速度。

### 超参数设置具体代码

```python
from unsloth import is_bfloat16_supported
from transformers import TrainingArguments
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

# 训练参数
    training_args = TrainingArguments(
        output_dir="/home/qwen-sft/model-sft/output",
        per_device_train_batch_size=16,
        gradient_accumulation_steps=1,
        save_steps=500,
        num_train_epochs=3,
        learning_rate=2e-5,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=10,
        optim="paged_adamw_8bit",#使用的优化器，用来调整模型参数
        lr_scheduler_type="cosine",#学习率调度器类型，控制学习率的变化方式
        warmup_ratio=0.03,
        save_total_limit=2,
        report_to="wandb",
        run_name="qwen-sft-run1",
        max_grad_norm=1.0
    )
    
data_collator = DataCollatorForCompletionOnlyLM(
    response_template="<|im_start|>assistant",
    instruction_template="<|im_start|>user",
    tokenizer=tokenizer,
)
# SFT Trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=tokenized_dataset,
        dataset_text_field="text",#指定数据集中文本字段的名称
        max_seq_length=max_seq_length,
        args=training_args,
        packing=False,#是否启用打包功能，打包可以让训练更快，但可能影响效果
        data_collator=data_collator,
        #dataset_num_proc=2#设置数据处理的并行进程数
    )


```



# 五、合并模型（代码）

```python
# Merge to 16bit
if False: model.save_pretrained_merged("model", tokenizer, save_method = "merged_16bit",)
if False: model.push_to_hub_merged("hf/model", tokenizer, save_method = "merged_16bit", token = "")

# Merge to 4bit
if False: model.save_pretrained_merged("model", tokenizer, save_method = "merged_4bit",)
if False: model.push_to_hub_merged("hf/model", tokenizer, save_method = "merged_4bit", token = "")

# Just LoRA adapters
if False:
    model.save_pretrained("model")
    tokenizer.save_pretrained("model")
if False:
    model.push_to_hub("hf/model", token = "")
    tokenizer.push_to_hub("hf/model", token = "")
```

```python
# Save to 8bit Q8_0
if False: model.save_pretrained_gguf("model", tokenizer,)
# Remember to go to https://huggingface.co/settings/tokens for a token!
# And change hf to your username!
if False: model.push_to_hub_gguf("hf/model", tokenizer, token = "")

# Save to 16bit GGUF
if False: model.save_pretrained_gguf("model", tokenizer, quantization_method = "f16")
if False: model.push_to_hub_gguf("hf/model", tokenizer, quantization_method = "f16", token = "")

# Save to q4_k_m GGUF
if False: model.save_pretrained_gguf("model", tokenizer, quantization_method = "q4_k_m")
if False: model.push_to_hub_gguf("hf/model", tokenizer, quantization_method = "q4_k_m", token = "")

# Save to multiple GGUF options - much faster if you want multiple!
if False:
    model.push_to_hub_gguf(
        "hf/model", # Change hf to your username!
        tokenizer,
        quantization_method = ["q4_k_m", "q8_0", "q5_k_m",],
        token = "", # Get a token at https://huggingface.co/settings/tokens
    )
```



```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Step 1: 加载原始模型
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-8B",
    torch_dtype=torch.float16,
    device_map="auto"
)

# Step 2: 加载训练好的 LoRA adapte
peft_model = PeftModel.from_pretrained(base_model, "/home/qwen-sft/model-sft/output/checkpoint-4000")

# Step 3: 合并权重
merged_model = peft_model.merge_and_unload()

# Step 4: 保存合并后的模型（可上传 HF）
merged_model.save_pretrained("/home/qwen-sft/model-sft/merged")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
tokenizer.save_pretrained("/home/qwen-sft/model-sft/merged")

```

#  六、测试模型

```python
# alpaca_prompt = Copied from above
FastLanguageModel.for_inference(model) # Enable native 2x faster inference
inputs = tokenizer(
[
    alpaca_prompt.format(
        "Continue the fibonnaci sequence.", # instruction
        "1, 1, 2, 3, 5, 8", # input
        "", # output - leave this blank for generation!
    )
], return_tensors = "pt").to("cuda")

outputs = model.generate(**inputs, max_new_tokens = 64, use_cache = True)
tokenizer.batch_decode(outputs)
```

- 交互式对话测试

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 加载模型（你已训练好的）
model_name = "hahayang012/Qwen3-8B-SFT"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)
model.eval()

# 初始化对话历史
history = [
    {"role": "system", "content": ""},##这里输入system prompt 和历史对话
    {"role": "user", "content": ""},
    {"role": "assistant", "content": ""},
    {"role": "user", "content": ""},
    {"role": "assistant", "content": ""}
]

# 构建 ChatML Prompt
def build_chatml_prompt(history):
    prompt = ""
    for turn in history:
        prompt += f"<|im_start|>{turn['role']}\n{turn['content']}\n<|im_end|>\n"
    prompt += "<|im_start|>assistant\n"  # 留给模型生成
    return prompt

# 对话函数
def chat():
    print("开始对话吧！（输入 exit 或 退出 来结束）")
    while True:
        user_input = input("你：")
        if user_input.lower() in ["exit", "退出", "quit"]:
            print("已结束对话。")
            break

        history.append({"role": "user", "content": user_input})
        prompt = build_chatml_prompt(history)

        # 编码 + 生成
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
        outputs = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=1024,
            do_sample=True,
            temperature=0.95,
            top_k=20,
            top_p=0.85,
            repetition_penalty=1.2,
            eos_token_id=tokenizer.eos_token_id,
        )

        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = decoded.split("<|im_start|>assistant\n")[-1].split("<|im_end|>")[0].strip()
        history.append({"role": "assistant", "content": response})
        print(f"assistant：{response}")

# 启动对话
chat()

```

# 七、打榜评测参数

| 参数名                 | 推荐值  | 用来干嘛（通俗解释）                                         |
| ---------------------- | ------- | ------------------------------------------------------------ |
| **Top K**              | 50      | 限制“下一个词”的候选词数量。例如：模型原本能选10000个词，但只考虑概率最高的50个词。防止选到奇怪词。 |
| **Top P**              | 0.9     | 再加一层“总概率过滤”。从概率高到低选出一组词，直到加起来超过90%概率。动态控制候选词，稳中求变。 |
| **Min P**              | 0.01    | 设定最低“入选门槛”，太冷门的词一律不考虑。这个值越高，模型越保守。一般不用调。 |
| **Temperature**        | 0.7–0.9 | 控制“选词的激进程度”。值越低越稳重（更准确），值越高越发散（更有创意）。 |
| **Repetition Penalty** | 1.1–1.2 | 惩罚模型重复用词的行为。数值越高，越不容易生成重复内容。防止出现“啰嗦”或“答复一模一样”的情况。 |
