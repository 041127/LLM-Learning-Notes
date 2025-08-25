# transformer的输入是什么

token id（数字）

# Tokenizer的作用

将文本序列转化为数字序列（token编号），作为transformer的输入

![image-20250804163446069](Tokenizer说明.assets/image-20250804163446069.png)

以下网址可以直观看到不同模型的不同分词方式

HuggingFace:  https://huggingface.co/spaces/Xenova/the-tokenizer-playground  
OpenAI:       https://platform.openai.com/tokenizer

# 三种不同分词粒度的Tokenizers

### word-based

将文本划分为一个个词（包括标点），需要处理很多特殊规则，spaCy和Moses可以很好地处理英文分词中的规则

#### 优点

- 符合人的自然语言和直觉

#### 缺点

- 相同意思的词被划分为不同的token（dog和dogs是两个token）
- 词表非常大（为解决词表非常大：限制词表大小，比如上限是10000，未知的词用特殊token表示，比如UNKNOWN，但是这样会带来信息丢失，模型的性能受影响）

### character-based

将文本划分为一个个字符（包括标点）

#### 优点

- 可以表示任意（英文文本），不会出现word-based中unknown的情况
- 对西文来说，不需要很大的词表，比如英文只需要不到256个字符

#### 缺点

- 相对单词来说信息量非常低，模型性能一般很差
- 相对于word-based来说，会产生很长的token序列
- 中文也需要一个很大一个词表

### subword-based

- 常用的词不应该再被切分为更小的token或子词（subword）

- 不常用的词或词群应该用子词来表示

又具体分为三种方法：

![image-20250804164931318](Tokenizer说明.assets/image-20250804164931318.png)

# 四种常用的Subword Tokenizers

### Byte-Pair Encoding(BPE)

包含两部分：

- “词频统计”和“词表合并”

![image-20250804165620961](Tokenizer说明.assets/image-20250804165620961.png)

![image-20250804165755299](Tokenizer说明.assets/image-20250804165755299.png)

![image-20250804165912779](Tokenizer说明.assets/image-20250804165912779.png)

#### 缺点

- 包含所有可能的基本字符（token）的基本词汇表可能会很大

    （例如中文里将所有unicode字符都视为基本字符）

#### 改进：Byte-level BPE

- 将字节视为基本token
- 两个字节合并则可以表示unicode

### WordPiece





### SentencePiece





### Unigram

