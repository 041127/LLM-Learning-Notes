# GPU 环境搭建和基于axolotl训练大模型 

### 遇到的问题小结

1. ![image-20250702140002889](C:\Users\14775\AppData\Roaming\Typora\typora-user-images\image-20250702140002889.png)

   猜测原因与解决办法：

   (1)axolotl最新的python版本要求≥3.11，配置的python版本是3.10->重新配了一次3.11的版本，可以显示下载进度了，但是下载速度慢->重启终端重新运行速度就变快了

   (2)远程服务器没有开网络代理，导致网速慢->

   ```
   source /etc/proxy/net_proxy #开启网络代理
   ```

   在下载前先开网络代理 如果还不行就重启服务器再重试

2. ![image-20250702140546933](C:\Users\14775\AppData\Roaming\Typora\typora-user-images\image-20250702140546933.png)最开始用的cuda11.8，执行axolotl fetch examples这一步时报错，说找不到`libcudart.so.12`，说明Axolotl 或其他依赖库尝试加载 CUDA 12.x 的运行时库，于是更换cuda版本为12.2.2

3. 交互式推理时报错![image-20250702143346150](C:\Users\14775\AppData\Roaming\Typora\typora-user-images\image-20250702143346150.png)

   ->source /etc/proxy/net_proxy开网络代理

4. 终端有问题时可以尝试重开一个，再不行重启服务器

5. 使用wandb可视化时，entity出错->在项目的网址中可以确定entity

## 远程连接服务器并配置pytorch和**Axolotl**

服务器配置：![image-20250702101017843](C:\Users\14775\AppData\Roaming\Typora\typora-user-images\image-20250702101017843.png)cuda12.2.2在vscode里通过ssh和密码连接

![image-20250702101727623](C:\Users\14775\AppData\Roaming\Typora\typora-user-images\image-20250702101727623.png)

连接成功后在终端输入`nvidia-smi` 命令检查 GPU 是否可用：![image-20250701150541200](C:\Users\14775\AppData\Roaming\Typora\typora-user-images\image-20250701150541200.png)

再使用conda创建隔离环境

```Bash
conda create -n lab01 python=3.11
```

如果此处报错，需要conda init后exit终端重进，再

```
conda activate lab01
```

![image-20250702101958848](C:\Users\14775\AppData\Roaming\Typora\typora-user-images\image-20250702101958848.png)

在官网里找到对应cuda12.2.2的pytorch

```
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
```

![image-20250702102859079](C:\Users\14775\AppData\Roaming\Typora\typora-user-images\image-20250702102859079.png)

安装 Axolotl 及其依赖项：

```Bash
pip3 install -U packaging setuptools wheel ninja
```

![image-20250702094917914](C:\Users\14775\AppData\Roaming\Typora\typora-user-images\image-20250702094917914.png)

```
两种方法
1.pip3 install -e '.[flash-attn,deepspeed]' 速度很慢 网络不稳定容易中断
2.pip3 install --no-build-isolation axolotl[flash-attn,deepspeed] 不隔离环境 速度快
```

最开始我配置环境时用的python==3.10,在执行pip3 install -e '.[flash-attn,deepspeed]'时总是下载不成功，可能是网络不稳定的原因，换成如下的命令

```
pip3 install -e '.[flash-attn,deepspeed]' --retries 5 --timeout 100
```

好了一些，但最后卡在这一步![image-20250701161525906](C:\Users\14775\AppData\Roaming\Typora\typora-user-images\image-20250701161525906.png)

还是花了很长很长时间也没有成功下载，经研究觉得是python版本不适配的原因，![image-20250702095027713](C:\Users\14775\AppData\Roaming\Typora\typora-user-images\image-20250702095027713.png)axolotl要求python版本≥3.11，而我用的3.10，于是重新配环境lightintern_lab01（python==3.11),可以下载，但是下载速度慢，于是关了终端重新开一个就正常了，最终完成了axolotl的安装![image-20250702095332749](C:\Users\14775\AppData\Roaming\Typora\typora-user-images\image-20250702095332749.png)

在之后重新配环境时又遇到了这个问题，这次采用开启服务器网络代理的方法解决

```
source /etc/proxy/net_proxy  #这个相当于一个vpn
pip3 install --no-build-isolation axolotl[flash-attn,deepspeed]
如果还没效果 就重启服务器重新运行这两行代码
```



## 模型训练

将demo拖到服务器的home目录下新建的lab01目录下，在配好的环境下执行，获得配置文件yml

```
axolotl fetch examples
```

![image-20250701170620931](C:\Users\14775\AppData\Roaming\Typora\typora-user-images\image-20250701170620931.png)

有一次报错发现是因为找不到`libcudart.so.12`，说明：

- PyTorch 编译时使用的是 CUDA 11.8

- 但 Axolotl 或其他依赖库尝试加载 CUDA 12.x 的运行时库![image-20250702100924944](C:\Users\14775\AppData\Roaming\Typora\typora-user-images\image-20250702100924944.png)

  于是又把cuda版本从11.8换成12.2.2

执行以下代码进行训练

```
HF_ENDPOINT=https://hf-mirror.com axolotl train examples/llama-3/lora-1b.yml
```

训练中![image-20250702140700113](C:\Users\14775\AppData\Roaming\Typora\typora-user-images\image-20250702140700113.png)训练完毕

![image-20250702142604145](C:\Users\14775\AppData\Roaming\Typora\typora-user-images\image-20250702142604145.png)

执行以下代码进行交互式推理

```Plain
axolotl inference examples/llama-3/lora-1b.yml --lora-model-dir="./outputs/lora-out"
```

有胡乱作答的可能性![image-20250702143938089](C:\Users\14775\AppData\Roaming\Typora\typora-user-images\image-20250702143938089.png)

## wandb可视化

![image-20250702165811586](C:\Users\14775\AppData\Roaming\Typora\typora-user-images\image-20250702165811586.png)

![image-20250702165841213](C:\Users\14775\AppData\Roaming\Typora\typora-user-images\image-20250702165841213.png)
