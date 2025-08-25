# pytorch深度学习

1. 在anaconda prompt 下新建conda环境并激活

`conda create -n learnpy python=3.11`

`conda activate learnpy`

2. 在cmd里输入`nvidia-smi`查找自己的GPU型号和对应的cuda版本

![image-20250705163800853](Pytorch深度学习教程.assets/image-20250705163800853.png)

在pytorch官网上找到对应pytorch下载方式

```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

3. 如果想在环境下使用jupyter，就在终端输入`conda install nb_conda`

安装好之后在终端输入`jupyter notebook`即可

4. 用这两个函数帮助我们了解一个package里面有什么、怎么用

   dir（）：打开，看见

   help（）：说明书

## pytorch加载数据

Dataset:获取数据（并进行编号）及其label

- 如何获取每一个数据及其label
- 告诉我们总共有多少数据

Dataloader：为后面的网络提供不同的数据形式

```python
from torch.utils.data import Dataset
from PIL import Image
import os
##path = os.path.join(root_dir,label_dir) 路径拼接

class Mydata(Dataset):
        
    def __init__(self,root_dir,img_dir):
        self.root_dir=root_dir
        self.img_dir=img_dir
        self.path=os.path.join(self.root_dir,self.img_dir)
        self.image_path=os.listdir(self.path)
    
    def __getitem__(self,idx):
        image_name=self.image_path[idx]   
        image_path=os.path.join(self.root_dir,self.img_dir,image_name)
        image=Image.open(image_path)
        label = self.img_dir.split('_')[0]
        return image,label
    
    def __len__(self):
        return len(self.image_path)
    
root_dir="D:\\machine_learning\\Code_training\\dataset\\hymenoptera_data\\train"  #注意这里的地址\\分隔
ants_dir="ants_image"  #分为数据目录和标签目录
bees_dir="bees_image"
ants_label_dir="ants_label"
bees_label_dir="bees_label"
ants_dataset= Mydata(root_dir,ants_dir)
bees_dataset= Mydata(root_dir,bees_dir)
train_dataset = ants_dataset + bees_dataset
print(len(ants_dataset))
print(ants_dataset[0])  # This will print the first image and its label

image,label= ants_dataset[0]
image.show()

for i in ants_dataset.image_path:
    file_name = i.split('.jpg')[0]
    with open(os.path.join(root_dir, ants_label_dir, "{}.txt".format(file_name)), 'w') as f:
        f.write(label)  #将数据对应的标签以文件名为图片地址，内容为标签的形式存放在标签目录下
```



4. ## Tensorboard的使用（暂时没看，感觉跟wandb差不多？）

## transform的使用

transform.py相当于一个工具箱，里面有totensor、resize等工具

以transform.totensor为例看transform怎么用以及tensor这种数据类型

```python
from torchvision import transforms
from PIL import Image

#lab1：ToTensor:输入为PIL或者narray，输出为tensor
image_path = "D:\\machine_learning\\Code_training\\dataset\\hymenoptera_data\\train\\ants_image\\0013035.jpg"
img=Image.open(image_path)

tensor_trans=transforms.ToTensor()  #利用已有的工具创建具体的工具
tensor_img=tensor_trans(img)  #根据工具的使用方式传入参数

print(tensor_img)
```

## 常见的transforms

1. 关注输入与输出类型
2. 通过crtl+单击函数名可以看官方文档
3. 关注输入参数（官方文档写了），如果想看输出类型，可以print（输出）来查看

PIL  ->Image.open()

tensor  ->ToTensor()

narrays  ->cv.imread()

```python
#lab2：Normalize:输入为tensor，输出为归一化的tensor
img=Image.open("dataset\\hymenoptera_data\\train\\ants_image\\0013035.jpg")
trans_norm=transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
tensor_trans=transforms.ToTensor()
tensor_img=tensor_trans(img)
img_norm=trans_norm(tensor_img)
print(img_norm)
print(tensor_img)
```

```python
# lab3 Resize:输入为PIL图像，输出为给定大小的PIL图像
img=Image.open("dataset\\hymenoptera_data\\train\\ants_image\\0013035.jpg")
tran_resize=transforms.Resize((256, 256))
img_resize=tran_resize(img)
img_resize.show()  # 显示调整大小后的图像
```

```python
#lab4 Compose:组合多个转换,输入参数是一个transform类型列表，且前后顺序要满足前者输出格式是后者输入格式
img=Image.open("dataset\\hymenoptera_data\\train\\ants_image\\0013035.jpg")
tran_resize=transforms.Resize(512)
tensor_trans=transforms.ToTensor()
tran_resize2=transforms.Compose([tran_resize,tensor_trans])
img_resize2=tran_resize(img)
img_resize2.show()  # 显示调整大小并转换为张量后的图像
img_tensor_resize2=tran_resize2(img)
print(img_tensor_resize2)  # 打印张量的形状
```

```python
# lab5: RandomCrop:随机裁剪,输入一个PIL图像，输出一个随机裁剪后的PIL图像
img=Image.open("dataset\\hymenoptera_data\\train\\ants_image\\0013035.jpg")
trans_random_crop=transforms.RandomCrop(200)  # 随机裁剪为200*200大小，也可以（512，200）这样规定长和宽
img_random=trans_random_crop(img)
img_random.show()  # 显示随机裁剪后的图像
```

## torchvision中的datasets使用：下载、处理数据集

在pytorch官网找到数据集[[Datasets — Torchvision 0.22 documentation](https://docs.pytorch.org/vision/stable/datasets.html)](https://docs.pytorch.org/vision/stable/datasets.html)

如果在py文件里下载缓慢，可以先用迅雷等下载好数据集，然后将数据集粘贴到对应的root目录下，再运行代码，会自动解压

如何找下载链接：ctrl单击数据集名称，找到对应的url，复制到迅雷下载

```python
import torchvision
#下载数据集、提取数据集中的数据和target（label）
train_set=torchvision.datasets.CIFAR10(root="D:\\machine_learning\\Code_training\\dataset\\CIFAR10",train=True,download=True)
test_set=torchvision.datasets.CIFAR10(root="D:\\machine_learning\\Code_training\\dataset\\CIFAR10",train=False,download=True)
print(test_set[0])
print(test_set.classes)
img,target=test_set[0]
print(img)
print(target)
print(test_set.classes[target])
img.show()  # 显示图像
```

```python
#加上对数据集的transform处理
dataset_transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
train_set=torchvision.datasets.CIFAR10(root="D:\\machine_learning\\Code_training\\dataset\\CIFAR10",train=True,transform=dataset_transform,download=True)
test_set=torchvision.datasets.CIFAR10(root="D:\\machine_learning\\Code_training\\dataset\\CIFAR10",train=False,transform=dataset_transform,download=True)
```

## DataLoader的使用

![image-20250707122411038](Pytorch深度学习教程.assets/image-20250707122411038.png)

```python
import torch
import torchvision

test_data=torchvision.datasets.CIFAR10(root="dataset\\CIFAR10",train=False,download=True,transform=torchvision.transforms.ToTensor()) #读取数据集
test_loader=torch.utils.data.DataLoader(dataset=test_data,batch_size=4,shuffle=True,num_workers=0,drop_last=False)  #对数据集进行打包，每个包里有batch_size个数据

img,target=test_data[0]
print(img.shape)
print(target)

for epoch in range(2): #可以对数据进行多轮次打包，如果shuffle设置为true，多次打包结果不一致，反之则一致
    for data in test_loader:
        imgs,targets=data
        print(imgs.shape)
        print(targets)
```

## 神经网络的基本骨架：nn.Module的使用

![image-20250707143831218](Pytorch深度学习教程.assets/image-20250707143831218.png)

![image-20250707144311184](Pytorch深度学习教程.assets/image-20250707144311184.png)

![image-20250707144337421](Pytorch深度学习教程.assets/image-20250707144337421.png)

![image-20250707144344351](Pytorch深度学习教程.assets/image-20250707144344351.png)



```python
from torch import nn
import torch
class test(nn.Module):
    def __init__(self):
        super(test, self).__init__() #继承父类
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)  #定义卷积层
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):  #定义前向传播函数
        output=x+1
        return output
    
TEST=test()  #创建实例
x=torch.tensor(1.0) 
output=TEST(x)
print(output)
```

## 神经网络-卷积层

**自动提取图像或数据中的局部特征**，比如边缘、纹理、形状等。

```python
torch.nn.functional.conv1d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1) → Tensor  ##2d和3d传入的参数一致
```

![image-20250707152724708](Pytorch深度学习教程.assets/image-20250707152724708.png)

![image-20250707152751816](Pytorch深度学习教程.assets/image-20250707152751816.png)

bias是偏置，stride是移动的路径大小（包括横向和纵向）

input和weight(kernel)都需要有四位，如果不满足，可以用torch.reshape(input,(x,x,x,x))进行转换

padding=1代表在input的上下左右都加一列0

```python
torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None) ##torch.nn相当于对torch.nn.functional进行封装，平时就需要设置前五个参数
```

![image-20250707154802953](Pytorch深度学习教程.assets/image-20250707154802953.png)![image-20250707164141340](Pytorch深度学习教程.assets/image-20250707164141340.png)

```python
from torch.utils.data import DataLoader
import torchvision
import torch.nn as nn
from torch.nn import Conv2d

# 加载 CIFAR10 数据集
dataset = torchvision.datasets.CIFAR10(
    "../data", 
    train=False, 
    transform=torchvision.transforms.ToTensor(), 
    download=True
)

# 创建 DataLoader
dataloader = DataLoader(dataset, batch_size=64)

# 定义自定义神经网络模块
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.conv1 = Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        return x

# 创建网络实例
tudui = Tudui()

for data in dataloader:
    imgs,targets=data
    outputs=tudui(imgs)
    print(imgs.shape)
    print(outputs.shape)
```



## 神经网络-最大池化的使用

用于**压缩特征图大小**、**减少计算量**、**增强特征的空间不变性**（比如旋转、平移等不敏感）。

在卷积层后面通常接池化层，例如：

```python
self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
```

这会把特征图尺寸缩小一半（例如从 32×32 → 16×16）。

```python
torch.nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)[source]#取出最大值
```

![image-20250707162417648](Pytorch深度学习教程.assets/image-20250707162417648.png)

![image-20250707163353742](Pytorch深度学习教程.assets/image-20250707163353742.png)

dilation:在池化核的每两个元素之间插入一个0元素

ceil_mode:保留不满kernel_size的部分 ；floor_mode：不保留

 

## 非线性激活

理论上加上激活函数后，神经网络可逼近任意复杂函数（万能逼近定理），好的激活函数还能改善反向传播过程，避免梯度消失/爆炸问题![image-20250707164434476](Pytorch深度学习教程.assets/image-20250707164434476.png)

```
torch.nn.ReLU(inplace=False)
```

```
torch.nn.Hardsigmoid(inplace=False)
```

```
torch.nn.Softmax(dim=None)
```



## 线性层和其他层的介绍

直接参照官方文档[[torch.nn — PyTorch 2.7 documentation](https://docs.pytorch.org/docs/stable/nn.html)](https://docs.pytorch.org/docs/stable/nn.html)



## 损失函数与反向传播与优化器

**损失函数**告诉模型学得好不好，
**反向传播**告诉模型该如何调整自己（参数）来变得更好。![image-20250707180156775](Pytorch深度学习教程.assets/image-20250707180156775.png)

```python
import torch.nn as nn

loss_fn = nn.MSELoss()
y_pred = torch.tensor([0.5, 0.7], requires_grad=True)
y_true = torch.tensor([1.0, 0.0])
loss = loss_fn(y_pred, y_true)
print(loss)  # 输出一个标量 loss 值
```

![image-20250707172619137](Pytorch深度学习教程.assets/image-20250707172619137.png)

```python
import torch
import torch.nn as nn
import torch.optim as optim

model = nn.Linear(2, 1)             # 一个简单线性层
loss_fn = nn.MSELoss()              # 损失函数
optimizer = optim.SGD(model.parameters(), lr=0.01)  # 优化器

x = torch.tensor([[1.0, 2.0]])      # 输入
y_true = torch.tensor([[5.0]])      # 真实值

# 1. 前向传播
y_pred = model(x)
loss = loss_fn(y_pred, y_true)

# 2. 反向传播
loss.backward()

# 3. 更新参数
optimizer.step()

# 4. 清零梯度
optimizer.zero_grad()

```

![image-20250707180459906](Pytorch深度学习教程.assets/image-20250707180459906.png)

![image-20250707181045211](Pytorch深度学习教程.assets/image-20250707181045211.png)

![image-20250707181136086](Pytorch深度学习教程.assets/image-20250707181136086.png)
