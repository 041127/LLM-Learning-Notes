# linux

内核->系统库->shell（命令行解释器）->应用程序

## 如何安装配置linux系统

虚拟机软件（已用VMWare安装了ubuntu）、容器安装、云服务器

## vi编辑器（常用的命令）

![image-20250630105437349](C:\Users\14775\AppData\Roaming\Typora\typora-user-images\image-20250630105437349.png)

1. i/a/o和I/A/0都可以用来从命令模式（command mode）切换到插入模式（insert mode）（插入模式用来插入文本）

2. ：用来用命令模式切换到尾行模式（last line mode）

### 命令模式下：

3. vi hello.txt 创建文件
4. ls命令用来查看当前目录下的文件和文件夹
5. cat命令用来查看文件的内容 cat+文件名 如cat hello.txt![image-20250630105639702](C:\Users\14775\AppData\Roaming\Typora\typora-user-images\image-20250630105639702.png)
6. HJKL控制光标的上下左右
7. ^跳转到该行行首 $跳转到行尾
8. yy 复制一行内容
9. p 粘贴内容（复制、删除的内容都可以粘贴）到光标下一行
10. dd 删除一行内容
11. 在yy p dd前面加入数字代表操作的次数
12. crtl+f 向前翻页 crtl+b向后翻页 ctrl+u向上翻半页 crtl+d向下翻半页
13. G跳转到文件中最后一行 gg第一行 数字+G跳转到数字对应的行数
14. /+要查找的内容 则从光标所在位置向下查找 ？+要查找的内容 从光标向上查找 按下n查找下一个 查找默认区分大小写 若要不区分则在文本后面加上\c
15. u 撤销
16. .vimrc文件用来保存vi的配置信息

### 尾行模式：

17. ：+数字 跳转到数字对应的行数
18. ：set number 显示行号 set nonumber 不显示行号
19. ：set ic 修改全局忽略大小写 则查找时不会区分大小写
20. ：n1，n2s/od/new/g n1和n2代表替换作用的行数范围 od是被替换的 new是替换的新内容 g表示全局

# linux系统常用的命令

1. ls（list）列出当前目录下的所有文件和目录 ls -l显示更全面的信息 ls -a显示隐藏文件

2. ln（link）创建链接文件 默认是硬链接 ln -s软链接 ln -s hello.txt link.txt 

   软链接相当于创建快捷方式，可以指向文件或者目录，软链接文件较小   

   硬链接与目标文件共享同一个i节点，只能指向文件

3. rm 删除文件

4. 文件权限![image-20250630113539273](C:\Users\14775\AppData\Roaming\Typora\typora-user-images\image-20250630113539273.png)

5. chmod（change mode）修改权限 chmod (u/g/o)+x hello.txt  加权限

   chmod (u/g/o)-rw hello.txt 减权限

​       chomd 777 hello.txt  ![image-20250630113930201](C:\Users\14775\AppData\Roaming\Typora\typora-user-images\image-20250630113930201.png)

6. 创建文件(若不存在该文件则新建该文件 若存在则修改该文件更新时间) touch hello.txt

7. echo用来输出文本 echo“hello”则在命令行输出hello echo“hello”>hello.txt将hello输出到该文件中（若不存在则新建文件）

8. cat hello.txt 显示文件内容

9. pwd显示当前目录所在位置

10. cd切换目录 cd /切换到根目录

    使用相对路径表示你想去的目录 .表示当前目录 ..表示上一级目录 cd  ../..切换到上两级目录 cd -表示上一次所在的目录

![image-20250630115758533](C:\Users\14775\AppData\Roaming\Typora\typora-user-images\image-20250630115758533.png)

![image-20250630115908456](C:\Users\14775\AppData\Roaming\Typora\typora-user-images\image-20250630115908456.png)

11. cp（copy）复制文件 copy file1.txt file2.txt 把1复制一份变成2 cp -r可以递归复制目录
12. mv（move）移动/重命名文件 mv file2.txt file3.txt 把2重命名为3
13. rm（remove）删除文件/目录 rm file3.txt rm -r递归删除目录 rm不是可逆的 小心使用
14. mkdir创建目录 mkdir -p 1/2/3创建多级目录 
15. tree 显示目录结构 用sudo apt install tree安装tree

 

# Shell（利用Bash讲解）

1. echo $SHELL 查看当前系统默认使用的shell路径
2. echo $0 查看当前执行脚本的路径
3. cat /etc/shells 可以看系统中所有的shell版本 直接输入版本就可以切换脚本 exit退出

4. shell脚本可以用来编写一些自动化的内容 新建shell文件：vi hello.sh

   #!/bin/bash

   你想执行的内容

5. 执行脚本: chmod a+x hello.sh (添加权限)

   ​                 ./hello.sh（执行）

   或者只用一条命令就能执行：bash hello.sh^C

   编写一个脚本后可以在复制到vscode里 打开终端 在终端输入sh hello.sh执行

6. 首先连接到linux环境 然后用 vi game.sh新建文件，或者用vscode里ssh远程连接服务器或者用nano编辑器

7. 用read来读取用户输入![image-20250630142317163](C:\Users\14775\AppData\Roaming\Typora\typora-user-images\image-20250630142317163.png)

8. 完整新建、编辑、执行的流程![image-20250630141835009](C:\Users\14775\AppData\Roaming\Typora\typora-user-images\image-20250630141835009.png)

9. 用echo来传递参数

   ![image-20250630142200853](C:\Users\14775\AppData\Roaming\Typora\typora-user-images\image-20250630142200853.png)执行效果如图![image-20250630142440396](C:\Users\14775\AppData\Roaming\Typora\typora-user-images\image-20250630142440396.png)

   ![image-20250630142524369](C:\Users\14775\AppData\Roaming\Typora\typora-user-images\image-20250630142524369.png)

10. 除了手动输入传递参数外，还可以通过定义环境变量来实现

    图一只是定义了普通变量name和channel 只在当前shell会话有效

![image-20250630142645391](C:\Users\14775\AppData\Roaming\Typora\typora-user-images\image-20250630142645391.png)

图二将普通变量转换为环境变量，这样在脚本文件中就可以获取到变量，但是使用export定义的变量只在当前shell会话有效

![image-20250630142845705](C:\Users\14775\AppData\Roaming\Typora\typora-user-images\image-20250630142845705.png)

图三修改了.bashrc这个配置文件，使name和channel成为了环境变量（依旧用export定义） 但是要注意：修改了之后要执行source .bashrc再exit再重新登陆才会生效![image-20250630143511563](C:\Users\14775\AppData\Roaming\Typora\typora-user-images\image-20250630143511563.png)

11. 编写更复杂的脚本

    ![image-20250630144024160](C:\Users\14775\AppData\Roaming\Typora\typora-user-images\image-20250630144024160.png)

    ![image-20250630145844551](C:\Users\14775\AppData\Roaming\Typora\typora-user-images\image-20250630145844551.png)

    while循环  continue和break继续循环和跳出循环![image-20250630150043657](C:\Users\14775\AppData\Roaming\Typora\typora-user-images\image-20250630150043657.png)





# SSH

```
1. ssh-keygen -t rsa -b 4096 -C "hahayang"用于创建公钥和私钥，配置好后在vscode可以免输入多次密码来连接远程服务器
2. 在算力平台上租了服务器后会提供ssh和密码，通过多次输入密码来连接
```

# Docker

## 镜像

是一个只读的模板，包含了运行应用程序所需的所有内容，如代码、运行时环境、系统工具、系统库和设置等。可以把镜像理解为一个软件安装包，它提供了运行特定软件的完整环境。例如，一个 Python 应用的镜像，不仅包含 Python 解释器，还包含了应用运行所需的各类依赖库。

通过编写 Dockerfile 文件来定义镜像的构建步骤。

```dockerfile
# 使用官方的Python 3.8基础镜像
FROM python:3.8

# 设置工作目录
WORKDIR /app

# 将当前目录下的所有文件复制到镜像的/app目录下
COPY. /app #第一个参数代表本地路径 此处.代表程序根目录下的所有文件 第二个参数代表docker镜像中的路径（目标路径）

# 安装应用程序所需的依赖（如果有），run是创建容器时使用的
# RUN pip install --no-cache-dir -r requirements.txt

# 定义容器启动时执行的命令，CMD是运用容器使用的
CMD ["python", "hello.py"]
```

<u>编写好 Dockerfile 后，在包含该文件的目录下的终端执行`docker build -t my-python-app.`</u>，其中`-t`参数用于指定镜像的名称和标签，`.`表示使用当前目录下的 Dockerfile 进行构建。

## 容器

容器是镜像的运行实例，是一个可运行的环境。它是基于镜像创建的，提供了应用程序运行的沙盒环境。容器之间相互隔离，拥有自己独立的文件系统、网络接口等资源，这保证了应用程序运行的稳定性和安全性。例如，你可以基于一个 Web 应用镜像启动多个容器，每个容器都是该 Web 应用的一个独立运行实例。

<u>使用`docker run`命令来创建并启动容器。</u>`docker run`命令的基本格式是`docker run [OPTIONS] IMAGE [COMMAND] [ARG...]`。例如，基于前面创建的`my-python-app`镜像来启动一个容器，可以执行`docker run my-python-app`。如果要将容器的端口映射到宿主机，可以使用`-p`参数，如`docker run -p 8080:80 my-web-app`，将容器内的 80 端口映射到宿主机的 8080 端口。

## Dockerfile

 是一个文本文件，其中包含了一系列指令，用于自动化构建 Docker 镜像。它定义了镜像的基础来源、安装的软件包、设置的环境变量、容器启动时要执行的命令等信息。

使用文本编辑器（如 vim、Visual Studio Code 等）在应用根目录下创建一个名为`Dockerfile`（注意大小写，且没有文件扩展名）的文件，然后按照 Dockerfile 的语法规则编写指令。常见指令如下：

- **FROM**：指定基础镜像，这是 Dockerfile 的第一条指令，例如`FROM ubuntu:latest` 。
- **RUN**：在镜像构建过程中执行命令，用于安装软件包等操作，如`RUN apt-get update && apt-get install -y python3` 。
- **COPY**：将宿主机的文件或目录复制到镜像中，如`COPY app.py /app/` 。
- **CMD**：定义容器启动时默认执行的命令，一个 Dockerfile 中只能有一条 CMD 指令，如果指定了多条，只有最后一条会生效。
- **EXPOSE**：声明容器运行时要监听的端口，但这只是一个声明，实际的端口映射需要在`docker run`时通过`-p`参数指定。

## 数据卷

```
docker volume create my-python-data 来创建一个数据卷
```

在启动容器时可以通过-v参数指定将这个数据卷挂载到哪个路径上（/etc/python）

```
docker run -dp 80:5000 -v my-python-data:/etc/python my-python-app
```

## docker-compose.yml

用于管理不同容器

通过简单的命令（如`docker-compose up/down`），就能一键启动或停止所有相关容器，而不需要分别手动启动每个容器。

```yaml
version: '3'  # 指定Docker Compose文件的版本，不同版本语法和功能略有差异，当前常用3.x版本
services:  # 定义服务列表
  web:  # 服务名称，可自定义
    build: .  # 表示使用当前目录下的Dockerfile构建镜像。也可以指定镜像仓库地址，如 image: nginx:latest
    ports:  # 端口映射，将容器内的端口映射到宿主机端口
      - "5000:5000"  # 格式为 宿主机端口:容器端口
    depends_on:  # 声明该服务依赖的其他服务，这里表示web服务依赖db服务
      - db
    volumes:  # 定义卷挂载，将宿主机目录与容器内目录进行关联，实现数据持久化
      -./data:/app/data
  db:
    image: postgres:13  # 使用官方的PostgreSQL 13镜像
    environment:  # 设置环境变量
      POSTGRES_USER: user
      POSTGRES_PASSWORD: password
      POSTGRES_DB: mydb
    volumes:
      - pgdata:/var/lib/postgresql/data  # 挂载一个命名卷
volumes:  # 定义命名卷
  pgdata:
```
