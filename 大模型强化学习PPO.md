# reward model

dataset:"question""chosen""rejected"

model:和要训练的模型能力差不多或者更强

loss：<img src="大模型强化学习PPO.assets/image-20250804155633201.png" alt="image-20250804155633201" style="zoom: 67%;" />

代码实现：https://github.com/RethinkFun/trian_ppo/tree/main/train_ppo



# PPO

![image-20250804160119662](大模型强化学习PPO.assets/image-20250804160119662.png)

为了减少显存的负担，将加载四个大模型优化为加载一个大模型和两个adapter![image-20250804160411259](大模型强化学习PPO.assets/image-20250804160411259.png)

score是给最后一个token打分

如何计算reward：训练模型相对于基准模型的kl散度*（-0.2）+score

![image-20250804160821817](大模型强化学习PPO.assets/image-20250804160821817.png)

如何计算advantage：

![image-20250804160944583](大模型强化学习PPO.assets/image-20250804160944583.png)

如何计算loss：

总的loss包含PPO loss和state value的loss

- state value的loss

<img src="大模型强化学习PPO.assets/image-20250804161133381.png" alt="image-20250804161133381" style="zoom:80%;" />

![image-20250804161223293](大模型强化学习PPO.assets/image-20250804161223293.png)

- ppo loss

![image-20250804161515596](大模型强化学习PPO.assets/image-20250804161515596.png)

ppo训练循环伪代码：

![image-20250804161755282](大模型强化学习PPO.assets/image-20250804161755282.png)

如何使用trl库中的PPOTrainer：

https://github.com/RethinkFun/trian_ppo/tree/main/train_ppo