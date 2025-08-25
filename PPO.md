![image-20250804121728089](PPO.assets/image-20250804121728089.png)

目标：训练一个Policy神经网络Π，在所有trajectory中（在所有状态S下，给出相应的Action），得到Return的期望最大

# on-policy的训练：

### 策略梯度：

$$

\frac{1}{N} \sum_{n=1}^{N} \sum_{t=1}^{T_n} R(\tau^n) \log P_\theta(a_n^t \mid s_n^t)
$$



### Loss：

$$
\text{Loss} = -\frac{1}{N} \sum_{n=1}^{N} \sum_{t=1}^{T_n} R(\tau^n) \log P_\theta(a_n^t \mid s_n^t)
$$


$$
由于当前做出的动作只对未来的R(\tau^n)有影响，并且只对未来的几步有影响，
因此将R(\tau^n)改写为R_t^n
$$
同时，为了让好的局势和坏的局势下都能高效学习到不同行动的影响，需要减去一个baseline，让这种影响区别更大，更容易学习![image-20250804122627857](PPO.assets/image-20250804122627857.png)

####  Action-Value Function（动作价值函数）

- $R_t^n$：每次都是一次随机采样，方差很大，训练不稳定。
- $Q_\theta(s, a)$：在状态 $s$ 下，做出动作 $a$ 时，**期望的回报**。称为**动作价值函数**。

####  State-Value Function（状态价值函数）

- $V_\theta(s)$：在状态 $s$ 下，**期望的回报**。称为**状态价值函数**。

####  Advantage Function（优势函数）

- $A_\theta(s, a) = Q_\theta(s, a) - V_\theta(s)$：表示在状态 $s$ 下做动作 $a$，**比平均水平（V）好多少**。

### 策略梯度 with Advantage：

$$
\frac{1}{N} \sum_{n=1}^N \sum_{t=1}^{T_n} A_\theta(s_n^t, a_n^t) \nabla \log P_\theta(a_n^t | s_n^t)
$$

#### TD 误差：

$$
\delta_t^V = r_t + \gamma V_\theta(s_{t+1}) - V_\theta(s_t)
$$

#### 多步 Advantage 估计：

$$
A_\theta^1(s_t, a) = \delta_t^V
$$

​					 $A_\theta^2(s_t, a) = \delta_t^V + \gamma \delta_{t+1}^V$

​				 $A_\theta^3(s_t, a) = \delta_t^V + \gamma \delta_{t+1}^V + \gamma^2 \delta_{t+2}^V$ 

​							$\vdots$ 

​		$A_\theta^T(s_t, a) = \delta_t^V + \gamma \delta_{t+1}^V + \gamma^2 \delta_{t+2}^V + \dots + \gamma^{T - t} \delta_T^V$

采样的步数越多偏差越小，方差越大

### GAE Advantage 定义（多步采样分配不同权重）：

平衡了采样不同步时带来的方差、偏差问题
$$
A_\theta^{\text{GAE}}(s_t, a) = \sum_{b=0}^\infty (\gamma \lambda)^b \delta_{t + b}^V
$$

------

#### 使用 GAE 的策略梯度：

$$
\frac{1}{N} \sum_{n=1}^N \sum_{t=1}^{T_n} A_\theta^{\text{GAE}}(s_n^t, a_n^t) \nabla \log P_\theta(a_n^t | s_n^t)
$$

------

#### Label（强化学习中的“监督信号”）定义：

$$
\text{Label} = \sum_{t'=t}^{T_n} \gamma^{t'-t} r_{t'}^n
$$



# Proximal Policy Optimization(PPO)：邻近策略优化（Off Policy）

利用重要性采样定理来更新off policy的策略梯度公式和Loss：

![image-20250804120653813](PPO.assets/image-20250804120653813.png)

为了防止我训练的策略与参考策略相差太大，加上 KL散度 或 利用Clip 损失函数

![image-20250804120829340](PPO.assets/image-20250804120829340.png)