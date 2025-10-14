### 通用基础

>  数学 概念 工具...

啥也不会 哭了

##### Math

得补课概率论和高数...

* 马尔可夫过程 *Markov process*
  * 该过程是**步骤独立**的, 状态间的转换仅取决于系统**当前的**状态, 而与系统过去或未来任意状态都独立不相关
  * 离散状态下被称为 *马尔可夫链*
    * 典型: 有A, B两种状态, 转换规则: 当前状态为[A]: to-A: 0.7, to-B: 0.3; [B]: to-A: 0.4, to-B: 0.6;
    * 状态转换概率仅取决于当前所处的状态
* KL散度 *KL diverge*
  * 描述两个分布的差别
  * 大小介于[0, 1)(1可否?)之间, 为0时代表同分布
* 蒙特卡洛特方法 *Monte Carlo Method*
  * 采样逼近期望值


$$
E_{x\backsim p(x)}(f(x))= \sum_{x}{f(x)p(x)} \approx \frac{1}{N}\sum_{n=0}^{N-1}f(x^{(n)})
$$

##### RL Basic

* 概述
  * RL问题可以描述为: Agent从与Environment的交互中不断学习以完成特定目标(如: maximize(Return))
  * RL的目标: 学习到一个策略$\pi_{\theta}(a|s)$来最大化期望回报(expected return)

* Environment 环境
  * 与智能体交互的环境, 接收智能体的Action影响而改变State, 提供Reward反馈
  * 一般来说, Environment类似黑箱, 其内部实现我们无法直接得知, 只能通过有限的接口和规则与其交互并观察其State和反馈(State改变以及Reward)
* Agent 智能体
  * 观察Environment(Observation), 获得当前的State和Reward, 并进行学习和决策(选定Action)
  * 一般来说, RL中训练Agent的目标为最大化预计能获得的Return(非单步reward, 而是预期整个过程能获得的总reward)
* State 状态
  * Environment当前的状态
  * Observation: State的子集, 因为Agent未必能获得全部的State, 有时其获得部分State信息(如: 超agent感知范围和感知结构) 
  * Agent可能直接获得State进行决策, 也可能只能通过Observation进行决策
  * 环境中有一个或多个特殊的终止状态（terminal state）
* 状态转移概率
  * $p(s_{t+1}|s_t, a_t)$
  * 智能体根据当前状态$s_t$做出一个动作$a_t$之后，下一个时刻环境处于不同状态$s_{t+1}$的概率
  * 一般由环境决定, 一般为Monte Carlo过程
* Action
  * Agent能采取的行动
* 策略 Policy $\pi_{\theta}(a|s)$
  * 假设策略使用的模型的参数集为$\theta$
* 轨迹 Trajectory $\tau$
  * 马尔可夫决策过程的一个轨迹（trajectory）
  * $\tau = s_0, a_0, s_1, r_1(s_0, a_0, s_1), ..., s_{T-1}, a_{T-1}, s_T, r_T(s_{T-1}, a_{T-1}, s_T)$
  * $\tau$的概率可以用状态转移概率连乘表示
  * $p(\tau) = p(s_0)\prod^{T-1}_{t=0}{\pi_{\theta}(a_t|s_t)p(s_{t+1}|s_t, a_t)}$
* Reward & Return
  * 奖励 Reward
    * $S: 状态空间; A: 动作空间$
    * $R : S × A × S’ → R$，即智能体根据当前状态做出一个动作之后，环境会反馈给智能体一个奖励，这个奖励和动作之后下一个时刻的状态也有关
    * 相当于函数: $R(s_t, a_t, s_{t+1})$
    * Reward是环境给Agent的奖惩反馈, 其定义会很大程度上影响模型的行为进而影响表现
  * 回报 Return
    * 给定策略$\pi_{\theta}(a|s)$，Agent和Env进行一次交互过程的轨迹$\tau$所收到的累积Reward为总回报Return
    * 轨迹$\tau$的总回报: $G(\tau) = \sum^{T-1}_{t=0}{\gamma^tr_{t+1}}$
      * $\gamma\in[0,1]$为折扣率
      * 当$\gamma\rightarrow0$时，Agent更在意短期回报；而当$\gamma\rightarrow{1}$时，长期回报变得更重要
    * 期望回报: $J(\theta)=\Bbb E_{\tau\backsim p_{\theta}(\tau)}[G(\tau)]=\Bbb E_{\tau\backsim p_{\theta}(\tau)}[\sum^{T-1}_{t=0}{\gamma^tr_{t+1}}]$
      * $\theta$为策略的参数集

* 值函数
  * Bellman方程: 递推求解
  * 状态值函数 State-Value Function: $V_{\pi_{\theta}}(s)$
    * 从状态s开始，执行策略$\pi_{\theta}$得到的期望总回报
    * $V_{\pi_{\theta}}(s) = \Bbb E_{\tau\backsim p(\tau)}[\sum_{t=0}^{T-1}{\gamma^tr_{t+1}}|\tau_{s_0} = s]$
    * Bellman推导: 见PPT; 思路: 树形分散 对于所有可能路径
  * 状态-动作值函数 State Action-Value Function: $Q_\theta(s, a)$
    * 初始状态为s并进行动作a，然后执行策略$\pi$得到的期望总回报。
    * $Q_{\pi_{\theta}}(s,a)=\Bbb E_{s^{'}\backsim p(s^{'}|s, a)}[r(s, a, s^{'})+\gamma V_{\pi_{\theta}}(s^{'})]$
    * Bellman: $Q_{\pi_{\theta}}(s,a)=\Bbb E_{s^{'}\backsim p(s^{'}|s, a)}[r(s, a, s^{'}) + \gamma \sum_{a^{'}\backsim \pi_{\theta}(a'|s')}[Q(s',a')]]$
  * 优势函数 Advantage Function: $A_\theta(s, a) = Q(s, a) - V(s)$

* 方法概论:
  * 值函数估计
    * 发展路径
      * Q学习: $Q(s_t, a_t)$
      * DQN
      * Double Q
  * 策略搜索 $\pi(a_t|s_t)$
    * 无梯度方法
    * 策略梯度
      * 传统策略梯度 Policy Gradient
      * TRPO: 引入 Trust Region
      * PPO: 
* On-Policy & Off-Policy
  * On-Policy
    * 采集数据用的Policy和训练的Policy是同一个
  * Off-Policy

##### Deep Learning Basic

* Loss & Gradient Descent



* 正则化 Normalization



* entropy: 熵 & 交叉熵



* Attention

  

* Transformer



* VAE