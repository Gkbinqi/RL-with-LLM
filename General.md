### 通用基础

概念 数学 工具...

##### Math

得补课概率论和高数...

* 马尔可夫过程 Markov process
  * 该过程是**步骤独立**的, 状态间的转换仅取决于系统**当前的**状态, 而与系统过去或未来任意状态都独立不相关
  * 离散状态下被称为 *马尔可夫链*
    * 典型: 有A, B两种状态, 转换规则: 当前状态为[A]: to-A: 0.7, to-B: 0.3; [B]: to-A: 0.4, to-B: 0.6;
    * 状态转换概率仅取决于当前所处的状态

* KL散度 KL diverge
  * 描述两个分布的差别
  * 大小介于[0, 1)(1可否?)之间, 为0时代表同分布
* 蒙特卡洛特方法 Monte Carlo


$$
E(f(x)) x~p(X) = \sum_{i=1}^{n}{f(x)p(x)} \almostequal {1}/{n}
$$

##### RL基本概念

* Environment 环境
  * 接收Action, 提供Reward反馈
  * 一般来说, Environment类似黑箱, 其内部实现我们无法直接得知, 只能通过有限的接口和规则与其交互并观察其反馈
* Agent 智能体
  * 观察Environment(Observation), 获得当前的State, 并决定采取的Action
  * 一般来说, RL中训练Agent的目标为最大化预计能获得的Return(非单步reward, 而是预期整个过程能获得的总reward)
* State 状态
  * Environment当前的状态
  * Observation: State的子集, 因为Agent未必能获得全部的State, 有时其获得部分State信息(如: 超agent感知范围和感知结构) 
  * Agent可能直接获得State进行决策, 也可能只能通过Observation进行决策
* Action
  * Agent能采取的行动
* Reward & Return
  * Reward
    * 环境给Agent的奖惩反馈
    * Reward定义会很大程度上影响模型的行为进而影响表现
  * Return
    * Agent在一条轨迹中获得的总回报
* On-Policy & Off-Policy
  * On-Policy
    * 采集数据用的Policy和训练的Policy是同一个
  * Off-Policy
* Q函数 & V函数
  * Action-Value Function: $Q_\theta(s, a)$
  * State-Value Function: $V_\theta(s)$
  * Advantage Function: $A_\theta(s, a) = Q(s, a) - V(s)$
* 方法概论:
  * Q学习: $Q(s_t, a_t)$
    * 发展路径
      * DQN
      * Double Q
  * Policy学习: $\pi(a_t|s_t)$
    * 发展路径
      * 传统Policy Gradient
      * TRPO: 引入 Trust Region
      * PPO: 

##### Deep Learning 概念

* Loss & Gradient Descent



* 正则化 Normalization



* entropy: 熵 & 交叉熵



* Attention

  

* Transformer



* VAE