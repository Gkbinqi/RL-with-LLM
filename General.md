#### 前言

仅关注基础理解, 路径, 经典方法的代码

尽量保证公式符号连贯

###### Resources

数理基础 -- 西湖大学 RL的数学原理: https://www.icourse163.org/course/XHUN-1470436188

基础概念+代码 -- Open-AI Spinning Up(latest: 2018): https://spinningup.openai.com/en/latest/

部分算法讲解 -- bilibili rethinkFun: https://space.bilibili.com/18235884

基础进阶 & Paper Roadmap -- RL: Intro 2025 by KM

Dive Deeper --  UCB-CS285 (see next repo)

#### RL概念基础

##### RL定义

$$
S:状态空间;~A:动作空间;~R:奖励空间;~H:Horizon~视界范围;\\
~\Delta(Space):在某个空间上的概率分布;~\theta:模型参数
$$

###### RL概述

RL问题可以描述为: Agent从与Environment的**交互**中不断学习以完成特定目标

* 交互: Agent依据其Policy $\pi_\theta$ 在观察到的State选择Action, 完成状态转移到达新的状态并获得Env的Reward反馈的过程
* Agent根据**反馈信息**学习最优策略$\pi^*$

RL的普遍目标: 学习到一个 optimal Policy $\pi^*_{\theta}(a|s)$来最大化期望回报$J(\theta)$(expected return)

###### **马尔可夫决策过程** *Markov Decision Process* (MDP)

RL一般将问题建模为MDP

MDP是**历史无关**(Memoryless)的, 状态间的转移仅取决于系统**当前的**状态, 而与系统过去或未来任意状态都独立不相关

* $p(s'|s,a)$状态转移的输出只与输入的当前的State和Action有关, 与之前任意步的State和Action无关
  * $p(s_{t+1},|s_t,a_t)=p(s_{t+1}|s_0,a_0,s_1,a_1,\ldots,s_t,a_t)$
  * 也可以理解为, 我们相信Env反馈的State已经包含了我们所做的行为的能影响到的所有信息
* 给定策略$\pi$, MDP将变成一个Markov Process

###### 空间

* 状态 State $s{\in}S$

  Agent 能观察到的 Env 的状态

  * Observation: State的子集(或者说, 残缺信息, Ob也可能是State的低维观察信息), 因为Agent未必能获得全部的State, 有时其只能获得Ob到的部分真实State信息, 并且可能有噪声(如: 超agent感知范围, 感知结构) 
    * 方便起见, 简单问题中我们假设Agent能直接获得Env的真实状态State, 即$o=s$
  * 环境中有一个或多个特殊的终止状态(terminal state; absorbing state)

* 动作 Action $a{\in}A$

  Agent能采取的行动

* 奖励 Reward $r{\in}\mathbb{R}$

  Reward是环境给Agent的奖惩反馈, **其定义会决定模型的改进方向**
  
  * 关于Reward有很多讨论研究, 很关键的部分

###### 智能体 Agent

观察 Env 的 State, 依据其 Policy 做出决策 Action 与 Env 进行交互的实体

* **策略 Policy** $\pi_{\theta}(a|s);\mu_\theta(s)$

  决定Agent在状态s下采取什么行动a

  * $Stochastic: \pi_{\theta}(a|s)$
    * $S\rightarrow\Delta(A)$
    * 随机策略$\pi_\theta(a|s)$输出a分布
  * $Deterministic:a=\mu_\theta(s)$
    * $S{\rightarrow}A$
    * 确定性策略, 输出具体动作

Agent可以有内部的部件如belief state, 也可以在内部构建世界模型对Env建模, 此处只讨论Policy

###### 环境 Environment (Env)

接收 Agent 的 Action 而改变State, 并反馈 next State & Reward 与Agent交互的实体

* **模型 Model**

  即 Env 的运行规律, RL里一般特指由环境决定的函数 $Model = \{r(s,a,s'), p(s'|s,a)\}$

  * **状态转移概率** *Transition Function*: $p(s'|s, a)$

    智能体根据当前状态$s_t$做出一个动作$a_t$之后，下一个时刻环境处于不同状态$s_{t+1}$的概率分布

    * $S\times A\rightarrow \Delta(S)$
      * $(s,a)$往往只能确定$s'$的分布, 而不是总是导向单个$s'$
      * 例如, 机器人走路遇到障碍(s), 选择跳过去(a), 可能成功($s^{'}_1$)也可能摔倒($s^{'}_2$)

  * **奖励函数** *Reward Function*: $r(s, a, s')$
    
    * $S × A × S → \mathbb R$
    * 需要注意, 返回的奖励与到达的新状态也有关
      * 如前例, 遇碍(s)跳(a), 成功($s^{'}_1$)或摔倒($s^{'}_2$), 两种新状态reward不同

###### 评估指标

* 轨迹 Trajectory $\tau$

  对MDP过程的一次采样记录

  $\tau = \{s_0, a_0, s_1, r_1(s_0, a_0, s_1), \cdots, s_{T-1}, a_{T-1}, s_T, r_T(s_{T-1}, a_{T-1}, s_T)\}$

  * ⭐在参数$\theta$下, 特定路径$\tau$出现的概率可以用Policy和Transition Function连乘表示

    $p_\theta(\tau) = p(s_0) \displaystyle  \prod^{T-1}_{t=0}{\pi_{\theta}(a_t|s_t)p(s_{t+1}|s_t, a_t)}$

* 折扣回报 Return $G_t$

  进行一次交互的轨迹$\tau$所累积的折扣全程Reward

  * 轨迹$\tau$的**折扣总回报**: $G(\tau) = \sum^{T-1}_{t=0}{\gamma^tr_{t+1}}$
  * 从$t_0$时刻开始计算: $G(\tau_{t_0})=G_{t_0}=\sum_{t=t_0}^{T-1}\gamma^{t-t_0}r_{t+1}=r_{t_0+1}+\gamma G_{t_0+1}$
  * $\gamma\in[0,1)$为折扣率 *discount rate*
    * 当$\gamma\rightarrow 0$时，Agent更在意短期回报；而当$\gamma\rightarrow 1$时，长期回报变得更重要

* 期望回报 Expected Return $J(\theta)$

  $J(\theta)=\mathbb{E}_{\tau\backsim p_{\theta}(\tau)}[G(\tau)]=\mathbb{E}_{\tau\backsim p_{\theta}(\tau)}[\sum^{T-1}_{t=0}{\gamma^tr_{t+1}}]$ 

##### 值函数

###### 状态值函数 *State-Value Function* $V_{\theta}(s)$

描述从状态s开始，执行策略$\pi_{\theta}$的期望总回报

$Def:V_{\theta}(s) = \Bbb E_{\tau\backsim p_\theta(\tau)}[G(\tau)|\tau_{s_t} = s]$

* Optimal $V^*$

  在当前模型$M$下状态的最优的值函数

  对于一个MDP

  * $\forall{s},V^*$存在且唯一(值唯一)
  * $\pi^*$存在且可能为多个
  * 一定存在至少一个$\mu^*$


###### 动作值函数 *Action-Value Function* $Q_\theta(s, a)$

描述初始状态为s并进行动作a后，执行策略$\pi_{\theta}$得到的期望总回报

$Q_{\theta}(s,a)=\mathbb{E}_{\tau\backsim p(\tau)}[G(\tau)|\tau_{s_0} = s,\tau_{a_0}=a]$

###### 优势函数 *Advantage Function* $A_\theta(s, a) = Q(s, a) - V(s)$

表示在$s$下 采取特定$a$相较于整体预期回报的优势

$A_\theta(s,a)=Q(s,a)-V(s)$

* $A_\theta(s,a)$的变化

  $\because Q(s,a)=\mathbb{E}_{s'\backsim p(s'|s,a)}[r(s,a,s')+\gamma V(s')]\\{\Rightarrow}A(s, a)=\mathbb{E}_{s'{\backsim}p(s'|s,a)}[r(s,a,s')+{\gamma}V(s')-V(s)]$


###### Bellman Evaluation

给定策略$\theta$, 对其合法的值函数能够通过递归的方式表示:

$\forall{s}\in{S},{V}_{\theta}(s) = \mathbb{E}_{a\backsim\pi_{\theta}(a|s)}\mathbb{E}_{s'\backsim p(s'|s,a)}[r(s,a,s')+\gamma V_{\theta}(s')]\\$ 

$$
\begin{align}
V_{\theta}(s) &= \mathbb{E}_{\tau\backsim p_\theta(\tau)}[G(\tau)|\tau_{s_t} = s]\\
	&=\sum_{a\backsim\pi_{\theta}(a|s)}\pi_{\theta}(a|s)\mathbb{E}_{\tau\backsim p_\theta(\tau)}[G(\tau_t)|\tau_{s_t}=s, \tau_{a_t}=a]\\
	&=\sum_{a\backsim\pi_{\theta}(a|s)}\pi_{\theta}(a|s)\sum_{s'\backsim p(s'|s,a)}p(s'|s,a)\mathbb (r(s,a,s')+\gamma \mathbb{E}[G(\tau_{t+1}|\tau_{s_{t+1}}=s')])\\
	&=\sum_{a\backsim\pi_{\theta}(a|s)}\pi_\theta(a|s)\sum_{s'\backsim p(s'|s,a)}p(s'|s,a)[r(s,a,s')+\gamma V_{\theta}(s')]\\
	&= \mathbb{E}_{a\backsim\pi_{\theta}(a|s)}\mathbb{E}_{s'\backsim p(s'|s,a)}[r(s,a,s')+\gamma V_{\theta}(s')]\\
\end{align}
$$

* 描述基于策略$\pi_\theta$的所有状态$s$的值函数之间的依赖关系
* 对合法的值函数 Bellman公式对所有State都成立
  * 若存在任一状态使等式不成立 则此时的值函数仍需优化
* 理解: 树形分散 对于所有可能路径
  * $\sum_{a\backsim\pi_{\theta}(a|s)}$: $\pi_{\theta}$在状态s下可能采取的所有a
  * $\sum_{s'\backsim p(s'|s,a)}$: 在状态s采取行动a后可能转移到的所有新状态$s'$

$\forall{(s,a)}\in{S{\times}A},Q_{\theta}(s,a)=\mathbb{E}_{s^{'}\backsim p(s^{'}|s, a)}[r(s, a, s^{'}) + \gamma \Bbb E_{a^{'}\backsim \pi_{\theta}(a'|s')}[Q(s',a')]]$
$$
\begin{align}
Q_{\theta}(s,a)
	&=\sum_{s'\backsim p(s'|s,a)}p(s'|s,a)[r(s,a,s')+\gamma\sum_{a'\backsim \pi_{\theta}(a'|s')}\pi_\theta(a'|s')Q_{\pi_{\theta}}(s',a')]\\
	&=\mathbb{E}_{s^{'}\backsim p(s^{'}|s, a)}[r(s, a, s^{'}) + \gamma \Bbb E_{a^{'}\backsim \pi_{\theta}(a'|s')}[Q(s',a')]]\\

\end{align}
$$

###### Bellman Optimization

###### 广义优势估计 $A^{GAE}_\theta$ *General Average Estimation*

对$A_\theta$的进一步优化, $A^{GAE}_\theta$表示在状态$s_t$时做动作a在整体上带来的优势

* Multi-Step Temporal Difference
* 通过调整$\lambda$平衡了采样不同步带来的Bias&Variance的平衡问题

$$
\begin{align}
  对A_\theta进行多步采样\\
  \because 在采样中, V(s_{t+1})&\approx r_{t+2}+\gamma V(s_{t+2})\\
  A^1_\theta(s_t,a)&=r_{t+1}+\gamma V_\theta(s_{t+1})-V_\theta(s_t)\\
  A^2_\theta(s_t,a)&=r_{t+1}+\gamma r_{t+2}+\gamma^2V_\theta(s_{t+2})-V_\theta(s_t)\\
  A^3_\theta(s_t,a)&=r_{t+1}+\gamma r_{t+2}+\gamma^2r_{t+3}+\gamma^3V_\theta(s_{t+3})-V_\theta(s_t)\\
  \vdots\\
  A^T_\theta(s_t,a)&=r_{t+1}+\gamma^1r_{t+2}+\cdots+\gamma^{T-1}r_{t+T}+\gamma^TV_\theta(s_{t+T})-V_\theta(s_t)\\
  定义:\delta^V_t(s_t,a)&=r_{t+1}+\gamma V_\theta(s_{t+1})-V_\theta(s_t)\\
  \delta^V_{t+1}(s_{t+1},a)&=r_{t+2}+\gamma V_\theta(s_{t+2})-V_\theta(s_{t+1})\\
  简写为\delta^V_t-&\text{表示第t步采取动作a在该步视角下带来的优势}\\
  
  
  定义:A^{GAE}_\theta
  	&=(1-\lambda)(A^1_\theta+\lambda A^2_\theta+\lambda^2A^3_\theta+\cdots)\\
  	&=(1-\lambda)(\delta^V_t+\lambda(\delta^V_t+\gamma\delta^V_{t+1})+\lambda^2(\delta^V_t+\gamma\delta^V_{t+1}+\gamma^2\delta^V_{t+2}))+\cdots)\\
  	&=(1-\lambda)(\delta^V_t(1+\lambda+\lambda^2+\cdots)+\gamma\delta^V_{t+1}(\lambda+\lambda^2+\cdots)+\cdots)\\
  	&=(1-\lambda)(\delta^V_t\frac{1}{1-\lambda}+\gamma\delta^V_{t+1}\frac{\lambda}{1-\lambda}+\cdots)&\lambda^n\rightarrow0\\
&=\sum_{b=0}^\infty(\gamma\lambda)^b\delta^V_{t+b}
  \end{align}
$$

###### $QVA$ 关系

$Q(s,a)=\mathbb{E}_{s'\backsim p(s'|s,a)}[r(s,a,s')+\gamma V(s')]$

$V(s)=\mathbb{E}_{a\backsim \pi_{\theta}(a|s)}[Q(s,a)]$

$A(s, a) = Q(s, a) - V(s) = \mathbb{E}_{s'\backsim p(s'|s,a)}[r(s,a,s')+\gamma V(s')-V(s)]$

##### 其他概念

###### Model-Based & Model-Free

区别在于对环境知识的掌握程度

* Model-Based: 显式使用Model函数
* Model-Free: 没有关于Model的先验知识, 通过交互采样$(s,a,s',r)$进行学习

###### On-Policy & Off-Policy

区别在于与环境交互采集数据的Policy和训练的Policy是否为同一个

* 在线学习(同策略学习) On-Policy
  * 采集数据用的 Policy 和训练的 Policy 是同一个
    * 使用$\theta$生成一组数据$\mathbb{D}$, 然后用$\mathbb{D}$更新$\theta$本身为$\theta'$, 然后用$\theta'$重复该过程
    * 训练$\theta'$时原来的$\mathbb{D}$会被丢弃, 需要重新用$\theta'$生成$\mathbb{D'}$
  * 问题
    * 大部分时间都在采集数据, 耗时长
    * 每次交互采集的数据只会使用一次, 训练效率低
* 离线学习(异策略学习) Off-Policy
  * 采集数据用参考策略, 目标是训练另一个 Policy
    * 使用 $\theta_{ref}$ 生成 $\mathbb{D}$, 然后用 $\mathbb{D}$ 更新目标策略 $\theta$
  * 一般数据可以多次复用

#### RL 方法概论

> 每个方法的论文都要去读

##### 通用架构 & 思想

###### DRL Doesn't Work Yet

> blog:

###### Bootstrapping

核心思想 各个状态的值函数间互相依赖 可以通过依赖关系自求解

###### 广义策略迭代*General Policy Iteration* GPI

###### 蒙特卡洛特方法 *Monte-Carlo* MC

$$
\begin{align}
	\sum_{x}{p(x)f(x)}
	&=\mathbb{E}_{x\backsim p(x)}(f(x)) \approx \frac{1}{N}\sum_{n=0}^{N-1}f(x^{(n)})\\

	e.g. \underbrace{\Bbb E_{\tau\backsim p_{\theta}(\tau)}[G(\tau)\nabla_{\theta}\log p_{\theta}(\tau)]}_{\tau\backsim p_{\theta}(\tau)空间过大~不适合直接求期望} 
	&\approx\frac{1}{N}\sum_{n=0}^{N-1}[G(\tau^n)\nabla_{\theta}\log p_{\theta}(\tau^n)]
\end{align}
$$

* 全部使用采样得到真实奖励 偏差小
* 实际中不同采样间差别极大 方差大

###### 时序差分法 *Temporal Difference* TD

基于单步采样, 逐步引入真实信息

* 方差小
* 由于依赖于估计值 偏差大

###### MM框架 *Minorization Maximization*

一种普遍使用的优化框架

* 每次迭代找到目标函数的一个下界函数
* 不断求这个下界函数的最大值, 以此优化目标函数

###### Exploration & Exploitation *Trade-off*

a trade off

###### Dive Deeper: Problems About Bias&Variance&Sampling in RL

> RL的样本 既不独立 也不同分布

* RL 与 Supervised Learning 的区别

  * 数据并非准备好的满足各种假设的数据集, 而是完全来自Agent与Env的交互
    * Hypothesis we have on the data set
      * 独立同分布
      * 真实
      * ...
  * 对于输入$s$, 没有人类可以提供的标签$a^*$, 需要agent自行探索
  * **(非常重要!)RL的样本既不独立也不同分布**
    * **Sampling带来的后果, 根据目前Policy动态变化的Dataset, 而Dataset又改变了Policy**
    * 后果: $\theta$的Graph是随着Sampling不断变化的
    * 某次错误的Sampling和updates可能会给全局带来灾难性的后果
      * 之后会深入研究

* Dynamic Graph
  * 由于sampling, $\theta$的Graph是不稳定的
  * Policy更新会影响采样结果, 而采样结果又会用于Policy更新, 此时若某一步出偏, 则可能带来灾难性的后果
  * 即, 整个策略更新的空间会被带入一个低价值空间
  * 掉悬崖下面去了

* 采样步数 & Bias&Variance问题
  * 采样步数越多
    * 真实样本更多
    * 方差越大: 
    * 偏差越小: 越多的数据直接来自采样得到的真实结果, 估计与真实间的差距越小
  * 采样步数越少
    * 方差越小: 两次估计间间隔小, 直接更新(?)
    * 偏差越大: 引入的数据少



##### Value-Based Roadmap $Q(s,a)$

> 一般适用于离散有限动作域$A$
>
> 由于可选动作有限, 决策时直接带入所有Action, 选择Q最大的Action即可

$$
Q-Learning~Series: S{\times}A\rightarrow{\mathbb{R}}\\
Temporal~Difference: 时序差分
$$

###### ValueOpt & PolicyOpt & General Policy Iteration

###### Q-Learning

* 用于离散$S$+离散$A$

* Sampling+Bellman: 
  
  $Q_{\pi_{\theta}}(s,a)=\underbrace{\Bbb E_{s^{'}\backsim p(s^{'}|s, a)}}_{where~sampling~works}[r(s, a, s^{'}) + \gamma \Bbb E_{a^{'}\backsim \pi_{\theta}(a'|s')}[Q(s',a')]]$
  
  * $\gamma \Bbb{E}_{a^{'}\backsim \pi_{\theta}(a'|s')}[Q(s',a')]$部分在算法中会默认选择最优Action
  
* 对于一步采样$(s,a,s',r)$(数据可复用, off-policy)
  * $target:r+\gamma {max}_{a'}Q_k(s',a')$
  * $update:Q_{k+1}(s,a)\leftarrow Q_k(s,a)+\alpha(target-Q_k(s,a))$

###### DQN

> paper:

使用深度网络模拟Q函数, 解决$S$空间过大/连续问题

* 用于连续$S$+离散$A$
* off-policy & Temporal Difference

通过网络直接模拟非线性函数将$S{\times}A$空间映射到$\mathbb{R}$, 解决映射表格无法用于连续情况的问题

* 引入函数 使用DNN来表达状态值函数 DNN很适合表达非线性函数
* 仍然无法解决无限$A$问题

训练: 对于一步采样$(s,a,s',r)$

* $TD: r+\gamma {max}_{a'}Q_{\theta_{target}}(s',a')$
* $loss:\frac{1}{2}[Q_{\theta_{main}}(s,a)-TD]^2$
* $GD: \theta_{main}^{'}\leftarrow \theta_{main}-\alpha\nabla_\theta[loss]|_{\theta=\theta_k}$

核心tricks
* 目标网络冻结 *freezing target networks*
  * 实际上会维护两个网络, main 和 target, targetNet用于提供计算loss时的target值
  * 会"冻结"targetNet的参数, 避免其变化过大以稳定学习目标
    * 对targetNet使用多步更新或者软更新(soft update)
    * soft update: $\theta_{target}\leftarrow\theta_{traget}+\tau\theta_{main}(for~example,\tau~can~be~0.005)$
* 经验回放 *experience replay*
  * 构建一个经验池来去除数据相关性, 同时提高数据利用率 -- off-policy
  * 实践中即为Replay-Buffer: $(s,a,s',r,terminal)$

###### More Q Learning

Double Q, Dual Q, Rainbow Q: 更多的Q网络雕花 解决过估计等问题

##### ⭐Policy-Based Roadmap $\pi_\theta(a|s)$

$$
Policy~Gradient~Series:S\rightarrow{\Delta (A)} \\
需要注意的是,相比Q-Learning,Policy~Gradient有更小的定义域,更平滑的计算空间
$$

###### [$\cal{Base}$] 策略梯度 Policy Gradient

用一个深度网络 $\theta$ 来模拟策略函数$\pi_\theta(a|s)$, 直接学习最优策略

$$
\begin{align}
    \nabla_{\theta}J(\theta)
    &=\nabla_{\theta}\mathbb{E}_{\tau\backsim p_{\theta}(\tau)}[G(\tau)]\\
    &=\nabla_{\theta}\sum_{\tau}[G(\tau)p_{\theta}(\tau)]&提取p_\theta(\tau)转移梯度算子\\
	&=\sum_{\tau}[G(\tau)\nabla_{\theta}p_{\theta}(\tau)]\\
	&=\sum_{\tau}[G(\tau)\nabla_{\theta}p_{\theta}(\tau)\frac{p_{\theta}(\tau)}{p_{\theta}(\tau)}]\\
	&=\sum_{\tau}[(p_{\theta}(\tau))*G(\tau)\nabla_{\theta}\log p_{\theta}(\tau)]&\nabla\log f(x)=\frac{\nabla f(x)}{f(x)}\\
	&=\Bbb E_{\tau\backsim p_{\theta}(\tau)}[G(\tau)\nabla_{\theta}\log p_{\theta}(\tau)]&全概率公式\\
	&\approx\frac{1}{N}\sum_{n=0}^{N-1}[G(\tau^n)\nabla_{\theta}\log p_{\theta}(\tau^n)] &\cal{Monte-Carlo}\\
	&\because p_\theta(\tau) = p(s_0)\prod^{T-1}_{t=0}{\pi_{\theta}(a_t|s_t)p(s_{t+1}|s_t, a_t)}\\
	&\log{p}_\theta(\tau)=\log{p(s_0)}+\sum^{T-1}_{t=0}\log{\pi_{\theta}(a_t|s_t)}+\sum^{T-1}_{t=0}\log{p(s_{t+1}|s_t, a_t)}\\
	&\nabla_\theta\log{p}_\theta(\tau)=\sum^{T-1}_{t=0}\nabla_\theta\log{\pi_{\theta}(a_t|s_t)}&消去\theta无关项\\
	
	MC
	&=\frac{1}{N}\sum_{n=0}^{N-1}\sum_{t=0}^{T_n-1}G(\tau^n)\nabla_{\theta}\log \pi_{\theta}(a_t^n|s_t^n)\\
	&由于一个action只能影响其之后的reward, 优化G(\tau^n)项\\
	&=\frac{1}{N}\sum_{n=0}^{N-1}\sum_{t=0}^{T_n-1}G_t^n\nabla_{\theta}\log \pi_{\theta}(a_t^n|s_t^n)&(G_t^n=G(\tau_t^n))\\
\end{align}
$$

* Optimal Policy目标: Gradient Ascent参数$\theta$获得最大的Expected Return $J(\theta)$
  
* 对$G_t^n$(势能?)可采取多种优化, 衍生出不同算法中的Loss
  
  * $G_t^n-Base(s_t^n)$
  * $A^{GAE}_\theta(s,a)$
  
* 直观理解: 以$A(s,a)$为例

  $\nabla J(\theta)=\frac{1}{N}\sum_{n=0}^{N-1}\sum_{t=0}^{T_n-1}A(s_t^n,a_t^n)\nabla_{\theta}\log \pi_{\theta}(a_t^n|s_t^n)$

  * 当$A>0$ 即在该状态下采取该动作相较于该状态下的平均水平来说是有优势的 增大该动作出现的概率
  * 为总体梯度方向增加一个该状态下该动作的概率的梯度 权重为$A$
  * A越大 该动作价值越大 相应的对该动作的概率增大越大 在其概率的梯度方向上的移动距离越大

###### REINFORCE

经典的Policy Grad, 先rollout整条轨迹, 然后进行更新

Monte-Carlo Policy Gradient

* 算法
  * 在策略$\pi_\theta$下采样N条轨迹(典型on-policy)
  * $计算梯度:\nabla_{\theta}J(\theta)=\frac{1}{N}\sum_{n=0}^{N-1}\sum_{t=0}^{T_n-1}G_t^n\nabla_{\theta}\log \pi_{\theta}(a_t^n|s_t^n)$
  * $Gradient~Ascent:\theta'\leftarrow\theta+\alpha\nabla J(\theta)$
  * 重复至收敛
* 实践中定义loss为负数, 以便使用GD工具
  * $loss=-\frac{1}{N}\sum_{n=0}^{N-1}\sum_{t=0}^{T_n-1}G_t^n\log \pi_{\theta}(a_t^n|s_t^n)$
  * $GD: \theta'\leftarrow\theta-\alpha\nabla loss$
* with baseline
* many baseline(adv)

###### 自然梯度 *Natural Policy Gradient*

$\theta$的欧几里得距离$||\theta-\theta'||^2_2$不能很好的反映作为概率分布的$\pi_\theta$的实际变动

采用KL divergance作为指标, 使用二阶指标以测度策略的真实变动

Fisher 信息矩阵 FIM

###### $\cal{[PG+ValueFunc]}$ DDPG *Deep Deterministic Policy Gradient*

> paper:

###### $\cal{[PO数理基础]}$ TRPO *Trust Region Policy Optimization*

> paper:

$$
argmax_{\theta'}\mathbb{E_{s\backsim v_\theta,a\backsim\pi_\theta(a|s)}}[\frac{\pi_{\theta'}(a|s)}{\pi_{\theta}(a|s)}A_{\pi_\theta}(s,a)]\\ 
s.t.D_{KL}(\pi_\theta(a|s)||\pi_{\theta'}(a|s))<\epsilon
$$

Why

* 传统Policy Gradient $\theta'\leftarrow\theta+\alpha\nabla{J}(\theta)$对更新步长$\alpha$十分敏感, 步长稍大就可能**破坏原有的策略性能**
  
  * Extra Topic: Problems About Sampling -- Dynamic Graph
  
* 基于CPI 混合策略的思想, 同时修正了混合策略在实践中不可用的问题

* KL散度 *KL diverge* $D_{KL}(P||Q)$
  $$
  用分布Q近似分布P: D_{KL}(P||Q) = \begin{cases}
  \displaystyle\sum_xP(x)\log\frac{P(x)}{Q(x)},&离散\\
  \displaystyle\int P(x)\log\frac{P(x)}{Q(x)}dx,&连续
  
  \end{cases}
  $$

  * 描述两个分布的差别

    $D_{KL}(P||Q){\geq}0;D_{KL}(P||Q) = 0~iff~P与Q同分布$

    * 证明: Jensen不等式

  * 正反向KL的区别

核心思想: 通过数学证明保证在使用符合限制的步长时每一步都确定能够优化模型表现, 避免步长过大导致模型偏离正确的梯度方向导致模型崩溃的同时使步长尽可能大以加速收敛

* 在保证每一步都**不会让策略变差**（即单调改进）的前提下
  * 允许我们进行**尽可能大的更新步**
  * 同时**适用于任意参数化策略（比如神经网络）**
* 数学推导相关见论文, 太多了
* 二次项bounding问题: $|\eta(\theta)-L_{\theta_0}(\theta)|{\propto}O(\alpha^2){\rightarrow}D_{KL}^{max}(\theta||\theta')$
  * 约束条件下最优化方法
* TRPO的问题

  * 计算困难(矩阵运算)
  * DNN本身不适合限制问题

###### $\cal{[PO技术基础]}$ **PPO** *Proximal Policy Optimization*

> paper:

基于TRPO思想的工程上的可行高效实现

* 特点理解: 总体上为on-policy, 通过局部的off-policy化实现数据复用解决训练效率问题
  * $\theta$生成本轮数据$\mathbb{D}$并作为参考Policy进行重要性采样, 使用$\mathbb{D}$对$\theta$进行**多轮训练**
  * 轮次内, 我们可以将最初的$\theta$视为Ref-Policy从而实现局部off-policy化

* 重要性采样
  $$
  \begin{align} 
  \mathbb{E}_{x\backsim p(x)}(f(x))
  	&=\sum_{x\backsim p(x)}{p(x)f(x)}\\
  	&=\sum_{x\backsim p(x)}{p(x)f(x)\frac{q(x)}{q(x)}}\\
  	&=\sum_{x\backsim q(x)}{q(x)[f(x)\frac{p(x)}{q(x)}]}\\
  	&=\mathbb{E_{x\backsim q(x)}}[f(x)\frac{p(x)}{q(x)}]\\
  	&\approx\frac{1}{N}\sum_{n=0}^{N-1}[f(x)\frac{p(x)}{q(x)}]_{x\backsim q(x)}\\
  \end{align}
  $$
  
* 公式推导 *从Policy Gradient Loss的替代优化到重要性采样*
  $$
  \begin{align}
  \nabla_{\theta}J(\theta)
      &=\frac{1}{N}\sum_{n=0}^{N-1}\sum_{t=0}^{T_n-1}G_t^n\nabla_{\theta}\log \pi_{\theta}(a_t^n|s_t^n)\\
  	&=\frac{1}{N}\sum_{n=0}^{N-1}\sum_{t=0}^{T_n-1}A^{GAE}_\theta(s_t^n, a_t^n)\nabla_{\theta}\log \pi_{\theta}(a_t^n|s_t^n)&引入A^{GAE}_\theta优化方差偏差\\
  	&引入重要性采样,局部off-policy化\\
  	&\theta:Training-Policy;\theta':Ref-Policy\\
  	&此时使用的轨迹数据由\theta'采样得到\\
  	&=\frac{1}{N}\sum_{n=0}^{N-1}\sum_{t=0}^{T_n-1}A^{GAE}_{\theta'}(s_t^n, a_t^n)\frac{\pi_{\theta}(a_t^n|s_t^n)}{\pi_{\theta'}(a_t^n|s_t^n)}\nabla_{\theta}\log \pi_{\theta}(a_t^n|s_t^n)\\
  	&=\frac{1}{N}\sum_{n=0}^{N-1}\sum_{t=0}^{T_n-1}A^{GAE}_{\theta'}(s_t^n, a_t^n)\frac{\nabla_{\theta}\pi_{\theta}(a_t^n|s_t^n)}{\pi_{\theta'}(a_t^n|s_t^n)}&\nabla\log f(x)=\frac{\nabla f(x)}{f(x)}\\
  \end{align}
  $$
  
* Loss处理
$$
\begin{align}
  Loss
  	&=-\frac{1}{N}\sum_{n=0}^{N-1}\sum_{t=0}^{T_n-1}A^{GAE}_{\theta'}(s_t^n, a_t^n)\frac{\pi_{\theta}(a_t^n|s_t^n)}{\pi_{\theta'}(a_t^n|s_t^n)}\\
  	&以PPO2为例,引入clip限制项\\
  	&=-\frac{1}{N}\sum_{n=0}^{N-1}\sum_{t=0}^{T_n-1}\min({A^{GAE}_{\theta'}\frac{\pi_{\theta}(a_t^n|s_t^n)}{\pi_{\theta'}(a_t^n|s_t^n)}},{clip(\frac{\pi_{\theta}(a_t^n|s_t^n)}{\pi_{\theta'}(a_t^n|s_t^n)},1-\epsilon, 1+\epsilon)}A^{GAE}_{\theta'})\\
  \end{align}
$$

* 训练实现

  * 使用模型: Policy模型(Actor) & Value模型(Critic for $A^{GAE}_\theta$)
* Loss GD
    * $Loss_{PolicyNet}=-\frac{1}{N}\sum_{n=0}^{N-1}\sum_{t=0}^{T_n-1}\min({A^{GAE}_{\theta'}\frac{\pi_{\theta}(a_t^n|s_t^n)}{\pi_{\theta'}(a_t^n|s_t^n)}},{clip(\frac{\pi_{\theta}(a_t^n|s_t^n)}{\pi_{\theta'}(a_t^n|s_t^n)},1-\epsilon, 1+\epsilon)}A^{GAE}_{\theta'})$
    * $Loss_{ValueNet}=MSE(TD\_target, V(s_t))$
  * 训练过程
    * 使用当前$\theta$生成一组轨迹$\mathbb{D}$(全局on-policy)
    * 计算此时$\theta$下得到的$(TD\_target,TD\_delta,\pi_{\theta'}(a|s)){\rightarrow}A^{GAE}_{\theta'}$作为参考策略$\theta'$的数据
    * 在$\mathbb{D}$上对$\theta$进行**多轮训练**, 该过程中使用预先计算好的数据作为参考策略$\theta'$数据进行重要性采样
      * 相当于将最初的$\theta_{init}$视为参考策略$\theta'$
      * 每一步实际计算的只有当前策略下的$\pi_{\theta}(a|s)$和$V_\theta(s)$
      * 局部off-policy化
      * 对于ValueNet, 每轮都采用TD-target进行MSE逼近
        * 每轮计算新的$V(s_t)$, 但是TD-target不变, 始终采用ref中的TD-target
    * 重复至收敛
* PPO on LLM
    * 双重限制
    * $\theta'$: 基准模型 所有的更新的前提限制
        * 不能偏离基准模型太原
        * 对$\theta$的更新, 只能优化在该方向上的概率
        * 但是 会对整体 对$\theta$在其他任务上的表现造成**不可预知无法测度的影响**
        * 即 *模型遗忘* 问题
        * $\theta''$中的参数实际上是模型掌握的知识 随着RL的进行 由于仅对$\theta$在目标任务的方向上进行优化 该过程中$\theta$的变化能够提升目标任务的表现 但无法顾及$\theta''$掌握的整体的能力 当这个偏差过大时 $\theta$ 往往会失去在其他类型的任务的能力
        * 可以理解为$\theta$ overfitting了
        * 也可以理解为$\theta$由于对参数进行了过大的改动 使得其"遗忘"了很多方面的能力

###### [RLHF] DPO *Direct Preference Optimization*

> paper:

###### $\cal{[Deepseek]}$ GRPO *Group Relative Policy Optimization*

> paper:

* 优化掉$V(s)$, 对每组采样计算均值作为baseline

###### GSPO

> paper:

###### $\cal{[Many~Tricks]}$ DAPO

> paper:

##### Model-Based Roadmap

###### World Model

> paper:

