#### 前言

仅关注基础理解, 路径, 经典方法的代码

尽量保证公式符号连贯

###### Resources

数理基础 -- RL的数学原理: https://www.icourse163.org/course/XHUN-1470436188

代码基础 -- 动手学强化学习: https://hrl.boyuai.com/chapter/intro/

challenge -- 简单实践: Agent 拆塔 小单层

进阶 & Papers -- RL-Intro & UCB-CS285

#### RL 基础概念 & 工具

##### RL 定义

$$
S:状态空间;~A:动作空间;~R:奖励空间;~\Delta(Space):在Space上的概率分布;\\
\theta:模型参数;~\mathbb{D}:数据;~\mathbb{M}:世界模型
$$



RL问题可以描述为: Agent 从与 Environment 的**交互**历史信息中 不断学习以完成特定目标

RL的普遍目标: 学习 **Optimal Policy** $\pi^*_{\theta}$ --  实现最大化期望回报 $J(\theta)$ 

* 交互: Agent 依据其策略 $\pi_\theta$ 在观察的State选择Action, **状态转移**到新状态并获得Reward的过程

  Agent 根据**反馈信息**($s', r$) 学习最优策略 $\pi^*_\theta$

###### **马尔可夫决策过程** *Markov Decision Process* MDP

RL 一般将问题建模为MDP: Space + Policy + Markov

Markov特性: **历史无关** (Memoryless)

* 状态转移仅取决于系统**当前**状态和动作, 与系统过去或未来任意状态动作都不相关

* $p(s'|s,a)$ 状态转移的输出只与输入的当前的s和a有关, 与之前任意步的s和a无关
  * $p(s_{t+1},|s_t,a_t)=p(s_{t+1}|s_0,a_0,s_1,a_1,\ldots,s_t,a_t)$
  
    也可以理解为, 我们相信Env反馈的State已经包含了我们所做的行为的能影响到的所有信息

###### 空间

* 状态 State $s{\in}S$

  Agent 观察到的自身在 Env 中的状态

  * Observation: State的子集(或者说, 残缺信息, Ob也可能是State的低维信息)
    * Agent未必能获得全部的State, 有时其只能获得Ob到的部分State信息, 并且可能有噪声
    
      方便起见, 简单问题中我们假设Agent能直接获得Env的真实状态State, 即$o=s$
  * 环境中有一个或多个特殊的终止状态(terminal state; absorbing state)

* 动作 Action $a{\in}A$

  Agent 能采取的行动

* 奖励 Reward $r{\in}\mathbb{R}$

  Reward 是环境给 Agent 的反馈, **其定义会决定模型的优化方向**
  
  * Reward 可以视为人类与 Agent 交互的接口, 通过 Reward 指导 Agent 的行为
  
    关于 Reward 有很多讨论研究, 很关键的部分

###### 智能体 Agent

观察 State, 依据其 Policy 做出决策 Action 与 Env 进行交互的实体

* **策略 Policy** $\pi_{\theta}(a|s);\mu_\theta(s)$

  决定Agent在状态s下采取什么行动a

  * $Stochastic: \pi_{\theta}(a|s)$
    * $S\rightarrow\Delta(A)$, 随机策略$\pi_\theta(a|s)$, 输出a的分布
  * $Deterministic:a=\mu_\theta(s)$
    * $S{\rightarrow}A$, 确定性策略, 输出具体动作

###### 环境 Environment Env

接收 Agent 的 Action 而改变 State, 并反馈 $(s',r)$ 的与 Agent 交互的实体

* **模型 Model** $\mathbb{M}$

  即 Env 的运行规律, RL里一般特指由环境决定的函数 $Model = \{r(s,a,s'), p(s'|s,a)\}$

  * **状态转移概率** *Transition Function*: $p(s'|s, a)$

    智能体根据当前状态$s$选择动作$a$后，下一个时刻状态 $s'$ 的概率分布

    $S\times A\rightarrow \Delta(S)$
    * $(s,a)$往往只能确定$s'$的分布, 而不是总是导向单个$s'$

      例如, 机器人走路遇到障碍(s), 选择跳过去(a), 可能平稳落地($s^{'}_1$)也可能摔倒($s^{'}_2$)

  * **奖励函数** *Reward Function*: $r(s, a, s')$ or $p(r|s,a)$

    $S × A × S → \mathbb{R}$ or $S × A {\rightarrow}\Delta(\mathbb{R}) $

    * 需要注意, 返回的奖励与到达的新状态也有关

      如前例, 遇碍(s)跳(a), 平稳落地($s^{'}_1$)或摔倒($s^{'}_2$), 两种新状态reward不同

###### 历史建模 & 评估指标

* 轨迹 Trajectory $\tau$

  对MDP一次过程的记录

  $\tau = \{s_0, a_0, r_1, s_1, \cdots, s_{T-1}, a_{T-1}, r_T, s_T \}$

  * ⭐在参数$\theta$下, 特定路径$\tau$出现的概率可以用Policy和Transition Function连乘表示

    $p_\theta(\tau) = p(s_0) \displaystyle  \prod^{T-1}_{t=0}{\pi_{\theta}(a_t|s_t)p(s_{t+1}|s_t, a_t)}$
    
  * Episode

    一条结束于 terminal state 的有限长的轨迹

    对于在 Episode 上工作的任务, 即任务会在合理步数内结束, 我们将其称为 Episodic Task 

* 折扣回报 Return $G_t$

  一条 $\tau$ 中全过程折扣后 Reward 的总和

  Return 可用于评估策略的效果

  * $\tau$ 的**折扣总回报**: $G(\tau) = \sum^{T-1}_{t=0}{\gamma^tr_{t+1}}$

    从任意时刻 $t_0$ 开始计算: $G(\tau_{t_0})=G_{t_0}=\sum_{t=t_0}^{T-1}\gamma^{t-t_0}r_{t+1}=r_{t_0+1}+\gamma G_{t_0+1}$
  * $\gamma\in[0,1)$为折扣率 *discount rate*
    
    * $\gamma$ 使得我们能够建模无限长轨迹的回报, 并能控制 Agent 的视野范围
    
      当$\gamma\rightarrow 0$时，Agent 更在意短期回报；当$\gamma\rightarrow 1$时，长期回报变得更重要

* 期望回报 Expected Return $J(\theta)$

  基于回报的定义, 通过计算给定策略下**回报的期望值**精确建模策略的效果

  $J(\theta)=\mathbb{E}_{\tau\backsim p_{\theta}(\tau)}[G(\tau)]=\mathbb{E}_{\tau\backsim p_{\theta}(\tau)}[\sum^{T-1}_{t=0}{\gamma^tr_{t+1}}]$ 

##### 值函数定义

值函数是 基于给定策略 得到的对状态的价值的评估 (执行策略 $\pi_\theta$ 的期望总回报)

值函数可以作为评估策略效果的一种方法/依据, 策略的值函数越大, 该策略越优秀

###### 状态值函数 *State-Value Function* $V_{\theta}(s)$

描述从状态s开始, 执行策略 $\pi_{\theta}$ 得到的的**期望回报**

$Def:V_{\theta}(s) = \mathbb{E}_{\tau\backsim p_\theta(\tau)}[G_t|s_t = s]$

* 函数性质: 

  状态值函数是接收 s 作为输入的函数

  状态值函数是基于 Policy 的函数

  $V_\theta(s) \iff V(s; \pi_\theta)$

###### 动作值函数 *Action-Value Function* $Q_\theta(s, a)$

描述从状态s开始并选择动作a后, 执行策略 $\pi_{\theta}$ 得到的期望回报

$Def: Q_{\theta}(s,a)=\mathbb{E}_{\tau\backsim p(\tau)}[G_t|s_t = s,a_t=a]$

* 函数性质

  动作值函数是接收 $(s,a)$ 状态动作对 作为输入的函数

  动作值函数是基于 Policy 的函数

  $Q_\theta(s,a) \iff Q(s,a; \pi_\theta)$

###### 优势函数 *Advantage Function* $A_\theta(s, a)$

描述在$s$下选择特定动作$a$相较于整体预期回报能带来的额外回报(优势)

$Def: A_\theta(s,a)=Q(s,a)-V(s)$

##### 贝尔曼工具 Bellman

定义了值函数后, 我们需要求解策略的值函数的方法

Bellman 公式 提供了求解值函数的工具

###### 贝尔曼公式 Bellman Equation

给定策略 $\theta$ 和模型, $\theta$ 合法的值函数能够通过递归的方式表示

$状态值函数: \forall{s}\in{S},{V}_{\theta}(s) = \mathbb{E}_{a\backsim\pi_{\theta}(a|s)}\mathbb{E}_{s'\backsim p(s'|s,a)}[r(s,a,s')+\gamma V_{\theta}(s')]$ 

$$
\begin{align}
V_{\theta}(s) 
	&=\mathbb{E}_{\tau{\backsim}p_\theta(\tau)}[G_t|s_t = s]\\
	&=\sum_{a\backsim\pi_{\theta}(a|s)}\pi_{\theta}(a|s)\mathbb{E}_{\tau\backsim p_\theta(\tau)}[G_t|s_t=s, a_t=a]
		&确定a\\
	&=\sum_{a\backsim\pi_{\theta}(a|s)}\pi_{\theta}(a|s)\sum_{s'\backsim p(s'|s,a)}p(s'|s,a)\mathbb (r(s,a,s')+\gamma \mathbb{E}[G_{t+1}|s_{t+1}=s'])
		&确定s'\\
	&=\sum_{a\backsim\pi_{\theta}(a|s)}\underbrace{\pi_\theta(a|s)}_{based ~on~Policy}\sum_{s'\backsim p(s'|s,a)}\underbrace{p(s'|s,a)}_{based~on~\mathbb{M}}[\underbrace{r(s,a,s')}_{based~on~\mathbb{M}}+\gamma V_{\theta}(s')]\\
	&= \mathbb{E}_{a\backsim\pi_{\theta}(a|s)}\mathbb{E}_{s'{\backsim}p(s'|s,a)}[\underbrace{r(s,a,s')}_{immediate~reward}+\underbrace{{\gamma}V_{\theta}(s')}_{future ~return}]\\
\end{align}
$$

* 描述基于策略 $\pi_\theta$ 的所有状态 $s$ 的值函数之间的依赖关系
  
  * 有n个状态就会有n个式子, 联立求解即可得到值函数
  
  * 对于真实的值函数, 贝尔曼公式 ${\forall}s{\in}S$ 成立
    
    若存在任一状态时等式不成立, 此时该值函数非真实值函数, 只能部分建模策略的期望回报, 仍需优化

$动作值函数: \forall{(s,a)}\in{S{\times}A},Q_{\theta}(s,a)=\mathbb{E}_{s^{'}\backsim p(s^{'}|s, a)}[r(s, a, s^{'}) + \gamma \Bbb E_{a^{'}\backsim \pi_{\theta}(a'|s')}[Q(s',a')]]$
$$
\begin{align}
Q_{\theta}(s,a)
	&=\sum_{s'\backsim p(s'|s,a)}p(s'|s,a)[r(s,a,s')+\gamma\sum_{a'\backsim \pi_{\theta}(a'|s')}\pi_\theta(a'|s')[Q_{\pi_{\theta}}(s',a')]]\\
	&=\mathbb{E}_{s^{'}\backsim p(s^{'}|s, a)}[r(s, a, s') + \gamma \Bbb E_{a^{'}\backsim \pi_{\theta}(a'|s')}[Q(s',a')]]\\
\end{align}
$$

###### 策略评估 Policy Evaluation

即通过 Bellman 公式 求解一个策略的值函数

由于值函数是评价策略效果的指标, 求解一个策略的值函数的过程也被称为 Policy Evaluation

以状态值函数为例: 设 $n = |S|$ , 给定 $\pi_\theta$, 有n个值函数, 可以通过矩阵形式表示贝尔曼公式
$$
Bellman:v=r+{\gamma}Pv\\
v=[v(s_1), v(s_2), {\cdots},v(s_n)]^T\\
r=[r(s_1), r(s_2), {\cdots},r(s_n)]^T\\
r_i=\mathbb{E}_{a\backsim\pi_{\theta}(a|s_i)}\mathbb{E}_{s'{\backsim}p(s'|s_i,a)}[r(s_i,a,s')]\\
P_{ij} = p(s_j|s_i) = \sum_{a{\backsim}\pi_\theta(a|s_i)}\pi_\theta(a|s_i)p(s_j|s_i,a)\\
$$
基于该表达式, 有两种求解值函数的方法:

* 直接求解 -- 闭式解, 矩阵求逆

  $v=r+{\gamma}Pv~{\Rightarrow}~v=(I-{\gamma}P)^{-1}r$

  直接通过线代变换求解, 得到的v为真实值函数 $v_\pi$

  不过需要做 $O(n^3)$ 的矩阵求逆操作, 以及优化 Policy 并不需要求解特别精确的值函数

  特别的, $(I-{\gamma}P)^{-1}$ 矩阵是 折扣状态访问频率矩阵

  优美, 但实践中一般不会使用

* 迭代求解

  $init:v_0,v_0可为任意初始向量,也可为\vec{0}$

  $iter: v_{k+1}=r+{\gamma}Pv_k,k{\rightarrow}{\infty}$

  可以证明, 迭代无数次后 $v_k$ 与真实值函数 $v_\pi$ 的误差趋零, 精确度足以用于优化策略

###### 贝尔曼最优公式 BOE *Bellman Optimization Equation*

基于求解策略的值函数作为评估依据, 我们可以优化策略以逼近目标 -- $\pi^*_\theta$

$V(s) = max_\pi\mathbb{E}_{a\backsim\pi_{\theta}(a|s)}\mathbb{E}_{s'\backsim p(s'|s,a)}[r(s,a,s')+{\gamma}V(s')]$

Optimal $V^*$

在当前模型$M$下状态的最优的值函数

对于一个MDP

* $\forall{s},V^*$存在且唯一(值唯一)
* $\pi^*$存在且可能为多个
* 一定存在至少一个$\mu^*$

###### Note: 基于 Bellman 的 $QVA$ 互推

$Q(s,a)=\mathbb{E}_{s'\backsim p(s'|s,a)}[r(s,a,s')+\gamma V(s')]$

$V(s)=\mathbb{E}_{a\backsim \pi_{\theta}(a|s)}[Q(s,a)]$

$A(s, a) = Q(s, a) - V(s) = \mathbb{E}_{s'\backsim p(s'|s,a)}[r(s,a,s')+\gamma V(s')]-V(s) = \mathbb{E}_{s'\backsim p(s'|s,a)}[r(s,a,s')+\gamma V(s')-V(s)]$

##### 其他概念

###### Model-Based & Model-Free

区别在于对环境知识的掌握程度

* Model-Based: 显式使用Model函数
* Model-Free: 没有关于Model的先验知识, 通过交互采样$\mathbb{D}$进行学习

没模型就得有数据, 没数据就得有模型, 什么都没有就学不了

###### On-Policy & Off-Policy

区别在于与环境交互采集数据的Behavior Policy和训练目标Target Policy是否为同一个策略

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

##### [Base] 蒙特卡洛特方法 *Monte-Carlo* MC

$$
\begin{align}
	原理:&\mathbb{E}(x)\approx\frac{1}{N}\sum^{N}_{i=1}(x_i),N\rightarrow\infty\\
	e.g.&\underbrace{\Bbb E_{\tau\backsim p_{\theta}(\tau)}[G(\tau)\nabla_{\theta}\log p_{\theta}(\tau)]}_{\tau\backsim p_{\theta}(\tau)空间过大~不适合直接求期望}\approx\frac{1}{N}\sum_{n=1}^N[G(\tau^n)\nabla_{\theta}\log p_{\theta}(\tau^n)]
\end{align}
$$

* 全部使用采样得到真实奖励 偏差小
* 实际中不同采样间差别极大 方差大

##### [Base] 时序差分法 *Temporal Difference* TD

基于单步采样, 逐步引入真实信息

* 方差小
* 由于依赖于估计值 偏差大

##### [1] Naive Model-Based

###### Value Evaluation

###### Policy Iteration

###### Truncated Policy Iteration

核心两个操作

* Value Iteration
  * 基于已有的策略迭代求解其值函数(不一定要求出最后的合法值函数, 差不多即可, 保证每次迭代都会优化)
* Policy Iteration
  * 基于已有的值函数(不一定合法)对策略进行迭代优化(BOE max操作)

##### [2] Value-Based Roadmap $Q(s,a)$

> 适用于离散有限动作域$A$
>
> 由于可选动作有限, 决策时直接带入所有Action, 选择Q最大的Action即可

###### Tabular Q-Learning

* 用于离散$S$+离散$A$

* Sampling+Bellman: 
  
  $Q_{\pi_{\theta}}(s,a)=\underbrace{\Bbb E_{s^{'}\backsim p(s^{'}|s, a)}}_{where~sampling~works}[r(s, a, s^{'}) + \gamma \Bbb E_{a^{'}\backsim \pi_{\theta}(a'|s')}[Q(s',a')]]$
  
  * $\gamma \Bbb{E}_{a^{'}\backsim \pi_{\theta}(a'|s')}[Q(s',a')]$部分在算法中会默认选择最优Action
  
* 对于一步采样$(s,a,s',r)$(数据可复用, off-policy)
  * $target:r+\gamma {max}_{a'}Q_k(s',a')$
  * $update:Q_{k+1}(s,a)\leftarrow Q_k(s,a)+\alpha(target-Q_k(s,a))$

###### Deep Q-Learning DQN

> paper: https://arxiv.org/abs/1312.5602
>
> code: todo

使用深度网络模拟Q函数, 解决$S$空间过大/连续问题

* 用于连续$S$+离散$A$
* off-policy & Temporal Difference

通过网络直接模拟非线性函数将$S{\times}A$空间映射到$\mathbb{R}$, 解决映射表格无法用于连续情况的问题

* 引入函数 使用DNN来表达状态值函数 DNN很适合表达非线性函数
* 仍无法解决无限$A$问题

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

###### Double Q

> paper:
>
> code:



###### Dual Q

>paper:
>
>code:



###### Rainbow Q

> paper:
>
> code:



##### [2] Actor-Critic Roadmap

结合值函数网络和策略网络, 解决了$A$无限的问题

###### DDPG *Deep Deterministic Policy Gradient*

> paper: https://arxiv.org/abs/1509.02971
>
> code:todo

###### A3C *AAA C*

> paper: https://arxiv.org/abs/1602.01783
>
> code: todo

###### SAC *Soft Actor-Critic*

> paper: https://arxiv.org/abs/1801.01290
>
> code: todo

###### TD3

> paper: https://arxiv.org/abs/1802.09477
>
> code: todo

##### [3] ⭐ Policy-Based Roadmap $\pi_\theta(a|s)$

> 需要注意的是, 相较于 Value-Based, Policy-Based 有着更小的定义域, 更平滑的计算空间

###### [$\cal{Base}$] 策略梯度推导 Policy Gradient

不同于 Value-Based 方法利用值函数 $V,Q,A$ 作为评估改进策略的依据

Policy-Based 方法基于 $J(\theta)$, 直接使用在当前策略下的预期折扣回报作为评估改进策略的依据

$Def: J(\theta)=\mathbb{E}_{\tau{\backsim}p_\theta(\tau)}[G(\tau)]$
$$
\begin{align}
    \nabla_{\theta}J(\theta)
    &=\nabla_{\theta}\mathbb{E}_{\tau\backsim p_{\theta}(\tau)}[G(\tau)]\\
    &=\nabla_{\theta}\sum_{\tau}[G(\tau)p_{\theta}(\tau)]
    	&提取p_\theta(\tau)转移梯度算子\\
	&=\sum_{\tau}[G(\tau)\nabla_{\theta}p_{\theta}(\tau)]\\
	&=\sum_{\tau}[G(\tau)\nabla_{\theta}p_{\theta}(\tau)\frac{p_{\theta}(\tau)}{p_{\theta}(\tau)}]
		&trick\\
	&=\sum_{\tau}[(p_{\theta}(\tau))*G(\tau)\nabla_{\theta}\log p_{\theta}(\tau)]
		&\nabla\log f(x)=\frac{\nabla f(x)}{f(x)}\\
	&=\Bbb E_{\tau\backsim p_{\theta}(\tau)}[G(\tau)\nabla_{\theta}\log p_{\theta}(\tau)]
		&全概率公式\\
	&\approx\frac{1}{N}\sum_{n=0}^{N-1}[G(\tau^n)\nabla_{\theta}\log p_{\theta}(\tau^n)] 
		&\cal{Monte-Carlo}\\
	&\because p_\theta(\tau) = p(s_0)\prod^{T-1}_{t=0}{\pi_{\theta}(a_t|s_t)p(s_{t+1}|s_t, a_t)}\\
	&\log{p}_\theta(\tau)=\log{p(s_0)}+\sum^{T-1}_{t=0}\log{\pi_{\theta}(a_t|s_t)}+\sum^{T-1}_{t=0}\log{p(s_{t+1}|s_t, a_t)}\\
	&\nabla_\theta\log{p}_\theta(\tau)=\sum^{T-1}_{t=0}\nabla_\theta\log{\pi_{\theta}(a_t|s_t)}
		&消去\theta无关项\\
	{\therefore}上式
	&=\frac{1}{N}\sum_{n=0}^{N-1}\sum_{t=0}^{T_n-1}G(\tau^n)\nabla_{\theta}\log \pi_{\theta}(a_t^n|s_t^n)\\
	&由于一个action只能影响其之后的reward, 优化G(\tau^n)项\\
	&=\frac{1}{N}\sum_{n=0}^{N-1}\sum_{t=0}^{T_n-1}G_t^n\nabla_{\theta}\log \pi_{\theta}(a_t^n|s_t^n)
		&(G_t^n=G(\tau_t^n))\\
\end{align}
$$

* 由梯度得到等效 Loss 函数

  $Loss = \frac{1}{N}\sum_{n=0}^{N-1}\sum_{t=0}^{T_n-1}G_t^n{\log}\pi_{\theta}(a_t^n|s_t^n)$

* 直观理解: 以$A(s_t^n,a_t^n)$为例

  $\nabla{J}(\theta)=\frac{1}{N}\sum_{n=0}^{N-1}\sum_{t=0}^{T_n-1}A(s_t^n,a_t^n)\nabla_{\theta}{\log}\pi_{\theta}(a_t^n|s_t^n)$

  * 当$A>0$, 即在该状态下采取该动作相较原策略能带来更大的预期回报 
  
    故, 增大该动作出现的概率, 为总体梯度方向增加 $\nabla_{\theta}\log{\pi}_{\theta}(a_t^n|s_t^n)$ 梯度, 大小权重为$A(s_t^n,a_t^n)$
  
    A越大, 代表该动作价值越大, 相应的在其概率的梯度方向上的移动权重越大
  
    同理, $A<0$, 即该$(s_t, a_t)$会带来更少的预期回报, 故向 $\nabla_{\theta}\log{\pi}_{\theta}(a_t^n|s_t^n)$ 相反方向更新

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

###### Natural Policy Gradient

> paper:

$\theta$的欧几里得距离$||\theta-\theta'||^2_2$不能很好的反映作为概率分布的$\pi_\theta$的实际变动

采用KL divergance作为指标, 使用二阶指标以测度策略的真实变动

Fisher 信息矩阵 FIM



###### $\cal{PolicyImprovement}$ TRPO *Trust Region Policy Optimization*

> paper:
>
> no code for this method

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

###### **PPO** *Proximal Policy Optimization*

> paper: https://arxiv.org/abs/1707.06347
>
> code: done

基于TRPO思想的工程上的可行高效实现

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

* Loss GD
  * $Loss_{PolicyNet}=-\frac{1}{N}\sum_{n=0}^{N-1}\sum_{t=0}^{T_n-1}\min({A^{GAE}_{\theta'}\frac{\pi_{\theta}(a_t^n|s_t^n)}{\pi_{\theta'}(a_t^n|s_t^n)}},{clip(\frac{\pi_{\theta}(a_t^n|s_t^n)}{\pi_{\theta'}(a_t^n|s_t^n)},1-\epsilon, 1+\epsilon)}A^{GAE}_{\theta'})$
  * $Loss_{ValueNet}=MSE(TD\_target, V(s_t))$

###### GRPO *Group Relative Policy Optimization*

> paper:

优化掉$V(s)$, 对每组采样计算均值作为baseline

##### [4] Modern Model-Based Roadmap

###### World Model

> paper:

