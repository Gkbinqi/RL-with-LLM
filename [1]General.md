### 通用基础-概述

>  数学 概念 工具...

体感上, RL领域的各种定义并没有一个标准化的规范, 大家只是"差不多这个意思"的进行讨论

我尽量保证公式符号表示定义的一致连贯

对经典算法会进行代码demo实现

* PPO
* PPO-on-LLM
* GRPO(todo)
* 调研报告

###### From How to Why

2025.10.22
how很好理解, 但是重要的是知道why
如TRPO的数学推导

在初期的探索后, 进一步学习, 了解数学原理和Why就需要开始研读论文了

怎么找Idea???GRPO视频讲解那一期很有意思

##### Math

手头的事情结束后会系统复习一次数学

现阶段, 问题遇到一个解决一个

###### 高数

* 泰勒展开 *Taylor's Formula*

$$
当f(x)...
$$

* 拉格朗日定理

* 隐函数

  无法通过$y=f(x)$显式的用x表示y, x与y的关系是通过一个形式为$F(x,y)=0$的方程"隐式定义"的
  $$
  e.g.~x^2+xy+y^3-7=0\\
  sin(xy^3)+x^2-1=0\\
  诸如此类
  $$

###### 概率相关

* 全概率公式

  $\Bbb E_{x\backsim p(x)}(f(x)) \iff {\displaystyle\sum_x}p(x)f(x)~~~ \mathbb{E}与\sum的互换$

* **蒙特卡洛特方法 *Monte Carlo Method* **

  对于不知道确切分布的求期望场景, 我们可以通过采样*(Sampling)*逼近期望值

  * 理论基础: 大数定理


$$
\begin{align}
\sum_{x}{p(x)f(x)}&=E_{x\backsim p(x)}(f(x)) \approx \frac{1}{N}\sum_{n=0}^{N-1}f(x^{(n)})\\
e.g. \underbrace{\Bbb E_{\tau\backsim p_{\theta}(\tau)}[G(\tau)\nabla_{\theta}\log p_{\theta}(\tau)]}_{\tau\backsim p_{\theta}(\tau)空间过大~基本无法求期望} &\approx\frac{1}{N}\sum_{n=1}^{N}[G(\tau^n)\nabla_{\theta}\log p_{\theta}(\tau^n)]
\end{align}
$$

* 重要性采样

$$
\begin{align} 
E_{x\backsim p(x)}(f(x))
	&=\sum_{x\backsim p(x)}{p(x)f(x)}\\
	&=\sum_{x\backsim p(x)}{p(x)f(x)\frac{q(x)}{q(x)}}\\
	&=\sum_{x\backsim p(x)}{q(x)f(x)\frac{p(x)}{q(x)}}\\
	&=\sum_{x\backsim q(x)}{q(x)[f(x)\frac{p(x)}{q(x)}]}\\
	&=\mathbb{E_{x\backsim q(x)}}[f(x)\frac{p(x)}{q(x)}]\\
	&\approx\frac{1}{N}\sum_{n=0}^{N-1}[f(x)\frac{p(x)}{q(x)}]_{x\backsim q(x)}\\
\end{align}
$$

* KL散度 *KL diverge* $D_{KL}(P||Q)$

  $$
  用分布Q近似分布P: D_{KL}(P||Q) = \begin{cases}
  \displaystyle\sum_xP(x)\log\frac{P(x)}{Q(x)},&离散\\
  \displaystyle\int P(x)\log\frac{P(x)}{Q(x)}dx,&连续
  
  \end{cases}
  $$

  * 描述两个分布的差别

  * 恒大于等于0, 当且仅当P,Q同分布时为0

    * 证明: Jensen不等式

  * 正反向KL的区别

* $\epsilon-探索$

$$
\pi_{\theta}^{\epsilon}(s) = 
\begin{cases}
	\pi(s), & \text{按概率1-$\epsilon$} \\
	randomly~selected~action~in~A, & \text{按概率$\epsilon$}\\
\end{cases}\\

一般\epsilon会随着模型学习的积累逐渐减小
$$

* 输赢概率--能力假设
* 最大似然估计

###### 离散相关

* 笛卡尔集 *Cartesian Product*

  离散数学内容, $A\times B$即A, B集合中所有元素 all to all 进行有序配对组成新集合

  * $|A\times B| = |A|\times|B|$
* **马尔可夫决策过程** *Markov Decision process* (MDP)

  MDP是研究RL最常用数学模型

  * MDP是**步骤独立**的, 状态间的转换仅取决于系统**当前的**状态, 而与系统过去或未来任意状态都独立不相关
    * $p(s'|s,a)$状态转移的输入只需要当前的State和Action, 与之前任意步的State和Action无关
    * 也可以理解为, 我们相信Env反馈的State域已经包含了我们所做的行为可能影响到的所有方面
  * 离散状态下被称为 *马尔可夫链*
* 熵 & 交叉熵: *Entropy*

##### RL 定义

$$
S:状态空间;~A:动作空间;~R:奖励空间;~H:Horizon~视界范围;\\
~\Delta(Space):在某个空间上的概率分布;~\theta:模型参数
$$

###### 概念定义

* 概述

  RL问题可以描述为: Agent从与Environment的**交互**中不断学习以完成特定目标

  * 交互: Agent在不同的State依据其Policy-$\pi_\theta$选择动作, 完成状态转移并获得Env的Reward反馈的过程
  * Agent根据已知信息更新Policy学习最优策略

  RL的普遍目标: 学习到一个策略$\pi_{\theta}(a|s)$来最大化期望回报(expected return)

  * 与经典Supervised Learning的区别
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

* 环境 Environment (Env)
  * 抽象的与Agent交互的实体. Env 接收 Agent 的 Action 而改变State, 并反馈 Reward
  * 一般来说, Env 是黑箱, 其内部实现我们无法直接得知, 只能通过有限的接口和规则与其交互并观察其 State 转移和 Reward 反馈
  
* 智能体 Agent

  Ob Env 的 State, 依据其 Policy 做出决策与 Env 进行交互的实体

* 模型 Model

    即 Env 的运行规律, alias "环境动力学模型"
    * RL里一般特指由环境决定的函数$Model = \{r(s,a,s'), p(s'|s,a)\}$
      * **状态转移概率** *Transition Function*: $p(s'|s, a)$
        * $S\times A\rightarrow \Delta(S)$
          * $(s,a)$往往只能确定$s'$的分布, 而不是总是导向单个$s'$
          * 例如, 机器人走路遇到障碍(s), 选择跳过去(a), 可能成功($s^{'}_1$)也可能摔倒($s^{'}_2$)
        * 智能体根据当前状态$s_t$做出一个动作$a_t$之后，下一个时刻环境处于不同状态$s_{t+1}$的概率分布
      * **奖励函数** *Reward Function*: $r(s, a, s')$
        * $S × A × S → \mathbb R$
        * 需要注意, 返回的奖励与到达的新状态也有关
          * 如前例, 遇到障碍(s)选择跳(a), 可能成功($s^{'}_1$)也可能摔倒($s^{'}_2$), 两种新状态reward不同
    * Model一般难以得知, 因此大多数时候我们无法直接使用Model得到解析解, 而只能通过交互采样
      * 有的方法会用一个网络模拟Model, 隐式得到这些概率

* **策略 Policy** $Stochastic: \pi_{\theta}(a|s);Deterministic:a=\mu_\theta(s)$
  
    * $S\rightarrow\Delta(A)$
    * 决定Agent在状态s下采取什么行动a
    * 同样的, $\pi_\theta(a|s)$输出一个**a的分布**
    * 也有**确定性策略**, 表示为$a= \mu_\theta(s):S{\rightarrow}A$
    
* 状态 State

  Agent 能观察到的 Env 的状态

  * Observation: State的子集(或者说, 残缺信息, Ob也可能是State的低维观察信息), 因为Agent未必能获得全部的State, 有时其只能获得Ob到的部分真实State信息, 并且可能有噪声(如: 超agent感知范围, 感知结构) 
  * 环境中有一个或多个特殊的终止状态(terminal state)

* 动作 Action

    Agent能采取的行动

* 奖励 Reward

    Reward是环境给Agent的奖惩反馈, 其定义会很大程度上影响模型的行为进而影响表现

* 轨迹 Trajectory $\tau$

     $\tau = \{s_0, a_0, s_1, r_1(s_0, a_0, s_1), \cdots, s_{T-1}, a_{T-1}, s_T, r_T(s_{T-1}, a_{T-1}, s_T)\}$

    * 一个MDP决策过程

    * 有的定义为$\tau = \{\cdots,s_t,a_t,r_t,s_{t+1},\cdots\}$

    * ⭐在参数$\theta$下, 特定路径$\tau$出现的概率可以用Policy和Transition Function连乘表示

      $p_\theta(\tau) = p(s_0) \displaystyle  \prod^{T-1}_{t=0}{\pi_{\theta}(a_t|s_t)p(s_{t+1}|s_t, a_t)}$

      这个式子还有更大的作用, $p_\theta(\tau)$形式的抽象将复杂的$\nabla_\theta\rho(s)$求导隐式包含了

* 回报 Return

    Agent 和 Env 进行一次交互过程的轨迹$\tau$所累积的全部Reward

    * 轨迹$\tau$的总回报: $G(\tau) = \sum^{T-1}_{t=0}{\gamma^tr_{t+1}}$
      * 从$t_0$时刻开始计算的总回报: $G(\tau_{t_0})=\sum_{t=t_0}^{T-1}\gamma^{t-t_0}r_{t+1}=r_{t_0+1}+\gamma G(\tau_{t_0+1})$
      * $\gamma\in[0,1)$为折扣率 *discount rate*
      * 当$\gamma\rightarrow 0$时，Agent更在意短期回报；而当$\gamma\rightarrow 1$时，长期回报变得更重要

* 期望回报 *Expected Return*

    $J(\theta)=\Bbb E_{\tau\backsim p_{\theta}(\tau)}[G(\tau)]=\Bbb E_{\tau\backsim p_{\theta}(\tau)}[\sum^{T-1}_{t=0}{\gamma^tr_{t+1}}]$ 

###### ⭐值函数

* 状态值函数 *State-Value Function* $V_{\pi_{\theta}}(s)$

  描述从状态s开始，执行策略$\pi_{\theta}$的期望总回报

  * $S\rightarrow\mathbb{R}$

      $V_{\pi_{\theta}}(s) = \Bbb E_{\tau\backsim p_\theta(\tau)}[G(\tau)|\tau_{s_t} = s]$

  * Bellman: $V_{\pi_{\theta}}(s) = \mathbb{E}_{a\backsim\pi_{\theta}(a|s)}\mathbb{E}_{s'\backsim p(s'|s,a)}[r(s,a,s')+\gamma V_{\pi_{\theta}}(s')]\\$ 

    $$
    \begin{align}
    V_{\pi_{\theta}}(s) &= \Bbb E_{\tau\backsim p_\theta(\tau)}[G(\tau)|\tau_{s_t} = s]\\
    	&=\sum_{a\backsim\pi_{\theta}(a|s)}\pi_{\theta}(a|s)\mathbb E_{\tau\backsim p_\theta(\tau)}[G(\tau_t)|\tau_{s_t}=s, \tau_{a_t}=a]\\
    	&=\sum_{a\backsim\pi_{\theta}(a|s)}\pi_{\theta}(a|s)\sum_{a'\backsim p(a'|s,a)}p(a'|s,a)\mathbb (r(s,a,s')+\gamma \mathbb E[G(\tau_{t+1}|\tau_{s_{t+1}}=s')])\\
    	&=\sum_{a\backsim\pi_{\theta}(a|s)}\pi_\theta(a|s)\sum_{s'\backsim p(s'|s,a)}p(s'|s,a)[r(s,a,s')+\gamma V_{\pi_{\theta}}(s')]\\
    	&= \mathbb{E}_{a\backsim\pi_{\theta}(a|s)}\mathbb{E}_{s'\backsim p(s'|s,a)}[r(s,a,s')+\gamma V_{\pi_{\theta}}(s')]\\
    \end{align}
    $$

    * 理解: 树形分散 对于所有可能路径

      * $\sum_{a\backsim\pi_{\theta}(a|s)}$: $\pi_{\theta}$在状态s下可能采取的所有a
      * $\sum_{s'\backsim p(s'|s,a)}$: 在状态s采取行动a后可能转移到的所有新状态$s'$

* 动作值函数 *Action-Value Function* $Q_\theta(s, a)$

  描述初始状态为s并进行动作a后，执行策略$\pi_{\theta}$得到的期望总回报

  * $S\times A\rightarrow\mathbb R$

      $Q_{\pi_{\theta}}(s,a)=\Bbb E_{\tau\backsim p(\tau)}[G(\tau)|\tau_{s_0} = s,\tau_{a_0}=a]$

  * Bellman: 
    $$
    \begin{align}
    Q_{\pi_{\theta}}(s,a)
    &=\Bbb E_{s^{'}\backsim p(s^{'}|s, a)}[r(s, a, s^{'}) + \gamma \Bbb E_{a^{'}\backsim \pi_{\theta}(a'|s')}[Q(s',a')]]\\
    
    &=\sum_{s'\backsim p(s'|s,a)}p(s'|s,a)[r(s,a,s')+\gamma\sum_{a'\backsim \pi_{\theta}(a'|s')}\pi_\theta(a'|s')Q_{\pi_{\theta}}(s',a')]
    
    \end{align}
    $$
    ​    

* Q&V 互推

    * $Q(s,a)=\Bbb E_{s'\backsim p(s'|s,a)}[r(s,a,s')+\gamma V(s')]$
    * $V(s)=\mathbb E_{a\backsim \pi_{\theta}(a|s)}[Q(s,a)]=\sum_{a\backsim \pi_{\theta}(a|s)}\pi_{\theta}(a|s)Q(s,a)$
    * 公式很美
    
* 优势函数 *Advantage Function* $A_\theta(s, a) = Q(s, a) - V(s)$

    表示在$s$下采取特定$a$相较于整体预期回报的优势

    * 消去$Q(s,a)$的采样表示

    $$
    \because Q(s,a) = \mathbb{E}_{s'\backsim p(s'|s,a)}[r(s,a,s')+\gamma V(s')]\\
        A_\theta(s, a) = Q(s, a) - V(s) = \mathbb{E}_{s'\backsim p(s'|s,a)}[r(s,a,s')+\gamma V(s')-V(s)]\\
    $$

      * 对$A_\theta$的进一步优化: 广义优势估计 $A^{GAE}_\theta$ *General Average Estimation*

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

      * 表示在状态$s_t$时做动作a在整体上带来的优势
        * Multi-Step Temporal Difference
        * 通过调整$\lambda$平衡了采样不同步带来的方差&偏差的平衡问题

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

###### Model-Based & Model-Free

区别在于对环境知识的掌握程度

* Model-Based: 显式使用Model中的函数
* Model-Free: 没有关于Model的先验知识, 通过交互采样$(s,a,s',r)$进行学习

##### RL 方法概论

###### 通用架构

* 蒙特卡洛特方法(Sampling) *Monte-Carlo*

* 时序差分 *Temporal Difference*
  
* 基于单步采样, 逐步引入真实信息
  
* Actor-Critic
  
  * 可解决$A$无限的问题
  
* MM框架 *Minorization Maximization*

  一种普遍使用的优化框架

  * 每次迭代找到目标函数的一个下界函数
  * 不断求这个下界函数的最大值, 以此优化目标函数

###### Value-Based Roadmap: $Q(s,a)$

> 一般适用于离散有限动作域$A$
>
> 由于可选动作有限, 可以直接带入所有Action, 选择Q最大的Action即可

$$
Q-Learning~Series: S{\times}A\rightarrow{\mathbb{R}}\\
Temporal~Difference: 时序差分
$$

* Pre
  * Monte-Carlo
    * 多步估计
    * 高Variance, 低Bias
  * 时序差分理解
    * 单步间隔更新
    * 打车比喻: 
      * 每走一段进行一次更新
      * 每次更新: 开始时预估应为的值(Q(s,a)) = 已确定发生时间(采样r(s,a,s'))+对剩余时间的预估(Q(s',a'))
    * 低Variance, 高Bias
    * Q-Learning, DQN
* Q-Learning
  * 用于离散$S$+离散$A$
  
  * Sampling+Bellman: 
    
    $Q_{\pi_{\theta}}(s,a)=\underbrace{\Bbb E_{s^{'}\backsim p(s^{'}|s, a)}}_{where~sampling~works}[r(s, a, s^{'}) + \gamma \Bbb E_{a^{'}\backsim \pi_{\theta}(a'|s')}[Q(s',a')]]$
    
    * $\gamma \Bbb{E}_{a^{'}\backsim \pi_{\theta}(a'|s')}[Q(s',a')]$部分在算法中会默认选择最优Action
    
  * 对于一步采样$(s,a,s',r)$(数据可复用, off-policy)
    * $target:r+\gamma {max}_{a'}Q_k(s',a')$
    * $update:Q_{k+1}(s,a)\leftarrow Q_k(s,a)+\alpha(target-Q_k(s,a))$
* DQN
  * off-policy
  * 用于连续$S$+离散$A$
  * 使用深度网络模拟Q函数, 解决$S$空间过大/连续问题
    * 通过网络直接模拟非线性函数将$S{\times}A$空间映射到$\mathbb{R}$, 解决映射表格无法用于连续情况的问题
  * 对于一步采样$(s,a,s',r)$
    * $TD target: r+\gamma {max}_{a'}Q_{\theta_{target}}(s',a')$
    * $loss:\frac{1}{2}[Q_{\theta_{main}}(s,a)-TD]^2$
    * $GD: \theta_{main}^{'}\leftarrow \theta_{main}-\alpha\nabla_\theta[loss]|_{\theta=\theta_k}$
    * 左脚踩右脚上天
  * 核心tricks
    * 目标网络冻结 *freezing target networks*
      * 实际上会维护两个网络, main 和 target, target网络用于提供计算loss时的target值
      * 在一个时间段内"冻结"target网络中的参数，来稳定学习目标
      * 对target使用多步更新或者软更新(soft update)
        * soft update: $\theta_{target}\leftarrow\theta_{traget}+\tau\theta_{main}(for~example,\tau~can~be~0.005)$
    * 经验回放 *experience replay*
      * 构建一个经验池来去除数据相关性, 同时提高数据利用率
      * 实践中即为Replay-Buffer: $(s,a,s',r)$
      * 数据可复用, 典型的off-policy
* Double Q, Dual Q: 更多的Q网络雕花 解决过估计等问题

###### ⭐Policy-Based Roadmap $\pi_\theta(a|s)$

$$
Policy~Gradient~Series:S\rightarrow{\Delta (A)} \\
需要注意的是,相比Q-Learning,Policy~Gradient有更小的定义域,更平滑的计算空间
$$

* 策略梯度 Policy Gradient

  用一个深度网络 $\theta$ 来模拟策略函数$\pi_\theta(a|s)$, 直接学习最优策略

  * Optimal Policy目标: Gradient Ascent参数$\theta$获得最大的Expected Return $J(\theta)$

  $$
  \begin{align}
      \nabla_{\theta}J(\theta)
      &=\nabla_{\theta}\Bbb E_{\tau\backsim p_{\theta}(\tau)}[G(\tau)]\\
      &=\nabla_{\theta}\sum_{\tau}[G(\tau)p_{\theta}(\tau)]\\
  	&=\sum_{\tau}[G(\tau)\nabla_{\theta}p_{\theta}(\tau)]\\
  	&=\sum_{\tau}[G(\tau)\nabla_{\theta}p_{\theta}(\tau)\frac{p_{\theta}(\tau)}{p_{\theta}(\tau)}]&\nabla\log f(x)=\frac{\nabla f(x)}{f(x)}\\
  	&=\sum_{\tau}[(p_{\theta}(\tau))*G(\tau)\nabla_{\theta}\log p_{\theta}(\tau)]\\
  	&=\Bbb E_{\tau\backsim p_{\theta}(\tau)}[G(\tau)\nabla_{\theta}\log p_{\theta}(\tau)]&\sum\&\Bbb E变换\\
  	&\approx\frac{1}{N}\sum_{n=0}^{N-1}[G(\tau^n)\nabla_{\theta}\log p_{\theta}(\tau^n)] &(Monte-Carlo)\\
  	&\because p_\theta(\tau) = p(s_0)\prod^{T-1}_{t=0}{\pi_{\theta}(a_t|s_t)p(s_{t+1}|s_t, a_t)}\\
  	&\log{p}_\theta(\tau)=\log{p(s_0)}+\sum^{T-1}_{t=0}\log{\pi_{\theta}(a_t|s_t)}+\sum^{T-1}_{t=0}\log{p(s_{t+1}|s_t, a_t)}\\
  	&\nabla_\theta\log{p}_\theta(\tau)=\sum^{T-1}_{t=0}\nabla_\theta\log{\pi_{\theta}(a_t|s_t)}&消去\theta无关项\\
  	
  	&=\frac{1}{N}\sum_{n=0}^{N-1}\sum_{t=0}^{T_n-1}G(\tau^n)\nabla_{\theta}\log \pi_{\theta}(a_t^n|s_t^n)\\
  	&由于一个action只能影响其之后的reward, 优化G(\tau^n)项\\
  	&=\frac{1}{N}\sum_{n=0}^{N-1}\sum_{t=0}^{T_n-1}G_t^n\nabla_{\theta}\log \pi_{\theta}(a_t^n|s_t^n)&(G_t^n=G(\tau_t^n))\\
  \end{align}
  $$

  

  * 直观理解: 当$G_t^n>0$, $\theta$优化方向是使$\tau^n$中所有状态下采取当前决策$\pi_{\theta}(a_t^n|s_t^n)$的概率增大的方向
  * 对$G_t^n$可采取多种优化, 衍生出不同算法中的Loss
    * $G_t^n-Base(s_t^n)$
    * $A^{GAE}_\theta(s,a)$
    * 时序差分

* Natural Policy Gradient

  * Fisher

* REINFORCE

  * Monte-Carlo Policy Gradient
    * 在策略$\pi_\theta$下采样N条轨迹(典型on-policy)
    * $计算梯度:\nabla_{\theta}J(\theta)=\frac{1}{N}\sum_{n=0}^{N-1}\sum_{t=0}^{T_n-1}G_t^n\nabla_{\theta}\log \pi_{\theta}(a_t^n|s_t^n)$
    * $Gradient~Ascent:\theta'\leftarrow\theta+\alpha\nabla J(\theta)$
    * 重复至收敛
  * 实践中, 会定义loss为负数, 以便使用梯度下降工具
    * $loss=-\frac{1}{N}\sum_{n=0}^{N-1}\sum_{t=0}^{T_n-1}G_t^n\log \pi_{\theta}(a_t^n|s_t^n)$
    * $GD: \theta'\leftarrow\theta-\alpha\nabla loss$

* $\cal{[⭐数理Base]}$ TRPO *Trust Region Policy Optimization*

  $$
  argmax_{\theta'}\mathbb{E_{s\backsim v_\theta,a\backsim\pi_\theta(a|s)}}[\frac{\pi_{\theta'}(a|s)}{\pi_{\theta}(a|s)}A_{\pi_\theta}(s,a)]\\ 
  s.t.D_{KL}(\pi_\theta(a|s)||\pi_{\theta'}(a|s))<\epsilon
  $$

  * Why

    * 传统Policy Gradient $\theta'\leftarrow\theta+\alpha\nabla{J}(\theta)$对更新步长$\alpha$十分敏感, 步长稍大就可能**破坏原有的策略性能**
      * Extra Topic: Problems About Sampling -- Dynamic Graph
    * 基于CPI 混合策略的思想, 同时修正了混合策略在实践中不可用的问题

  * 核心思想: 通过数学证明保证在使用符合限制的步长时每一步都确定能够优化模型表现, 避免步长过大导致模型偏离正确的梯度方向导致模型崩溃的同时使步长尽可能大以加速收敛

    * 在保证每一步都**不会让策略变差**（即单调改进）的前提下
    * 允许我们进行**尽可能大的更新步**
    * 同时**适用于任意参数化策略（比如神经网络）**

  * 数学推导 *From How to Why*
    $$
    Define:\\
    \rho(s_0):初始状态s_0的分布
    $$

    * 用$\theta'$在轨迹上的advantage function和$J(\theta)$来表示$J(\theta')$

    $$
    \begin{align}
    Proof:J(\theta')
    	&=J(\theta)+\mathbb{E}_{\tau{\backsim}p_{\theta'}(\tau)}[\sum_{t=0}^\infty\gamma^tA_\theta(s_t,a_t)]\\
    
    即证&\Downarrow\\
    
    \mathbb{E}_{\tau{\backsim}p_{\theta'}(\tau)}[\sum_{t=0}^\infty\gamma^tA_\theta(s_t,a_t)]
    	&=J(\theta')-J(\theta)\\
    
    \because A_\theta(s, a) = Q(s, a) - V(s) 
    	&= \mathbb{E}_{s'\backsim p(s'|s,a)}[r(s,a,s')+\gamma V(s')-V(s)]\\
    
    \mathbb{E}_{\tau{\backsim}p_{\theta'}(\tau)}[\sum_{t=0}^\infty\gamma^tA_\theta(s_t,a_t)]
    	&=\mathbb{E}_{\tau{\backsim}p_{\theta'}(\tau)}[\sum_{t=0}^\infty\gamma^t(r_t+{\gamma}V_\theta(s_{t+1})-V_\theta(s_t))]\\
    	&=\mathbb{E}_{\tau{\backsim}p_{\theta'}(\tau)}[\sum_{t=0}^\infty\gamma^tr_t+\sum_{t=0}^\infty\gamma^t({\gamma}V_\theta(s_{t+1})-V_\theta(s_t))]\\
    	
    	\because\sum_{t=0}^\infty\gamma^t({\gamma}V_\theta(s_{t+1})-V_\theta(s_t))
    	&=\sum_{t=0}^\infty[{\gamma}^{t+1}V_\theta(s_{t+1})-\gamma^tV_\theta(s_t)]\\
    	&=-V(s_0)+\gamma{V(s_1)}-\gamma{V(s_1)}+\gamma^2V(s_2)-\gamma^2V(s_2)+\cdots\\
    	&=-V_\theta(s_0)\\
    	
    	\therefore\mathbb{E}_{\tau{\backsim}p_{\theta'}(\tau)}[\sum_{t=0}^\infty\gamma^tA_\theta(s_t,a_t)]
    	&=\mathbb{E}_{\tau{\backsim}p_{\theta'}(\tau)}[\sum_{t=0}^\infty\gamma^tr_t-V_\theta(s_0)]\\
    	&=\mathbb{E}_{\tau{\backsim}p_{\theta'}(\tau)}[\sum_{t=0}^\infty\gamma^tr_t]-\mathbb{E}_{\tau{\backsim}p_{\theta'}(\tau)}[V_\theta(s_0)]\\
    &我们默认环境初始状态分布\rho与\theta无关\\
    	&=\mathbb{E}_{\tau{\backsim}p_{\theta'}(\tau)}[\sum_{t=0}^\infty\gamma^tr_t]-\mathbb{E}_{s_0\backsim\rho(s_0)}[V_\theta(s_0)]\\
    &=J(\theta')-J(\theta)\\
    	&得证
    \end{align}
    $$

    

  * 上下界探究问题--如何确定合适的步长以保证每次更新模型只会优化表现而非出错

  * 

  * TRPO的问题

    * Policy的KL散度不好算, 一般没有解析式只能进行数值计算
    * DNN本身不适合限制问题

* $\cal{[exSOTA]}$ **PPO** *Proximal Policy Optimization*

  * 基于TRPO思想的工程上的可行高效实现

  * 特点理解: 总体上为on-policy, 通过局部的off-policy化实现数据复用解决训练效率问题
    * $\theta$生成本轮数据$\mathbb{D}$并作为参考Policy进行重要性采样, 使用$\mathbb{D}$对$\theta$进行**多轮训练**
    * 轮次内, 我们可以将最初的$\theta$视为Ref-Policy从而实现局部off-policy化

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
      * 计算此时$\theta$下得到的$TD\_target, TD\_delta, A^{GAE}_{\theta'},\pi_{\theta'}(a|s)$作为参考策略$\theta'$的数据
      * 在$\mathbb{D}$上对$\theta$进行**多轮训练**, 该过程中使用预先计算好的数据作为参考策略$\theta'$数据进行重要性采样
        * 相当于将最初的$\theta_{init}$视为参考策略$\theta'$
        * 每一步实际计算的只有当前策略下的$\pi_{\theta}(a|s)$和$V_\theta(s)$
        * 局部off-policy化
        * 对于ValueNet, 每轮都采用TD-target进行MSE逼近
          * 每轮计算新的V(s_t), 但是TD-target不变, 始终采用ref中的TD-target
      * 重复至收敛
  
* $\cal{[deepseek]}$ GRPO *Group Relative Policy Optimization*

###### RLHF Series (RL on Human Feedbacks)

* DPO
* ToD调研
  * 项目方向
  * 数据集+调研报告
  * 试着搓一个demo吧

###### Problems About Sampling -- Dynamic Graph

> RL的样本既不独立也不同分布

* Graph不稳定
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

##### DL & LLM

###### The Role of RL in LLM

Work Flow: $Pre-Train{\rightarrow}SFT:Supervised~Fine-Tuning{\rightarrow}ReinforcementLearning$

一个LLM模型的上限, 在Pre-Train时就已经确定了

RL的作用是, 彻底发挥出LLM的潜力

当LLM本身能力不足时($roughly,<7B$), 用RL也没效果...

###### Bias & Variance & Sampling: Dive Deeper

###### Attention

###### Transformer

###### VAE

