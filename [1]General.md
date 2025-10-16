### 通用基础-概述

>  数学 概念 工具...

什么都不会...

体感上, RL领域的各种定义并没有一个标准化的规范, 大家只是"差不多这个意思"的进行讨论

我尽量保证公式符号表示定义的一致连贯...

##### Math

得补课概率论和高数...

###### 概率相关

* 全概率公式

  * $\Bbb E_{x\backsim p(x)}(f(x)) \iff {\displaystyle\sum_x}p(x)f(x)~~~ \mathbb{E}与\sum的互换$
  
* **蒙特卡洛特方法 *Monte Carlo Method* **
  * 对于分布空间极大的求期望场景, 我们可以通过采样*(Sampling)*逼近期望值


$$
\sum_{x}{p(x)f(x)}=E_{x\backsim p(x)}(f(x)) \approx \frac{1}{N}\sum_{n=0}^{N-1}f(x^{(n)})\\
e.g. \underbrace{\Bbb E_{\tau\backsim p_{\theta}(\tau)}[G(\tau)\nabla_{\theta}\log p_{\theta}(\tau)]}_{\tau\backsim p_{\theta}(\tau)空间过大~基本无法求期望} \approx\frac{1}{N}\sum_{n=1}^{N}[G(\tau^n)\nabla_{\theta}\log p_{\theta}(\tau^n)]
$$

* $\epsilon-探索$

$$
\pi_{\theta}^{\epsilon}(s) = \begin{cases}\pi(s), & 按概率1-\epsilon, \\randomly~selected~action~in~A, & 按概率\epsilon.\end{cases}
\\一般\epsilon会随着模型学习的积累逐渐减小
$$

* KL散度 *KL diverge* $D_{KL}(P||Q)$

  > 详解见BV1r6jHzpE1J

  * $$
    用分布Q近似分布P: D_{KL}(P||Q) = \begin{cases}
    \displaystyle\sum_xP(x)\log\frac{P(x)}{Q(x)},&离散\\
    \displaystyle\int P(x)\log\frac{P(x)}{Q(x)}dx,&连续
    
    \end{cases}
    $$

  * 描述两个分布的差别

  * 恒大于等于0, 当且仅当P,Q同分布时为0

    * 证明: Jensen不等式

###### 离散相关

* 笛卡尔集 Cartesian Product
* 离散数学内容, $A\times B$即A, B集合中所有元素 all to all 进行有序配对组成新集合
  * $|A\times B| = |A|\times|B|$
* **马尔可夫决策过程** *Markov Decision process* (MDP)
  * 研究RL最常用数学模型
  * 该过程是**步骤独立**的, 状态间的转换仅取决于系统**当前的**状态, 而与系统过去或未来任意状态都独立不相关
  * 离散状态下被称为 *马尔可夫链*
    * 典型: 有A, B两种状态, 转换规则: 当前状态为[A]: to-A: 0.7, to-B: 0.3; [B]: to-A: 0.4, to-B: 0.6;
    * 状态转换概率仅取决于当前所处的状态

##### RL Basic

$S:状态空间;~A:动作空间;~R:奖励空间;~\pi_\theta: 策略,即Policy;~\theta:模型参数$

###### 概念定义

* 概述
  * RL问题可以描述为: Agent从与Environment的**交互**中不断学习以完成特定目标(如: maximize(Return))
    * 交互: Agent在不同的State依据其Policy-$\pi_\theta$选择动作, 完成状态转移并获得Env的Reward反馈的过程
    * Agent根据已知信息更新Policy学习最优策略
  * RL的目标: 学习到一个策略$\pi_{\theta}(a|s)$来最大化期望回报(expected return)
  * 与经典Supervised的区别
    * 对于输$s$, 没有人类可以提供的标签$a^*$, 需要agent自行探索
* 环境 Environment (Env)
  * 与智能体交互的环境, 接收智能体的Action影响而改变State, 提供Reward反馈
  * 一般来说, Env类似黑箱, 其内部实现我们无法直接得知, 只能通过有限的接口和规则与其交互并观察其State和反馈(State改变以及Reward)
* 智能体 Agent
  * 观察Env(Observation), 获得当前的State和Reward, 并进行学习和决策(选定Action)
  * 一般来说, RL中训练Agent的目标为最大化预计能获得的Return(非单步reward, 而是预期整个过程能获得的总reward)
* 模型 Model

    * 即Env的运行规律
      * alias"环境动力学模型"
      * RL里一般特指$Model = \{r(s,a,s'), p(s'|s,a)\}$
        * **状态转移概率** *Transition Function*: $p(s_{t+1}|s_t, a_t)$
          * $S\times A\rightarrow \Delta(S)$
            * 用$\Delta(S)$表示在S上的概率分布
            * $(s,a)$往往只能确定$s'$的分布, 而不是总是导向单个$s'$
            * 例如, 机器人走路遇到障碍(s), 选择跳过去(a), 可能成功($s^{'}_1$)也可能摔倒($s^{'}_2$)
          * 智能体根据当前状态$s_t$做出一个动作$a_t$之后，下一个时刻环境处于不同状态$s_{t+1}$的概率分布
          * 一般由环境决定, 一般符合Monte Carlo过程
        * **奖励函数** *Reward Function*: $r(s_t, a_t, s_{t+1})$
          * $S × A × S → \mathbb R$
          * 需要注意, 返回的奖励与到达的新状态也有关
            * 如前例, 遇到障碍(s)选择跳(a), 可能成功($s^{'}_1$)也可能摔倒($s^{'}_2$), 两种新状态reward不同
      * Model一般难以得知, 因此大多数时候我们无法直接使用model得到解析解, 而只能通过交互采样
        * 但也可以用一个网络进行模拟, 隐式得到
    * Model-Based/Model-Free: 区别在于环境知识的掌握程度
      * Model-Based: 显式使用Model中的函数
      * Model-Free: 不依赖Model, 通过采样$(s,a,s',r)$进行学习
* **策略 Policy** $\pi_{\theta}(a|s)$
    * $S\rightarrow\Delta(A)$
    * 决定Agent在状态s下采取什么行动a
    * 同样的, s导向一个**a的分布**, 不一定总是某个确定的动作
* 状态 State
  * Env给Agent反馈的状态
  * Observation: State的子集(或者说, 残缺信息, Ob也可能是State的低维观察信息), 因为Agent未必能获得全部的State, 有时其只能获得Ob到的部分真实State信息, 并且可能有噪声(如: 超agent感知范围, 感知结构) 
  * 环境中有一个或多个特殊的终止状态（terminal state）
* 动作 Action

    * Agent能采取的行动
* 奖励 Reward
    * Agent根据当前State做出一个Action, Env接收Action转移到新状态s', 并反馈给Agent一个奖励
    * Reward是环境给Agent的奖惩反馈, 其定义会很大程度上影响模型的行为进而影响表现
* 轨迹 Trajectory $\tau$
    * 马尔可夫决策过程的一个轨迹（trajectory）
    * 例: $\tau = \{s_0, a_0, s_1, r_1(s_0, a_0, s_1), \cdots, s_{T-1}, a_{T-1}, s_T, r_T(s_{T-1}, a_{T-1}, s_T)\}$
    * $\tau$的概率可以用策略选择和状态转移概率连乘表示
      * $p_\theta(\tau) = p(s_0) \displaystyle  \prod^{T-1}_{t=0}{\pi_{\theta}(a_t|s_t)p(s_{t+1}|s_t, a_t)}$
* 回报 Return
    * Agent和Env进行一次交互过程的轨迹$\tau$累积的Reward
    * 轨迹$\tau$的总回报: $G(\tau) = \sum^{T-1}_{t=0}{\gamma^tr_{t+1}}$
      * 从$t_0$时刻开始计算的总回报: $G(\tau_{t_0})=\sum_{t=t_0}^{T-1}\gamma^{t-t_0}r_{t+1}$
      * $\gamma\in[0,1]$为折扣率 discount rate
      * 当$\gamma\rightarrow 0$时，Agent更在意短期回报；而当$\gamma\rightarrow 1$时，长期回报变得更重要
* 期望回报 Expected Return
    * $J(\theta)=\Bbb E_{\tau\backsim p_{\theta}(\tau)}[G(\tau)]=\Bbb E_{\tau\backsim p_{\theta}(\tau)}[\sum^{T-1}_{t=0}{\gamma^tr_{t+1}}]$ 

###### 值函数

* 状态值函数 State-Value Function: $V_{\pi_{\theta}}(s)$
  * $S\rightarrow\mathbb{R}$

  * 描述从状态s开始，执行策略$\pi_{\theta}$得到的期望总回报

  * $V_{\pi_{\theta}}(s) = \Bbb E_{\tau\backsim p_\theta(\tau)}[G(\tau)|\tau_{s_t} = s]$

      * 因为 $p(s'|s, a)$实际上返回一个分布, 所以我们即使在相同s采取相同a, 得到的$s'$也可能不一样
      * 因而即使输入固定的$s_0, \pi_{\theta}$, 每次交互的轨迹也可能不同, 因而求$V_{\pi_{\theta}}(s)$需要用$\mathbb{E}$--期望值

  * Bellman: $V_{\pi_{\theta}}(s) = \sum_{a\backsim\pi_{\theta}(a|s)}\pi_\theta(a|s)\sum_{s'\backsim p(s'|s,a)}p(s'|s,a)[r(s,a,s')+\gamma V_{\pi_{\theta}}(s')]$ 

    * $$
      \begin{align}
      V_{\pi_{\theta}}(s) &= \Bbb E_{\tau\backsim p_\theta(\tau)}[G(\tau)|\tau_{s_t} = s]\\
      	&=\sum_{a\backsim\pi_{\theta}(a|s)}\pi_{\theta}(a|s)\mathbb E_{\tau\backsim p_\theta(\tau)}[G(\tau_t)|\tau_{s_t}=s, \tau_{a_t}=a]\\
      	&=\sum_{a\backsim\pi_{\theta}(a|s)}\pi_{\theta}(a|s)\sum_{a'\backsim p(a'|s,a)}p(a'|s,a)\mathbb (r(s,a,s')+\gamma \mathbb E[G(\tau_{t+1}|\tau_{s_{t+1}}=s')])\\
      	&=\sum_{a\backsim\pi_{\theta}(a|s)}\pi_\theta(a|s)\sum_{s'\backsim p(s'|s,a)}p(s'|s,a)[r(s,a,s')+\gamma V_{\pi_{\theta}}(s')]\\
      	&= \Bbb E_{a\backsim\pi_{\theta}(a|s)}\Bbb E_{s'\backsim p(s'|s,a)}[r(s,a,s')+\gamma V_{\pi_{\theta}}(s')]\\
      \end{align}
      $$

    * 理解: 树形分散 对于所有可能路径

      * $\sum_{a\backsim\pi_{\theta}(a|s)}$: $\pi_{\theta}$在状态s下可能采取的所有a
      * $\sum_{s'\backsim p(s'|s,a)}$: 在状态s采取行动a后可能转移到的所有新状态$s'$

* 动作值函数 Action-Value Function: $Q_\theta(s, a)$
  * $S\times A\rightarrow\mathbb R$
  * 描述初始状态为s并进行动作a后，执行策略$\pi_{\theta}$得到的期望总回报。
  * $Q_{\pi_{\theta}}(s,a)=\Bbb E_{\tau\backsim p(\tau)}[G(\tau)|\tau_{s_0} = s,\tau_{a_0}=a]$
  * Bellman: $Q_{\pi_{\theta}}(s,a)=\Bbb E_{s^{'}\backsim p(s^{'}|s, a)}[r(s, a, s^{'}) + \gamma \Bbb E_{a^{'}\backsim \pi_{\theta}(a'|s')}[Q(s',a')]]$
  
* Q&V 关系
  
    * $Q(s,a)=\Bbb E_{s'\backsim p(s'|s,a)}[r(s,a,s')+\gamma V(s')]$
    * $V(s)=\mathbb E_{a\backsim \pi_{\theta}(a|s)}[Q(s,a)]$
    * 也可代换入Bellman
    * 公式很美
    
* 优势函数 Advantage Function: $A_\theta(s, a) = Q(s, a) - V(s)$



##### RL方法概论

###### 值函数估计-发展路径

* Value Function
  * $V_{\pi_{\theta}}(s) = \sum_{a\backsim\pi_{\theta}(a|s)}\pi_\theta(a|s)\sum_{s'\backsim p(s'|s,a)}p(s'|s,a)[r(s,a,s')+\gamma V_{\pi_{\theta}}(s')]$ 
  * 古早阶段简单环境下的方法, 由于简单环境, 已知$Model = \{r(s,a,s'), p(s'|s,a)\}$和所有state
    * 知道Model才能数值计算Expectation
  * 可以通过迭代的方式求出每个state的最佳Value Function
    * $V^*(s)=max_{\pi_\theta}{\mathbb{E}_{\tau\backsim p_{\pi_\theta}(\tau)}}[\sum_{t=0}^{T-1}\gamma^tr(s_t,a_t,s_{t+1})|\pi_\theta,s_0=s]$
    * Bellman-update
    * 问题
      * 实际情况, Env总是复杂不可知难以观测从满噪声的, Model不可得
      * State空间往往巨大/连续, 遍历不可得

$$
Q-Learning~Series: S{\times}A\rightarrow{\mathbb{R}}
$$

* Q-Learning

  * 由于Model(特别是$p(s'|s,a)$)未知, 为了算期望, 我们只能通过Monte-Carlo进行近似
    * 然而, $V^*(s)=max_{\pi_\theta}{\mathbb{E}_{\tau\backsim p_{\pi_\theta}(\tau)}}[\sum_{t=0}^{T-1}\gamma^tr(s_t,a_t,s_{t+1})|\pi_\theta,s_0=s]$在采样时存在问题
      * $V^*(s)$的计算依赖于$\pi_\theta$, 而$\pi_\theta$决策又是通过$V^*(s)$进行
      * 这导致每一次Sampling后的更新都可能改变$\pi_\theta$,而在不同$\pi_\theta$下计算的$V(s)$会有很大的差别
      * 故, 使用在不同Policy下采样到的$V(s)$进行平均求期望实际上没有意义
      * 通过Q函数, 将Action作为参数, 把Policy的影响排除在外
  * 学习最优Q函数, 相较于Value Function, 能更好的进行Sampling
  * Sampling+Bellman: $Q_{\pi_{\theta}}(s,a)=\underbrace{\Bbb E_{s^{'}\backsim p(s^{'}|s, a)}}_{where~sampling~works}[r(s, a, s^{'}) + \gamma \Bbb E_{a^{'}\backsim \pi_{\theta}(a'|s')}[Q(s',a')]]$
    * $\gamma \Bbb{E}_{a^{'}\backsim \pi_{\theta}(a'|s')}[Q(s',a')]$部分在算法中会默认选择最优Action
    * 对于一次采样$(s,a,s',r)$
      * $target:r+\gamma {max}_{a'}Q_k(s',a')$
      * $update:Q_{k+1}(s,a)\leftarrow Q_k(s,a)+\alpha(target-Q_k(s,a))$
  * 由于$S{\times}A$空间一般极大或无限, 传统Q表格的方式并不适用
    * DQN解决了这个问题

* DQN
  
  > Deep Learning was introduced to RL from now on.
  
  * 即使用深度网络来模拟Q函数, 解决$S{\times}A$空间过大问题
    * 通过网络直接模拟非线性函数将$S{\times}A$空间映射到$\mathbb{R}$, 而无需维护一套映射表格/K-V对
  * Sampling+Bellman: $Q_{\pi_{\theta}}(s,a)=\underbrace{\Bbb E_{s^{'}\backsim p(s^{'}|s, a)}}_{where~sampling~works}[r(s, a, s^{'}) + \gamma \Bbb E_{a^{'}\backsim \pi_{\theta}(a'|s')}[Q(s',a')]]$
    * $\gamma \Bbb{E}_{a^{'}\backsim \pi_{\theta}(a'|s')}[Q(s',a')]$部分在算法中会默认选择最优Action
      * 这里是对于离散有限动作域$A$, 可以直接带入所有Action选择Q最大的Action
      * 对于连续/无限动作域, 用什么方法?
        * 现在我也不知道.. DDPG?
    * 对于一次采样$(s,a,s',r)$
      * $target: r+\gamma {max}_{a'}Q_{\theta_k}(s',a')$
      * $loss: \frac{1}{2}[Q_{\theta_k}(s,a)-target]^2$
      * $GD: \theta_{k+1}\leftarrow \theta_k-\alpha\nabla_\theta[loss]|_{\theta=\theta_k}$
  * 核心trick
    * 目标网络冻结（freezing target networks），即在一个时间段内固定目标中的参数，来稳定学习目标
      * 实际上会维护两个网络, main 和 target
      * 对target使用多步更新或者软更新(soft update)
        * soft update: $\theta_{target}\leftarrow\theta_{traget}+\tau\theta_{main}(for~example,\tau~can~be~0.005)$
    * 经验回放（experience replay），构建一个经验池来去除数据相关性。
      * 实践中即为Replay-Buffer: $(s,a,s',r)$
  * DQN系列现在仍然是很多RL问题的首选方法
    * 简单便宜
  
* Double Q
  
  * 解决Q值估计过大问题

###### 策略搜索 $\pi(a_t|s_t)$-发展路径

$$
Policy~Gradient~Series:S\rightarrow{A} \\
需要注意的是, Policy~Gradient的搜索空间相比Q-Learning要小许多
$$

* 策略梯度 Policy Gradient

  * 目标: optimize参数$\theta$来最大化期望总回报

  * $$
    \begin{align}
        \nabla_{\theta}J(\theta)
        &=\nabla_{\theta}\Bbb E_{\tau\backsim p_{\theta}(\tau)}[G(\tau)]\\
        &=\nabla_{\theta}\sum_{\tau}[G(\tau)p_{\theta}(\tau)]\\
    	&=\sum_{\tau}[G(\tau)\nabla_{\theta}p_{\theta}(\tau)]\\
    	&=\sum_{\tau}[G(\tau)\nabla_{\theta}p_{\theta}(\tau)\frac{p_{\theta}(\tau)}{p_{\theta}(\tau)}]\\
    	&=\sum_{\tau}[(p_{\theta}(\tau))*G(\tau)\nabla_{\theta}\log p_{\theta}(\tau)]\\
    	&=\Bbb E_{\tau\backsim p_{\theta}(\tau)}[G(\tau)\nabla_{\theta}\log p_{\theta}(\tau)]&\sum\&\Bbb E变换\\
    	&\approx\frac{1}{N}\sum_{n=0}^{N-1}[G(\tau^n)\nabla_{\theta}\log p_{\theta}(\tau^n)] &(Monte-Carlo)\\
    	&\because p_\theta(\tau) = p(s_0) \displaystyle  \prod^{T-1}_{t=0}{\pi_{\theta}(a_t|s_t)p(s_{t+1}|s_t, a_t)}\\
    	&\cdots &(余见PPO笔记, 纯练写公式了\ldots)\\
    	&=\frac{1}{N}\sum_{n=0}^{N-1}\sum_{t=0}^{T-1}G(\tau^n)\nabla_{\theta}\log \pi_{\theta}(a_t|s_t)
    \end{align}
  $$
    

    
  * 理解: 参数θ优化的方向是使得总回报$G(\tau)$越大的轨迹$\tau$的概率$p_{\theta}(\tau)$也越大

  * 后续都是基于这个式子进行各种优化-*雕花*

* REINFORCE *Monte Carlo Policy Gradient*

* $\cal{[SOTA]}$ From TRPO *Trust Region Policy Optimization* to **PPO** *Proximal Policy Optimization*

  * 

###### RLHF Series (RL on Human Feedbacks)

* RLHF
* DPO
* GRPO

###### Actor-Critic

###### On-Policy & Off-Policy

* On-Policy
  * 采集数据用的Policy和训练的Policy是同一个
* Off-Policy

##### Deep Learning Basic

###### Loss & Gradient Descent



###### 正则化 Normalization



###### entropy: 熵 & 交叉熵



###### Attention



###### Transformer



###### VAE



##### With LLM

###### RLHF

* RL on Human Feedbacks