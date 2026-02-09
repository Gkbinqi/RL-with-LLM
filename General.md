#### 前言

Basic, basic...

仅关注入门的基础理解, 大致路径, 经典方法的代码

尽量保证公式符号连贯

同时也算是学习了latex风格公式写法...

###### Resources

数理基础 -- RL的数学原理: https://www.icourse163.org/course/XHUN-1470436188

代码基础 -- 动手学强化学习: https://hrl.boyuai.com/chapter/intro/

> next -- UCB-CS285

#### Fundamental: RL 基础概念 & 工具

后续内容的基础和框架

##### RL定义 & Markov 框架

$$
S:状态空间;~A:动作空间;~R:奖励空间;~\Delta(Space):在Space上的概率分布;\\
\pi:策略;\theta:策略模型的参数;~\mathbb{D}:数据/经验;~\mathbb{M}:Env模型/model
$$

RL问题可以描述为: Agent 从与 Environment 的**交互**历史信息中 不断学习以完成特定目标

RL的普遍目标: 学习 **Optimal Policy** $\pi^*_{\theta}$ --  实现最大化期望回报 $J(\theta)$ 

* 交互: Agent 依据其策略 $\pi_\theta$ 在观察到的 State 选择 Action, 完成**状态转移**并获得 Reward 的过程

###### ⭐**马尔可夫决策过程** Markov Decision Process (MDP)

RL 一般将问题建模为 MDP

MDP 有三要素: Markov Property(Markov) + Policy(Decision) + Space & Trajectory(Process)

MDP 是我们在讨论RL问题时的世界观, 我们的概念都是在该框架下定义推演的

Markov Property: **历史无关** (Memoryless)

* 状态转移仅取决于系统**当前**状态和动作, 与系统过去或未来任意状态动作都不相关

* $p(s'|s,a)$ 状态转移的输出只与输入的当前的s和a有关, 与之前任意步的s和a无关

  $p(s_{t+1}|s_t,a_t)=p(s_{t+1}|s_0,a_0,s_1,a_1,\ldots,s_t,a_t)$

当策略确定时, MDP 退化成一个 Markov Process.

###### 空间 Space

#状态 State $s{\in}S$

Agent 观察到的自身在 Env 中的状态

* Observation: State的子集(或者说, 残缺信息, Ob也可能是State的低维信息)
  * Agent未必能获得State的所有维度, 有时其只能获得部分维度的信息, 并且可能有噪声
  
    方便起见, 简单问题中我们假设Agent能直接获得Env的所有信息, 即$o=s$
  
* 环境中可以有一个或多个特殊的终止状态(terminal; absorbing)

* 也可以truncated, 在一定时间或步数后主动终止

#动作 Action $a{\in}A$

Agent 能采取的行动

#奖励 Reward $r{\in}\mathbb{R}$

Reward 是环境给 Agent 的反馈, **其定义会决定模型的优化方向**

* Reward 可以视为人类与 Agent 交互的接口, 人通过 Reward 指导评估 Agent 的行为

  关于 Reward 有很多讨论研究, 是很关键的部分

###### 智能体 Agent

观察 State, 依据其 Policy 做出决策 Action 与 Env 进行交互的实体

###### 环境 Environment (Env)

接收 Agent 的 Action 而改变 State, 并反馈 $(s',r)$ 的与 Agent 交互的实体

#模型 Model $\mathbb{M}$

即 Env 的运行规律, RL里一般特指由环境决定的函数 $Model = \{r(s,a,s'), p(s'|s,a)\}$

* **状态转移概率** *Transition Function*: $p(s'|s, a)$

  智能体根据当前状态$s$选择动作$a$后，下一个时刻状态 $s'$ 的概率分布

  $S\times A\rightarrow \Delta(S)$
  * $(s,a)$往往只能确定$s'$的分布, 而不是总是导向单个$s'$

    例如, 机器人走路遇到障碍(s), 选择跳过去(a), 可能平稳落地($s^{'}_1$)也可能摔倒($s^{'}_2$)

* **奖励函数** *Reward Function*: $r(s, a, s')$ or $r(s,a)$

  $S × A × S → \mathbb{R}$ or $S × A {\rightarrow}\Delta(\mathbb{R}) $

  * 需要注意, 返回的奖励与到达的新状态也有关

    如前例, 遇碍(s)跳(a), 平稳落地($s^{'}_1$)或摔倒($s^{'}_2$), 两种新状态reward不同

###### 策略 Policy

决定Agent在状态s下采取什么行动a

#Stochastic: $\pi_{\theta}(a|s)$

* $S\rightarrow\Delta(A)$, 随机策略$\pi_\theta(a|s)$, 输出a的分布

#Deterministic: $a=\mu_\theta(s)$

* $S{\rightarrow}A$, 确定性策略, 输出具体动作

###### 轨迹 Trajectory $\tau$

对一次MDP的历史记录

$\tau = \{s_0, a_0, r_1, s_1, \cdots, s_{T-1}, a_{T-1}, r_T, s_T \}$

* 在参数$\theta$下, 特定路径$\tau$出现的概率可以用Policy和Transition Function连乘表示

  $p_\theta(\tau) = p(s_0) \displaystyle  \prod^{T-1}_{t=0}{\pi_{\theta}(a_t|s_t)p(s_{t+1}|s_t, a_t)}$

* Episode

  一条结束于 terminal state 的有限长的轨迹

  对于在 Episode 上工作的任务, 即任务会在合理步数内结束, 我们将其称为 Episodic Task 

###### 回报 Return $G_t$

一种评估策略 $\theta$ 的效果的指标

一条 $\tau$ 中全过程 Reward 的总和

#**折扣总回报** Discounted Return: $G(\tau) = \sum^{T-1}_{t=0}{\gamma^tr_{t+1}}$

一条 $\tau$ 中全过程 Reward 的折扣后总和, 实际中使用的 Return

从任意时刻 $t_0$ 开始计算: $G(\tau_{t_0})=G_{t_0}=\sum_{t=t_0}^{T-1}\gamma^{t-t_0}r_{t+1}=r_{t_0+1}+\gamma G_{t_0+1}$

* $\gamma\in[0,1)$为折扣率 *discount rate*
  
  * $\gamma$ 使得我们能够建模无限长轨迹的回报, 并能控制 Agent 的视野范围
  
    当$\gamma\rightarrow 0$时，Agent 更在意短期回报；当$\gamma\rightarrow 1$时，长期回报变得更重要

###### 期望回报 Expected Return $J(\theta)$

基于回报的定义, 通过计算给定策略 $\theta$ 下**回报的期望值**精确建模策略的效果

$J(\theta)=\mathbb{E}_{\tau\backsim p_{\theta}(\tau)}[G(\tau)]=\mathbb{E}_{\tau\backsim p_{\theta}(\tau)}[\sum^{T-1}_{t=0}{\gamma^tr_{t+1}}]$ 

##### 值函数定义

我们定义, 在给定的 MDP 中, 每一个策略都有对应的**价值函数**, 该函数输入当前状态, 输出从输入状态开始, 执行**当前策略**到结束能获得的**期望回报**

这一定义得到了数学和实验层面的验证, 值函数确实存在且可求解

> 在当前环境(或者说, 任务)下 基于给定策略 得到的对状态的价值的评估 (执行策略 $\pi_\theta$ 的期望总回报)

值函数可以作为评估策略效果的一种方法/依据, 策略的值函数越大, 该策略越优秀

###### 状态值函数 State-Value Function $v_{\pi}(s)$

描述从状态s开始, 执行策略 $\pi_{\theta}$ 得到的的**期望回报**

Def: $v_{\pi}(s) = \mathbb{E}_{\tau\backsim p_\theta(\tau)}[G_t|s_t = s]$

#函数性质: 

状态值函数是接收 s 作为输入, 基于 Policy 的函数

$v_\pi(s) \iff v(s; \pi_\theta)$

###### 动作值函数 *Action-Value Function* $q_{\pi}(s, a)$

描述从在状态s选择动作a后开始, 执行策略 $\pi_{\theta}$ 得到的期望回报

Def: $q_{\pi}(s,a)=\mathbb{E}_{\tau\backsim p(\tau)}[G_t|S_t = s,A_t=a]$

#函数性质

动作值函数是接收 $(s,a)$ 作为输入, 基于 Policy 的函数

$q_\pi(s,a) \iff q(s,a; \pi_\theta)$

###### 优势函数 Advantage Function $A_{\theta}(s, a)$

描述在 $s$ 下选择特定动作 $a$ 较动作空间整体预期回报具有的额外回报(优势)

Def: $A_\theta(s,a)=q(s,a)-v(s)$

##### 贝尔曼工具

定义了值函数后, 我们需要求解策略的值函数的方法

Bellman 公式 提供了求解值函数的工具

###### 贝尔曼公式 Bellman Equation (Bellman Equ)

贝尔曼公式描述了策略的合法值函数之间的依赖关系

给定策略 $\pi$ 和 $\mathbb{M}$, 策略的值函数能够通过递归的方式表示

#状态值函数: $\forall s{\in}S,{v}_{\pi}(s) = \mathbb{E}_{a\backsim\pi_{\theta}(a|s)}\mathbb{E}_{s'\backsim p(s'|s,a)}[r(s,a,s')+\gamma V_{\theta}(s')]$ 
$$
\begin{align}
v_{\pi}(s) 
	&=\mathbb{E}_{\tau{\backsim}p_\theta(\tau)}[G_t|S_t = s]\\
	&=\sum_{a\backsim\pi_{\theta}(a|s)}\pi_{\theta}(a|s)\mathbb{E}_{\tau\backsim p_\theta(\tau)}[G_t|s_t=s, a_t=a]
		&确定a\\
	&=\sum_{a\backsim\pi_{\theta}(a|s)}\pi_{\theta}(a|s)\sum_{s'\backsim p(s'|s,a)}p(s'|s,a)\mathbb (r(s,a,s')+\gamma \mathbb{E}[G_{t+1}|s_{t+1}=s'])
		&确定s'\\
	&=\sum_{a\backsim\pi_{\theta}(a|s)}\underbrace{\pi_\theta(a|s)}_{based ~on~Policy}\sum_{s'\backsim p(s'|s,a)}\underbrace{p(s'|s,a)}_{based~on~\mathbb{M}}[\underbrace{r(s,a,s')}_{based~on~\mathbb{M}}+\gamma V_{\theta}(s')]\\
	&= \mathbb{E}_{a\backsim\pi_{\theta}(a|s)}\mathbb{E}_{s'{\backsim}p(s'|s,a)}[\underbrace{r(s,a,s')}_{immediate~reward}+\underbrace{{\gamma}V_{\theta}(s')}_{future ~return}]\\
\end{align}
$$

* 描述基于策略 $\pi_\theta$ 的, 所有状态 $s$ 的值函数之间的依赖关系
  
  * 有n个状态, 就会有n个式子. 联立求解, 即可得到值函数
  
  * 对于合法值函数, 贝尔曼公式 ${\forall}s{\in}S$ 成立
    
    若存在任一状态, 该等式不成立, 此时该 estimation 与真实值函数仍存在误差

#动作值函数: $\forall{(s,a)}\in{S{\times}A},q_{\pi}(s,a)=\mathbb{E}_{s^{'}\backsim p(s^{'}|s, a)}[r(s, a, s') + \gamma \mathbb{E}_{a'\backsim \pi_{\theta}(a'|s')}[q_{\pi}(s',a')]]$
$$
\begin{align}
Q_{\theta}(s,a)
	&=\sum_{s'\backsim p(s'|s,a)}p(s'|s,a)[r(s,a,s')+\gamma\sum_{a'\backsim \pi_{\theta}(a'|s')}\pi_\theta(a'|s')[Q_{\pi_{\theta}}(s',a')]]\\
	&=\mathbb{E}_{s^{'}\backsim p(s^{'}|s, a)}[r(s, a, s') + \gamma \Bbb E_{a^{'}\backsim \pi_{\theta}(a'|s')}[Q(s',a')]]\\
\end{align}
$$

###### 策略评估 Policy Evaluation (PE) -- Solving Bellman Equation

即, 通过 Bellman Equation 求解一个策略对应的值函数

由于值函数是评价策略效果的指标, 求解一个策略的值函数的过程也被称为 Policy Evaluation

以状态值函数为例: 设 $n = |S|$ , 给定 $\pi_\theta$, 有n个值函数, 可以通过矩阵形式表示贝尔曼公式
$$
Bellman:v=r+{\gamma}Pv\\
v=[v(s_1), v(s_2), {\cdots},v(s_n)]^T\\
r=[r(s_1), r(s_2), {\cdots},r(s_n)]^T\\
r_i=\mathbb{E}_{a\backsim\pi_{\theta}(a|s_i)}\mathbb{E}_{s'{\backsim}p(s'|s_i,a)}[r(s_i,a,s')]\\
P_{ij} = p(s_j|s_i) = \sum_{a{\backsim}\pi_\theta(a|s_i)}\pi_\theta(a|s_i)p(s_j|s_i,a)\\
$$
基于该表达式, 有两种求解给定策略的值函数的方法:

#直接求解 -- 闭式解, 矩阵求逆

$v=r+{\gamma}Pv~{\Rightarrow}~v=(I-{\gamma}P)^{-1}r$

直接通过线代变换求解, 得到的v为真实值函数 $v_\pi$

不过需要做 $O(n^3)$ 的矩阵求逆操作, 以及优化 Policy 并不需要求解特别精确的值函数

特别的, $(I-{\gamma}P)^{-1}$ 矩阵是 折扣状态访问频率矩阵

优美, 但实践中一般不会使用

#迭代求解

init: $v_0,v_0可为任意初始向量,也可为\vec{0}$

iter: $v_{k+1}=r+{\gamma}Pv_k,k{\rightarrow}{\infty}$

可以证明, 迭代无数次后, estimator $v_k$ 与真实值函数 $v_\pi$ 的误差趋零, 精确度足以用于优化策略

#迭代算法实现

实践中, 通过 element-wise 方法进行迭代求解

${\forall}s,v_{k+1}(s)=\sum_{a\backsim\pi_{\theta}(a|s)}{\pi_\theta(a|s)}\sum_{s'\backsim p(s'|s,a)}{p(s'|s,a)}[{r(s,a,s')}+{\gamma}v_k(s')]\\$

###### 贝尔曼最优公式 Bellman Optimality Equation (BOE)

基于值函数的概念, 我们可以此优化策略以逼近目标 -- $\pi^*_\theta$

首先定义目标 Optimal Policy, Optimal State Value 为何

* 我们定义, 最优策略 $\pi^*$ 满足: ${\forall}s,v_{\pi^*}(s){\geq}v_\pi(s)\text{ for any other Policy $\pi$}$

  $v^*(s){\iff}v_{\pi^*}(s)$ 最优策略的值函数, 即为最优值函数

#BOE定义:

* BOE: $v(s) = max_\pi\mathbb{E}_{a\backsim\pi_{\theta}(a|s)}\mathbb{E}_{s'\backsim p(s'|s,a)}[r(s,a,s')+{\gamma}v(s')]= max_{\pi}\sum_{a{\backsim}{\pi}(a|s)}{\pi}(a|s)q(s,a)$

  Matrix Vector Form: $v=max_{\pi}(r+{\gamma}Pv)=f(v)$
  
* 对于 $=$ 的两种理解

  当左右两边值相等时, 此时 $=$ 代表等式成立, 这里的 $v$ 即为 $v^*$, 对应 $\pi^*$

  当左右两边的值不相等时, 此时 $v{\neq}v^*$, 等号可以视为一次迭代的赋值符号 $v_{k+1}{\leftarrow}f(v_k)$

#BOE理解

本质上, BOE 是一种特殊的 Bellman 公式

* based on $\mathbb{M}$, 用于求解 $v^*$ 以及对应的 $\pi^*$

* 给定 $\pi=\pi^*$ 时, BOE 即为基于 $\pi^*$ 的 Bellman Equation

  此时的值函数 $v$ 即为最优值函数 $v^*=v_{\pi^*}$

可以证明(基于 Fixed Point Theorem & 证 $f(v)$  的Contraction property, 细节见书本)

对于模型 $\mathbb{M}$, $v^*$ 和 $\pi^*$ 有:

* $v^*$存在且唯一
* $\pi^*$ 存在, 可能为多个
* 存在至少一个$\mu^*$
* BOE 可通过迭代的方式求解

#算法步骤

任意初始 $v_0$, 对当前步 $v_k$, 迭代: $v_{k+1}=f(v_k)=max_{\pi}(r_{\pi}+{\gamma}P_{\pi}v_k)$

* init $v_0$ 
* iter: $v_{k+1}=f(v_k)\text{ until converge, }k=0,1,2,\ldots$

###### 值迭代 Value Iteration (VI) -- Solving BOE

即复制 BOE 求解, 通过迭代方法直接求解最优值函数

#算法步骤

对当前步 $v_k$, 求解: $v_{k+1}=f(v_k)=max_{\pi}(r_{\pi}+{\gamma}P_{\pi}v_k)$

* Policy Update: 求解 $\pi_{k+1}=argmax_{\pi}(r_{\pi}+{\gamma}P_{\pi}v_k)$
* Value Update: 将 $\pi_{k+1}$ 代入赋值式: $v_{k+1}=r_{\pi_{k+1}}+{\gamma}P_{\pi_{k+1}}v_k$

###### Note: 基于 Bellman Equation 和 $\mathbb{M}$ 的 $QVA$ 互推

$Q(s,a)=\mathbb{E}_{s'\backsim p(s'|s,a)}[r(s,a,s')+\gamma V(s')]$

$V(s)=\mathbb{E}_{a\backsim \pi_{\theta}(a|s)}[Q(s,a)]$

$A(s, a) = Q(s, a) - V(s) = \mathbb{E}_{s'\backsim p(s'|s,a)}[r(s,a,s')+\gamma V(s')]-V(s) = \mathbb{E}_{s'\backsim p(s'|s,a)}[r(s,a,s')+\gamma V(s')-V(s)]$

#### 高度概念汇总

如题, 统一汇总学习过程中常见/重要/基础的高级别概念

在细节内容中涉及到时也会再次提及并结合上下文进行实例分析

###### Model-Based & Model-Free

区别在于对环境的模型使用与否

注意, RL 不同于 Dynamic Programming, 其区分的 Model-Based 和 Model-Free 都没有关于Model的先验

* Model-Based: 通过 $\mathbb{D}$ 学习 Model, 在学习策略过程中显式使用Model相关的函数
* Model-Free: 不使用 Model 相关函数, 仅通过交互采样 $\mathbb{D}$ 进行学习

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

###### On-line & Off-line

aaa

###### Exploration & Exploitation

RL 中常见的平衡

Exploitation: 利用已有的信息高效利用高价值动作, 但是可能导致被困在次优解

Exploration: 探索目前看来次优的动作, 以发现潜在的更优的动作, 但短期内会导致奖励的降低

Exploration 非常重要, 但同时很多情况下局部最优解也是可接受的, 不一定要追求最优解

#### Algos: RL 方法概论

##### [0] Dynamic Programming

特点在于能直接使用 Env Model 的函数

基于给定 Env Model 和 BOE 思想的最优策略求解方法

###### Value Iteration (VI)

实际上就是求解 BOE, 通过迭代值函数的方式求解最优值函数

* 任意初始 $v_0$, 迭代收敛到 $v^*$ 使得 $v^*=f(v^*)$ 等式成立

  $\text{init $v_0$},iter:v_{k+1}=f(v_k),k=0,1,2,\ldots$

#算法步骤

任意初始 $v_0$, 给定当前 $v_k$, iter: $v_{k+1}=f(v_k)=max_{\pi}(r_{\pi}+{\gamma}P_{\pi}v_k)$

* Step 1: Policy Update: 

  求解 $max~\pi$: $\pi_{k+1}=arg~max_{\pi}(r_{\pi}+{\gamma}P_{\pi}v_k)$

* Step 2: Value Update: 

  不是 Evaluation, 因为我们并没有求解合法值函数, 这是一个 estimation

  将 $\pi_{k+1}$ 代入赋值式: $v_{k+1}=max_{\pi}(r_{\pi}+{\gamma}P_{\pi}v_k)=r_{\pi_{k+1}}+{\gamma}P_{\pi_{k+1}}v_k$

#算法实现

element-wise form

给定当前 $v_k$ 

* Step 1: Policy Update: 

  ${\forall}s,{\pi}_{k+1}(s)=arg~max_{\pi}{\sum}_{a}{\pi}(a|s){{\sum}_{s'}p(s'|s,a)[r(s,a,s')+{\gamma}v_k(s')]}=arg~max_{\pi}{\sum}_{a}{\pi}(a|s){q_k(s,a)}$

* Step 2: Value Update: 

  ${\forall}s,v_{k+1}(s)={\sum}_{a}{\pi_{k+1}}(a|s){{\sum}_{s'}p(s'|s,a)[r(s,a,s')+{\gamma}v_k(s')]}=max_{a}q_k(s,a)$

需要注意的是, 迭代过程中的 $v_k$ 不保证为合法的值函数, 没有对应的策略也不满足 Bellman Equation

$v_k$ 是求解目标 $v^*$ 过程中的中间状态变量, 标识求解 $v^*$ 第k次迭代的变量的中间值, 同样 $q_k$ 也只是工具变量

###### Policy Iteration (PI)

重要的最优策略求解框架

类似于 VI, 不过 PI **每次迭代中会求解出当前策略合法的值函数**, 基于合法值函数此进行 PU

#算法步骤

init $\pi_0$, 给定当前策略 $\pi_k$

* Step 1: Policy Evaluation (PE): $v_{\pi_k}=r_{\pi_k}+{\gamma}P_{\pi_k}v_{\pi_k}$

  PE, 使用 Bellman Equ 迭代方法求解 $\pi_k$ 的值函数 $v_{\pi_k}$, $v_{\pi},q_{\pi}$下标代表该值函数是有对应策略的合法值函数

  迭代直到 $v_{\pi_k}^{j}$ 收敛, 然后由 $v_{\pi_k}$ 推导得到 $q_{\pi_k}$

* Step 2: Policy Update (Improvement) (PU): $\pi_{k+1}=argmax_{\pi}(r_{\pi}+{\gamma}P_{\pi}v_{\pi})$

  这里使用的是合法的值函数

  ${\forall}s,{\pi}_{k+1}(s)=arg~max_{\pi}{\sum}_{a}{\pi}(a|s){{\sum}_{s'}p(s'|s,a)[r(s,a,s')+{\gamma}v_{\pi_k}(s')]}=arg~max_{\pi}{\sum}_{a}{\pi}(a|s)q_{\pi_k}(s,a)$

  可以观察到, 在更新策略的过程中我们实际上使用的是 $q_{\pi_k}(s,a)$ 作为更新的依据

#算法实现 -- element-wise form

给定当前策略 $\pi_k$

* Step 1: Policy Evaluation: 

  iter: ${\forall}s,v_{\pi_k}^{j+1}(s)={\sum}_a{\pi}(a|s){\sum}_{s'}[r(s,a,s')+{\gamma}v_{\pi_k}^j(s')],{\text{until converge($j{\geq}n$ or $||v_{\pi_k}^{j+1}-v_{\pi_k}^j||{\leq}{\epsilon}$)}}$

  计算 $q_{\pi_k}(s,a)$: ${\forall}(s,a),q_{\pi_k}(s,a)={\sum}_{s'}p(s'|s,a)[r(s,a,s')+{\gamma}v_{\pi_k}(s')]$

* Step 2: Policy Update

  ${\forall}s,{\pi}_{k+1}(s)=arg~max_{\pi}{\sum}_{a}{\pi}(a|s){{\sum}_{s'}p(s'|s,a)[r(s,a,s')+{\gamma}v_{\pi_k}(s')]}=arg~max_{\pi}{\sum}_{a}{\pi}(a|s){q_{\pi_k}(s,a)}$

###### Truncated Policy Iteration

观察 VI 和 PI, 其本质差异在于 PU/PE 步骤对当前策略的值函数的求解精度, 是一个粗略高误差的 estimation 还是合法的值函数

当精度最低时, 我们仅进行一次 PE 迭代

* N-times PE iter (Value Update)

  N次迭代 PE, estimate 当前策略的值函数
  
* PU

  使用得到的 estimator (精度取决于N) 对当前策略进行优化 (argmax 操作)

###### ⭐General Policy Iteration (GPI)

泛指, 有着类似 PI 形式的, 迭代优化策略方法框架

#框架

此类方法可以概括为两个步骤

* Step 1: PE

  使用 estimator 估计策略, 目标是最终得到动作值函数的 estimation

* Step 2: PU

  基于 estimator 优化策略

##### [Base] 蒙特卡洛 Monte-Carlo MC

基于 大数定理, 我们可以通过采样来估计期望值, 由此来进行基于数据的无模型的学习
$$
\begin{align}
	&\mathbb{E}(x)\approx\frac{1}{N}\sum^{N}_{i=1}(x_i),N\rightarrow\infty,x_i满足iid\\
	&e.g.\underbrace{\Bbb E_{\tau\backsim p_{\theta}(\tau)}[G(\tau)\nabla_{\theta}\log p_{\theta}(\tau)]}_{\tau\backsim p_{\theta}(\tau)空间过大~不适合直接求期望}\approx\frac{1}{N}\sum_{n=1}^N[G(\tau^n)\nabla_{\theta}\log p_{\theta}(\tau^n)]
\end{align}
$$

通过 MC 的得到的 estimation 是一个无偏估计

###### MC-Basic

基于 Policy Iteration 的无模型方法

由于没有模型, 也无法通过状态值函数推导动作值函数: $q_{\pi}(s,a)=\mathbb{E}_{s'\backsim p(s'|s,a)}[r(s,a,s')+\gamma V(s')]$

同时, 观察 PI 的策略优化过程, 可以发现, 能够通过 $q_{\pi_k}(s,a)$ 避免显式使用模型 ${\forall}s,{\pi}_{k+1}(s)=arg~max_{\pi}{\sum}_{a}{\pi}(a|s){q_{\pi_k}(s,a)}$

所以, 在 Model-free 的 PI 中, 通过 MC 直接求 $q_{\pi_k}(s,a)$, 然后在 PI 阶段直接使用 $q_{\pi_k}(s,a)$ 进行策略优化

#算法步骤

* 给定策略 $\pi$

  ${\forall}(s,a){\in}S{\times}A,\text{with N samples: }q_k(s,a)\approx\frac{1}{N}\sum_{i=1}^Ng(s,a)$

  $q_k$ 标识第 k 次迭代的估计. $g(s,a)$ 标识采样得到的折扣后回报
  
  对每个 $(s,a)$ pair, 均需要进行充分次数的采样, 然后以此估计 $q(s,a)$
  
  然后基于 $q(s,a)$, 更新策略 ${\forall}s,\pi_{k+1}(s)=argmax_a[q(s,a)]$

###### MC-Exploring Starts

在 Policy Improvement 阶段, 我们需要所有 $(s,a)$ pair才能实现有效的更新

为了保证每个 $(s,a)$ pair 都得到了访问采样, 我们选择对所有 $(s,a)$ 为起点, 均采样足够条轨迹

相较于 MC-Basic, 会通过 first/every visitation 策略, 让每条轨迹都能对各个 $(s,a)$ 提供样本

###### MC-Epsilon Greedy

为了提高效率, 避免过多的采样, 将策略改为 soft Policy

这样, 在一条足够长的轨迹中, 我们就可以采样到所有的 $(s,a)$ pair

##### [Base] 时序差分 *Temporal Difference* TD

理论基础: Stochastic Approximation, Stochastic Sequence Convergence

Also based on Policy Improvement

基本框架仍然是 GPI:

* PE:

  核心在于, 通过 Stochastic Approximation 的方法进行 PE

  数学基础是, 直接使用 Bellman + 期望定义, 然后引入 Stochastic Approximation 迭代求解 $q_\pi$

* PU:

  同 GPI, 基于 $q_\pi$ 优化

###### TD-Basic

###### SARSA

###### Taular Q

#算法实现:

对于一步采样$(s,a,s',r)$(数据可复用, off-policy)

* $target:r+\gamma {max}_{a'}Q_k(s',a')$
* $update:Q_{k+1}(s,a)\leftarrow Q_k(s,a)+\alpha(target-Q_k(s,a))$

##### [1] Value-Based Roadmap

> 对 $S$ 无特定要求, 适用于离散有限动作域 $A$
>
> Implementation 中常设计Q网络的输出段大小为 $|A|$ , 选择输出值最大的Action即可

Value-Based 在于对策略的优化都是基于值函数的指导的

之前的内容也都属于 Value-Based

其特点是: 基于**值函数**, 进行**迭代式**的更新

主要套路则在于, 如何进行 PE 步骤. PI 步骤则较为简单, 基于动作值函数进行max更新即可.

#PE 步骤 -- 如何得到值函数

* DP: 在有 Env 模型的情况下, 基于 Bellman Equation 求解值函数
* Model-Free 在没有模型的情况下, 通过**期望形式**的值函数定义进行求解
  * 核心就是, 如何求 $\mathbb{E}[X]$
  * Monte-Carlo: 基于大数定理, 进行多次采样计算平均值求解
  * TD: 基于 Stochastic Approximation Theory, 
* Model-Based: TBD

对E[X]的求解方式

* 基于定义 sum_x p(x)x
* 基于MC
  * 大数定理
  * 大量sample求平均
* 基于SA
  * incremental
  * 迭代算法

###### Deep Q-Learning DQN

> paper: https://arxiv.org/abs/1312.5602
>
> code: 

首次在 RL 中引入 DNN 并取得显著成果

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
  * 实际上会维护两个网络, main 和 target, targetNet 用于提供计算loss时的target值
  * 会"冻结" targetNet 的参数, 避免其变化过大以稳定学习目标
    * 对 targetNet 使用多步更新或者软更新(soft update)
    * soft update: $\theta_{target}\leftarrow\theta_{traget}+\tau\theta_{main}(for~example,\tau~can~be~0.005)$
* 经验回放 *experience replay*
  * 构建一个经验池来去除数据相关性, 同时提高数据利用率 -- off-policy
  * 实践中即为Replay-Buffer: $(s,a,s',r,terminal)$

##### [2] ⭐ Policy-Based Roadmap $\pi_\theta(a|s)$

> 需要注意的是, 相较于 Value-Based, Policy-Based 有着更小的定义域, 更平滑的计算空间

###### [Base] Policy Gradient 策略梯度推导

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

#算法

* 在策略$\pi_\theta$下采样N条轨迹(典型on-policy)
* $计算梯度:\nabla_{\theta}J(\theta)=\frac{1}{N}\sum_{n=0}^{N-1}\sum_{t=0}^{T_n-1}G_t^n\nabla_{\theta}\log \pi_{\theta}(a_t^n|s_t^n)$
* $Gradient~Ascent:\theta'\leftarrow\theta+\alpha\nabla J(\theta)$
* 重复至收敛

#实践中定义loss为负数, 以便使用GD工具

* $loss=-\frac{1}{N}\sum_{n=0}^{N-1}\sum_{t=0}^{T_n-1}G_t^n\log \pi_{\theta}(a_t^n|s_t^n)$
* $GD: \theta'\leftarrow\theta-\alpha\nabla loss$

* with baseline

###### Trust Region Policy Optimization (TRPO)

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

###### Proximal Policy Optimization (PPO)

> paper: https://arxiv.org/abs/1707.06347
>
> code: done

基于TRPO思想的工程上的可行高效实现

#重要性采样
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
#公式推导 *从Policy Gradient Loss的替代优化到重要性采样*
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
#Loss处理
$$
\begin{align}
  Loss
  	&=-\frac{1}{N}\sum_{n=0}^{N-1}\sum_{t=0}^{T_n-1}A^{GAE}_{\theta'}(s_t^n, a_t^n)\frac{\pi_{\theta}(a_t^n|s_t^n)}{\pi_{\theta'}(a_t^n|s_t^n)}\\
  	&以PPO2为例,引入clip限制项\\
  	&=-\frac{1}{N}\sum_{n=0}^{N-1}\sum_{t=0}^{T_n-1}\min({A^{GAE}_{\theta'}\frac{\pi_{\theta}(a_t^n|s_t^n)}{\pi_{\theta'}(a_t^n|s_t^n)}},{clip(\frac{\pi_{\theta}(a_t^n|s_t^n)}{\pi_{\theta'}(a_t^n|s_t^n)},1-\epsilon, 1+\epsilon)}A^{GAE}_{\theta'})\\
  \end{align}
$$

#Loss GD

* $Loss_{PolicyNet}=-\frac{1}{N}\sum_{n=0}^{N-1}\sum_{t=0}^{T_n-1}\min({A^{GAE}_{\theta'}\frac{\pi_{\theta}(a_t^n|s_t^n)}{\pi_{\theta'}(a_t^n|s_t^n)}},{clip(\frac{\pi_{\theta}(a_t^n|s_t^n)}{\pi_{\theta'}(a_t^n|s_t^n)},1-\epsilon, 1+\epsilon)}A^{GAE}_{\theta'})$
* $Loss_{ValueNet}=MSE(TD\_target, V(s_t))$

