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
  
* KL散度 *KL diverge* $D_{KL}(P||Q)$

  > 详解见BV1r6jHzpE1J

  * $$
    用分布Q近似分布P: D_{KL}(P||Q) = \begin{cases}
    \displaystyle\sum_xP(x)\log\frac{P(x)}{Q(x)},&离散\\
    \displaystyle\int P(x)\log\frac{P(x)}{Q(x)}dx,&连续
    
    \end{cases}
    $$

  * 描述两个分布的差别

  * 恒大于等于0, 为0时代表同分布

    * 证明: Jensen不等式
    * 

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
  
* 模型 Model

  * 即Env的运行规律, alias"环境动力学模型"
    * RL里一般特指$Model = \{r(s,a,s'), p(s'|s,a)\}$
      * 状态转移概率: $p(s_{t+1}|s_t, a_t)$
        * 智能体根据当前状态$s_t$做出一个动作$a_t$之后，下一个时刻环境处于不同状态$s_{t+1}$的概率
        * 一般由环境决定, 一般为Monte Carlo过程
    * 一般难以得知
    * 可以模拟
  * Model-Based/Model-Free: 区别在于环境知识的掌握程度
    * Model-Based: 显式使用Model中的函数
    * Model-Free: 不依赖Model, 通过采样$(s,a,s',r)$进行学习

* 动作 Action
  
  * Agent能采取的行动
  
* 策略 Policy $\pi_{\theta}(a|s)$

  * 决定在状态s下采取什么行动a
  * 假设策略使用的模型的参数集为$\theta$

* 轨迹 Trajectory $\tau$
  * 马尔可夫决策过程的一个轨迹（trajectory）
  * 例: $\tau = \{s_0, a_0, s_1, r_1(s_0, a_0, s_1), \cdots, s_{T-1}, a_{T-1}, s_T, r_T(s_{T-1}, a_{T-1}, s_T)\}$
  * $\tau$的概率可以用策略选择和状态转移概率连乘表示
    * $p(\tau) = p(s_0) \displaystyle  \prod^{T-1}_{t=0}{\pi_{\theta}(a_t|s_t)p(s_{t+1}|s_t, a_t)}$
  
* Reward & Return
  * 奖励 Reward
    * $S: 状态空间; A: 动作空间$
    * Agent根据当前State做出一个Action, Env接收Action转移到新状态s', 并反馈给Agent一个奖励
      * 需要注意, 返回的奖励与到达的新状态也有关
        * 例如, 遇到障碍(s), 选择跳过去(a), 可能成功($s^{'}_1$)也可能摔倒($s^{'}_2$), 两种新状态reward不同
      * 奖励映射: $R : S × A × S’ → R$
      * 相当于函数: $R(s_t, a_t, s_{t+1})$
    * Reward是环境给Agent的奖惩反馈, 其定义会很大程度上影响模型的行为进而影响表现
  * 回报 Return
    * 给定策略$\pi_{\theta}(a|s)$，Agent和Env进行一次交互过程的轨迹$\tau$所收到的累积Reward为总回报Return
    * 轨迹$\tau$的总回报: $G(\tau) = \sum^{T-1}_{t=0}{\gamma^tr_{t+1}}$
      * $\gamma\in[0,1]$为折扣率
      * 当$\gamma\rightarrow 0$时，Agent更在意短期回报；而当$\gamma\rightarrow 1$时，长期回报变得更重要
    * 期望回报: $J(\theta)=\Bbb E_{\tau\backsim p_{\theta}(\tau)}[G(\tau)]=\Bbb E_{\tau\backsim p_{\theta}(\tau)}[\sum^{T-1}_{t=0}{\gamma^tr_{t+1}}]$
      * $\theta$为策略的参数集

* 值函数
  * 状态值函数 State-Value Function: $V_{\pi_{\theta}}(s)$
    * 从状态s开始，执行策略$\pi_{\theta}$得到的期望总回报
    * $V_{\pi_{\theta}}(s) = \Bbb E_{\tau\backsim p(\tau)}[\sum_{t=0}^{T-1}{\gamma^tr_{t+1}}|\tau_{s_0} = s]$
    * Bellman: $V_{\pi_{\theta}}(s) = \sum_{a\backsim\pi_{\theta}(a|s)}\sum_{s'\backsim p(s'|s,a)}[r(s,a,s')+\gamma V_{\pi_{\theta}}(s')]$ 
      * 思路: 树形分散 对于所有可能路径
        * $\sum_{a\backsim\pi_{\theta}(a|s)}$: $\pi_{\theta}$在状态s下可能采取的所有a
        * $\sum_{s'\backsim p(s'|s,a)}$: 在状态s采取行动a后可能转移到的所有新状态$s'$
  * 状态-动作值函数 State Action-Value Function: $Q_\theta(s, a)$
    * 初始状态为s并进行动作a，然后执行策略$\pi_{\theta}$得到的期望总回报。
    * $Q_{\pi_{\theta}}(s,a)=\Bbb E_{s^{'}\backsim p(s^{'}|s, a)}[r(s, a, s^{'})+\gamma V_{\pi_{\theta}}(s^{'})]$
    * Bellman: $Q_{\pi_{\theta}}(s,a)=\Bbb E_{s^{'}\backsim p(s^{'}|s, a)}[r(s, a, s^{'}) + \gamma \sum_{a^{'}\backsim \pi_{\theta}(a'|s')}[Q(s',a')]]$
  * 优势函数 Advantage Function: $A_\theta(s, a) = Q(s, a) - V(s)$

* 最优策略 $\pi^*$
  * $\forall s, \pi^* = {arg~max}_\pi V_{\pi}(s)$
  * 难以实现
  
* 策略改进
  * 值函数可以看作是对策略$\pi$的评估
  * 如果在状态s，有一个动作$a^*$使得$Q_{\pi_{\theta}}(s,a^*) > V_{\pi_{\theta}}(s)$，说明执行动作$a^*$比当前的策略$\pi_{\theta}$要好，我们就可以调整参数使得策略$\pi_{\theta}(a^*|s)$的概率增加
  
* 值函数计算
  * 基于模型RL
    * 基于MDP
    * 策略迭代
    * 值迭代
  * 模型无关RL
    * 无MDP过程
    * Monte-Carlo 采样
    * 时序差分
  
* $\epsilon-贪心法$

  * $$
    \pi_{\theta}^{\epsilon}(s) = \begin{cases}
    \pi(s), & 按概率1-\epsilon, \\
    randomly~selected~action~in~A, & 按概率\epsilon.
    \end{cases}
    $$

* 方法概论:
  * 值函数估计
    * 发展路径
      * Q学习: $Q(s_t, a_t)$
        * 只学习Q函数, 决策时, 选择能带来最大Q值的Action
      * DQN
        * 目标网络冻结（freezing target networks），即在一个时间段内固定目标中的参数，来稳定学习目标
          * 多步更新一次或者软更新(soft update)
            * soft update: $\theta_{target}\leftarrow\theta_{traget}+\tau\theta_{main}(for~example,\tau~can~be~0.005)$
        * 经验回放（experience replay），构建一个经验池来去除数据相关性。
          * 实践中即为ReplayBuffer
      * Double Q
  * 策略搜索 $\pi(a_t|s_t)$
    * 无梯度方法
    
    * 策略梯度: 基于策略函数$a=\pi_{\theta}(s)$的学习
      
      * 传统策略梯度 Policy Gradient
      
        * 目标: 找到是的期望总回报最大的$\theta$
      
        * $$
          \begin{align}
              \nabla_{\theta}J(\theta)
              &=\nabla_{\theta}\Bbb E_{\tau\backsim p_{\theta}(\tau)}[G(\tau)]\\
              &=\nabla_{\theta}\sum_{\tau}[G(\tau)p_{\theta}(\tau)]\\
          	&=\sum_{\tau}[G(\tau)\nabla_{\theta}p_{\theta}(\tau)]\\
          	&=\sum_{\tau}[G(\tau)\nabla_{\theta}p_{\theta}(\tau)\frac{p_{\theta}(\tau)}{p_{\theta}(\tau)}]\\
          	&=\sum_{\tau}[p_{\theta}(\tau)*G(\tau)\nabla_{\theta}\log p_{\theta}(\tau)]\\
          	&=\Bbb E_{\tau\backsim p_{\theta}(\tau)}[G(\tau)\nabla_{\theta}\log p_{\theta}(\tau)]\\
          	&\approx\frac{1}{N}\sum_{n=1}^{N}[G(\tau^n)\nabla_{\theta}\log p_{\theta}(\tau^n)] &(Monte-Carlo)\\
          	&\cdots &(余见PPO笔记, 纯练写公式了\ldots)\\
          	&=\frac{1}{N}\sum_{n=1}^N\sum_{t=1}^TG(\tau^n)\nabla_{\theta}\log \pi_{\theta}(a_t|s_t)
          \end{align}
          $$
      
          
      
        * 理解: 参数θ优化的方向是使得总回报$G(\tau)$越大的轨迹$\tau$的概率$p_{\theta}(\tau)$也越大
      
      * REINFORCE
      
      * TRPO: 引入 Trust Region
      
      * PPO *Proximal Policy Optimization* 
      
        * 
    
  * Actor-Critic(结合二者)
  
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



##### With LLM

* RLHF
  * RL on Human Feedbacksx