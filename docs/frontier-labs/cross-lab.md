# 六家前沿实验室跨谱系综合：不要比较品牌，要比较资源、状态、监督与证据

本章综合 [DeepSeek / Kimi](deepseek-kimi.md)、[GLM](glm.md)、
[OpenAI](openai.md)、[Anthropic](anthropic.md) 与
[Google DeepMind / Gemini](gemini.md) 的公开证据。它不再逐条复述 3,050 条书目记录，
而是回答一个更难的问题：**六家实验室究竟在解决哪些相同的约束，采用了哪些不可
互换的技术路线，哪些差异来自技术本身，哪些只是披露制度不同？**

所有总数与归属计数以 2026-07-19 的[覆盖快照](coverage.md)为准；记录数、主题标签数和
PDF 数衡量的是这套采集规则下的**可观察语料**，不是机构能力、真实产出或安全性的代理。

> 证据标签沿用 [D]/[C]/[R]/[I]/[U]。跨实验室关系通常是 [I]，即使构成它的每个
> 单点事实都是 [D]。把“两个团队都公开了 Muon”写成“一个复制另一个”，或把
> “同一 benchmark 名称”写成“协议可比”，都越过了证据边界。

## 1. 先纠正比较对象：这不是六个同构样本

| 对象 | 目录中的主要可见单元 | 最强公开面 | 最容易误读的地方 |
|---|---|---|---|
| DeepSeek | 高密度旗舰技术报告 + 方法/系统论文 | 参数表、active experts、tokens、低精度、训练系统、RL 阶段 | 把独立研究架构 NSA 当成旗舰 DSA；把 R1 当作 GRPO 起点 |
| Moonshot / Kimi | 旗舰报告 + 优化器/attention/agent 研究 | K2/K2.5、Moonlight、Kimi Linear、agent Gym 与长轨迹 | 因同属一家就断言 K2 使用 MoBA，或断言 K3 完全沿用 Kimi Linear 配方 |
| Zhipu / GLM | 旗舰与多模态专用模型 + 大量直接方法研究 | GLM-4.5/5 的 MoE、Muon、agent 数据与 RL，图像/视频/语音/OCR 支线 | 用产品博客补齐未披露的训练账本；混淆 config、API 与有效上下文 |
| OpenAI | 早期论文 + 后期系统卡/部署研究/产品研究 | 偏好学习、scaling、reasoning、tools、安全评测与部署制度 | 从产品名反推隐藏架构/数据/优化器；把 system-card 分数当综合安全概率 |
| Anthropic | 对齐/可解释性论文 + 系统卡/风险报告/治理页面 | CAI/RLAIF、mechanistic interpretability、红队、RSP 与风险报告 | 把 constitution 当客观价值函数；把一次评测或 ASL 分级当零风险证明 |
| Google DeepMind / Gemini | Gemini core cards + 近两千条机构署名长尾 | RL/搜索、Alpha 系列、科学/机器人、多模态、安全框架 | 把全部 GDM 署名论文称为 Gemini 训练组件；把 Alpha 方法族当一个模型 |

因此，本章只在**共同技术变量**上比较，不拿记录总数代表模型能力，也不拿披露篇幅
代表绝对安全性。Google DeepMind 的 1,943 条 affiliated 记录显示本快照保留了很长的
机构署名尾部；DeepSeek 的 22 条 core 则显示其清单更集中在旗舰报告。两者不是同一
统计量，也不能推出两家真实研究规模、能力或披露质量的强弱。[C/I]

## 2. 一个统一分析框架：四类资源、四层状态、五种监督

### 2.1 模型不是参数总数，而是资源分配函数

对 MoE 或混合稀疏模型，至少要记录：

$$
\rho_{\text{act}}=\frac{P_{\text{active/token}}}{P_{\text{total}}},\qquad
C_{\text{token}}\approx C_{\text{dense}}+C_{\text{expert}}
+C_{\text{attention}}(L)+C_{\text{routing}}+C_{\text{communication}}.
$$

同样的总参数可以对应完全不同的每 token 激活量、expert parallel 通信、KV/state
占用和训练稳定性。DeepSeek-V2/V3/V4、Kimi K2/K3、GLM-4.5/5 都把稀疏性放进
旗舰路线 [D]；OpenAI、Anthropic 与后期 Gemini 的公开卡片通常不足以计算
$\rho_{\text{act}}$ [U]。所以“1T 比 700B 大多少”往往不是可回答的效率问题。

### 2.2 长上下文不是长度，而是四层状态系统

长时程系统至少包含：

1. **权重内状态**：参数化知识与策略；
2. **序列内状态**：attention KV、latent KV、selected tokens、recurrent state；
3. **序列外状态**：retrieval index、文件、数据库、browser cache、agent memory；
4. **组织状态**：多个 worker 的独立上下文、共享工件、checkpoint 与任务队列。

把成本写成向量更诚实：

$$
\mathbf C_{\text{context}}(L)=
(\text{attention FLOPs},\ \text{KV/state bytes},\ \text{selection cost},
\ \text{storage I/O},\ \text{compaction loss}).
$$

DeepSeek 旗舰的 MLA→DSA→CSA/HCA（V4 为 `deepseek-2606.19348`），Kimi Linear 的
KDA+MLA（`kimi-2510.26692`），以及 GLM-5 的 MLA→DSA 与 GLM-5.2 的 IndexShare
（`arxiv-2602.15763`、`official-research-161`）分别优化这个向量的不同坐标 [D]。
MoBA（`kimi-2502.13189`）和 IndexCache（`arxiv-2603.12201`）是同机构的独立方法记录，
不能仅凭归属写进 K2/K2.5 或 GLM-5/5.2 的旗舰配方 [D/U]。OpenAI deep research/Codex、
Anthropic 长时程 agents 与 Gemini computer-use 则更多从序列外和组织状态暴露系统行为
[D/C]。它们不能只用“1M context”一列排序。

### 2.3 监督来源至少有五类

| 监督 | 可观察信号 | 典型公开路线 | 主要失败模式 |
|---|---|---|---|
| 人类示范/偏好 | demonstration、pairwise ranking、rubric | OpenAI preference learning/InstructGPT；Anthropic HHH/RLHF | 标注分布偏差、reward-model 外推、昂贵且慢 |
| AI 反馈/规则规范 | critique、revision、AI preference、constitutional rule | Anthropic CAI/RLAIF；OpenAI deliberative alignment/RBR | 规范冲突、模型自举偏差、规则覆盖空洞 |
| 可验证奖励 | exact answer、compiler/test、proof checker、environment success | DeepSeek GRPO/R1；Kimi/GLM code/math RL；OpenAI code/math 与 process-supervision 研究（旗舰 reward mixture [U]）；Alpha 系列 | reward hacking、judge leakage、可验证任务选择偏差 |
| 过程/轨迹监督 | step labels、process RM、tool trace、failure localization | OpenAI process supervision；多家 agent trajectory pipelines | 中间步骤不忠实、长轨迹 credit assignment、policy lag |
| 风险/红队信号 | attack success、capability threshold、monitor/classifier outcome | Anthropic classifiers/RSP；OpenAI system cards；GDM FSF/cards | threat-model 漂移、adaptive attack、负结果难解释 |

不同团队常把这些监督叠加。差异不只是“用了 PPO 还是 GRPO”，而是**谁产生问题、
谁产生回答、谁判分、何时更新 policy、失败轨迹是否留下、工具环境是否版本固定**。

## 3. 架构与 scaling：共同趋势是条件计算，分歧在条件放在哪里

### 3.1 专家稀疏：容量扩张不等于每 token 成本等比增长

- DeepSeekMoE/V2/V3 把细粒度 routed experts 与 shared experts 结合；V2 又加入 MLA。
  [DeepSeek-V2](https://arxiv.org/abs/2405.04434) 披露 236B total/21B active、8.1T
  tokens 和两阶段 online GRPO（库存 `deepseek-2405.04434`，§§2–4、Tables 1–3）[D]。
- [Kimi K2](https://github.com/MoonshotAI/Kimi-K2/blob/main/tech_report.pdf)
  披露 1.04T/32.6B active、384 routed experts、15.5T tokens 与 MuonClip
  （`kimi-2507.20534`，架构/训练表）[D]。
- [GLM-4.5](https://arxiv.org/abs/2508.06471) 披露 355B/32B active、23T tokens、
  Muon 和 agentic SFT/RL 数据—环境管线（`arxiv-2508.06471`）[D]；GLM-5 再引入
  MLA→DSA 转换、Muon Split 与共享参数的 MTP（`arxiv-2602.15763`）[D]。

**[I] 共同约束：** 三家公开 MoE 旗舰都把“路由质量、load balance、通信、优化器
尺度、激活精度”当成一个系统问题，而不是只画 expert 图。若控制总参数和训练 tokens
后，路由/通信的独立干预对质量与墙钟都无影响，这个判断会被反驳。

### 3.2 序列稀疏和状态压缩：第二条条件计算轴

- MLA 压缩每 token 的 KV state；
- DSA/NSA/CSA 选择或压缩历史 token；
- MoBA 做 query-dependent block routing；
- KDA 把历史递归进固定大小状态，并用部分 MLA 层保留精确检索；
- IndexShare/IndexCache 在层间或阶段间复用选择结果；
- external memory/retrieval 把部分状态移出模型序列。

这些机制不是同义词。尤其 [Native Sparse Attention](https://arxiv.org/abs/2502.11089)
（`deepseek-2502.11089`）是 compression/selection/sliding 三路的原生可训练研究架构；
V3.2 DSA 是 lightning indexer + selected attention 的旗舰转换路径 [D]。
[MoBA](https://arxiv.org/abs/2502.13189)（`kimi-2502.13189`）也不能因为出自
Moonshot 就自动写进 K2/K2.5 配方 [U]。

### 3.3 闭源卡片的可比上限

OpenAI、Anthropic 和后期 Gemini/Claude/GPT 的系统卡可以确认版本、模态、部分
context/tool interface、capability 与 safety evaluations [D/C]，但通常不披露足以计算
active parameters、训练 tokens、optimizer state 或训练 FLOPs 的参数表 [U]。这不是
“一定没有技术创新”，也不是“参数不重要”，而是**公开证据不能支持同粒度架构横排**。

## 4. 优化、精度与训练系统：算法名称只是最外层标签

### 4.1 Muon 的跨实验室扩散能确认到哪里

[Muon is Scalable](https://arxiv.org/abs/2502.16982) 用 Moonlight 研究模型公开
distributed orthogonalization、update scaling、weight decay 与约 5.7T-token 训练
（`kimi-2502.16982`）[D]。K2 加入 QK-Clip，监测 attention logit 并在越阈时调整
query/key projection；GLM-4.5 对大多数矩阵采用 Muon，GLM-5 进一步公开 Muon Split；
DeepSeek V4 后来把多数矩阵从 AdamW 切到 Muon，同时保留 embeddings/head/norm 的
AdamW [D]。

**可以说：** Muon 已成为多个公开大模型配方中的真实优化分支 [C/D]。

**不能说：** 四个实现具有相同的 shape scaling、Newton–Schulz 数值、分布式切分、
clip 逻辑或普适收敛优势 [U]。若只复刻“optimizer=Muon”却不复刻 precision、routing、
QK stability、batch/schedule 和 failure recovery，实验并没有复现旗舰训练。

### 4.2 低精度是数值协议，不是 dtype 开关

DeepSeek-V3 把 FP8、fine-grained scaling、DualPipe 和 expert parallel 联合披露；V4
明确对 routed-expert weights 与 CSA indexer 的 QK path 做 FP4（MXFP4）QAT
（`deepseek-2606.19348`）[D]。Kimi K3 页面只在高层列出 MXFP4 weights / MXFP8
activations（`kimi-k3-note`）[D/U]；GLM 与西方闭源卡片的同粒度训练数值证据较少 [U]。
真正的复现记录至少应包含：哪些 tensor 量化、scale 粒度、accumulator、master weights、
通信 dtype、overflow/underflow 处理与 fallback kernel。

### 4.3 系统决定“能否 on-policy”

对 agent/RL 轨迹，样本分布更准确地写作：

$$
\tau\sim P(\tau\mid \theta_{\text{serve}},E,T,M,S),
$$

其中 $E$ 是 environment 版本，$T$ 是 tools，$M$ 是 context/memory manager，
$S$ 是 sampling/serving stack。训练权重 $\theta_{\text{train}}$ 与服务权重、量化、
路由或工具版本不同，就会出现 policy lag。因而“算法声称 on-policy”不证明整条系统链
on-policy [I]。

DeepSeek V3.2 的 Keep Routing/Keep Mask 追求训练—部署 routing fidelity；Kimi 的
partial rollout 保存未完成状态，而 PARL 训练 orchestrator；GLM-5 的异步 rollout 明确引入
policy lag，OPD 则改变 student/teacher 的采样关系 [D]。它们都影响轨迹分布，却不是同一种
“on-policy 修复”；OpenAI/Anthropic/GDM 的生产 agent stack 又未披露到同等粒度 [U]。

## 5. 后训练与 reasoning：六条路线不是一张 PPO 家谱

### 5.1 DeepSeek：从领域 GRPO 到 specialist consolidation

[DeepSeekMath](https://arxiv.org/abs/2402.03300) 先于 R1 公开 GRPO：同题多回答、
组内相对优势、规则/模型奖励（`deepseek-2402.03300`）[D]。R1-Zero 展示无 cold-start
SFT 的 accuracy + format reward 路线；R1 加入 cold-start、reasoning SFT 与 rejection
sampling。V3.2 将 specialist 轨迹、thinking-with-tools 与 mixed GRPO 连接；V4 的最终
consolidation 又转向 multi-teacher on-policy distillation，而不是简单“把所有 specialist
继续一起 GRPO” [D]。

### 5.2 Kimi：从长到短、可验证 RL 到 agent orchestration

Kimi k1.5 强调 long-CoT RL、long-to-short 与采样策略；K2/K2.5 加入 tool Gym、
self-critique、agentic SFT、可验证环境和并行 orchestration；Kimi-Researcher 把 browser/
search/code、context manager 与异步/partial rollout 放进端到端轨迹 [D]。这些公开 objective、
segment masking、Toggle/PARL 和 orchestration 与 DeepSeek GRPO 有交叉却不等价，把整条
路线压缩成“也是 GRPO”是跨层归类 [D/I]；未披露的超参与生产实现仍是 [U]。

### 5.3 GLM：expert、distillation、environment factory 与 compaction-aware RL

GLM-4.5 公开 expert 模型、轨迹工厂与多阶段 agentic RL；GLM-5 将 reasoning、coding、
agent RL 顺序化并讨论异步 rollout、train/inference mismatch 与 OPD；GLM-5.2 把
context compaction 后的 credit assignment 放入 critic/训练问题 [D]。它最有价值的公开点
不是某一 benchmark 分数，而是承认**context manager 会改变训练分布**。

### 5.4 OpenAI：从偏好学习到 train-time RL × test-time compute

OpenAI 的公开主线依次包含人类偏好比较、summarization/WebGPT/InstructGPT、process
supervision、rule-based rewards/instruction hierarchy、o1 reasoning 与 deliberative
alignment [D]。[o1 system card](https://openai.com/index/openai-o1-system-card/) 披露
训练时 RL 和测试时 reasoning 计算随能力提升，但没有公开足以复制的 optimizer、reward
mixture、KL、rollout volume 或硬件账本 [D/U]。

test-time compute 至少要分开：单轨迹延长、并行采样、投票/聚合、verifier/search、
tool interaction。它们有不同的成本、相关误差和失败模式，不能都写成“多想一会”。

### 5.5 Anthropic：规范显式化与监督来源替换

[Constitutional AI](https://www.anthropic.com/research/constitutional-ai-harmlessness-from-ai-feedback) 把 constitution 驱动的自我批评/
修订 SFT 与 AI preference feedback/RLAIF 连接（`anthropic-arxiv-2212.08073`，Figure 1
p. 2、§1.2 p. 5）[D]。它改变的是监督生产过程，不是把价值冲突数学上消除。
Collective CAI、character/values 研究和 constitutional classifiers 继续把“谁写原则、原则
如何变成训练/防御信号、如何审计失败”拆开 [D]。

### 5.6 Google DeepMind：RL、搜索、world model 与 verifier 的长期组合

DQN、AlphaGo/AlphaZero、MuZero、SIMA、AlphaCode/AlphaGeometry/AlphaProof、
AlphaEvolve 不能被叫作一个算法，但共享一个长期设计空间：policy/value/model、搜索、
simulator/environment 与可执行 verifier 的不同组合 [D/I]。Gemini 具体 post-training recipe
在模型卡中的公开粒度通常低于这条研究前史 [U]；不能把所有 GDM RL 论文直接写成
Gemini 训练组件。

## 6. Agent 能力：从模型函数转成系统函数

Agent 评测对象应写成：

$$
Q_{\text{agent}}=
f(Q_{\theta},\ \text{tool protocol},\ \text{environment},\ \text{memory/context},
\ \text{orchestration},\ \text{permissions},\ \text{latency},\ \text{judge},\ \text{date}).
$$

### 6.1 四层分解

| 层 | 问题 | 六家公开证据的典型切片 |
|---|---|---|
| Policy | 是否会规划、选工具、纠错？ | reasoning/tool RL、agentic SFT、computer-use cards |
| Environment | 动作是否真实执行、反馈是否可靠？ | DeepSeek/Kimi/GLM Gym；OpenAI browser/code sandbox；SIMA/robotics |
| State | 长轨迹如何存、压缩、恢复？ | Kimi/GLM context manager；OpenAI compaction/memory；Anthropic long-running agents |
| Control plane | 多 worker、权限、审批、重试如何协调？ | Kimi swarm/PARL；Codex/deep research；科学/软件 agents |

**[I] 关键综合：** 当前 agent 上限越来越像四层瓶颈的最小值，而非单一模型分数。
如果在固定 policy 下，改变 environment fidelity、context manager 或 permissions 对长任务
成功率没有显著影响，这个判断会被削弱。

### 6.2 “会调用工具”不等于“能自主完成任务”

Function calling 测试的是 schema/argument 生成；GUI grounding 测坐标/元素选择；browser
agent 还要处理动态页面、引用、登录与 stale state；coding agent 需要 repository state、
编译测试、sandbox 与 patch review；科学 agent 还需仪器/数据库和外部验证。把这些任务
混为一个 agent benchmark 会隐藏真正的失败层。

## 7. 多模态、机器人与科学：统一接口不等于统一目标

### 7.1 “原生多模态”至少有三种含义

1. 预训练时共同 token/序列；
2. 共享 backbone，但模态 encoder/decoder 专用；
3. 产品接口统一，但底层由多个模型/路由器组成。

DeepSeek/Kimi/GLM 的公开报告在本快照中较常给出视觉 encoder、projector、token compression、
diffusion/AR decoder 或训练阶段 [D/I]；GPT/Claude/Gemini 卡片更常确认模态与评测而不
公开全部内部路由 [D/U]。因此“都支持图像”不能推出同一训练机制。

### 7.2 Perception 到 action 的关键跃迁

Kimi-VL/K2.5、GLM-4.5V/AutoGLM、OpenAI CUA/Operator、Claude computer-use 与
Gemini Robotics/SIMA 把视觉从理解输入变成 action observation [D/C]。新增风险不是
只看错图，而是错误动作通过权限、重复执行和外部状态累积。评测必须记录 action
space、reset、confirmation policy、hidden state 和 irreversible actions。

### 7.3 AI for science 的证据链必须分段

按冻结 inventory 的多标签计数，Google DeepMind 有 233 条 `science_biology` 与 573 条
`mathematics_science` 命中，AlphaFold、AlphaMissense/AlphaGenome、GNoME、
GraphCast/GenCast、fusion 与材料/生物长尾使其成为本快照中覆盖最宽的科学语料 [C/I]；
标签会重叠，且采集密度不同，因此不能外推为机构绝对领先。OpenAI、Anthropic、GLM、
DeepSeek/Kimi 也有数学、代码、OCR、科研 agent 或机构署名应用研究。强结论应按以下链
验证：

$$
\text{benchmark prediction}
\rightarrow \text{prospective validation}
\rightarrow \text{scientific/engineering use}
\rightarrow \text{causal or operational impact}.
$$

在第一箭头得分高不自动证明最后一项。数据库采用、湿实验、临床/工业 deployment 与
独立 replication 是不同等级证据。

## 8. Evaluation：横排前必须冻结协议

任何数值记录至少应绑定：

$$
M=f(\text{checkpoint},\text{prompt},\text{sampling},\text{tools},\text{context},
\text{budget},\text{judge},\text{date}).
$$

### 8.1 不可直接横排的常见情况

- base 与 chat/reasoning checkpoint；
- pass@1 与 best-of-N、投票或 verifier search；
- 无工具与 search/code/browser tools；
- 不同 context window、memory/compaction；
- benchmark 原版、clean/verified/private split；
- human judge、LLM judge 与可执行 exact judge；
- vendor 自报与独立复测；
- 不同 API 日期或静默更新。

GLM-4.5 首页、Kimi/DeepSeek 报告、OpenAI/Anthropic/Gemini 系统卡都提供大量结果 [D]，
但结果只有在上面变量足够一致时才支持性能排序。这个规则会牺牲“榜单一句话”，却能
防止大部分伪比较。

### 8.2 安全评测也不是一个总分

安全证据至少分为 capability、propensity、attack success、monitor/classifier detection、
safeguard effectiveness、deployment incident。低 harmful-answer rate 不等于低 autonomous
capability；高 capability 也不等于在当前产品权限下必然造成风险。

## 9. 安全、可解释性与治理：六家公开制度的结构差异

下表的制品身份是 [D/C]，对“连续性”“联动程度”和公开密度的横向概括是 [I]；未公开项
只能标 [U]，不能反推内部没有相应流程。

| 机构 | 公开安全制品的主要结构 | 可确认 [D/C] | 仍未知 [U] |
|---|---|---|---|
| DeepSeek | 模型报告中的 safety/alignment slices、部分评测与开源制品 | checkpoint 行为可复测，部分训练/奖励阶段公开 | 连续 system-card 制度、完整 threat model、部署监测/事件 |
| Kimi | flagship 报告、供应链/agent 验证与部分安全章节 | 部分 tool/agent failure protocol | 统一版本化风险框架、生产 safeguards 与 incident 统计 |
| GLM | 模型报告、model cards、benchmark、安全切片与工具研究 | 若干中文/英文/agent safety evaluations | 跨版本统一的 safety-card schema、阈值决策链和生产红队覆盖 |
| OpenAI | system cards、Preparedness、Deployment Safety、外部测试 | 版本化评测、部分 capability/safeguard 结果 | 完整训练 recipe、所有内部阈值/事件、跨版本可比性 |
| Anthropic | system cards、RSP/ASL、风险报告、红队、classifiers、interpretability | 最明确的治理—评测—风险报告联动之一 | 阈值校准、内部例外、所有部署事件、机制解释的完备性 |
| GDM / Gemini | model cards、Frontier Safety Framework、红队/provenance 研究 | cards 与框架、部分 dangerous capability/safeguard 评测 | 生产全链路、阈值触发细节、跨产品统一程度 |

**[I] 披露制度是技术系统的一部分。** 它决定外部研究者能否发现 regression、复核风险
主张和区分 model-level 与 deployment-level mitigations。但“披露多”不是“绝对更安全”，
“披露少”也不是“绝对不安全”；它首先改变证据不确定性。

### 9.1 可解释性不能替代行为审计

Anthropic 的 circuits、dictionary learning、circuit tracing/model diffing 构成本快照中最连续的
机制可解释性主线之一 [D/I]；Google DeepMind/OpenAI/中国实验室也有大量表示、探针、
归因或机制研究。feature 可命名不证明它是唯一因果变量，activation intervention 成功也
不证明全分布安全。机制证据、behavioral eval、red team 与 deployment monitoring 应互补。

## 10. 开放性：用向量，不用“开源/闭源”二分

定义一个制品向量：

$$
\mathbf O=(W,C,D,R,E,S),
$$

其中 $W$=weights，$C$=code/config，$D$=data provenance，$R$=training recipe，
$E$=evaluation protocol/artifacts，$S$=safety/deployment evidence。一个模型可以开放
权重却不开放数据，也可以闭源权重但公开详细风险评测。

下表是按冻结制品做的定性综合 [I]，不是开放度评分；“低/中/高”只比较当前可见材料，
缺失项仍是 [U]。

| 实验室 | 权重/代码可确认面 | recipe 公开密度 | safety/deployment 公开密度 | 综合边界 |
|---|---|---|---|---|
| DeepSeek | 多个核心 checkpoint/config/code [C] | core 报告相对高 [D] | 中低且不连续 | 技术复现强于治理复现 |
| Kimi | 多个模型/研究制品与代码 [C] | K2/K2.5、Moonlight/Kimi Linear 较高 | 分散 | agent 环境细节仍不完整 |
| GLM | 多模型/代码/配置 [C] | 旗舰和专用支线不均 | 分散 | 研究广，版本间粒度不一 |
| OpenAI | 少量 open models/research code，主产品闭源 | GPT-3 后显著降低 [U] | system-card/deployment 较高 [D] | 行为/治理证据强于训练复现 |
| Anthropic | 主模型闭源，研究代码/数据/页面不一 | 模型 recipe 低 [U] | 风险/治理/可解释性高 [D] | 机制与治理强，训练账本弱 |
| GDM / Gemini | 研究代码/模型/数据库丰富，Gemini 主模型闭源 | Gemini recipe 中低 | cards/FSF 中高 [D] | 机构研究开放面远宽于旗舰 |

这个表只比较当前公开制品，不对内部能力或动机作推断。

## 11. 九条跨实验室综合判断与反驳条件

### 11.1 稀疏性正在从参数扩展到序列、状态和时间 [I]

MoE 稀疏 capacity，MLA/KDA 压 state，DSA/MoBA 选历史，tool/retrieval 只在需要时访问
外部资源，multi-agent orchestration 按需分配 worker。若受控实验显示这些条件机制在同等
总计算下不改变质量—成本 Pareto frontier，该综合会被反驳。

### 11.2 优化器、attention 稳定性和并行拓扑已不可分 [I]

Muon/MuonClip/Muon Split、FP8/FP4、routing balance 和 expert parallel 的公开证据显示
大模型训练的“optimizer”是数值—通信协议。若跨规模实验只替换数学更新就能稳定复现
相同墙钟/质量，而 QK clip、precision 与 topology 无贡献，该判断会被削弱。

### 11.3 数据 scaling 正从静态语料表变成数据运算 [I]

迭代网页恢复、synthetic reasoning/tool trajectories、difficulty filtering、rejection、
curriculum 与 self-improvement 都在运行中重写训练分布。应报告生成器、过滤器、版本、
去重、污染和每阶段 mixture，而不只报 token 总数。

### 11.4 Verifier 决定哪些能力容易被 RL 扩展 [I]

代码、数学、proof、game/search 与部分工具任务有便宜反馈，开放 RL 进展集中于此并非巧合。
反例将是：在没有可靠 outcome/process verifier 的开放域，采用同等计算仍稳定获得同样的
RL scaling。

### 11.5 Test-time compute 的收益受相关误差约束 [I]

并行采样只有在错误不完全相关、aggregator 能识别好解时才有用；长单轨迹会受 context
污染；tool search 会受环境噪声。报告 N 或 token budget 但不报告相关性与选择器，会高估
“多算”的普适性。

### 11.6 Agent 的训练样本包含整个系统版本 [I]

同一 policy 在不同 browser、tool schema、sandbox、memory manager 与权限下产生不同轨迹。
如果 environment 版本未冻结，严格 on-policy 与可复现性都不成立。

### 11.7 Consolidation 是多能力时代的核心瓶颈 [I]

specialist RL、distillation 与 mixed updates 在回答“如何合并权重而不遗忘”；router/product
routing 则在系统层选择模型或预算，可能绕开而不是解决权重级 consolidation。DeepSeek
V3.2/V4、GLM-5 与 Kimi 多域/agent 路线提供前一类证据，OpenAI GPT-5 router
（`web-eb6095ff0b58`）提供后一类证据 [D/I]。两类都应逐域测正迁移、负迁移与 calibration，
但不能把 routing 当作模型已完成能力合并的证明。

### 11.8 Safety case 必须版本化且分层 [I]

模型权重、system prompt、tools、permissions、monitor 和 policy update 任一变化都可能使旧
风险结论失效。安全报告必须绑定 checkpoint 与 deployment configuration，不能成为永久
证书。

### 11.9 披露质量本身是研究变量 [I]

可复核参数表、protocol、negative results、版本差异与 artifact hashes 会降低外部不确定性。
未来比较实验室时应同时报告能力置信区间和**证据置信区间**。

## 12. 一个可执行的跨实验室比较模板

对任意两个模型/系统，按下面顺序填写；任一关键格为空就停止强排序。

1. **Identity**：checkpoint、发布日期、base/chat/reasoning、API 或 weight hash；
2. **Architecture**：total/active、layers/hidden/heads/experts、attention/state、modalities；
3. **Training**：tokens/mixture/provenance、optimizer/schedule/precision、hardware/FLOPs；
4. **Post-training**：SFT/preference/RL/distillation、reward/verifier、rollout与 policy lag；
5. **Runtime**：prompt、sampling、context、tools、memory、router、latency/retries；
6. **Evaluation**：dataset version、contamination、judge、budget、uncertainty、independent run；
7. **Safety**：threat model、capability、propensity、safeguard、adaptive attack、deployment；
8. **Artifacts**：weights/code/data/eval cards/hash/license；
9. **Evidence label**：每格标 D/C/R/I/U；
10. **Stop rule**：缺失项会使哪一种结论不可识别。

## 13. 研究议程：最值得补的不是另一张榜单

1. **Active-compute accounting**：统一报告 active params、attention/state、router 与通信，
   建立跨 MoE/linear/sparse attention 的每 token 成本基准。
2. **Long-context causal audit**：区分训练长度、检索准确、state bytes、compaction loss 与
   long-horizon task success。
3. **Optimizer transportability**：在固定数据/架构/精度下比较 AdamW、Muon 变体，并公开
   failure recovery 与 topology sensitivity。
4. **RL provenance**：对题目、轨迹、verifier、judge 与模型生成器做 versioned lineage，
   量化 reward leakage 和 self-contamination。
5. **Agent environment checksums**：给 tools、browser image、sandbox、fixtures 和 permissions
   可复现摘要；报告 trajectory 与环境版本的绑定。
6. **Consolidation matrices**：逐域报告 specialist→unified 的正/负迁移，而不是只报总平均。
7. **Adaptive safety evaluation**：攻击者知道防御后重新优化；同时测 capability、propensity、
   monitor 和实际权限。
8. **Card diff standard**：系统卡/模型卡版本之间提供机器可读差异、撤回项和新增风险。
9. **Prospective science validation**：把 benchmark、wet-lab/field validation 与实际 adoption
   分开注册和报告。
10. **Evidence-calibrated leaderboards**：分数旁边显示 protocol completeness 与公开证据
    等级，缺关键变量时拒绝精确排名。

## 14. 常见误述的证据裁决

- “GRPO 由 R1 首次提出。”——错；DeepSeekMath 已先公开。
- “NSA、DSA、MoBA、KDA 都是同一种 sparse attention。”——错；压缩、选择、块路由、
  递归状态的计算图不同。
- “K2 使用 MoBA。”——公开 K2 报告披露 MLA；同机构独立论文不等于旗舰采用 [U]。
- “Muon 在四家实现里相同。”——错；scaling、clip、split、shape/precision/topology 不同或
  未披露。
- “1M context 证明能解决 1M-token agent 任务。”——错；长度、有效检索、state 管理和
  task success 是四个变量。
- “system card 是完整训练报告。”——错；它通常证明行为/风险评测，不补齐训练 recipe。
- “所有 GDM 论文都是 Gemini 论文。”——错；1,943 条 affiliated 长尾保留机构参与边界。
- “开源权重等于可复现训练。”——错；数据、recipe、精度、并行与 eval artifacts 可能缺失。
- “闭源模型无法被科学研究。”——也错；可以研究可观察行为与部署制度，但因果结论必须
  限于接口，不得虚构内部实现。
- “一个安全分数可代表整体风险。”——错；threat model、capability、propensity、permissions
  与 safeguards 必须分层。

## 15. 综合结论

六家公开研究真正汇流的地方不是某个算法名字，而是四个约束：

1. **容量如何条件化分配**——experts、tokens、state、tools、workers；
2. **监督如何规模化且不被投机**——human/AI feedback、verifier、process、risk signals；
3. **长时程状态如何保持忠实**——KV/recurrent/external memory、compaction、environment；
4. **能力如何在部署中被测量和约束**——protocol、permissions、monitor、cards、governance。

公开证据支持这些结构性关系 [D→I]，但还不支持一张无条件的“六家技术总排名” [U]。
最深的比较不是填满未知，而是精确指出：**哪条因果链已经被披露或可复核，哪条只能
作为假设，以及什么实验会推翻它。**
