# OpenAI：从通用预训练到推理—工具—部署安全共同扩展的全研究谱系

**核验截止：2026-07-19（Asia/Shanghai）。** 本章冻结
`research/literature/inventory/openai.jsonl` 的 **657 条**记录：`core` 79、`direct` 456、
`affiliated` 122。按 inventory `type` 计，`blog_with_report` 366 条、`research_paper` 236 条、
`system_card` 34 条、`technical_report` 9 条、`benchmark` 9 条、`dataset` 2 条和
`model_card` 1 条。
清单保存 **222 个公开 PDF 路由**、130 个 arXiv 标识；当前归档中有 **220 份 PDF 与对应全文
抽取**。全文数量不是纳入标准：HTML 原生报告可以是一手披露，RSS/sitemap 条目只能证明
标题、日期和官方发布路径，而一篇有 OpenAI 机构署名的论文也不自动成为 GPT 训练报告。

这里的“全”有可机器核验的含义：第 16 节逐条列出冻结清单 **657/657** 的 inventory ID、
标题与 `primary_url`；正文覆盖 GPT/scaling/pretraining、RLHF 与 reasoning post-training、
test-time compute、多模态/音频/图像/视频、代码/数学/科学、agents/tools/Codex/deep research、
系统与推理、评测、可解释性、安全/system cards/Preparedness/治理/事故/威胁报告以及广义
机构署名研究。它不声称枚举私有、撤下、未索引或未披露的研究，也不把产品名反推成
未公开的 checkpoint 或训练图。

## 1. 怎么读：证据标签、归属边界和十个结论

本章沿用 [Research, Evidence, and Citation Standard](../research-method.md)：

- **[D] Disclosed / 已披露**：论文、技术报告、system/model card 或官方实质研究页直接陈述。
- **[C] Confirmed artifact / 公开制品确认**：代码、权重、数据、配置或可下载报告可直接观察。
- **[R] Reproduced / 已复现**：按可审计协议复跑并保存命令、环境、日志与制品。本章没有
  把任何模型能力、训练结果或安全率标成 [R]。
- **[I] Inferred / 推断**：从列明的事实和假设推出；来源本身没有作该陈述。
- **[U] Unknown / 未知**：证据缺失、冲突、只到发布元数据，或不足以支持所问结论。

`core` 是旗舰/专用模型发布、正式 model/system card 或明确核心报告；`direct` 是 OpenAI
第一方研究、部署、安全、治理、工程或 benchmark，以及可核验的直接论文；`affiliated`
只证明论文作者行或权威元数据中有明确 OpenAI 机构关系。三者是**归属强度，不是质量、
影响力或安全等级**。尤其不能把 122 条 affiliated 工作一律说成“用于训练 GPT”。

十个先行结论：

1. **[D] OpenAI 的研究史不是 GPT 单线史。** 2016–2019 年的 Gym/Universe、机器人、
   multi-agent、自博弈、生成模型、鲁棒性与 AI safety，和语言模型线并行；后来 tool-using
   agent 的环境观念明显承接了这批工作，但具体权重或代码继承多为 [U]。

2. **[D/U] 预训练能力持续扩展，配方透明度却没有单调增加。** GPT、GPT-2、GPT-3 和
   Scaling Laws 给出较多结构、数据与实验细节；GPT-4 明确不披露 size、hardware、compute、
   dataset construction 与完整 training method（`web-3375c5d27d9b`，PDF p. 2）；GPT-5.x
   的发布和卡片更详于能力、安全与运行时，却不足以重建预训练。

3. **[D] 后训练主线可审计地经历了五次重心迁移：** 人类比较学习奖励 → SFT/RM/PPO
   产品化 → process/rule/specification supervision → scaled reasoning RL → reasoning、tools、
   safe-completion 与运行时 safeguards 共设计。它们不是同一个“RLHF”标签的重复版本。

4. **[D/I] test-time compute 至少有五种机制。** 单轨迹延长、并行采样、投票、verifier/
   reranker 选择、工具/环境交互以及产品 router 会产生不同成本和错误相关性。o1 在 AIME
   上的 pass@1、consensus@64 与 1,000-sample learned reranker 结果不能互换。

5. **[D] 多模态是多棵树。** Image GPT/CLIP/DALL·E/diffusion 是视觉表示与生成；Whisper、
   Jukebox、Voice Engine/GPT-4o/GPT-Live 是语音音乐与实时音频；GPT-4V/4o 是视觉输入；
   Sora/Sora 2 是视频；Point-E 是 3D。共享品牌或 ChatGPT UI 不证明共享 tokenizer、训练集、
   decoder 或 optimizer [U]。

6. **[D/I] agent 是 policy、environment、harness、tools、budget、state 与 authorization 的
   联合系统。** WebGPT、VPT、Operator、deep research、o3/o4、Codex 和 ChatGPT agent 的
   observation/action space 均不同；“同一模型更强”不能解释全部系统差异。

7. **[D] 系统工程是能力边界的一部分。** Kubernetes、block-sparse kernels、Fiber、Triton、
   large-model training techniques、sandbox/App Server、Responses API computer environment、
   voice latency、network transport 与数据库扩展，决定哪些训练和长轨迹能可靠发生；它们
   不是一个参数量表能够吸收的噪声。

8. **[D/U] OpenAI 的安全披露从问题清单扩成 release-time control system。** 2016 年
   Concrete AI Safety 与 faulty reward，经过 GPT-2 staged release、GPT-4/vision/image/audio/
   reasoning/agent cards、Preparedness Framework、Frontier Governance Framework、威胁报告
   与事故复盘，形成模型行为、监控、access control、sandbox、provenance 和治理门槛的多层栈。
   这些是一手披露 [D]，不是独立有效性认证 [U]。

9. **[D/I] benchmark 是版本化测量，不是永恒排行榜。** OpenAI 从 Gym/Procgen/OpenAI Five
   走到 HumanEval、SWE-bench Verified、MLE-bench、SWE-Lancer、BrowseComp、PaperBench、
   GDPVAL 与领域 agent eval；2026 年又公开停止使用 SWE-bench Verified。benchmark 也会
   饱和、污染或失去区分度。

10. **[U] 当前材料不能端到端复现任一最新闭源旗舰，也不能建立跨供应商总排名。** 原始
    corpus/许可与 unique-token 统计、完整 data mixture、失败 runs、SFT/RM/RL prompts、
    reward weights、rollout 总量、router 更新、生产监控和总成本均不完整。API 产品、单一
    checkpoint、不同 tools/context/reasoning budget/judge/date 的数字禁止无条件横排。

### 1.1 657 条实际覆盖哪些研究域

下表各域有重叠，不能相加成 657；它用来防止把“OpenAI 研究”缩成语言模型发布。

| 研究域 | 冻结清单中的代表节点 |
|---|---|
| 基础模型与 scaling | GPT、GPT-2、Scaling Laws、GPT-3、GPT-4/4.1/4.5、GPT-5–5.6、gpt-oss |
| 数据与预训练 | WebText、few-shot pretraining、FIM、curated behavior data、数据提取与隐私、synthetic/filtered data |
| 后训练与 reasoning | human preferences、summarization、WebGPT、InstructGPT、PRM800K、RBR、o1、deliberative alignment、safe-completions |
| 多模态 | Image GPT、CLIP、DALL·E 1–3、diffusion/consistency、GPT-4V/4o Images、Whisper、Jukebox、Sora 1/2、Point-E |
| 代码、数学与科学 | Codex/HumanEval/FIM、formal theorem proving、MiniF2F、PRM、Codex Security、PaperBench、First Proof、biology/physics/chemistry agents |
| agents 与 embodied RL | Gym、Universe、sim-to-real、OpenAI Five、Rubik’s Cube、emergent tool use、VPT、Operator、deep research、ChatGPT agent、Codex |
| 系统与推理 | large-scale RL infra、Kubernetes、block-sparse、Fiber、Triton、PyTorch 2、training networks、sandbox、App Server、low-latency voice |
| 评测 | continuous-control、generalization、OpenAI Five、Procgen、HumanEval、SWE-bench Verified、MLE-bench、SWE-Lancer、BrowseComp、GDPVAL |
| 可解释性 | sentiment neuron、Activation Atlas、Circuits/Microscope、multimodal neurons、neuron explanation、GPT-4 concepts、sparse circuits、CoT monitoring |
| 安全与治理 | Concrete AI Safety、debate、staged release、deployment guidelines、34 system cards、Preparedness、Frontier Governance、incidents、threat reports |
| 广义机构研究 | 法律/经济、伦理与社会、机器人与进化、视觉/生成、神经科学与医学、系统/图算法、隐私/密码/量子等 122 条 affiliated 工作 |

## 2. 证据工程：RSS、sitemap、Deployment Safety、CDN 与 64 卡专项审计

### 2.1 四种来源状态不能混写

本次清单把“发现路径”和“论证强度”分开：

| 来源状态 | 能确认什么 | 不能自动确认什么 |
|---|---|---|
| S1：论文/PDF/正式 card 全文 | 正文、表图、页码、方法与限定语 [D] | 未披露的生产实现、复现成功 [U] |
| S2：官方 HTML 原生实质报告 | 页面当日公开的叙述、图表和链接 [D] | 等价 PDF、永久分页、完整训练 recipe [U] |
| S3：官方 RSS/sitemap 元数据 | 官方标题、日期、URL、taxonomy [D-meta] | 页面正文中的能力、方法或数字 [U] |
| S4：DOI/OpenAlex/会议元数据 | 论文身份、作者、venue、可核验机构关系 [D] | OpenAI 产品采用、旗舰训练输入 [U] |

`[D-meta]` 只是 `[D]` 的元数据子类，不表示已读取页面正文。冻结 inventory 中许多 2025–2026
官方页面因站点 access challenge 只保存 RSS description 与 sitemap 分类；本章不会把标题
推演成训练算法。HTML 原生报告也不因 `pdf_url: null` 被降格：若内容只以网页发布，正确
边界是引用其 section/chart，而不是制造一个不存在的 PDF 页码。

### 2.2 64 条官方卡片/报告专项审计的结果

`research/literature/inventory/openai-cards-audit.md` 对官方 system/model cards、addenda、
Preparedness/治理报告、deployment-safety 页面、研究型安全 PDF、威胁报告和事故页做了
**64 条候选专项审计**。修复前状态为：16 条完整、13 条 HTML-only、27 条记录存在但缺
PDF 路由、2 条指向错误 PDF、6 条 inventory 缺记录。六条缺项随后进入冻结清单：

1. `gpt-5.6-preview-system-card` — GPT-5.6 Preview；
2. `gpt-rosalind-5.5-system-card` — GPT-Rosalind-5.5；
3. `chatgpt-images-2.0-system-card` — ChatGPT Images 2.0；
4. `o1-preview-o1-mini-system-card` — o1-preview/o1-mini；
5. `preparedness-framework-beta` — Preparedness Framework (Beta)；
6. `gpt-4-system-card` — 独立于 GPT-4 Technical Report 的 system card。

**[C]** 当前清单保留官方 `deploymentsafety.openai.com` context URL，并在可用时保留
`cdn.openai.com` PDF。Deployment Safety sitemap 曾暴露 `localhost` slug，审计先规范化为
公开域名，再验证 HTTP content type 与 PDF magic bytes；不能把 sitemap 字符串直接当成
有效下载。GPT-4 technical report/card、o1 preview/final、GPT-5 base/addenda、GPT-5.6
preview/final 也都保持为不同记录，因为评测对象、日期和安全结论不同。

专项审计还纠正 GPT-4.5 PDF，并补齐/验证 GPT-5.x Codex addenda、Sensitive Conversations、
Sora 2、gpt-oss、ChatGPT agent、Operator、o3/o4、4o image、Instruction Hierarchy、
Safe-Completions、Frontier Governance、deployment simulation、CoT control/monitorability 与六份
连续威胁报告的官方 CDN 路由。当前 **222 个公开 PDF 路由中已有 220 份本地 PDF+全文**；
没有本地副本的 2 条仍需按 `pdf_url` 和 hash 继续归档，不能记作已复现。

### 2.3 可审计停止规则

一个定量主张只有在下列至少一项成立时进入正文：

1. 本地 PDF 能定位到 printed/PDF page、section、table 或 figure；
2. 官方 HTML 原生报告有可命名的 section/chart/protocol；
3. artifact 本身可观察，并只标 [C]；
4. 若只有 RSS/sitemap，正文只使用 title/date/category 元数据。

相反，“官方说了”不是停止规则。若缺 checkpoint、prompt、sampling、tools、context、budget、
judge、denominator 或 date，本章把横向结论标 [U]，即使单个数字来自官方页面。

## 3. 三条并行历史：研究、产品部署与安全治理

### 3.1 时间轴

| 时段 | 研究与训练线 | 产品/环境/系统线 | 安全、评测与治理线 |
|---|---|---|---|
| 2015–2016 | generative models、InfoGAN、GAIL、weight normalization、exploration、meta-RL [D] | Gym、Universe、deep-learning infrastructure 把统一环境与分布式实验变成研究对象 [D/C] | Concrete AI Safety、faulty reward、privacy-preserving knowledge transfer [D] |
| 2017–2018 | human preferences、unsupervised sentiment neuron、GPT、domain randomization、dexterous manipulation、multi-agent language [D] | sim-to-real、OpenAI Five、Kubernetes 2,500 nodes、block-sparse kernels [D] | black-box/policy attacks、debate、malicious-use report、generalization benchmarks [D] |
| 2019–2020 | GPT-2、preference fine-tuning、Scaling Laws、GPT-3、summarization HF、Image GPT、Jukebox、theorem proving [D] | Dota 2 scale、Fiber、Kubernetes/compute-efficiency work、Microscope [D/C] | GPT-2 staged release、model extraction/privacy、verifiable claims、misinformation studies [D] |
| 2021–2022 | CLIP/DALL·E/diffusion、Codex、book summarization、WebGPT、InstructGPT、DALL·E 2、Whisper、VPT、FIM、Point-E [D] | Triton、browser actions、Minecraft inverse dynamics、code execution and embeddings [D/C] | curated behavior/PALMS、deployment guidelines、code hazard analysis、DALL·E 2 data mitigations [D] |
| 2023 | GPT-4、process supervision/PRM800K、neuron explanation、GPT-4V、DALL·E 3 [D] | multimodal ChatGPT 与 API/product routing 开始高频发布 [D] | GPT-4/4V/DALL·E 3 cards、red-teaming network、Preparedness Beta [D] |
| 2024 | Sora、GPT-4o、RBR、Instruction Hierarchy、o1 reasoning RL、deliberative alignment [D] | native audio/vision、video generation、reasoning product tiers [D] | GPT-4o/o1/Sora cards、state-actor/covert-influence reports、Preparedness evals [D] |
| 2025 | Operator、deep research、o3/o4、Codex、ChatGPT agent、GPT-5、gpt-oss、safe-completions、science evals [D/C] | screenshot/terminal/browser/Python/file tools、sandboxed repos、router、Sora 2 [D] | updated Preparedness、agent/Codex/GPT-5 cards、sensitive-conversation addendum、threat reports、Mixpanel incident [D] |
| 2026 | GPT-5.3–5.6、GPT-Live/Rosalind、deployment simulation、CoT control/monitoring、science/chemist/lab agents [D] | App Server、Responses computer environment、WebSockets、Symphony、MRC、voice latency、Codex Security [D/C] | GPT-5.3–5.6 cards、Frontier Governance Framework、child/teen safety blueprints、continuous automated red-teaming [D] |

箭头式品牌史会丢掉两个关键事实。第一，早期 embodied/multi-agent RL 的 environment、reward
hacking 与 scalable infrastructure 问题在后来 agent 系统中重新出现；这支持方法类比 [I]，
不证明直接参数继承 [U]。第二，2023 年以后 product、system card、runtime safeguards 与模型
一起发布；“模型”越来越像一个版本化系统，而不只是 frozen weights [D]。

### 3.2 重大定量主张的一手定位

下表只保留能落到一手 section/page/table/figure 的数字；“PDF p.”指归档 PDF 的物理页，
可能与印刷页脚相差一页。数字只对相应实验对象和协议成立。

| 主张 | 一手定位 | 严格边界 |
|---|---|---|
| Scaling Laws 拟合 $\alpha_N=0.076$、$\alpha_D=0.095$、$\alpha_C=0.057$；compute-efficient $N_{opt}\propto C^{0.73}$、$D_{opt}\propto C^{0.27}$ | `web-bd2a7f12ec03`，Appendix A，Tables 4–6，PDF p. 20 | 该数据、tokenizer、模型族与拟合区间内的经验律；不是所有后来模型的常数 |
| GPT-3 最大模型 175B，训练时总采样约 300B tokens | `web-8f27ffb5bc66` 所链论文，§2.1–2.3，Tables 2.1–2.3，pp. 4–9 | weighted sampling exposure，不等于 300B unique 或无污染数据 |
| 2019 preference fine-tuning 使用约 5,000 个 style comparisons 与 60,000 个 summarization comparisons | `web-dd954fe97ea9` 所链论文，§3–4 与 experiment tables | 两个任务的数据量，不是后来 ChatGPT 的人类反馈规模 |
| WebGPT 约 6,000 demonstrations、21,500 comparisons；175B best-of-64 相对 human demonstrations 获 56% preference | `web-bb1464aabad9` 所链论文，§3–4、Table 1 与 human-eval tables | 主结果是 BC+RM selection；不是“PPO 最强” |
| InstructGPT 的 175B PPO 以约 31,144 unique customer prompts 跑 256,000 episodes；约 40 名筛选后的 labelers | `web-87477ad8b476` 所链论文，§3–4、Appendix A–C，Tables 6–11 | prompts 不是 preference labels；不能外推到 GPT-4/5 recipe |
| PRM800K 约 800k step labels、75k solutions、12k MATH problems | `web-b6a5fdb87e9f` 所链论文，abstract、§2 与 data appendix | process verifier/candidate selection；论文未报告对 solver 做 policy-gradient RL |
| CLIP 预训练用 400M image-text pairs | `doi-98a139038f2b`，§2.1，PDF/paper p. 3 | 自建 WIT 语料规模；不等于后续 DALL·E/4o 数据 |
| 初代 DALL·E 是 12B autoregressive Transformer，使用 250M text-image pairs | `doi-4e082ecb573e`，abstract、§2.1–2.2，pp. 1–4 | 不能外推到 diffusion-based DALL·E 2/3 |
| Whisper 以 680,000 小时多语种、多任务监督音频训练 | `web-2b132b9f05ef` 所链论文，abstract、§2.1，pp. 1–3 | weak supervision exposure，不是每小时人工精标 |
| Codex 论文发布 HumanEval 164 个手写编程问题 | `rss-47d0c87b08f0` 所链论文，§2.2、Table 1，pp. 3–5 | pass@$k$ 依赖采样数、temperature 与测试 harness |
| VPT 用约 2,000 小时 contractor data 训练 inverse dynamics，再标注约 70,000 小时网络视频 | `web-4e33c3e8216e` 所链论文，abstract、§2，pp. 1–4 | Minecraft keyboard/mouse action space；不是通用 GUI agent 数据 |
| GPT-4 的部分能力由不超过最终训练 compute 的 $1/1000$ 小模型预测；模拟 bar exam 约 top 10% | `web-3375c5d27d9b`，abstract/PDF p. 1；§3 Figure 1–2；§4 Table 1/Figure 4，PDF pp. 4–6 | architecture/data/compute 未披露；exam result 不是现实执业能力 |
| o1 AIME 2024 为 74.4% pass@1、83.3% consensus@64、93% learned reranker over 1,000 samples | `web-d1359f03b307`，AIME chart 与 accompanying text | 三种推理预算，93% 绝非单样本分数 |
| Operator/CUA 报告 OSWorld 38.1 | `web-24bc39ed487b`，computer-use evaluation，PDF p. 12 | 该 card 的 checkpoint、harness 和当时 OSWorld；不能和 WebArena/WebVoyager 平均 |
| GPT-5 外部安全测试超过 5,000 小时 | `web-eb6095ff0b58`，§4.4，PDF p. 21 | 输入劳动/测试量，不等于覆盖率或安全概率 |
| GPT-5.6 自动 universal-jailbreak 搜索投入超过 700,000 A100e GPU-hours | `system-card-gpt-5-6`，Introduction/PDF p. 3；§9.4.4/PDF p. 76 | red-team search compute，不是训练 compute，也不证明不存在未知 jailbreak |
| GPT-5.6 Sol/Terra/Luna 均按 Bio/Chem 与 Cyber “High”处理，AI Self-Improvement 未达 High | `system-card-gpt-5-6`，Introduction/PDF p. 2；§9/Figure 21，PDF p. 34 | OpenAI Preparedness 分类，不是跨机构统一风险等级 |

### 3.3 禁止横排的形式化规则

一个可比较观测至少应写成：

$$
z=(\text{checkpoint},\text{product/router},\text{prompt},\text{sampling},
\text{tools},\text{context},\text{budget},\text{judge},\text{date}).
$$

只有当两个结果的关键坐标一致，或差异被单独做因子实验时，差值才可归因。API 中的 routed
GPT-5、system-card 的 pre-mitigation checkpoint、带 browser/Python 的 o3、无工具 text-only
checkpoint、pass@1 与 best-of-1,000 都不是同一对象。后文所有 vendor 数字都服从此规则。

## 4. GPT、scaling 与 pretraining：能力扩展和披露收缩同时发生

### 4.1 GPT 到 GPT-2：生成式预训练、零样本行为与 staged release [D/C]

[Improving language understanding with unsupervised learning](https://openai.com/index/language-unsupervised)
（`web-91c4184aef84`）把 decoder-only Transformer 的 generative pretraining 与任务监督微调
接起来。初代 GPT 使用 BooksCorpus、12 层、约 117M 参数；论文 §3–4 的关键实验是“同一
预训练表示经少量 task-specific adaptation”而不是今天意义的 instruction-tuned chat。

GPT-2 的 1.5B 模型（`web-31066a8ed5c3`）扩大 WebText 与 left-to-right LM，并用 prompt/
format 诱导 translation、QA、summarization 等 zero-shot 行为。**[D]** “Language Models are
Unsupervised Multitask Learners”报告约 8M 文档、40GB WebText；这些是清洗后的 outbound-link
网页，不是“整个互联网”。**[C]** 2019 年先后发布小模型、阶段性观察和 1.5B 权重；
`web-aec2fcecd95e` 与 `web-31066a8ed5c3` 共同记录 staged-release 实验。

这段历史留下两条互相拉扯的线：规模提升带来更通用的 in-context behavior [D]；同一能力
也推动 output detection、misuse、bias 与 release strategy 评估 [D]。**[I] 可证伪判断：**
如果后续完整 release records 显示模型从一开始就按固定日期无条件全量发布，那么“GPT-2
staged release 是部署实验”的解释被推翻；现有六个月 follow-up 与分批权重制品支持它。

### 4.2 Scaling Laws 与 GPT-3：参数、数据、compute 的经验耦合 [D]

[Scaling Laws for Neural Language Models](https://openai.com/index/scaling-laws-for-neural-language-models)
（`web-bd2a7f12ec03`）将 loss 写成模型参数 $N$、数据 token $D$ 和训练 compute $C$ 的经验
幂律。其简化形式为：

$$
L_N(N)\approx \left(\frac{N_c}{N}\right)^{\alpha_N},\qquad
L_D(D)\approx \left(\frac{D_c}{D}\right)^{\alpha_D},\qquad
L_C(C)\approx \left(\frac{C_c}{C}\right)^{\alpha_C}.
$$

报告还用 $C\approx 6ND$ 估算 non-embedding training FLOPs（§2.1/Table 1，PDF p. 7），并在
Appendix A/Tables 4–6 给出拟合值与 compute-efficient frontier。**[I]** 这奠定了“先做小规模
可预测 runs，再决定大 run”的工程观，但不是自然定律；tokenizer、数据质量、optimizer、
architecture 与训练是否接近临界 batch 都能改变拟合。

[Language models are few-shot learners](https://openai.com/index/language-models-are-few-shot-learners)
（`web-8f27ffb5bc66`）把同一 autoregressive objective 扩到 175B，并系统比较 zero/one/few-shot
in-context learning。论文 Tables 2.1–2.3 给出 model/data mixture；§3 的几十个 task 采用不同
prompt、shot 与 metric。**[D]** “300B tokens”是按 mixture weights 采样的 exposure，部分
高质量数据跨 epoch 重复；**[U]** exact dedup、版权/许可与污染边界不足以重建语料。

GPT-3 的贡献不应缩成 175B：它把 task adaptation 从 parameter update 移到 context [D]，也
暴露 prompt sensitivity、contamination、calibration 与社会偏差。**[I] 可证伪判断：** 若在
固定 $N,D,C$、同数据与 optimizer 的受控实验中，in-context gains 与 scale 无关，则“规模
推动 few-shot 过渡”需要改写；当前论文的跨规模 curves 支持但没有识别唯一因果机制。

### 4.3 GPT-4：可预测 scaling 与不可复现 recipe 的转折 [D/U]

[GPT-4 Technical Report](https://openai.com/index/gpt-4-research)
（`web-3375c5d27d9b`）确认：Transformer-style next-token pretraining、公开互联网与第三方
许可数据、RLHF、图文输入/文本输出，以及从不超过最终 compute 的 $1/1000$ runs 预测部分
能力。§3 Figures 1–2 给出 loss 与 HumanEval scaling prediction；§4 列 exams、MMLU、
多语言和 vision evaluations；PDF pp. 42–60 的 system-card portion 讨论风险。

同一报告在 §2/PDF p. 2 明确省略 architecture（含 size）、hardware、training compute、
dataset construction 与 training method 细节。于是：

- “GPT-4 用 RLHF”是 [D]；“使用 InstructGPT 完全相同 PPO settings”是 [U]。
- “能处理图像和文字”是 [D]；具体视觉 encoder、fusion 与 joint-training recipe 是 [U]。
- “小 run 预测部分 final performance”是 [D]；所有 emergent behavior 都被准确预测是 [U]。
- 独立 `gpt-4-system-card` 保存 release safety 版本；不能被 technical report 的内嵌 card 去重。

GPT-4 报告的 factuality、exam 与 safety 数字是 checkpoint/protocol observations，不是训练
配方证据。更重要的是，论文 Figure 8 显示 RLHF 后 calibration 可能变差（PDF p. 12）；
alignment 改善偏好/行为并不保证所有统计性质同步改善。

### 4.4 GPT-4.1/4.5 到 GPT-5.6：产品谱系不是隐藏训练图 [D/U]

GPT-4o、4o mini、4.1、4.5、o-series 与 GPT-5.x 的官方 release/card 构成清晰时间谱系。
但标题“introducing”“update”“incorporates”不等于 optimizer 或 weight lineage：joint training、
continued pretraining、distillation、model merging、routing 或完全独立 checkpoint 都可能产生
产品上的继承 [U]。

| 节点 | 一手能确认 [D] | 仍未知 [U] |
|---|---|---|
| GPT-4o / mini | text/image/audio 的统一产品能力与相应安全评测；4o mini 是低成本发布 | modality fusion、完整数据、训练 compute、mini 的蒸馏/独训关系 |
| GPT-4.1 / 4.5 | API/product generation、长上下文/能力与 system-card 评测 | 与 4o 的参数继承、完整 pre/post-training recipe |
| o1 / o3 / o4-mini | scaled reasoning RL、不同 reasoning effort、工具进入 reasoning loop | optimizer、rollout 数、reward composition、train/test compute 曲线原始数据 |
| GPT-5 | fast model、reasoning model 与实时 router 的系统描述；safe-completion framing | router architecture/update schedule、各 checkpoint 大小、完整 reward/data mixture |
| GPT-5.1–5.5 | general/Codex/Instant/Thinking 的连续 release 与 card/addenda | transfer/merge/distillation 图、训练 compute、跨版本数据重用 |
| GPT-5.6 | Sol/Terra/Luna 三层 family 与 Preparedness/safeguard 评测 | 三者是否共享 base、如何训练/压缩、总训练成本 |

`web-05738f6c55b7`/`web-eb6095ff0b58` 是 GPT-5 release/card；`web-10ab1c39be14`、
`web-eea6230f612a`、`web-dfe7e0c6ff39`、`web-e19fbb75eb44` 依次记录 5.2/5.4/5.5/5.6
发布。后几页当前主要是官方 RSS/sitemap 元数据和卡片全文，因此本章只陈述发布谱系，不从
营销摘要发明 benchmark 或 training details。

### 4.5 gpt-oss 是重要制品，但不是闭源 GPT 的透视窗 [C/D/U]

`web-1c46fdf17c79` 的 gpt-oss-120b/20b model card 与公开权重/推理制品允许直接检查模型
配置和部署行为 [C]；`web-1b667a11f9a6` 研究 open-weight fine-tuning 的 worst-case frontier
risks；`web-e943411c2e8b` 给出 safeguard technical report。它们使开放权重的 adaptation、
monitor 与安全研究可复查 [D/C]。但 **[U]** gpt-oss 与 GPT-5.x 是否共享 tokenizer、数据、
teacher、base checkpoint 或 post-training mixture，不能仅凭同一组织与相近日期推出。

## 5. 从 human preferences 到 reasoning/tool RL：后训练的六层堆栈

### 5.1 比较学习奖励：2017 与 2019 奠定操作模板 [D]

[Learning from human preferences](https://openai.com/index/learning-from-human-preferences)
（`web-98fa91b77ece`）在 Atari/robotics 轨迹片段上让人类比较“哪段更接近目标”，拟合 reward
再做 RL；它证明难以手写的目标可以由少量相对判断近似。2019
[Fine-tuning GPT-2 from human preferences](https://openai.com/index/fine-tuning-gpt-2)
（`web-dd954fe97ea9`）把模板迁到 text continuation/summarization，使用约 5k/60k comparisons。

对 pair $(y^+,y^-)$，Bradley–Terry 形式可写为：

$$
\mathcal L_{RM}(\phi)=
-\mathbb E\log \sigma\!\left(r_\phi(x,y^+)-r_\phi(x,y^-)\right).
$$

随后 policy 优化通常加入对 reference 的 KL 约束：

$$
R(x,y)=r_\phi(x,y)-\beta
\log\frac{\pi_\theta(y\mid x)}{\pi_{ref}(y\mid x)}.
$$

这些方程概括已披露算法角色 [I]，不是所有 OpenAI 产品的精确 loss。[D] 人类比较提供
可扩展 signal；[U] reward model 是否代表 truth/safety、分布外是否可靠。`web-02a375c53f6d`
的 faulty reward 与后来的 reward overoptimization（`web-1a032a768300`）正说明 proxy 可被优化
到偏离目标。

### 5.2 Summarization、WebGPT 与 InstructGPT：三个不同层次 [D]

2020 summarization HF（`web-e5838d162a66`）与 2021 book summarization
（`web-7f768525daf5`、`doi-cfb93bf896c3`）研究 comparison reward 与 recursive decomposition。
WebGPT（`web-bb1464aabad9`）再加入受限 text browser：search/click/find/scroll/quote/finish，
demonstrations 记录完整 trajectory，RM 比较带引用答案。

WebGPT 公开了三种机制：behavior cloning、RM+PPO experiment、BC trajectories 的 best-of-$N$
选择。175B PPO 相对 BC 获 58% preference，而 best-of-64 BC 相对 BC 为 68%；PPO 没有实质
改善 best-of-$N$ 组合，因此主评测采用 BC+RM selection。**“WebGPT 的最佳 browser agent
由 PPO 产生”是可证伪且已被原文结果否定的说法。**

InstructGPT（`web-87477ad8b476`）给出最完整的经典产品级 SFT→RM→PPO/PPO-ptx recipe：

1. 约 40 名筛选后的 contractor 写 demonstrations，并对每 prompt 的 4–9 个 outputs 排序；
2. ranking 展开成 pairwise comparisons，训练 6B scalar reward model；
3. policy 在约 31,144 unique customer prompts 上跑 256k PPO episodes；
4. frozen reference 提供 token KL，6B value model 提供 advantage baseline；
5. PPO-ptx 混入 pretraining gradient 以减轻 public NLP regressions。

概念角色是：

```text
trainable actor policy
frozen reference policy  -> token-level KL
frozen reward model      -> terminal preference score
trainable value model    -> advantage baseline
pretraining examples     -> optional PPO-ptx retention gradient
```

论文 Appendix A–C/Tables 6–11 给出 prompts、learning rates、batch/clip/KL 等。**[U]** 这些
超参不能套到 GPT-4/5；后者公开材料没有说“沿用 InstructGPT PPO”。

### 5.3 Process supervision、RBR 与 instruction hierarchy：把规则放到不同位置 [D]

[Improving mathematical reasoning with process supervision](https://openai.com/index/improving-mathematical-reasoning-with-process-supervision)
（`web-b6a5fdb87e9f`）发布 PRM800K：labelers 标注每个 MATH 解题步骤 positive/negative/neutral，
process reward model 用于给候选排序。若用 step-valid 概率聚合，一个概念分数是：

$$
S(y)=\prod_{t=1}^{T}p_\phi(\text{valid}_t\mid x,y_{\le t}),\qquad
\log S(y)=\sum_t\log p_\phi(\text{valid}_t\mid x,y_{\le t}).
$$

这是 verifier 与 inference selection 研究；原论文没有报告用 PPO 更新 solver。把 PRM800K
写成“OpenAI 已用 process-reward RL 训练该 solver”会混淆 reward learning 与 policy learning。

Rule-Based Rewards 把研究者写的安全 propositions 交给固定 LLM grader，再用小规模人类标签
拟合线性组合，与 helpfulness RM 一起进入 PPO。简化表示为：

$$
r_{RBR}(x,y)=b+\sum_i w_i g_i(x,y),\qquad
r_{total}=\lambda_h r_{help}+\lambda_r r_{RBR}.
$$

它不是纯 `if/else`；规则由人写、LLM 解释、权重由数据拟合。Instruction Hierarchy
（`web-7d089e2d1180`，2024 原论文；`web-18b7ee4fe5e4`，2026 frontier update）训练模型优先
服从 system/developer 等 privileged instructions，并将 prompt injection 视为冲突指令问题。
RBR 管 output policy；hierarchy 管 instruction authority；两者不能互换。

### 5.4 o1：train-time RL 与 test-time reasoning 一起 scaling [D/U]

[Learning to reason with LLMs](https://openai.com/index/learning-to-reason-with-llms)
（`web-d1359f03b307`）披露 o1 通过 large-scale RL 学习长 chain of thought，能力随 train-time
RL compute 和 test-time reasoning compute 平滑提升，并出现检查错误、改策略与重试。

但公开证据没有 optimizer、prompt 来源/数量、rollout 数、reward decomposition、process vs
outcome、KL schedule、curriculum、hardware 或 token budget [U]。因此任何 PPO/GRPO demo 都
只能是教学实现，不能标 [R] o1。

test-time scaling 至少分为：

| 机制 | 增加什么 | 典型节点 | 主要风险 |
|---|---|---|---|
| 单轨迹延长 | 同一次生成的 reasoning tokens | o1/o3 reasoning effort | 错误自洽、延迟、不可见 CoT |
| 并行采样/投票 | 独立候选数 $N$ | o1 consensus@64 | correlated errors、成本 $O(N)$ |
| verifier/reranker | 候选与 judge compute | PRM、o1 1,000-sample rerank | judge hacking、length bias |
| tool/environment | search/code/file/GUI actions | WebGPT、o3/o4、deep research | 环境漂移、注入、tool error |
| product routing | fast/reasoning model 与 budget choice | GPT-5 system | router error、版本不可见 |

**[I] 可证伪判断：** 若固定 model、prompt 与 tools 后，增加 reasoning budget 不再提高目标
metric 或只提高 grader score、不提高 blinded human/executable outcome，则该任务上的有效
test-time scaling 已饱和或被 reward/judge 偏差取代。

### 5.5 Deliberative alignment 与 safe-completions：安全本身成为 reasoning 任务 [D]

[Deliberative alignment](https://openai.com/index/deliberative-alignment)
（`web-9cf289b414fd`）的公开 pipeline 是：把 policy specification 放进 prompt，生成 user
prompt/internal reasoning/final answer triples；policy-aware RM 过滤；移除 specification 后做
incremental SFT；再用能访问 specification 的 RM 做 RL。报告明确 safety CoT/answer 不必由
人逐条写，human 仍负责 specification、distribution、grader 与 evaluation。

[Safe-completions](https://openai.com/index/gpt-5-safe-completions)
（`web-aec54e92a7f3`）从 prompt-level comply/refuse 转向 output-level harm severity 与 usefulness：

$$
\max_\pi\ \mathbb E[H(x,y)]
\quad\text{s.t.}\quad \mathbb E[C_{safety}(x,y)]\le c.
$$

这是公开 framing 的形式化 [I]，不是 production coefficient。目标是在可允许边界内给出有限、
有用回答，而不是所有 dual-use prompt 一律 hard refusal。PDF Figure 3 与 §§3–5 对比 refusal/
safe-completion 的 automated/human eval；具体分数只能按对应 model、policy taxonomy 与 judge
解释。

### 5.6 o3/o4、GPT-5 与后续：reasoning、tools、router、safeguards 共设计 [D/U]

o3/o4-mini release/card（`web-646f1157fd8b`、`web-c02bd9ed41d6`）披露模型经 RL 学会在
reasoning 中选择 web、Python、image inspection 与 uploaded files；相较 o1，o3 使用约一个
数量级更多训练 compute 和更多 inference reasoning。公开材料仍不含 optimizer、task volume、
per-tool curriculum、reward weights 或 rollout infrastructure [U]。

GPT-5 card（`web-eb6095ff0b58`）把 fast model、deeper reasoning model 与 real-time router
列为系统；router 参考 conversation type、complexity、tool need、explicit intent，并从 model
switching、preferences、correctness 等信号改进。**[D]** routing 是产品机制；**[U]** router
是不是 RL policy、多久更新、如何校准、是否和 underlying weights joint-train。

GPT-5.2–5.6 与 Codex 变体继续发布能力和卡片，却没有等价于 InstructGPT appendix 的 stage-by-
stage recipe。**[I] 可证伪判断：** 若未来 technical report 给出共同 base、明确 continuation/
distillation/merge 图与训练日志，本章“产品谱系不能确定参数谱系”的 [U] 可被收缩；在此之前
不得靠名称补图。

## 6. 多模态、音频、图像、视频与 3D：并行树而非一个“原生”标签

### 6.1 表示学习：Image GPT、CLIP 与 multimodal neurons [D]

Image GPT（`web-d46ff01d5c8d`）把图像量化为像素序列，检验 autoregressive pretraining 是否
产生可迁移视觉表示。CLIP（`web-bc37d5969edc`/`doi-98a139038f2b`）对 400M image-text pairs
做 contrastive pretraining，使自然语言充当开放类别接口：

$$
\mathcal L_{CLIP}=\tfrac12\left(
\mathrm{CE}(S,\mathrm{diag})+\mathrm{CE}(S^\top,\mathrm{diag})
\right),\quad S_{ij}=\frac{v_i^\top t_j}{\tau}.
$$

`web-93d6498f6233` 观察 CLIP 中跨图文概念、人物/地域/情绪相关 neurons，同时揭示 typographic
attacks 与社会偏差。**[D]** zero-shot transfer 很广；**[U]** 一个 neuron label 不是因果解释，
CLIP classifier 也不自动适合高风险部署。

### 6.2 DALL·E、diffusion、consistency 与图像产品安全 [D/U]

初代 DALL·E（`web-f14c51ae873c`/`doi-4e082ecb573e`）用 discrete VAE tokens 与 12B
autoregressive Transformer 联合建模 text/image。DALL·E 2（`web-4a8bccb8dab1`、
`web-2c38d2f354fa`）转向 CLIP latent prior + diffusion decoder；Improved Diffusion
（`doi-a0200ddffdb4`）与 Diffusion Models Beat GANs（`doi-33e0a55739d5`）构成方法背景。
DALL·E 3（`web-b3d8859fa802`）强调 prompt following 与 captioning/rewrite pipeline；其 card
`web-ba80e5e27a11` 记录 red teaming、public figures、bias 与 provenance 控制。

2024 consistency-model scaling（`web-48d4ba4c85a1`）、2025 4o native image addendum
（`web-5223fe491ee4`）和 2026 ChatGPT Images 2.0 card
（`chatgpt-images-2.0-system-card`）说明图像线继续变化。C2PA metadata、visible/invisible
signals、prompt/output classifiers、public-figure/sexual-content policy 与 model training 是不同
层；“带 provenance”不等于内容真实，“拒绝某 prompt”也不等于训练集已移除相应图像。

**[I] 可证伪判断：** 若去掉 prompt rewriting 后在同一 checkpoint、seed 与评测集上 instruction
following 不变，则 DALL·E 3 产品增益不应归因于 rewrite；需要 component ablation 而非 UI
对比。

### 6.3 GPT-4V/4o：视觉输入与端到端音频要分开 [D]

GPT-4V system card（`web-a95452783369`）扩展 GPT-4 的 image input，并评估 person
identification、medical advice、CAPTCHA、location、hate/extremism 等风险。它是 capability/
safety evaluation，不公开视觉 architecture [U]。GPT-4o card（`web-e8cc130e4af6`）覆盖 text、
vision、audio 与 voice risks；后续 4o image addendum 又是独立生成能力版本。

“omni”至少可能指：多 modality 输入、共享 decoder、低延迟 native audio、统一 post-training、
或产品 router。公开 card 支持具体能力和评测 [D]，不支持所有组件共享一个端到端 loss [U]。
跨 modality benchmark 也不能平均：视觉 OCR、speaker identification、audio persuasion 与 text
MMLU 测量对象不同。

### 6.4 Whisper、Jukebox、Voice Engine 与 GPT-Live [D/C/U]

Jukebox（`web-3039e4df07aa`/`doi-cfe17a4d8015`）用 hierarchical VQ-VAE 与 autoregressive
priors 生成长音乐音频；Whisper（`web-2b132b9f05ef`）用 680k 小时 weakly supervised
multilingual/multitask audio 做 ASR/translation/language/timestamp prediction，并开放模型 [C]。

Voice Engine（`web-931159437ec9` 等官方研究页）关注短 reference voice 的 speech generation
及 consent/impersonation 风险；GPT-4o voice card 覆盖 speaker identification、unauthorized
voice generation 与 audio policy；`system-card-gpt-live` 是 2026 实时模型独立 card。**[U]**
端到端 latency budget、生产 codec、routing、训练语音许可和 speaker dedup 并未完整公开。

### 6.5 Sora/Sora 2 与 Point-E：时空/3D 分支 [D/U]

[Video generation models as world simulators](https://openai.com/index/video-generation-models-as-world-simulators)
（`web-0fe343211040`）描述 Sora 将视频压到 latent patches、用 diffusion transformer 处理可变
分辨率/时长/宽高比，并呈现部分 3D consistency、object permanence 与 simulation failure。
“world simulator”是研究 framing，不等于具备可验证物理 world model。

Sora card（`web-a51b48a9155c`）与 Sora 2 release/card（`web-661a3476d3d5`、
`web-8530f5b0e216`）把 likeness、sexual/deceptive content、C2PA、feed/product controls 和 audio
风险加入版本化评测。Point-E（`web-28adad6ac48b`）则先生成 synthetic view/point cloud，追求
3D asset 速度。三者 output space 与 metric 不同，不能以“多模态分数”横排。

### 6.6 多模态披露的关键未知 [U]

| 阶段 | 已披露 [D/C] | 关键未知 [U] |
|---|---|---|
| data | 部分 pair/hour 类别、过滤与安全 mitigations | exact sources/licences、dedup、people/voice consent、跨 modality overlap |
| tokenization | 部分 image/audio/video latent 或 patch framing | GPT-4o/5.x production encoders、codec 与 token budget |
| objective | contrastive、autoregressive、diffusion/consistency 的论文级目标 | 生产 joint loss、mixture weights、stage order |
| post-training | caption rewrite、preference/safety eval、部分 native multimodal claims | annotator counts、reward composition、RL trajectories |
| serving | real-time voice、image/video products、provenance artifacts | live router、latency topology、fallback 与 regional variation |
| reproduction | Whisper/CLIP/gpt-oss 等部分 weights/code [C] | DALL·E 3、GPT-4o、Sora、GPT-5.x 完整训练 [U] |

## 7. 代码、数学与科学：verifier、environment 与外部验证的三种证据

### 7.1 Codex/HumanEval：代码生成先把 execution 带进评测 [D/C]

[Evaluating large language models trained on code](https://openai.com/index/evaluating-large-language-models-trained-on-code)
（`rss-47d0c87b08f0`）以 GPT-3 continuation 的 code-trained variants 研究 synthesis，并发布
164 个手写 Python 问题的 HumanEval。pass@$k$ 估计至少一个候选通过 tests 的概率；其无偏
估计常写作：

$$
\widehat{\mathrm{pass@}k}
=1-\frac{\binom{n-c}{k}}{\binom{n}{k}},
$$

其中 $n$ 为 samples、$c$ 为通过数。改变 $n,k$, temperature、prompt、timeout、依赖、隐藏
tests 都会改变结果。**[C]** benchmark/tests 可检查；**[U]** 通过 public tests 不保证安全、
maintainability 或无训练污染。

FIM（`web-54ba9a907ae8`）显示把一部分 next-token sequences 重排为 prefix/suffix/middle，可
在不显著损害 left-to-right 能力的情况下训练 infilling。code embeddings
（`web-d3cbea351a46`）、经济影响 agenda（`web-212f765399d5`）与 code hazard framework
（`web-f3db758df8fe`）分别处理 retrieval、劳动测量与 misuse taxonomy；不能由 HumanEval
一项替代。

### 7.2 数学与形式推理：答案验证、过程验证、proof checker 不等价 [D]

2020 generative theorem proving（`web-88e99ce4204e`）、2021 math word problems
（`web-34f845c1e710`）、MiniF2F（`doi-bddb32d0dc59`）、2022 formal Olympiad work
（`web-89ed49981586`）与 2023 PRM/PRM800K 构成四种反馈：

| 反馈 | 能确认 | 不能确认 |
|---|---|---|
| exact answer | final answer matches | 中间推理正确、无 guessing |
| unit tests | program behavior on tests | specification 完整、无 test hacking |
| process label/PRM | 每步看似有效 | formal soundness、judge 无偏 |
| proof assistant kernel | term/type checks | informal statement 翻译正确、定理有价值 |

2026 First Proof submissions（`web-feae6511a0a2`）属于公开提交/验证流程；其证据强度取决于
problem statement、proof artifact 与 external checker，而不是产品发布标题。**[I] 可证伪
判断：** 若移除 executable/formal verifier 后性能不降，则“该能力主要由可验证反馈推动”的
归因需被否定；目前公开材料通常缺这个 ablation。

### 7.3 从 PaperBench 到 autonomous lab：科学 agent 需要分解证据链 [D/U]

2025–2026 清单包含 PaperBench（`web-a58fadf0d59c`）、AI scientific-research tasks
（`web-1b288e83884e`）、biology acceleration（`web-3f703ff65ec6`）、GPT-5.2 science/math
（`web-03cba5d23965`）、theoretical-physics result（`web-407f7c69ffbc`）、protein synthesis
（`web-aec434a838fa` 与论文 `doi-74b5c1fbc1f6`）、GeneBench
（`doi-3698c78278b4`）、GPT-Rosalind（`web-078f0d540d88`/`gpt-rosalind-5.5-system-card`）和
near-autonomous chemist（`rss-5ff2f715f1e6`）。

一项“AI 科学结果”至少拆为：

```text
problem chosen -> literature/data access -> hypothesis/model output
-> executable/symbolic check -> wet-lab or domain-expert validation
-> novelty search -> independent replication
```

官方 case study 可证明系统在所述 workflow 中产出候选并通过列明验证 [D]；不能自动证明
结果新颖、因果正确、可推广或独立复现 [U]。protein-synthesis 项有正式预印本，可比纯 RSS
摘要更强；physics/First Proof 也应以 proof/peer review 的后续状态更新，而非固定成“模型已
独立发现定律”。

### 7.4 Codex Security 与软件工程：agent 不等于 SAST [D]

2025 Codex-1（`web-abb942f337e2`/`web-183d58a311d5`）从 o3 衍生，披露在真实软件任务和
多样 sandbox repos 中做 RL：inspect/edit/run tests/iterate，并惩罚违背 instructions 的结果。
概念 reward 可分解为：

$$
r(\tau)=w_t r_{tests}+w_i r_{instruction}+w_q r_{quality}
-w_s c_{safety}-w_h c_{hack},
$$

但这只是 specification checklist [I]，不是 OpenAI 公布的权重。visible tests 单独可被删测、
hard-code、改 fixture 或 hidden regression 欺骗；sandbox policy、immutable/hidden tests、diff
review 与 instruction judge 是环境的一部分。

2026 Codex Security preview（`rss-5becf449a527`）与 “Why ... Doesn’t Include a SAST Report”
（`rss-8a8601b30dd9`）明确把 agentic investigation 和传统 static report 区分。Triton
（`web-03e8feb802d3`）、Triton-Sanitizer（`doi-aa7e89e34dbd`）、Proton
（`doi-3557c9ba3a3e`）、PyTorch 2（`doi-a26b81631a77`）则是 compiler/runtime/profiling
研究。**[U]** Codex 找到 bug 不表示完整 soundness；没有 SAST report 也不能被重写成
“优于所有 SAST”。

## 8. Agents、tools、Codex 与 deep research：环境谱系比品牌谱系更有解释力

### 8.1 Gym/Universe 到 multi-agent/robotics：先有 environment science [D/C]

OpenAI Gym Beta（`web-c33fb9b7adb1`）标准化 observation/action/reward API；Universe
（`web-d346a1aa6260`）把 browser/desktop applications 纳入可交互环境。continuous-control
benchmark、count-based exploration、RL²、ES、domain randomization、sim-to-real、curiosity、
asymmetric self-play、OpenAI Five 与 Rubik’s Cube 等记录探索：

- reward design 与 reward hacking；
- procedural/domain randomization 与 generalization；
- demonstrations、imitation、self-play 与 curriculum；
- large distributed rollout、population training 与 opponent non-stationarity；
- vision/action latency 和 sim-to-real gap。

Emergent Tool Use（`doi-ce7617c3d37b`）在 multi-agent autocurricula 中观察工具行为；OpenAI
Five 的 benchmark、large-scale RL 与 long-term planning 分别由 `rss-c2f416b49d98`、
`web-2c89cede3d83`、`doi-b4b263a4548f` 记录。它们支持后来 agent 的问题类比 [I]，但
**没有证据**说 o-series/Codex 直接继承这些 policy weights [U]。

### 8.2 WebGPT 与 VPT：受限工具和 action labels 的两种答案 [D]

WebGPT 的 text browser 把网页状态压成可序列化文字与固定 commands；VPT
（`web-4e33c3e8216e`）用 contractor-labeled Minecraft video 训练 inverse dynamics，再给
70k 小时无 action 标签视频补 pseudo-actions，最后 behavior clone 并 fine-tune。两者都把
“互联网上的行为痕迹”变成 trajectory data，但 supervision 不同：

```text
WebGPT: human browser demonstrations + answer preferences + RM selection/PPO
VPT:    video observations + small action-labeled set -> inverse dynamics
        -> pseudo-action labels -> behavior cloning/fine-tuning
```

**[I] 可证伪判断：** 固定 policy 后若 unrestricted browser/GUI 的失败率与 text-browser
一致，则 interface restriction 对 reliability 的解释被削弱；需要同任务、同页面 snapshot、
同 action budget 的对照。

### 8.3 Operator/CUA：截图—坐标—确认构成新 MDP [D/U]

Computer-Using Agent（`web-74eeb7c9fb20`）与 Operator card（`web-24bc39ed487b`）披露：
GPT-4o visual perception 加 specialized supervised screen/action data，RL 学 multi-step reasoning、
recovery 与 adaptation。一个简化状态是：

$$
s_t=(I_t,h_t,z_t),\qquad
a_t\in\{\mathrm{click}(x,y),\mathrm{type}(u),\mathrm{scroll}(d),
\mathrm{key}(k),\ldots\}.
$$

$I_t$ 是 screenshot，$h_t$ 是 history，$z_t$ 是 task/runtime metadata。screen size、zoom、
pop-up、animation、login、site version 都改变 transition。PDF p. 12 报告 OSWorld 38.1；它
不能与其他 suite 分数平均。高影响动作的 user confirmation 是 runtime authorization [D]，
不是 policy 已学会完美约束。

### 8.4 Deep research：长轨迹 browsing、Python、files 与 citations [D/U]

Deep research release/card（`web-e0a00b60e015`、`web-386e34a7513e`）把早期 o3 variant 用
end-to-end RL 训练于困难 browsing/reasoning tasks，能 plan/search/read/use Python/inspect
uploaded files/revise/backtrack/produce cited report。相对 WebGPT：

```text
WebGPT       constrained text browse -> evidence quotes -> answer
deep research plan -> heterogeneous browse/code/file actions
              -> revision/backtracking -> cited synthesis
```

一个严谨 evaluation 要分别测 answer correctness、citation entailment、source quality/diversity、
tool validity、coverage 与 policy compliance。**[U]** actual reward decomposition、datasets、
optimizer、rollout budget 与 human/verifiable feedback 比例未公开。BrowseComp
（`web-13a8c4bf46a3`）测 difficult-to-find factual browsing；它不是完整 deep-research report
质量的充分统计量。

### 8.5 o3/o4、Codex、ChatGPT agent：tool choice 进入 reasoning policy [D]

o3/o4 需要学习是否调用 tool、选择哪个、如何解释结果、何时继续或回答。Codex 把 terminal/
repository/tests/sandbox 作为 environment；ChatGPT agent（`web-b074ccdb2d9e`、
`web-7a7213b68631`）组合 Operator 的 visual interaction、deep research 的 browsing，以及
terminal/connectors 等工具。三者 capability surface 不同，不能只报底层 model family。

agent trajectory 的可复现实验单元应是：

$$
\tau=(m,e_0,h_0,\{o_t,a_t,r_t,d_t\}_{t=0}^{T},b,
\mathcal T,v,j),
$$

其中 $m$ 为 checkpoint/router，$e_0$ 为环境镜像，$h_0$ 为 harness，$b$ 为预算，
$\mathcal T$ 为 tools/权限，$v$ 为外部资源版本，$j$ 为 judge。缺任一项都会使“同模型复跑”
含糊。

### 8.6 2026 harness engineering：state system 成为显式研究对象 [D/C]

`web-ae10eb4597fe`（agent loop）、`web-e205ffac9d9f`（Codex App Server）、
`web-f6b93302265d`（harness engineering）、`web-01572b4c34e9`（Responses API computer
environment）、`web-8f555fe918e1`（WebSockets）、`web-f9ff221789e0`（Symphony orchestration
spec）、`web-4a3442a5ca8c`（Windows sandbox）和 `web-c3ab33c62226`（memory/Dreaming）
把运行时 protocol、context、sandbox、orchestration 与 memory 明确为工程对象。

这些官方报告能证明接口/系统设计 [D]，开放 spec 或 code 可标 [C]；却不证明 underlying
model 因 harness 改变而 weights 变强。**[I] 可证伪判断：** 在固定 checkpoint、task 与预算的
factorial ablation 中，如果换 harness/memory/orchestration 不改变成功率、干预率或 latency，
“harness 是能力乘数”的解释应收缩。

### 8.7 Agent 的主要未知 [U]

- 训练 task/repository/site/OS inventory 与 contamination controls；
- horizon、失败/中断/人工接管分布与 total rollout tokens；
- per-tool curriculum、reward/penalty 权重与 anti-hacking；
- train environment 与 live deployment 的 route/mask/precision fidelity；
- prompt-injection、credential、data exfiltration 和 cross-tool authorization 的真实 base rate；
- router、context compaction、memory 和 model update 对 benchmark 的交互。

## 9. 系统、训练基础设施与推理：从集群规模到 agent control plane

### 9.1 早期基础设施：统一实验、稀疏 kernel 与 population RL [D/C]

2016 Infrastructure for Deep Learning（`web-5eec2d002d2e`）、2018/2021 Kubernetes 2,500/
7,500 nodes（`web-4c9c2f788cfe`、`web-0b8a27d6cb1c`）、block-sparse GPU kernels
（`web-f0d270fbca2e`）与 Fiber（`doi-4c4a270fbe8f`）对应四种瓶颈：experiment scheduling、
cluster orchestration、operator efficiency、distributed RL/population methods。

这些工作解释为何 model research 与 systems 不可分。更多 actors 可提高 sample throughput，
也会增加 policy lag；更大 batch 可提高硬件利用率，也会改变 optimizer dynamics；稀疏 kernel
只有在 shape、memory traffic 与 hardware 匹配时才兑现 FLOPs 节省。

### 9.2 大模型训练：predictability、parallelism 与 compiler stack [D/U]

2022 Techniques for Training Large Neural Networks（`web-4f4c5c964bce`）系统梳理 data/
pipeline/tensor parallel、activation checkpointing 与 memory/communication trade-offs。Triton
（`web-03e8feb802d3`）用 Python-like DSL 生成高性能 GPU kernels；PyTorch 2
（`doi-a26b81631a77`）以 Dynamo/graph compilation 加速动态 Python workloads；2026 MRC
（`web-a2a811b50a4c`）处理大规模训练网络多路径可靠连接。

GPT-4 的 small-run scaling prediction 是训练系统可预测性的外显结果 [D]，但 total hardware、
topology、failures、utilization 和 energy/cost 未披露 [U]。不能从 API token price 反推 training
FLOPs，也不能把不同代价格下降全归因于模型压缩。

### 9.3 推理与产品系统：router、voice、database、sandbox [D]

GPT-5 router 说明 inference path 可能包含 model/budget selection。2026 low-latency voice
（`web-5c3d93bd7160`）、PostgreSQL at 800M users（`web-89a221162f32`）、Codex/Sora access
scaling（`web-338aaf898d6c`）、App Server/WebSockets/computer environment 与 Windows sandbox
表明 production quality 是：

$$
Q_{system}=f(Q_{weights},\,routing,\,context,\,tools,\,latency,\,retries,
\,state,\,authorization,\,monitoring).
$$

这是系统分解 [I]，不是可加总 score。产品成功率可能因 retry、cached state 或 tool fallback
提高；也可能因 latency timeout、stale page、permission 或 context truncation 降低。system
card 通常评估某一 pre-release stack，live stack 会继续变，因此 date 必填。

### 9.4 系统主张的验证协议

| 主张 | 至少需要 |
|---|---|
| “训练更高效” | 相同 target loss/quality、tokens、precision；报告 hardware、FLOPs、wall time、utilization 与 failures |
| “推理更快” | 相同 checkpoint、batch/concurrency、input/output length、reasoning effort、tools；p50/p95/p99 |
| “agent 更可靠” | pinned environment/harness、success + intervention + timeout + safety violation、repeated seeds |
| “更便宜” | 相同质量约束和完整系统成本；不能只比 API list price |
| “更可扩展” | load shape、backpressure、regional/failure assumptions、recovery behavior |

本章没有这些系统的统一复跑日志，所以均不标 [R]。

## 10. Evaluation 与 benchmarks：从可复现实验环境到会过期的测量产品

### 10.1 Gym、generalization 与 Procgen [D/C]

Gym/Universe 统一接口后，continuous-control benchmark（`openalex-w2963641140`）、Gotta Learn
Fast（`web-4ec028b44ef3`）、Quantifying Generalization（`web-8c8b67d910ad`）、OpenAI Five
Benchmark（`rss-c2f416b49d98`/`web-f7475741716d`）和 Procgen
（`web-2f2d250f7b2e`/`doi-a0f4dd5c6a07`）逐步把 train/test levels、opponents 与 procedural
variation 纳入 protocol。[C] 环境/代码可见不等于论文结果已在本章重跑。

### 10.2 从 HumanEval 到 browsing/software/science agents [D]

| 年份 | benchmark | 测量对象 | 不能替代 |
|---|---|---|---|
| 2021 | HumanEval | short Python function + unit tests | repository engineering、security、maintainability |
| 2024 | SWE-bench Verified | human-filtered GitHub issue resolution | 所有语言/仓库、live dependency、hidden product tasks |
| 2024 | MLE-bench | Kaggle-style ML engineering | general software/GUI/web agency |
| 2025 | SWE-Lancer | economically valued freelance software tasks | full labor-market productivity |
| 2025 | BrowseComp | hard-to-find browsing facts | long-form synthesis/citation quality |
| 2025 | PaperBench | AI research replication tasks | novel science、wet-lab confirmation |
| 2026 | GDPVAL | real-world economically valuable work samples | GDP impact、firm adoption、all occupations |
| 2026 | GeneBench | multi-stage genomics/quant-bio inference | autonomous wet-lab safety/effectiveness |

`web-10b59b402b80` introduced SWE-bench Verified；`web-33e5e5bcd689` later says OpenAI no
longer evaluates it.**[D]** 这证明 benchmark lifecycle 本身需要版本化；仅凭页面标题不能假定
原因，具体 contamination/saturation/protocol 论证要回到 HTML 正文 [U when unavailable]。

### 10.3 System-card evaluation 不是一个总 safety score [D/U]

system cards 覆盖 disallowed content、jailbreak、CBRN、cyber、persuasion、privacy、bias、
mental health、child safety、deception/scheming、model autonomy、image/audio/video/person
likeness、prompt injection、computer-use 与 agentic misalignment。不同 card 的 taxonomy、
grader、pre/post-mitigation checkpoint、elicitation、denominator 与 threshold 会改。

“Preparedness High”是 framework 内 capability classification；“policy violation rate”是行为
eval；“jailbreak ASR”是 adversarial protocol；“red-team hours/GPU-hours”是 search effort。
四者不可加成一个安全分，也不能和别家同名等级横排。

### 10.4 一个最小 eval record

```yaml
model: exact checkpoint and product/router
date: evaluation and external-resource snapshot
prompt: template, system/developer messages, few-shot examples
sampling: temperature, top_p, seed, samples
reasoning: effort/token cap/parallel candidates/reranker
tools: names, versions, permissions, latency/error injection
context: window, truncation/compaction/memory policy
dataset: version, split, contamination audit, exclusions
judge: executable/human/model, rubric, calibration, blinding
metrics: numerator, denominator, confidence interval, failures/timeouts
safeguards: pre/post mitigation and access tier
```

没有这份 record，vendor table 仍是发布证据 [D]，却不是可归因的横向科学比较 [U]。

## 11. Interpretability、circuits 与 CoT monitoring：可视化、解释、因果与监督不是同一件事

### 11.1 从 disentanglement 到 feature visualization：解释对象一直在变 [D]

OpenAI 的解释性谱系至少有五种不同对象，不能都叫“看懂神经元”：

| 阶段 | 代表记录 | 解释对象 | 能支持的结论 | 仍不能支持 |
|---|---|---|---|---|
| latent factor | `doi-3ca1a4531296` InfoGAN | 生成模型潜变量与可辨别因素 | mutual-information regularization 可在所列数据上产生可解释 latent factors [D] | 因素与真实因果变量一一对应 [U] |
| unit/feature visualization | `web-5dc883adf998` sentiment neuron、`web-e4ccd46451d8` Activation Atlases | 单元激活与输入空间中的高激活区域 | 可观察到与 sentiment/visual features 相关的 activation geometry [D] | 单个单元是充分、稳定、唯一的概念载体 [U] |
| circuits/artifact | `web-6e6e5524f81a` Microscope、`doi-da8a9f9b3311` Zoom In、`doi-3f4601bcac3a` Thread | feature、weights、paths 与局部计算图 | 公共可视化制品和局部机制假设可检查 [C/D] | 从局部 circuit 推出完整模型行为 [U] |
| model-assisted explanation | `web-784c8fdf9a87` neuron explanations、`web-d58703e87ddb` GPT-4 concepts、`web-1285c175b36f` sparse circuits | 由另一模型生成/评分的自然语言概念或稀疏结构 | 可扩展地产生候选解释并做自动评分/干预 [D] | judge 与解释模型无共同盲点、解释具有因果充分性 [U] |
| reasoning-trace oversight | `web-454ed7542563` monitorability、`web-287a6bf67154` controllability | 外显 CoT、tool actions、final answer 与 monitor | 在给定 agent/monitor/eval 下测量可监控性与可控性 [D] | CoT 完全忠实于内部计算，或监控已足够安全 [U] |

InfoGAN 的局部目标可写为在生成质量之外最大化下界

$$
I(c;G(z,c)) \ge \mathbb{E}_{c,z}[\log Q(c\mid G(z,c))] + H(c).
$$

`doi-3ca1a4531296` 的 §2、Equation 4 与 Figures 2-5（PDF pp. 3-8）给出方法和 MNIST、3D
faces、SVHN、CelebA 的结果。[D] 这里的“disentangled”是实验协议下的可辨认变化；没有
intervention graph 或真实生成因子标签时，不能把 mutual information 高直接等同因果语义。

sentiment neuron、Activation Atlas、Microscope/Circuits 又是另一层：前两者从高激活样本和
feature visualization 找结构，后者把局部 weights/features/paths 组织成可浏览制品 [C]。同一
feature 可能是 polysemantic，语义也可能分布在多个 feature；因此“图看起来像 X”只产生机制
候选，最小确认还需 ablation、activation patching、counterfactual input 与跨分布稳定性。

### 11.2 自动解释的三层误差：feature、explainer、simulator [D/I/U]

`web-784c8fdf9a87` 把语言模型用于生成 neuron 的自然语言解释，再用模型模拟该 neuron 在新文本
上的激活来评分；`web-d58703e87ddb` 与 `web-1285c175b36f` 把对象扩展到 GPT-4 concepts 与
sparse circuits。该范式真正测到的是一条复合通道：

$$
\text{score}=g(\text{feature extraction},\text{explainer},\text{simulator},
\text{examples},\text{metric}).
$$

高分至少可能来自三种原因：解释捕捉真实机制、simulator 擅长从例子插值、或 feature extraction
已经把复杂机制投影成容易命名的对象。[I] 所以“自然语言解释可预测 activation”不等于“解释对
部署行为具有完整因果控制力”。若要升级结论，应在 held-out distribution 上做 causal intervention，
报告 explanation stability、false discovery、feature splitting/merging 与 downstream behavior
change，而不只展示高分例子。

### 11.3 CoT monitorability 是 model-monitor-observation 三元属性 [D]

`web-454ed7542563` 在 §2/Table 1 把 monitorability eval 分成 intervention、process 与
outcome-property 三类，并区分 CoT-only、action-only 与 full-output monitor；Figure 1 表明在其
协议下 CoT 提供的监督信号通常强于只看 actions，长 reasoning 和更强 monitor/test-time compute
往往提高可监控性。论文也明确说 monitorability 不要求 CoT 对内部计算完全忠实：它测的是
monitor 能否从可见 observation 提取目标 property。

因此应写成：

$$
M=M(\text{agent checkpoint},\text{trace policy},\text{task},\text{monitor},
\text{monitor budget},\text{metric}).
$$

`web-287a6bf67154` 的 CoT-Control 则测潜在失败通道：模型能否在完成任务的同时控制 reasoning
trace 以隐藏某些属性。该报告 abstract、§2/Table 1、Figure 1 和 Appendix H 给出 14,076 个
test instances 及多种 controllability instruction；示例中 Claude Sonnet 4.5 的 CoT control 为
2.7%，final-output control 为 61.9%。[D] 这在该 suite 上支持“当前 reasoning trace 较难精确
控制”，却不证明未来训练、隐式通信、latent reasoning 或不同工具轨迹不会绕过 monitor [U]。

### 11.4 一个可证伪判断：解释性正在从“看 feature”转成“管理监督通道” [I]

本章的判断是：2025-2026 的重点不再只是给 feature 命名，而是测量可供 oversight 使用的
observation channel，并研究训练是否会破坏该通道。[I] 以下任一证据会迫使这个判断收缩：

1. 在相同 capability 下，CoT/actions/full-output 对 misbehavior detection 没有稳定增量；
2. 可扩展的 latent/circuit intervention 比 trace monitoring 更早、更可靠地预测真实事故；
3. 训练到更强能力后 controllability 大幅上升而 monitorability 系统性下降；
4. monitor 的高分主要由 benchmark artifacts、judge leakage 或同家族 correlated errors 解释。

在这些检验完成前，CoT monitoring 是有条件的防线 [D/I]，不是安全证明 [U]。

## 12. Safety、alignment、system cards、威胁情报与治理：从模型行为到发布控制系统

### 12.1 六层 safety stack [D/I]

把 106 条 safety topic、74 条 safety/alignment、30 条 security、34 份 system cards 和多份
治理/威胁/事故记录放在一起后，最有解释力的不是一条“alignment 方法”线，而是六层控制栈：

| 层 | 典型机制 | 代表记录 | 主要 failure mode |
|---|---|---|---|
| specification/data | behavior guidelines、Model Spec、curated/value-targeted data | `web-49090b5da7f8`、`doi-67245cbcdf25` | policy 冲突、coverage 缺口、labeler/judge 偏差 |
| training | preferences、RBR、instruction hierarchy、deliberative alignment、safe-completions | `web-dd954fe97ea9`、`web-dcb04574301f`、`web-7d089e2d1180`、`web-9cf289b414fd`、`web-aec54e92a7f3` | reward hacking、overrefusal、spec gaming、distribution shift |
| model evaluation | policy、jailbreak、CBRN/cyber、autonomy、bias/privacy 等 evals | 34 份 system cards、Preparedness | elicitation 不足、denominator/judge 不同、未知攻击 |
| runtime safeguards | classifier、monitor、sandbox、access tier、confirmation、provenance | GPT-5.x、Operator/Codex/image/audio cards | adaptive bypass、false positive、cross-tool escalation |
| operations | abuse detection、threat intel、incident response、rollbacks | 六期 malicious-use reports、sycophancy/outage/Mixpanel records | 观测选择偏差、漏报、第三方系统不可见 |
| governance | deployment threshold、risk acceptance、external/board oversight | Preparedness v2、Frontier Governance Framework | 自评独立性、例外、定义漂移、执行证据不足 |

这六层是 defense-in-depth 的系统分解 [I]，不能简单相乘得到“安全概率”。各层失败通常相关：
同一个 spec blind spot 可同时污染 SFT、reward model、judge 和 runtime classifier；同一个访问策略
也会改变 eval 与生产 base rate。需要报告联合攻击与 correlated failure，而不只逐层通过率。

### 12.2 从 preferences 到 safe-completions：优化对象发生了什么变化 [D]

早期 human preferences 把相对人类选择用于 reward learning；InstructGPT 把 SFT/RM/PPO
产品化。RBR 把明确规则转成可扩展 feedback；Instruction Hierarchy 训练模型区分 privileged
instructions；deliberative alignment 让 reasoning model 在 SFT 与 RL 阶段使用 category-specific
safety specifications。

`web-9cf289b414fd` 的 Figure 3 展示 spec-conditioned generation/filtering/SFT/RL，Table 1
比较 disallowed content、jailbreak、style 与 overrefusal，Figure 13 把 inference-time reasoning
budget 与部分安全表现联系起来。[D] 其结论对 o-series、所列 policies 与 graders 成立；CoT
引用 policy 不等于内部因果完全忠实 [U]。

`web-aec54e92a7f3` 进一步把二元 prompt/refusal 边界改为 output-centric reward。§2/Figure 3
把 final reward 分成 safety score $s_i$ 与 helpfulness $h_i$，unsafe output 得到强惩罚，policy
允许时继续优化 direct/indirect helpfulness；§3 的 controlled ablation 才是判断方法增益的关键，
不是 GPT-5 与旧产品的无控制版本比较。它支持“在该训练/评测设置下减少 dual-use 过拒并降低
残余失败严重度”[D]，不支持“安全与有用性已无 trade-off”[U]。

### 12.3 34 份 system card 反映的是评测对象分化 [D]

system-card 谱系可以按 attack surface 分成五组：

1. **通用/推理模型：** GPT-4、GPT-4o、GPT-4.5、o1-preview/o1/o3-mini/o3-o4、GPT-5
   及 5.1-5.6 系列；重点从 content policy 扩到 CBRN、cyber、persuasion、autonomy、scheming、
   sensitive conversations 与 model-specific safeguards。
2. **视觉/生成媒体：** GPT-4V、DALL·E 3、Sora/Sora 2、4o image、ChatGPT Images 2.0；
   额外覆盖 person/public-figure、sexual/minor content、provenance、likeness、image/video-specific
   classifiers 与 distribution channels。
3. **computer/tool agents：** Operator、deep research、ChatGPT agent；额外覆盖 prompt injection、
   destructive action、confirmation、credential/data exfiltration、web/environment reliability。
4. **code agents：** Codex、o3 Operator、GPT-5-Codex、5.1-Codex-Max、5.2-Codex、5.3-Codex；
   额外覆盖 repository/tool permissions、cyber capability、misaligned actions、sandbox 与 monitoring。
5. **specialists/open weights/audio：** gpt-oss、gpt-oss-safeguard、GPT-Rosalind、GPT-Live；
   分别改变权重可获得性、policy classifier、bio domain 与 real-time audio attack surface。

不同 card 不能被压成一张代际表。`system-card-gpt-5-6` Introduction/PDF p. 2 把 Sol、Terra、
Luna 的 Bio/Chem 与 Cyber 按 High 处理而 AI Self-Improvement 未达 High；§7/Figures 8-10 分析
agentic coding 中的 misaligned behavior；§9/Figure 21 给 capability classifications；§9.4.4/
PDF p. 76 披露自动 universal-jailbreak 搜索超过 700,000 A100e GPU-hours。该搜索 effort 是
攻击覆盖投入 [D]，不是训练 compute，更不是“剩余越狱概率为零” [U]。

### 12.4 Preparedness v2 与 Frontier Governance Framework 解决不同问题 [D/U]

`web-159d28217d06` Preparedness Framework v2 的 Table 1（PDF pp. 4-6）定义 Bio/Chem、Cyber、
AI Self-Improvement 三个 Tracked Categories 及 High/Critical thresholds；Table 2（pp. 6-7）
列 Research Categories；§4（pp. 10-12）说明 safeguards、sufficiency 与 marginal risk；Appendix B-C
（pp. 15-20）给 decision practices 和示例 controls。High 要求部署前把相关 severe-harm risk
充分降低；Critical 还要求开发阶段 safeguards。[D] “充分”仍依赖内部 threat model、证据与
决策程序，不能从公开 PDF 独立算出残余风险 [U]。

`web-ac3d09a478fb` Frontier Governance Framework 则是面向法律/组织过程的框架：printed
pp. 03-13 描述 systemic-risk identification/analysis/acceptance、risk tiers、mitigations 与 critical
incident response，pp. 14-20 描述 security、model reporting、external input、responsibility 和
change management。其风险类别包括 cyber offense、CBRN、harmful manipulation、loss of
control。[D] 它与 Preparedness 有重叠却不等价：前者公开合规/治理 process，后者定义内部
frontier capability tracking 与 deployment gates。两份都是第一方承诺与流程披露，不是第三方
执行审计 [U]。

### 12.5 Threat reports 与事故页：真实世界证据有选择机制 [D/U]

2024-2026 的 covert-influence/state-affiliated/malicious-use 系列记录把安全证据从 benchmark
扩展到 accounts、campaigns、TTPs、disruptions 与跨平台 reach。`rss-79aae2d11d48` 的 2024
报告在 “Threats and Trends” 与 case studies 中称所列五项行动在 Breakout Scale 上均不高于 2，
并区分 content/productivity gain 与 authentic-audience reach。[D] 这支持“在已发现、已调查的
样本中，模型使用未自动造成突破性传播”，不支持“所有未发现行动都低影响” [U]。

sycophancy、ChatGPT outage、Mixpanel security incident 等记录同样重要，因为它们观察的是
release/operation failure，而非静态 checkpoint。事故证据至少需要：影响窗口、受影响对象、
检测机制、root cause、mitigation、复发控制与 denominator。只有标题/RSS 元数据时，本章只确认
事件页存在 [D-meta]；不根据标题补写 root cause。

### 12.6 三个可证伪 safety 判断 [I]

1. **安全正在从“单一 aligned model”转成版本化 control system。** 若 pinned checkpoint 在
   去掉 runtime layers 后与完整 stack 的攻击成功率、误拒、事故率完全相同，这个判断应收缩。
2. **reasoning 同时扩大能力与监督面。** 若在 matched capability/budget 下 CoT/spec reasoning
   不增加 policy adherence 或 monitorability，或显著增加隐蔽规避，则不能把 reasoning 当安全杠杆。
3. **更厚的披露不等于更强的独立保证。** 只有当外部审计能从 cards/frameworks 重建样本、
   elicitation、grader、safeguard test 和 decision trail，并得到一致结论，才可提升为 assurance。

## 13. 广义 OpenAI 机构署名研究：122 条 affiliated 不能倒灌成旗舰配方

### 13.1 为什么必须保留，又为什么必须隔离

122 条 `affiliated` 记录回答的是“公开论文的作者/权威元数据中是否有明确 OpenAI 机构关系”，
不是“OpenAI 是否把它用于 GPT”。保留它们能看见实验室人员跨越的 systems、algorithms、RL、
robotics、interpretability、社会科学、法律、医学、生物、物理、材料、网络与量子研究；隔离它们
则防止把一位作者的工作级署名误写成组织级 roadmap。

按 inventory topics 做**非互斥**计数，122 条中有 society/governance 18、efficiency/systems 17、
RL 10、evaluation 9、coding 9、safety/alignment 9、data/training 10、interpretability 7、
science/biology 5；另有大量由 OpenAlex topic taxonomy 标出的 graph/optimization、robotics、
vision/generative、health/materials、law/economics 与 privacy/security 工作。数字非互斥，不能相加。

### 13.2 六个研究簇与严格边界 [D/U]

| 研究簇 | 代表记录 | work-level 可确认 | 对 GPT/OpenAI 产品仍未知 |
|---|---|---|---|
| RL/robotics/control | `doi-b6016bd48177` GAIL、`doi-0d4d4a16d1a2` CPO、`doi-de7104b25a2f` PLATO、`doi-6f84d440fb92` dexterous manipulation | 方法、实验与作者机构 [D] | 是否进入后续 language-agent RL recipe [U] |
| systems/hardware/algorithms | `doi-7519c5aaba75` 8-Tb/s SuperNIC、`doi-420f3658e493` Mercury、`doi-7c7db3a53d76` HALO、graph/streaming/quantum records | 独立系统或算法贡献 [D] | 生产训练/serving 是否部署 [U] |
| interpretability/robustness | Circuits/Activation Atlas discussion、`doi-82af25041c5d` double descent、robust-feature records | 局部机制与 robustness 研究 [D] | 对 GPT-5.x failure 的直接外推 [U] |
| governance/economics/law | `doi-17545e6c66b1` measurement in AI policy、`doi-b1f0d10f19ed` democracy、法律/竞争/古典法秩序记录 | 作者工作的论证与数据 [D] | 公司采纳为正式政策 [U] |
| medicine/biology/science | GeneBench-Pro、rare-disease reanalysis、Evo 2、protein tokenizers、COVID/ophthalmology/physics/materials | 特定领域研究与署名 [D] | 临床有效性、产品集成或通用模型训练来源 [U] |
| human/social impact | extended-chatbot RCT、generative-AI social impact、red-teaming human factors、labor capability | 指定样本/协议下的社会或交互证据 [D] | 全体用户因果效应和长期 population impact [U] |

特别要防止两种方向相反的错误。第一，标题看似远离 AI（例如法律史、野生动物 reversal
learning、眼科 workforce、理论物理或 permutation theory）不意味着 inventory 错；它可能真实
反映某位作者在该时点的 OpenAI 署名。第二，真实署名也不意味着这篇工作属于 flagship 模型线。
附录逐条保留 URL 和边界，正是为了同时允许发现与禁止倒灌。

### 13.3 版本、同源条目与 taxonomy 噪声

同一 work 可能有 arXiv、conference、journal、dataset、官方介绍页；inventory 只在实质不同或
需要保留独立 provenance 时分开。OpenAlex topic 是发现/聚类辅助，不是作者的 claim。例如
某些 RL 论文被自动 topic 映射到 society/governance，某些 systems 论文带出无关长尾主题；正文
只用手工可解释的大类，不把自动 topic 名称当结论。[U] 若要研究组织知识流，仍需要 author
employment dates、project acknowledgments、code/repo commits、card citations 或明确采用声明。

## 14. 全谱系综合：真正连接各研究线的不是模型名，而是六个约束

### 14.1 六个共同约束 [I]

从 657 条记录中可以抽出一组跨语言、多模态、agent、安全与系统研究都反复出现的约束：

1. **可扩展监督：** 从 human comparison、RBR、process verifier 到 model judge/monitor，关键是
   feedback 在规模扩大时是否保真，而不是标签叫 RLHF 还是 RL。
2. **环境真实性：** Gym/Universe、browser、Minecraft、OS、repository、lab 的 observation/action/
   failure distribution 决定 policy 学到了什么。
3. **稀缺资源分配：** pretraining compute、test-time reasoning、parallel samples、tools、monitor
   budget、network/storage/latency 都是同一优化问题中的不同资源。
4. **评价安全：** benchmark contamination、grader bias、reward hacking、stale dependencies、第三方
   页面变化会让“能力提升”变成测量工件。
5. **授权与可逆性：** 输出文字、写代码、点 GUI、运行命令、控制实验室的危害半径不同；能力相同
   时，permission/confirmation/sandbox/rollback 决定 deployment risk。
6. **披露可重建性：** 越接近闭源旗舰，公开材料越常足以评估部分能力/风险却不足以重建训练；
   open-weight 也可能缺 data/reward/compute/runtime。

### 14.2 一个统一的 agent/reasoning/safety 分解

对任一部署系统，观察到的成功与风险可写成：

$$
Y = F(W, P, E, H, B, A, S, J, T),
$$

其中 $W$ 是 weights，$P$ 是 prompt/policy，$E$ 是 environment，$H$ 是 harness/state/memory，
$B$ 是 reasoning/tool budget，$A$ 是 authorization，$S$ 是 safeguards，$J$ 是 judge/monitor，
$T$ 是时间与外部资源快照。多数发布比较同时改变多个坐标，所以可以确认系统版本更强或更安全
[D]，却无法把差值唯一归因到某一训练算法 [U]。

### 14.3 八个可证伪综合判断

| 判断 [I] | 支持它的公开模式 | 会推翻或显著收缩它的证据 |
|---|---|---|
| GPT 透明度从 recipe 重心转向 eval/deployment 重心 | GPT-4 起 architecture/data/compute 缺失，cards/evals 增厚 | 新旗舰公开足以端到端重建的 data/model/optimizer/RL recipe |
| reasoning scaling 是训练与运行时共同扩展 | o1/o-series 的 RL、budget、sampling/reranking 分层 | matched checkpoint 下所有增益均由提示/路由工件解释 |
| agent 能力是模型-环境共设计 | WebGPT/VPT/Operator/Codex 的 action spaces 不同 | frozen model 跨 harness/environment 的排序与成功率完全不变 |
| verifier 质量会成为主要瓶颈 | PRM、code tests、tool feedback、science replication | verifier error 对 policy 学习与选择没有因果影响 |
| safety 由单模型属性转成多层控制栈 | cards、runtime classifier、sandbox、threat ops、governance | 去掉外层控制不改变真实攻击/事故/误拒指标 |
| monitorability 是需要预算和版本管理的资源 | CoT monitor/controllability scaling 结果 | capability 增长后任意弱 monitor 仍保持稳定完美检测 |
| benchmark 会成为需要退役的产品 | SWE-bench Verified 从引入到停止使用 | 长期 contamination audit 下 protocol 保持区分度且结论稳定 |
| affiliated breadth 反映人才网络而非模型 lineage | 122 条跨域 work-level 署名 | cards/reports 明确逐篇链接到训练数据、架构或生产系统 |

## 15. 公开制品、归档完整性与关键未知

### 15.1 当前本地证据层 [C]

冻结清单有 657 条，其中 222 条保存公开 PDF route；本地
`research/literature/archive/pdf/openai/` 与 `txt/openai/` 各有 **220** 个按 inventory ID 命名的
PDF/全文抽取。下载器验证 PDF magic bytes 与 `pdfinfo`，manifest 保留 URL、status、hash 和错误；
文本用于检索，涉及图表/布局时回到 PDF。抽样视觉核验已覆盖 GPT-4 report、GPT-5.6 card、
Preparedness v2、Frontier Governance Framework 与 InfoGAN，均可正常阅读。[C] 这确认的是本地
制品与渲染，不是论文结果复现 [R]。

剩余 **2** 个有 route 但尚无本地 PDF 的记录是 publisher/access 层失败（403）；
它们仍保留 primary/DOI 和失败状态，不能把下载失败误写成“论文不存在”。其余 435 条没有
PDF route，包含 HTML-native 一手报告、RSS/sitemap
元数据和只有 DOI/landing page 的工作；`pdf_url: null` 本身不代表低质量，也不代表正文已归档。

### 15.2 复现矩阵

| 对象 | 本章做到 | 未做到，因此标签 |
|---|---|---|
| inventory identity/provenance | 657 条 schema、ID、URL、tier、affiliation evidence 可审计 [C] | 私有/撤下/未索引 universe 穷尽 [U] |
| PDF archive | 220 份 PDF+text，本地可渲染/检索 [C] | 2 份 publisher route 未归档；HTML 未统一离线渲染 [U] |
| 论文/card 阅读 | 重大数字给 page/section/table/figure；逐条附录给边界 [D] | 657 条每个实验的逐项独立复算 [U] |
| artifacts | 对公开代码/权重/dataset 只确认可观察性 [C] | artifact completeness、build/train reproducibility [U] |
| 模型能力与安全 | 整理一手披露及 protocol boundary [D] | 相同环境的跨模型重跑 [R 未建立] |
| 训练 recipe | 早期论文可重建较多方法骨架 [D] | 最新旗舰 corpus/optimizer/RL/runtime 端到端复现 [U] |

### 15.3 最新闭源旗舰的最小未知集 [U]

- exact architecture、parameterization、tokenizer、context curriculum 与 multimodal/audio/video
  token interfaces；
- corpus sources、license/provenance、deduplication、mixture weights、unique tokens、contamination
  audit 与删除/opt-out 执行；
- optimizer/schedule/precision/parallelism、failed runs、total FLOPs/GPU-hours、energy/cost；
- SFT/preference/process/RL datasets、reward components/weights、rollout counts、policy lag、
  rejection/selection/distillation details；
- router、reasoning effort、memory/compaction、tool policy、retry/fallback、sandbox 与 live update cadence；
- eval prompts/splits、all denominators、judge calibration、confidence intervals、negative/failed results；
- runtime safeguard thresholds、false positive/negative、adaptive attack coverage、incident denominator；
- system-card checkpoint 与当前 production stack 的完整差异。

这些未知项不是“苛求开源”的修辞，而是决定哪些因果结论可被检验的变量清单。没有它们，最稳妥
的单位是版本化 system release，不是臆测的单一 checkpoint。

## 16. 657/657 逐条审计地图

以下附录按 `core`、`direct`、`affiliated` 和 inventory 原顺序逐条列出。每行至少包含 inventory
ID、primary URL、tier/type 与证据边界；它是 identity/provenance audit，不表示每一行都支持正文
的因果结论。`local PDF+text` 只标 [C] 制品存在，绝不自动标 [R]。


### 16.1 Core：旗舰、正式卡片与核心报告（79）

| 日期 / 记录 | 条目与 primary URL | 类型 | 审计边界 |
|---|---|---|---|
| 2026-07-09 <code>web-e19fbb75eb44</code> | GPT-5.6: Frontier intelligence that scales with your ambition<br><https://openai.com/index/gpt-5-6> | <code>core</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: foundation_models, efficiency_systems, release. |
| 2026-07-09 <code>system-card-gpt-5-6</code> | GPT-5.6 System Card<br><https://deploymentsafety.openai.com/gpt-5-6> | <code>core</code> / <code>system_card</code> | [D/C] official core record；local PDF+text；论文/卡片结果未在本章复跑 [R 未建立]。 Topics: foundation_models, deployment_safety. |
| 2026-07-09 <code>web-07aa5a8a45a2</code> | GPT-5.5 Bio Bug Bounty<br><https://openai.com/index/bio-bug-bounty> | <code>core</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: foundation_models, safety. |
| 2026-07-08 <code>system-card-gpt-live</code> | GPT-Live System Card<br><https://deploymentsafety.openai.com/gpt-live> | <code>core</code> / <code>system_card</code> | [D/C] official core record；local PDF+text；论文/卡片结果未在本章复跑 [R 未建立]。 Topics: foundation_models, deployment_safety. |
| 2026-05-05 <code>web-01a8b52eee81</code> | GPT-5.5 Instant: smarter, clearer, and more personalized<br><https://openai.com/index/gpt-5-5-instant> | <code>core</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: foundation_models, release. |
| 2026-05-05 <code>web-3bbc92d9c76e</code> | GPT-5.5 Instant System Card<br><https://openai.com/index/gpt-5-5-instant-system-card> | <code>core</code> / <code>system_card</code> | [D/C] official core record；local PDF+text；论文/卡片结果未在本章复跑 [R 未建立]。 Topics: foundation_models, publication, safety. |
| 2026-04-23 <code>web-dfe7e0c6ff39</code> | Introducing GPT-5.5<br><https://openai.com/index/introducing-gpt-5-5> | <code>core</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: foundation_models, release. |
| 2026-04-23 <code>web-a07eb39dec04</code> | GPT-5.5 System Card<br><https://openai.com/index/gpt-5-5-system-card> | <code>core</code> / <code>system_card</code> | [D/C] official core record；local PDF+text；论文/卡片结果未在本章复跑 [R 未建立]。 Topics: foundation_models, publication, safety. |
| 2026-03-16 <code>rss-8a8601b30dd9</code> | Why Codex Security Doesn’t Include a SAST Report<br><https://openai.com/index/why-codex-security-doesnt-include-sast> | <code>core</code> / <code>technical_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: agents_tool_use, coding_software, security. |
| 2026-03-06 <code>rss-5becf449a527</code> | Codex Security: now in research preview<br><https://openai.com/index/codex-security-now-in-research-preview> | <code>core</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: agents_tool_use, coding_software, security. |
| 2026-03-05 <code>web-eea6230f612a</code> | Introducing GPT-5.4<br><https://openai.com/index/introducing-gpt-5-4> | <code>core</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: foundation_models, release. |
| 2026-03-05 <code>web-f59abf9daf09</code> | GPT-5.4 Thinking System Card<br><https://openai.com/index/gpt-5-4-thinking-system-card> | <code>core</code> / <code>system_card</code> | [D/C] official core record；local PDF+text；论文/卡片结果未在本章复跑 [R 未建立]。 Topics: foundation_models, publication, safety. |
| 2026-03-03 <code>web-46b3e364b737</code> | GPT-5.3 Instant: Smoother, more useful everyday conversations<br><https://openai.com/index/gpt-5-3-instant> | <code>core</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: foundation_models, release. |
| 2026-03-03 <code>web-b4a245738a8f</code> | GPT-5.3 Instant System Card<br><https://openai.com/index/gpt-5-3-instant-system-card> | <code>core</code> / <code>system_card</code> | [D/C] official core record；local PDF+text；论文/卡片结果未在本章复跑 [R 未建立]。 Topics: foundation_models, publication, safety. |
| 2026-02-13 <code>web-407f7c69ffbc</code> | GPT-5.2 derives a new result in theoretical physics<br><https://openai.com/index/new-result-theoretical-physics> | <code>core</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: foundation_models, science_biology, publication. |
| 2026-02-12 <code>web-4a841ea7ec7c</code> | Introducing GPT-5.3-Codex-Spark<br><https://openai.com/index/introducing-gpt-5-3-codex-spark> | <code>core</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: foundation_models, agents_tool_use, coding_software. |
| 2026-02-05 <code>web-9aba7995b6a5</code> | Introducing GPT-5.3-Codex<br><https://openai.com/index/introducing-gpt-5-3-codex> | <code>core</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: foundation_models, agents_tool_use, coding_software. |
| 2026-02-05 <code>web-dd64fe3084a4</code> | GPT-5.3-Codex System Card<br><https://openai.com/index/gpt-5-3-codex-system-card> | <code>core</code> / <code>system_card</code> | [D/C] official core record；local PDF+text；论文/卡片结果未在本章复跑 [R 未建立]。 Topics: foundation_models, agents_tool_use, coding_software. |
| 2026-02-05 <code>web-aec434a838fa</code> | GPT-5 lowers the cost of cell-free protein synthesis<br><https://openai.com/index/gpt-5-lowers-protein-synthesis-cost> | <code>core</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: foundation_models, science_biology, publication. |
| 2025-12-18 <code>web-cd8d6c8023f2</code> | Introducing GPT-5.2-Codex<br><https://openai.com/index/introducing-gpt-5-2-codex> | <code>core</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: foundation_models, agents_tool_use, coding_software. |
| 2025-12-18 <code>web-7fda7ddac79f</code> | Addendum to GPT-5.2 System Card: GPT-5.2-Codex<br><https://openai.com/index/gpt-5-2-codex-system-card> | <code>core</code> / <code>system_card</code> | [D/C] official core record；local PDF+text；论文/卡片结果未在本章复跑 [R 未建立]。 Topics: foundation_models, agents_tool_use, coding_software. |
| 2025-12-11 <code>web-18ba0c6ad33e</code> | Update to GPT-5 System Card: GPT-5.2<br><https://openai.com/index/gpt-5-system-card-update-gpt-5-2> | <code>core</code> / <code>system_card</code> | [D/C] official core record；local PDF+text；论文/卡片结果未在本章复跑 [R 未建立]。 Topics: foundation_models, publication, safety. |
| 2025-12-11 <code>web-10ab1c39be14</code> | Introducing GPT-5.2<br><https://openai.com/index/introducing-gpt-5-2> | <code>core</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: foundation_models, release. |
| 2025-11-19 <code>web-13d4a9bfec63</code> | GPT-5.1-Codex-Max System Card<br><https://openai.com/index/gpt-5-1-codex-max-system-card> | <code>core</code> / <code>system_card</code> | [D/C] official core record；local PDF+text；论文/卡片结果未在本章复跑 [R 未建立]。 Topics: foundation_models, agents_tool_use, coding_software. |
| 2025-11-12 <code>web-5bea303cfdb1</code> | GPT-5.1: A smarter, more conversational ChatGPT<br><https://openai.com/index/gpt-5-1> | <code>core</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: foundation_models, release. |
| 2025-11-12 <code>web-c8672ece6975</code> | GPT-5.1 Instant and GPT-5.1 Thinking System Card Addendum<br><https://openai.com/index/gpt-5-system-card-addendum-gpt-5-1> | <code>core</code> / <code>system_card</code> | [D/C] official core record；local PDF+text；论文/卡片结果未在本章复跑 [R 未建立]。 Topics: foundation_models, publication, safety. |
| 2025-10-29 <code>web-e943411c2e8b</code> | gpt-oss-safeguard technical report<br><https://openai.com/index/gpt-oss-safeguard-technical-report> | <code>core</code> / <code>technical_report</code> | [D/C] official core record；local PDF+text；论文/卡片结果未在本章复跑 [R 未建立]。 Topics: foundation_models, release, safety. |
| 2025-10-27 <code>web-408b38efbeaf</code> | Addendum to GPT-5 System Card: Sensitive conversations<br><https://openai.com/index/gpt-5-system-card-sensitive-conversations> | <code>core</code> / <code>system_card</code> | [D/C] official core record；local PDF+text；论文/卡片结果未在本章复跑 [R 未建立]。 Topics: foundation_models, publication, safety. |
| 2025-10-06 <code>web-0a0c5c86c5bf</code> | Codex is now generally available<br><https://openai.com/index/codex-now-generally-available> | <code>core</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: agents_tool_use, coding_software, release. |
| 2025-09-30 <code>web-8530f5b0e216</code> | Sora 2 System Card<br><https://openai.com/index/sora-2-system-card> | <code>core</code> / <code>system_card</code> | [D/C] official core record；local PDF+text；论文/卡片结果未在本章复跑 [R 未建立]。 Topics: multimodal, publication, safety. |
| 2025-09-30 <code>web-661a3476d3d5</code> | Sora 2 is here<br><https://openai.com/index/sora-2> | <code>core</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: multimodal, release, research. |
| 2025-09-15 <code>web-2d6622764daa</code> | Addendum to GPT-5 system card: GPT-5-Codex<br><https://openai.com/index/gpt-5-system-card-addendum-gpt-5-codex> | <code>core</code> / <code>system_card</code> | [D/C] official core record；local PDF+text；论文/卡片结果未在本章复跑 [R 未建立]。 Topics: foundation_models, agents_tool_use, coding_software. |
| 2025-08-07 <code>web-05738f6c55b7</code> | Introducing GPT-5<br><https://openai.com/index/introducing-gpt-5> | <code>core</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: foundation_models, release. |
| 2025-08-07 <code>web-eb6095ff0b58</code> | GPT-5 System Card<br><https://openai.com/index/gpt-5-system-card> | <code>core</code> / <code>system_card</code> | [D/C] official core record；local PDF+text；论文/卡片结果未在本章复跑 [R 未建立]。 Topics: foundation_models, publication, safety. |
| 2025-08-05 <code>web-1c46fdf17c79</code> | gpt-oss-120b & gpt-oss-20b Model Card<br><https://openai.com/index/gpt-oss-model-card> | <code>core</code> / <code>model_card</code> | [D/C] official core record；local PDF+text；论文/卡片结果未在本章复跑 [R 未建立]。 Topics: foundation_models, publication, safety. |
| 2025-07-17 <code>web-b074ccdb2d9e</code> | Introducing ChatGPT agent<br><https://openai.com/index/introducing-chatgpt-agent> | <code>core</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: foundation_models, agents_tool_use, release. |
| 2025-07-17 <code>web-7a7213b68631</code> | ChatGPT agent System Card<br><https://openai.com/index/chatgpt-agent-system-card> | <code>core</code> / <code>system_card</code> | [D/C] official core record；local PDF+text；论文/卡片结果未在本章复跑 [R 未建立]。 Topics: foundation_models, agents_tool_use, publication. |
| 2025-05-23 <code>web-89bffd91b9ae</code> | Addendum to OpenAI o3 and o4-mini system card: OpenAI o3 Operator<br><https://openai.com/index/o3-o4-mini-system-card-addendum-operator-o3> | <code>core</code> / <code>system_card</code> | [D/C] official core record；local PDF+text；论文/卡片结果未在本章复跑 [R 未建立]。 Topics: agents_tool_use, publication, safety. |
| 2025-05-16 <code>web-abb942f337e2</code> | Introducing Codex<br><https://openai.com/index/introducing-codex> | <code>core</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: agents_tool_use, coding_software, release. |
| 2025-05-16 <code>web-183d58a311d5</code> | Addendum to o3 and o4-mini system card: Codex<br><https://openai.com/index/o3-o4-mini-codex-system-card-addendum> | <code>core</code> / <code>system_card</code> | [D/C] official core record；local PDF+text；论文/卡片结果未在本章复跑 [R 未建立]。 Topics: agents_tool_use, coding_software, publication. |
| 2025-04-16 <code>web-c02bd9ed41d6</code> | OpenAI o3 and o4-mini System Card<br><https://openai.com/index/o3-o4-mini-system-card> | <code>core</code> / <code>system_card</code> | [D/C] official core record；local PDF+text；论文/卡片结果未在本章复跑 [R 未建立]。 Topics: publication, safety. |
| 2025-04-16 <code>web-646f1157fd8b</code> | Introducing OpenAI o3 and o4-mini<br><https://openai.com/index/introducing-o3-and-o4-mini> | <code>core</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: release. |
| 2025-04-14 <code>web-1cb94724584b</code> | Introducing GPT-4.1 in the API<br><https://openai.com/index/gpt-4-1> | <code>core</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: foundation_models, publication, research. |
| 2025-03-25 <code>web-5223fe491ee4</code> | Addendum to GPT-4o System Card: 4o image generation<br><https://openai.com/index/gpt-4o-image-generation-system-card-addendum> | <code>core</code> / <code>system_card</code> | [D/C] official core record；local PDF+text；论文/卡片结果未在本章复跑 [R 未建立]。 Topics: foundation_models, multimodal, publication. |
| 2025-02-27 <code>web-2f8a38bafb20</code> | OpenAI GPT-4.5 System Card<br><https://openai.com/index/gpt-4-5-system-card> | <code>core</code> / <code>system_card</code> | [D/C] official core record；local PDF+text；论文/卡片结果未在本章复跑 [R 未建立]。 Topics: foundation_models, publication, safety. |
| 2025-02-25 <code>web-386e34a7513e</code> | Deep research System Card<br><https://openai.com/index/deep-research-system-card> | <code>core</code> / <code>system_card</code> | [D/C] official core record；local PDF+text；论文/卡片结果未在本章复跑 [R 未建立]。 Topics: publication, safety. |
| 2025-01-31 <code>web-8fe8674e0195</code> | OpenAI o3-mini System Card<br><https://openai.com/index/o3-mini-system-card> | <code>core</code> / <code>system_card</code> | [D/C] official core record；local PDF+text；论文/卡片结果未在本章复跑 [R 未建立]。 Topics: publication. |
| 2025-01-31 <code>web-cd0416914af5</code> | OpenAI o3-mini<br><https://openai.com/index/openai-o3-mini> | <code>core</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: release. |
| 2025-01-23 <code>web-24bc39ed487b</code> | Operator System Card<br><https://openai.com/index/operator-system-card> | <code>core</code> / <code>system_card</code> | [D/C] official core record；local PDF+text；论文/卡片结果未在本章复跑 [R 未建立]。 Topics: agents_tool_use, publication, safety. |
| 2025-01-23 <code>web-74eeb7c9fb20</code> | Computer-Using Agent<br><https://openai.com/index/computer-using-agent> | <code>core</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: agents_tool_use, release. |
| 2024-12-09 <code>web-a51b48a9155c</code> | Sora System Card<br><https://openai.com/index/sora-system-card> | <code>core</code> / <code>system_card</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: multimodal, publication, safety. |
| 2024-12-05 <code>web-dcf8ab338d73</code> | OpenAI o1 System Card<br><https://openai.com/index/openai-o1-system-card> | <code>core</code> / <code>system_card</code> | [D/C] official core record；local PDF+text；论文/卡片结果未在本章复跑 [R 未建立]。 Topics: publication. |
| 2024-09-12 <code>web-6a9179dfe54e</code> | OpenAI o1-mini<br><https://openai.com/index/openai-o1-mini-advancing-cost-efficient-reasoning> | <code>core</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: release. |
| 2024-08-08 <code>web-e8cc130e4af6</code> | GPT-4o System Card<br><https://openai.com/index/gpt-4o-system-card> | <code>core</code> / <code>system_card</code> | [D/C] official core record；local PDF+text；论文/卡片结果未在本章复跑 [R 未建立]。 Topics: foundation_models, publication. |
| 2024-07-18 <code>web-031c86eaad2b</code> | GPT-4o mini: advancing cost-efficient intelligence<br><https://openai.com/index/gpt-4o-mini-advancing-cost-efficient-intelligence> | <code>core</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: foundation_models, efficiency_systems, release. |
| 2024-02-15 <code>web-0fe343211040</code> | Video generation models as world simulators<br><https://openai.com/index/video-generation-models-as-world-simulators> | <code>core</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: multimodal, publication. |
| 2023-10-03 <code>web-ba80e5e27a11</code> | DALL·E 3 system card<br><https://openai.com/index/dall-e-3-system-card> | <code>core</code> / <code>system_card</code> | [D/C] official core record；local PDF+text；论文/卡片结果未在本章复跑 [R 未建立]。 Topics: multimodal, release. |
| 2023-09-25 <code>web-a95452783369</code> | GPT-4V(ision) system card<br><https://openai.com/index/gpt-4v-system-card> | <code>core</code> / <code>system_card</code> | [D/C] official core record；local PDF+text；论文/卡片结果未在本章复跑 [R 未建立]。 Topics: foundation_models, release. |
| 2023-03-14 <code>web-3375c5d27d9b</code> | GPT-4<br><https://openai.com/index/gpt-4-research> | <code>core</code> / <code>blog_with_report</code> | [D/C] official core record；local PDF+text；论文/卡片结果未在本章复跑 [R 未建立]。 Topics: foundation_models, milestone. |
| 2022-12-16 <code>web-28adad6ac48b</code> | Point-E: A system for generating 3D point clouds from complex prompts<br><https://openai.com/index/point-e> | <code>core</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: publication. |
| 2022-09-21 <code>web-2b132b9f05ef</code> | Introducing Whisper<br><https://openai.com/index/whisper> | <code>core</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: release. |
| 2022-07-25 <code>web-f3db758df8fe</code> | A hazard analysis framework for code synthesis large language models<br><https://openai.com/index/a-hazard-analysis-framework-for-code-synthesis-large-language-models> | <code>core</code> / <code>technical_report</code> | [D/C] official core record；local PDF+text；论文/卡片结果未在本章复跑 [R 未建立]。 Topics: coding_software, publication. |
| 2022-06-28 <code>web-cf1134280999</code> | DALL·E 2 pre-training mitigations<br><https://openai.com/index/dall-e-2-pre-training-mitigations> | <code>core</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: multimodal, data_training, publication. |
| 2022-05-18 <code>rss-0df61418b28f</code> | DALL·E 2 research preview update<br><https://openai.com/index/dall-e-2-update> | <code>core</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: multimodal, Product. |
| 2021-01-05 <code>web-f14c51ae873c</code> | DALL·E: Creating images from text<br><https://openai.com/index/dall-e> | <code>core</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: multimodal, milestone. |
| 2021-01-05 <code>web-bc37d5969edc</code> | CLIP: Connecting text and images<br><https://openai.com/index/clip> | <code>core</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: multimodal, milestone. |
| 2020-04-30 <code>web-3039e4df07aa</code> | Jukebox<br><https://openai.com/index/jukebox> | <code>core</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: release. |
| 2019-11-05 <code>web-31066a8ed5c3</code> | GPT-2: 1.5B release<br><https://openai.com/index/gpt-2-1-5b-release> | <code>core</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: foundation_models, release. |
| 2019-08-20 <code>web-aec2fcecd95e</code> | GPT-2: 6-month follow-up<br><https://openai.com/index/gpt-2-6-month-follow-up> | <code>core</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: foundation_models, publication. |
| 未标日期 <code>web-486eae78d321</code> | Sora research preview<br><https://openai.com/index/sora> | <code>core</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: multimodal, milestone. |
| 未标日期 <code>web-3d4c2aabbd9c</code> | Introducing GPT-4.5<br><https://openai.com/index/introducing-gpt-4-5> | <code>core</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: foundation_models, release. |
| 未标日期 <code>web-97ef93e629bf</code> | GPT-4 product release<br><https://openai.com/index/gpt-4> | <code>core</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: foundation_models, release. |
| 未标日期 <code>web-b3d8859fa802</code> | DALL·E 3<br><https://openai.com/index/dall-e-3> | <code>core</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: multimodal, publication, release. |
| 未标日期 <code>web-4a8bccb8dab1</code> | DALL·E 2<br><https://openai.com/index/dall-e-2> | <code>core</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: multimodal, release. |
| 2026-06-26 <code>gpt-5.6-preview-system-card</code> | GPT-5.6 Preview System Card<br><https://deploymentsafety.openai.com/gpt-5-6-preview> | <code>core</code> / <code>system_card</code> | [D/C] official core record；local PDF+text；论文/卡片结果未在本章复跑 [R 未建立]。 Topics: foundation_models, safety, evaluation. |
| 2026-06-03 <code>gpt-rosalind-5.5-system-card</code> | GPT-Rosalind-5.5 System Card<br><https://deploymentsafety.openai.com/gpt-rosalind-5-5> | <code>core</code> / <code>system_card</code> | [D/C] official core record；local PDF+text；论文/卡片结果未在本章复跑 [R 未建立]。 Topics: foundation_models, safety, evaluation. |
| 2026-04-21 <code>chatgpt-images-2.0-system-card</code> | ChatGPT Images 2.0 System Card<br><https://deploymentsafety.openai.com/chatgpt-images-2-0> | <code>core</code> / <code>system_card</code> | [D/C] official core record；local PDF+text；论文/卡片结果未在本章复跑 [R 未建立]。 Topics: foundation_models, safety, evaluation. |
| 2024-09-12 <code>o1-preview-o1-mini-system-card</code> | OpenAI o1-preview and o1-mini System Card<br><https://openai.com/index/openai-o1-system-card> | <code>core</code> / <code>system_card</code> | [D/C] official core record；local PDF+text；论文/卡片结果未在本章复跑 [R 未建立]。 Topics: foundation_models, safety, evaluation. |
| 2023-03-14 <code>gpt-4-system-card</code> | GPT-4 System Card<br><https://cdn.openai.com/papers/gpt-4-system-card.pdf> | <code>core</code> / <code>system_card</code> | [D/C] official core record；local PDF+text；论文/卡片结果未在本章复跑 [R 未建立]。 Topics: foundation_models, safety, evaluation. |

### 16.2 Direct：OpenAI 第一方研究、工程、评测、安全与治理（456）

| 日期 / 记录 | 条目与 primary URL | 类型 | 审计边界 |
|---|---|---|---|
| 2026-07-16 <code>web-a907415b440a</code> | Why teens deserve access to safe AI<br><https://openai.com/index/why-teens-deserve-access-safe-ai> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: safety. |
| 2026-07-15 <code>rss-23b1197c7433</code> | The US is advancing AI safety through state and federal action<br><https://openai.com/index/advancing-ai-safety-through-state-and-federal-action> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: safety_alignment, Global Affairs. |
| 2026-07-15 <code>web-f8df3e0d3cc8</code> | GPT-Red: Unlocking Self-Improvement for Robustness<br><https://openai.com/index/unlocking-self-improvement-gpt-red> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: foundation_models, safety_alignment, publication. |
| 2026-07-08 <code>web-e9167e656930</code> | Separating signal from noise in coding evaluations<br><https://openai.com/index/separating-signal-from-noise-coding-evaluations> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: evaluation_benchmarks, publication, research. |
| 2026-07-08 <code>rss-e00d55023af7</code> | Our approach to government and national security partnerships<br><https://openai.com/index/government-national-security-partnerships> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: society_governance, security, Global Affairs. |
| 2026-07-08 <code>web-3fb14968493d</code> | Introducing GPT-Live<br><https://openai.com/index/introducing-gpt-live> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: foundation_models, release. |
| 2026-06-30 <code>web-0b8dc2473cab</code> | Introducing GeneBench-Pro<br><https://openai.com/index/introducing-genebench-pro> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: publication, research. |
| 2026-06-30 <code>web-f1f7d4f8001b</code> | Core dump epidemiology: fixing an 18-year-old bug<br><https://openai.com/index/core-dump-epidemiology-data-infrastructure-bug> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: engineering. |
| 2026-06-26 <code>web-30f48121decc</code> | Previewing GPT-5.6 Sol: a next-generation model<br><https://openai.com/index/previewing-gpt-5-6-sol> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: foundation_models, release. |
| 2026-06-22 <code>rss-3696e67c2c9e</code> | Patch the Planet: a Daybreak initiative to support open source maintainers<br><https://openai.com/index/patch-the-planet> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: Security. |
| 2026-06-22 <code>rss-b63e5cb33bb0</code> | Daybreak: Tools for securing every organization in the world<br><https://openai.com/index/daybreak-securing-the-world> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: Security. |
| 2026-06-17 <code>web-381e7f583722</code> | Introducing LifeSciBench<br><https://openai.com/index/introducing-life-sci-bench> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: publication, research. |
| 2026-06-17 <code>rss-5ff2f715f1e6</code> | A near-autonomous AI chemist improves a challenging reaction in medicinal chemistry<br><https://openai.com/index/ai-chemist-improves-reaction> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: science_biology, Research. |
| 2026-06-16 <code>web-c000018ba1f0</code> | Predicting model behavior before release by simulating deployment<br><https://openai.com/index/deployment-simulation> | <code>direct</code> / <code>blog_with_report</code> | [D/C] direct record；local PDF+text；论文/卡片结果未在本章复跑 [R 未建立]。 Topics: research. |
| 2026-06-04 <code>web-c3ab33c62226</code> | Dreaming: Better memory for a more helpful ChatGPT<br><https://openai.com/index/chatgpt-memory-dreaming> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: foundation_models, efficiency_systems, release. |
| 2026-06-03 <code>web-ba562226725d</code> | Introducing new capabilities to GPT-Rosalind<br><https://openai.com/index/introducing-new-capabilities-to-gpt-rosalind> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: foundation_models, release, research. |
| 2026-06-02 <code>web-f4e0423d66a1</code> | Advancing youth safety and opportunity through global leadership<br><https://openai.com/index/advancing-youth-safety-and-opportunity-through-global-leadership> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: safety_alignment, safety. |
| 2026-05-29 <code>web-1c9b44953320</code> | Strengthening societal resilience with Rosalind Biodefense<br><https://openai.com/index/strengthening-societal-resilience-with-rosalind-biodefense> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: society_governance, release, research. |
| 2026-05-29 <code>web-8c84b8422093</code> | A shared playbook for trustworthy third party evaluations<br><https://openai.com/index/trustworthy-third-party-evaluations-foundations> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: evaluation_benchmarks, safety. |
| 2026-05-28 <code>web-ac3d09a478fb</code> | OpenAI’s Frontier Governance Framework<br><https://openai.com/index/openai-frontier-governance-framework> | <code>direct</code> / <code>technical_report</code> | [D/C] direct record；local PDF+text；论文/卡片结果未在本章复跑 [R 未建立]。 Topics: society_governance, safety. |
| 2026-05-28 <code>doi-1a4b22933546</code> | LLM and Agents for Recommendation Systems (LARS)<br><https://doi.org/10.1145/3774905.3795718> | <code>direct</code> / <code>research_paper</code> | [D/U] direct record；无 local PDF+text；主张只到可核验 landing page/metadata。 Topics: foundation_models, agents_tool_use, Recommender Systems and Techniques. |
| 2026-05-27 <code>web-c7affab7451e</code> | Building self-improving tax agents with Codex<br><https://openai.com/index/building-self-improving-tax-agents-with-codex> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: agents_tool_use, coding_software, engineering. |
| 2026-05-22 <code>doi-15001b4baa0b</code> | DraftNEPABench: A Benchmark for Drafting NEPA Document Sections with Coding Agents<br><https://doi.org/10.1145/3786335.3813132> | <code>direct</code> / <code>research_paper</code> | [D/U] direct record；无 local PDF+text；主张只到可核验 landing page/metadata。 Topics: agents_tool_use, evaluation_benchmarks, Machine Learning in Materials Science. |
| 2026-05-20 <code>web-1c93d6c1c725</code> | An OpenAI model has disproved a central conjecture in discrete geometry<br><https://openai.com/index/model-disproves-discrete-geometry-conjecture> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: milestone, research. |
| 2026-05-19 <code>web-f80637b40404</code> | Advancing content provenance for a safer, more transparent AI ecosystem<br><https://openai.com/index/advancing-content-provenance> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: safety. |
| 2026-05-14 <code>web-a1312e54bd28</code> | Helping ChatGPT better recognize context in sensitive conversations<br><https://openai.com/index/chatgpt-recognize-context-in-sensitive-conversations> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: foundation_models, safety. |
| 2026-05-13 <code>rss-8939013f174d</code> | Our response to the TanStack npm supply chain attack<br><https://openai.com/index/our-response-to-the-tanstack-npm-supply-chain-attack> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: security, Security. |
| 2026-05-13 <code>web-4a3442a5ca8c</code> | Building a safe, effective sandbox to enable Codex on Windows<br><https://openai.com/index/building-codex-windows-sandbox> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: agents_tool_use, coding_software, engineering. |
| 2026-05-12 <code>web-fd797e93058c</code> | What Parameter Golf taught us about AI-assisted research<br><https://openai.com/index/what-parameter-golf-taught-us> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: research. |
| 2026-05-08 <code>web-66be12207714</code> | Running Codex safely at OpenAI<br><https://openai.com/index/running-codex-safely> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: agents_tool_use, coding_software, safety. |
| 2026-05-07 <code>rss-a505eac8f113</code> | Scaling Trusted Access for Cyber with GPT-5.5 and GPT-5.5-Cyber<br><https://openai.com/index/gpt-5-5-with-trusted-access-for-cyber> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: foundation_models, efficiency_systems, security. |
| 2026-05-07 <code>web-a6fcbba28838</code> | Introducing Trusted Contact in ChatGPT<br><https://openai.com/index/introducing-trusted-contact-in-chatgpt> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: foundation_models, safety. |
| 2026-05-07 <code>web-24a3c172de48</code> | Advancing voice intelligence with new models in the API<br><https://openai.com/index/advancing-voice-intelligence-with-new-models-in-the-api> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: multimodal, release. |
| 2026-05-05 <code>web-a2a811b50a4c</code> | Unlocking large scale AI training networks with MRC (Multipath Reliable Connection)<br><https://openai.com/index/mrc-supercomputer-networking> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: data_training, efficiency_systems, engineering. |
| 2026-05-05 <code>web-2de76809a598</code> | Advancing youth safety and wellbeing in EMEA<br><https://openai.com/index/advancing-youth-safety-in-emea> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: safety_alignment, safety. |
| 2026-05-04 <code>web-5c3d93bd7160</code> | How OpenAI delivers low-latency voice AI at scale<br><https://openai.com/index/delivering-low-latency-voice-ai-at-scale> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: multimodal, efficiency_systems, engineering. |
| 2026-04-29 <code>web-8b7276cd935e</code> | Where the goblins came from<br><https://openai.com/index/where-the-goblins-came-from> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: publication. |
| 2026-04-29 <code>rss-180302e63d71</code> | Cybersecurity in the Intelligence Age<br><https://openai.com/index/cybersecurity-in-the-intelligence-age> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: security, Global Affairs. |
| 2026-04-28 <code>web-a15adbd33f27</code> | Our commitment to community safety<br><https://openai.com/index/our-commitment-to-community-safety> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: safety_alignment, safety. |
| 2026-04-27 <code>web-f9ff221789e0</code> | An open-source spec for orchestration: Symphony<br><https://openai.com/index/open-source-codex-orchestration-symphony> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: engineering. |
| 2026-04-23 <code>doi-3698c78278b4</code> | GeneBench: Assessing AI Agents for Multi-Stage Inference Problems in Genomics and Quantitative Biology<br><https://doi.org/10.64898/2026.04.22.720113> | <code>direct</code> / <code>research_paper</code> | [D/U] direct record；无 local PDF+text；主张只到可核验 landing page/metadata。 Topics: agents_tool_use, efficiency_systems, science_biology. |
| 2026-04-22 <code>web-8f555fe918e1</code> | Speeding up agentic workflows with WebSockets in the Responses API<br><https://openai.com/index/speeding-up-agentic-workflows-with-websockets> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: agents_tool_use, engineering. |
| 2026-04-22 <code>web-3efc4342b784</code> | Introducing OpenAI Privacy Filter<br><https://openai.com/index/introducing-openai-privacy-filter> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: release, research. |
| 2026-04-22 <code>doi-14ff6a7a93ca</code> | Evaluating large language models for accuracy incentivizes hallucinations<br><https://doi.org/10.1038/s41586-026-10549-w> | <code>direct</code> / <code>research_paper</code> | [D/U] direct record；无 local PDF+text；主张只到可核验 landing page/metadata。 Topics: evaluation_benchmarks, Topic Modeling, Natural Language Processing Techniques. |
| 2026-04-21 <code>web-0f697d747d9c</code> | Introducing ChatGPT Images 2.0<br><https://openai.com/index/introducing-chatgpt-images-2-0> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: foundation_models, multimodal, release. |
| 2026-04-16 <code>doi-e477aeec069b</code> | ORION: An agentic reasoning construct for the analysis of complex human immune profiling<br><https://doi.org/10.64898/2026.04.13.718286> | <code>direct</code> / <code>research_paper</code> | [D/U] direct record；无 local PDF+text；主张只到可核验 landing page/metadata。 Topics: agents_tool_use, vaccines and immunoinformatics approaches, Diabetes and associated disorders. |
| 2026-04-16 <code>web-078f0d540d88</code> | Introducing GPT-Rosalind for life sciences research<br><https://openai.com/index/introducing-gpt-rosalind> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: foundation_models, science_biology, release. |
| 2026-04-16 <code>web-23624d66f30a</code> | Accelerating the cyber defense ecosystem that protects us all<br><https://openai.com/index/accelerating-cyber-defense-ecosystem> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: security, safety. |
| 2026-04-14 <code>web-17bf9cbe1f8c</code> | Trusted access for the next era of cyber defense<br><https://openai.com/index/scaling-trusted-access-for-cyber-defense> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: security, safety. |
| 2026-04-13 <code>doi-18f2241090ad</code> | Human-AI Interaction Alignment: Designing, Evaluating, and Evolving Value-Centered AI For Reciprocal Human-AI Futures<br><https://doi.org/10.1145/3772363.3778710> | <code>direct</code> / <code>research_paper</code> | [D/C] direct record；local PDF+text；论文/卡片结果未在本章复跑 [R 未建立]。 Topics: safety_alignment, evaluation_benchmarks, Ethics and Social Impacts of AI. |
| 2026-04-10 <code>rss-d78a27ea5811</code> | Our response to the Axios developer tool compromise<br><https://openai.com/index/axios-developer-tool-compromise> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: Security. |
| 2026-04-08 <code>web-487938567ca0</code> | Introducing the Child Safety Blueprint<br><https://openai.com/index/introducing-child-safety-blueprint> | <code>direct</code> / <code>technical_report</code> | [D/C] direct record；local PDF+text；论文/卡片结果未在本章复跑 [R 未建立]。 Topics: safety_alignment, safety. |
| 2026-04-06 <code>web-047e7bac94f5</code> | Announcing the OpenAI Safety Fellowship<br><https://openai.com/index/introducing-openai-safety-fellowship> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: safety_alignment, safety. |
| 2026-03-25 <code>web-fb38dfafd80a</code> | Introducing the OpenAI Safety Bug Bounty program<br><https://openai.com/index/safety-bug-bounty> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: safety_alignment, coding_software, safety. |
| 2026-03-25 <code>web-263c2043f1c3</code> | Inside our approach to the Model Spec<br><https://openai.com/index/our-approach-to-the-model-spec> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: publication, research. |
| 2026-03-24 <code>web-d75d7d2ee748</code> | Helping developers build safer AI experiences for teens<br><https://openai.com/index/teen-safety-policies-gpt-oss-safeguard> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: safety. |
| 2026-03-23 <code>web-85771c0fe867</code> | Creating with Sora Safely<br><https://openai.com/index/creating-with-sora-safely> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: multimodal, safety. |
| 2026-03-19 <code>web-b4b340379768</code> | How we monitor internal coding agents for misalignment<br><https://openai.com/index/how-we-monitor-internal-coding-agents-misalignment> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: safety_alignment, agents_tool_use, publication. |
| 2026-03-17 <code>web-4e32c322a737</code> | OpenAI Japan announces Japan Teen Safety Blueprint to put teen safety first<br><https://openai.com/index/japan-teen-safety-blueprint> | <code>direct</code> / <code>technical_report</code> | [D/C] direct record；local PDF+text；论文/卡片结果未在本章复跑 [R 未建立]。 Topics: safety_alignment, safety. |
| 2026-03-11 <code>web-01572b4c34e9</code> | From model to agent: Equipping the Responses API with a computer environment<br><https://openai.com/index/equip-responses-api-computer-environment> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: agents_tool_use, engineering. |
| 2026-03-11 <code>rss-3dec299964e6</code> | Designing AI agents to resist prompt injection<br><https://openai.com/index/designing-agents-to-resist-prompt-injection> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: agents_tool_use, security, Security. |
| 2026-03-10 <code>doi-aa7e89e34dbd</code> | Triton-Sanitizer: A Fast and Device-Agnostic Memory Sanitizer for Triton with Rich Diagnostic Context<br><https://doi.org/10.1145/3779212.3790241> | <code>direct</code> / <code>research_paper</code> | [D/U] direct record；无 local PDF+text；主张只到可核验 landing page/metadata。 Topics: coding_software, efficiency_systems, Parallel Computing and Optimization Techniques. |
| 2026-03-10 <code>web-18b7ee4fe5e4</code> | Improving instruction hierarchy in frontier LLMs<br><https://openai.com/index/instruction-hierarchy-challenge> | <code>direct</code> / <code>blog_with_report</code> | [D/C] direct record；local PDF+text；论文/卡片结果未在本章复跑 [R 未建立]。 Topics: publication, research. |
| 2026-03-05 <code>web-287a6bf67154</code> | Reasoning models struggle to control their chains of thought, and that’s good<br><https://openai.com/index/reasoning-models-chain-of-thought-controllability> | <code>direct</code> / <code>blog_with_report</code> | [D/C] direct record；local PDF+text；论文/卡片结果未在本章复跑 [R 未建立]。 Topics: publication, research, safety. |
| 2026-03-04 <code>web-5bd932ab7f49</code> | Extending single-minus amplitudes to gravitons<br><https://openai.com/index/extending-single-minus-amplitudes-to-gravitons> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: publication, research. |
| 2026-02-27 <code>web-807caffbde0e</code> | An update on our mental health-related work<br><https://openai.com/index/update-on-mental-health-related-work> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: safety. |
| 2026-02-25 <code>rss-d163670fb22c</code> | Disrupting malicious uses of AI \\| February 2026<br><https://openai.com/index/disrupting-malicious-ai-uses> | <code>direct</code> / <code>blog_with_report</code> | [D/C] direct record；local PDF+text；论文/卡片结果未在本章复跑 [R 未建立]。 Topics: security, Security. |
| 2026-02-23 <code>web-33e5e5bcd689</code> | Why we no longer evaluate SWE-bench Verified<br><https://openai.com/index/why-we-no-longer-evaluate-swe-bench-verified> | <code>direct</code> / <code>benchmark</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: evaluation_benchmarks, publication, research. |
| 2026-02-20 <code>web-feae6511a0a2</code> | Our First Proof submissions<br><https://openai.com/index/first-proof-submissions> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: conclusion, research. |
| 2026-02-19 <code>web-d56ac300fb09</code> | Advancing independent research on AI alignment<br><https://openai.com/index/advancing-independent-research-ai-alignment> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: safety_alignment, publication, research. |
| 2026-02-18 <code>web-55bc17d8de5d</code> | Introducing EVMbench<br><https://openai.com/index/introducing-evmbench> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: publication, research. |
| 2026-02-13 <code>web-4bdc56229cf5</code> | Scaling social science research<br><https://openai.com/index/scaling-social-science-research> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: efficiency_systems, science_biology, publication. |
| 2026-02-13 <code>web-b117978acb8a</code> | Introducing Lockdown Mode and Elevated Risk labels in ChatGPT<br><https://openai.com/index/introducing-lockdown-mode-and-elevated-risk-labels-in-chatgpt> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: foundation_models, safety_alignment, safety. |
| 2026-02-13 <code>web-338aaf898d6c</code> | Beyond rate limits: scaling access to Codex and Sora<br><https://openai.com/index/beyond-rate-limits> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: agents_tool_use, multimodal, coding_software. |
| 2026-02-11 <code>web-f6b93302265d</code> | Harness engineering: leveraging Codex in an agent-first world<br><https://openai.com/index/harness-engineering> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: agents_tool_use, coding_software, engineering. |
| 2026-02-05 <code>doi-74b5c1fbc1f6</code> | Using a GPT-5-driven autonomous lab to optimize the cost and titer of cell-free protein synthesis<br><https://doi.org/10.64898/2026.02.05.703998> | <code>direct</code> / <code>research_paper</code> | [D/U] direct record；无 local PDF+text；主张只到可核验 landing page/metadata。 Topics: foundation_models, science_biology, RNA and protein synthesis mechanisms. |
| 2026-02-05 <code>web-d12899870b4c</code> | Introducing Trusted Access for Cyber<br><https://openai.com/index/trusted-access-for-cyber> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: security, safety. |
| 2026-02-04 <code>web-e205ffac9d9f</code> | Unlocking the Codex harness: how we built the App Server<br><https://openai.com/index/unlocking-the-codex-harness> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: agents_tool_use, coding_software, engineering. |
| 2026-02-03 <code>web-96d26eaa8234</code> | The Sora feed philosophy<br><https://openai.com/index/sora-feed-philosophy> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: multimodal, safety. |
| 2026-01-31 <code>doi-3557c9ba3a3e</code> | Proton: Towards Multi-level, Adaptive Profiling for Triton<br><https://doi.org/10.1109/cgo68049.2026.11395207> | <code>direct</code> / <code>research_paper</code> | [D/U] direct record；无 local PDF+text；主张只到可核验 landing page/metadata。 Topics: coding_software, Radiation Therapy and Dosimetry, Electron Spin Resonance Studies. |
| 2026-01-29 <code>web-fa5410010e0a</code> | Inside OpenAI’s in-house data agent<br><https://openai.com/index/inside-our-in-house-data-agent> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: agents_tool_use, data_training, engineering. |
| 2026-01-28 <code>web-0d2abbb0fa62</code> | Keeping your data safe when an AI agent clicks a link<br><https://openai.com/index/ai-agent-link-safety> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: agents_tool_use, data_training, safety. |
| 2026-01-23 <code>web-ae10eb4597fe</code> | Unrolling the Codex agent loop<br><https://openai.com/index/unrolling-the-codex-agent-loop> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: agents_tool_use, coding_software, engineering. |
| 2026-01-22 <code>web-89a221162f32</code> | Scaling PostgreSQL to power 800 million ChatGPT users<br><https://openai.com/index/scaling-postgresql> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: foundation_models, coding_software, efficiency_systems. |
| 2026-01-20 <code>web-3eec9fb139cb</code> | Our approach to age prediction<br><https://openai.com/index/our-approach-to-age-prediction> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: safety. |
| 2026-01-01 <code>doi-1dba6031a4bc</code> | What Should Physics Foundation Model Benchmarks Measure?<br><https://doi.org/10.1109/mcse.2026.3698434> | <code>direct</code> / <code>research_paper</code> | [D/U] direct record；无 local PDF+text；主张只到可核验 landing page/metadata。 Topics: foundation_models, evaluation_benchmarks, science_biology. |
| 2026-01-01 <code>doi-619855882e0a</code> | Legal Alignment for Safe and Ethical AI<br><https://doi.org/10.2139/ssrn.6036657> | <code>direct</code> / <code>research_paper</code> | [D/C] direct record；local PDF+text；论文/卡片结果未在本章复跑 [R 未建立]。 Topics: safety_alignment, society_governance, Ethics and Social Impacts of AI. |
| 2026-01-01 <code>doi-2abdc8fe81ff</code> | GPT as a Measurement Tool<br><https://doi.org/10.2139/ssrn.6246332> | <code>direct</code> / <code>research_paper</code> | [D/U] direct record；无 local PDF+text；主张只到可核验 landing page/metadata。 Topics: foundation_models, evaluation_benchmarks, Computational and Text Analysis Methods. |
| 2025-12-22 <code>rss-0ad9fbef8c87</code> | Continuously hardening ChatGPT Atlas against prompt injection<br><https://openai.com/index/hardening-atlas-against-prompt-injection> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: foundation_models, security, Security. |
| 2025-12-18 <code>web-020ab4b4b78d</code> | Updating our Model Spec with teen protections<br><https://openai.com/index/updating-model-spec-with-teen-protections> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: safety. |
| 2025-12-18 <code>web-454ed7542563</code> | Evaluating chain-of-thought monitorability<br><https://openai.com/index/evaluating-chain-of-thought-monitorability> | <code>direct</code> / <code>blog_with_report</code> | [D/C] direct record；local PDF+text；论文/卡片结果未在本章复跑 [R 未建立]。 Topics: evaluation_benchmarks, publication, research. |
| 2025-12-18 <code>web-1c04d5e205f8</code> | AI literacy resources for teens and parents<br><https://openai.com/index/ai-literacy-resources-for-teens-and-parents> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: safety. |
| 2025-12-16 <code>web-e2e14d0c1d72</code> | The new ChatGPT Images is here<br><https://openai.com/index/new-chatgpt-images-is-here> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: foundation_models, multimodal, release. |
| 2025-12-16 <code>web-3f703ff65ec6</code> | Measuring AI’s capability to accelerate biological research<br><https://openai.com/index/accelerating-biological-research-in-the-wet-lab> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: publication, research. |
| 2025-12-16 <code>web-1b288e83884e</code> | Evaluating AI’s ability to perform scientific research tasks<br><https://openai.com/index/frontierscience> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: evaluation_benchmarks, publication, research. |
| 2025-12-12 <code>web-eafd28afa8a6</code> | How We Used Codex to Ship Sora for Android in 28 Days<br><https://openai.com/index/shipping-sora-for-android-with-codex> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: agents_tool_use, multimodal, coding_software. |
| 2025-12-11 <code>web-03cba5d23965</code> | Advancing science and math with GPT-5.2<br><https://openai.com/index/gpt-5-2-for-science-and-math> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: foundation_models, science_biology, publication. |
| 2025-12-10 <code>rss-365d33ac3468</code> | Strengthening cyber resilience as AI capabilities advance<br><https://openai.com/index/strengthening-cyber-resilience> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: security, Security. |
| 2025-12-08 <code>web-48e9458551fc</code> | The state of enterprise AI<br><https://openai.com/index/the-state-of-enterprise-ai-2025-report> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: research. |
| 2025-12-03 <code>web-f45c751a8120</code> | How confessions can keep language models honest<br><https://openai.com/index/how-confessions-can-keep-language-models-honest> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: publication, research. |
| 2025-12-01 <code>web-cd2c685fc846</code> | Funding grants for new research into AI and mental health<br><https://openai.com/index/ai-mental-health-research-grants> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: safety. |
| 2025-11-26 <code>rss-ec9167a4ecbb</code> | Mixpanel security incident: what OpenAI users need to know<br><https://openai.com/index/mixpanel-incident> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: security, Company. |
| 2025-11-20 <code>web-f484173e2976</code> | Early experiments in accelerating science with GPT-5<br><https://openai.com/index/accelerating-science-gpt-5> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: foundation_models, science_biology, publication. |
| 2025-11-19 <code>web-8dbcc33d5213</code> | Strengthening our safety ecosystem with external testing<br><https://openai.com/index/strengthening-safety-with-external-testing> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: safety_alignment, evaluation_benchmarks, safety. |
| 2025-11-19 <code>web-43cb7efd66ff</code> | How evals drive the next chapter in AI for businesses<br><https://openai.com/index/evals-drive-next-chapter-of-ai> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: research. |
| 2025-11-19 <code>web-55da04f42e32</code> | Building more with GPT-5.1-Codex-Max<br><https://openai.com/index/gpt-5-1-codex-max> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: foundation_models, agents_tool_use, coding_software. |
| 2025-11-13 <code>web-1285c175b36f</code> | Understanding neural networks through sparse circuits<br><https://openai.com/index/understanding-neural-networks-through-sparse-circuits> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: interpretability, efficiency_systems, publication. |
| 2025-11-12 <code>rss-058aaf1f9b7b</code> | Fighting the New York Times’ invasion of user privacy<br><https://openai.com/index/fighting-nyt-user-privacy-invasion> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: Security. |
| 2025-11-07 <code>rss-8cd903898ffd</code> | Understanding prompt injections: a frontier security challenge<br><https://openai.com/index/prompt-injections> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: security, Security. |
| 2025-11-06 <code>web-8f309e2a6858</code> | Introducing the Teen Safety Blueprint<br><https://openai.com/index/introducing-the-teen-safety-blueprint> | <code>direct</code> / <code>technical_report</code> | [D/C] direct record；local PDF+text；论文/卡片结果未在本章复跑 [R 未建立]。 Topics: safety_alignment, safety. |
| 2025-11-03 <code>web-f688037bba7d</code> | Introducing IndQA<br><https://openai.com/index/introducing-indqa> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: release, research. |
| 2025-10-30 <code>web-fed198f10f4d</code> | Introducing Aardvark: OpenAI’s agentic security researcher<br><https://openai.com/index/introducing-aardvark> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: agents_tool_use, security, release. |
| 2025-10-30 <code>web-3556c47e80db</code> | How we built OWL, the new architecture behind our ChatGPT-based browser, Atlas<br><https://openai.com/index/building-chatgpt-atlas> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: foundation_models, agents_tool_use, engineering. |
| 2025-10-29 <code>web-2081c76ea2b4</code> | Introducing gpt-oss-safeguard<br><https://openai.com/index/introducing-gpt-oss-safeguard> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: foundation_models, release. |
| 2025-10-27 <code>web-087387dbcb42</code> | Strengthening ChatGPT’s responses in sensitive conversations<br><https://openai.com/index/strengthening-chatgpt-responses-in-sensitive-conversations> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: foundation_models, safety. |
| 2025-10-14 <code>web-ef12d7074e94</code> | Expert Council on Well-Being and AI<br><https://openai.com/index/expert-council-on-well-being-and-ai> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: safety. |
| 2025-10-09 <code>web-b80206eeacbe</code> | Defining and evaluating political bias in LLMs<br><https://openai.com/index/defining-and-evaluating-political-bias-in-llms> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: safety_alignment, evaluation_benchmarks, publication. |
| 2025-10-07 <code>rss-1ee19958e991</code> | Disrupting malicious uses of AI: October 2025<br><https://openai.com/global-affairs/disrupting-malicious-uses-of-ai-october-2025> | <code>direct</code> / <code>blog_with_report</code> | [D/C] direct record；local PDF+text；论文/卡片结果未在本章复跑 [R 未建立]。 Topics: security, Global Affairs. |
| 2025-09-29 <code>web-419d8c6232d2</code> | Introducing parental controls<br><https://openai.com/index/introducing-parental-controls> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: safety. |
| 2025-09-29 <code>web-1b2240d7e630</code> | Combating online child sexual exploitation & abuse<br><https://openai.com/index/combating-online-child-sexual-exploitation-abuse> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: safety. |
| 2025-09-25 <code>web-bc53ab6b4dee</code> | Measuring the performance of our models on real-world tasks<br><https://openai.com/index/gdpval> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: publication, research. |
| 2025-09-17 <code>web-bef6cbe2a83e</code> | Detecting and reducing scheming in AI models<br><https://openai.com/index/detecting-and-reducing-scheming-in-ai-models> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: science_biology, publication, research. |
| 2025-09-16 <code>web-fe411c5ad4f1</code> | Teen safety, freedom, and privacy<br><https://openai.com/index/teen-safety-freedom-and-privacy> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: safety_alignment, safety. |
| 2025-09-16 <code>web-d63866e4506b</code> | Building towards age prediction<br><https://openai.com/index/building-towards-age-prediction> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: safety. |
| 2025-09-15 <code>web-6deeebce1719</code> | Introducing upgrades to Codex<br><https://openai.com/index/introducing-upgrades-to-codex> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: agents_tool_use, coding_software, release. |
| 2025-09-15 <code>web-bc32444a81aa</code> | How people are using ChatGPT<br><https://openai.com/index/how-people-are-using-chatgpt> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: foundation_models, publication, research. |
| 2025-09-14 <code>doi-c0ec44c0aed6</code> | Novel Paradigm for Privacy-Preserving Data Sharing and Anonymization in Smart Homes<br><https://doi.org/10.1109/icaifi66942.2025.11326168> | <code>direct</code> / <code>research_paper</code> | [D/U] direct record；无 local PDF+text；主张只到可核验 landing page/metadata。 Topics: data_training, efficiency_systems, Privacy-Preserving Technologies in Data. |
| 2025-09-12 <code>web-9992c02f6290</code> | Working with US CAISI and UK AISI to build more secure AI systems<br><https://openai.com/index/us-caisi-uk-aisi-ai-update> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: safety. |
| 2025-09-05 <code>web-39305e45dcf4</code> | Why language models hallucinate<br><https://openai.com/index/why-language-models-hallucinate> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: publication, research. |
| 2025-09-02 <code>web-4955eca90327</code> | Building more helpful ChatGPT experiences for everyone<br><https://openai.com/index/building-more-helpful-chatgpt-experiences-for-everyone> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: foundation_models, safety. |
| 2025-08-28 <code>web-402e4cf4a0dd</code> | Introducing gpt-realtime and Realtime API updates<br><https://openai.com/index/introducing-gpt-realtime> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: foundation_models, release. |
| 2025-08-27 <code>web-ad13fe7d3321</code> | OpenAI and Anthropic share findings from a joint safety evaluation<br><https://openai.com/index/openai-anthropic-safety-evaluation> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: safety_alignment, evaluation_benchmarks, safety. |
| 2025-08-27 <code>web-8504921d9f48</code> | Collective alignment: public input on our Model Spec<br><https://openai.com/index/collective-alignment-aug-2025-updates> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: safety_alignment, publication. |
| 2025-08-26 <code>web-dd0a7bcfd7cd</code> | Helping people when they need it most<br><https://openai.com/index/helping-people-when-they-need-it-most> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: safety. |
| 2025-08-22 <code>web-92e917d72f8b</code> | Accelerating life sciences research<br><https://openai.com/index/accelerating-life-sciences-research-with-retro-biosciences> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: science_biology, publication. |
| 2025-08-07 <code>rss-a315f70061d3</code> | Medical research with GPT-5<br><https://openai.com/index/gpt-5-medical-research> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: foundation_models, science_biology, ChatGPT. |
| 2025-08-07 <code>web-aec54e92a7f3</code> | From hard refusals to safe-completions: toward output-centric safety training<br><https://openai.com/index/gpt-5-safe-completions> | <code>direct</code> / <code>research_paper</code> | [D/C] direct record；local PDF+text；论文/卡片结果未在本章复跑 [R 未建立]。 Topics: safety_alignment, data_training, publication. |
| 2025-08-05 <code>web-12865e8612c7</code> | Introducing gpt-oss<br><https://openai.com/index/introducing-gpt-oss> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: foundation_models, release. |
| 2025-08-05 <code>web-1b667a11f9a6</code> | Estimating worst case frontier risks of open weight LLMs<br><https://openai.com/index/estimating-worst-case-frontier-risks-of-open-weight-llms> | <code>direct</code> / <code>blog_with_report</code> | [D/C] direct record；local PDF+text；论文/卡片结果未在本章复跑 [R 未建立]。 Topics: safety_alignment, publication, safety. |
| 2025-08-04 <code>web-bb96c88aa03c</code> | What we’re optimizing ChatGPT for<br><https://openai.com/index/optimizing-chatgpt> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: foundation_models, safety. |
| 2025-07-22 <code>web-85bbe066139c</code> | Pioneering an AI clinical copilot with Penda Health<br><https://openai.com/index/ai-clinical-copilot-penda-health> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: science_biology, publication. |
| 2025-06-18 <code>web-cd4bacbccf75</code> | Toward understanding and preventing misalignment generalization<br><https://openai.com/index/emergent-misalignment> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: safety_alignment, publication. |
| 2025-06-18 <code>web-f282bcdc97e3</code> | Preparing for future AI risks in biology<br><https://openai.com/index/preparing-for-future-ai-capabilities-in-biology> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: safety_alignment, science_biology, safety. |
| 2025-06-10 <code>doi-a34f34be767a</code> | Vision-Language Models Do Not Understand Negation<br><https://doi.org/10.1109/cvpr52734.2025.02757> | <code>direct</code> / <code>research_paper</code> | [D/C] direct record；local PDF+text；论文/卡片结果未在本章复跑 [R 未建立]。 Topics: multimodal, Speech and dialogue systems, Language, Metaphor, and Cognition. |
| 2025-06-10 <code>doi-2c2eab5b8abc</code> | Improving Diffusion Inverse Problem Solving with Decoupled Noise Annealing<br><https://doi.org/10.1109/cvpr52734.2025.01946> | <code>direct</code> / <code>research_paper</code> | [D/C] direct record；local PDF+text；论文/卡片结果未在本章复跑 [R 未建立]。 Topics: Neural Networks and Applications. |
| 2025-06-09 <code>rss-f3ad0bad7648</code> | Scaling security with responsible disclosure<br><https://openai.com/index/scaling-coordinated-vulnerability-disclosure> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: efficiency_systems, security, Security. |
| 2025-06-05 <code>rss-0606f63a1fb8</code> | How we’re responding to The New York Times’ data demands in order to protect user privacy<br><https://openai.com/index/response-to-nyt-data-demands> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: data_training, Security. |
| 2025-06-05 <code>rss-8a0ab2f2f7fd</code> | Disrupting malicious uses of AI: June 2025<br><https://openai.com/global-affairs/disrupting-malicious-uses-of-ai-june-2025> | <code>direct</code> / <code>blog_with_report</code> | [D/C] direct record；local PDF+text；论文/卡片结果未在本章复跑 [R 未建立]。 Topics: security, Global Affairs. |
| 2025-06-01 <code>doi-bcc9ef44deb6</code> | Spatial reasoning via recurrent neural dynamics in mouse retrosplenial cortex<br><https://doi.org/10.1038/s41593-025-01944-z> | <code>direct</code> / <code>research_paper</code> | [D/C] direct record；local PDF+text；论文/卡片结果未在本章复跑 [R 未建立]。 Topics: Neural dynamics and brain function, Memory and Neural Mechanisms, Neuroscience and Neuropharmacology Research. |
| 2025-05-12 <code>web-eba97d45c48b</code> | Introducing HealthBench<br><https://openai.com/index/healthbench> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: publication. |
| 2025-05-02 <code>rss-34b2cf85390c</code> | Expanding on what we missed with sycophancy<br><https://openai.com/index/expanding-on-sycophancy> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: safety_alignment, Product. |
| 2025-04-29 <code>rss-3da06b55f73b</code> | Sycophancy in GPT-4o: what happened and what we’re doing about it<br><https://openai.com/index/sycophancy-in-gpt-4o> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: foundation_models, safety_alignment, Product. |
| 2025-04-16 <code>web-365b999051a4</code> | Thinking with images<br><https://openai.com/index/thinking-with-images> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: multimodal, release. |
| 2025-04-15 <code>web-159d28217d06</code> | Our updated Preparedness Framework<br><https://openai.com/index/updating-our-preparedness-framework> | <code>direct</code> / <code>technical_report</code> | [D/C] direct record；local PDF+text；论文/卡片结果未在本章复跑 [R 未建立]。 Topics: safety_alignment, publication, safety. |
| 2025-04-11 <code>doi-70642f904fba</code> | SEAL: Systematic Error Analysis for Value ALignment<br><https://doi.org/10.1609/aaai.v39i26.34973> | <code>direct</code> / <code>research_paper</code> | [D/C] direct record；local PDF+text；论文/卡片结果未在本章复跑 [R 未建立]。 Topics: safety_alignment, Explainable Artificial Intelligence (XAI), Topic Modeling. |
| 2025-04-10 <code>web-13a8c4bf46a3</code> | BrowseComp: a benchmark for browsing agents<br><https://openai.com/index/browsecomp> | <code>direct</code> / <code>benchmark</code> | [D/C] direct record；local PDF+text；论文/卡片结果未在本章复跑 [R 未建立]。 Topics: agents_tool_use, evaluation_benchmarks, publication. |
| 2025-04-09 <code>doi-1c3f707cf74c</code> | Universal photonic artificial intelligence acceleration<br><https://doi.org/10.1038/s41586-025-08854-x> | <code>direct</code> / <code>research_paper</code> | [D/U] direct record；无 local PDF+text；主张只到可核验 landing page/metadata。 Topics: Neural Networks and Reservoir Computing, Optical Network Technologies, Photonic and Optical Devices. |
| 2025-04-09 <code>doi-9244c81d9d5e</code> | Position: Contextual Confidence and Generative AI<br><https://doi.org/10.1109/satml64287.2025.00022> | <code>direct</code> / <code>research_paper</code> | [D/U] direct record；无 local PDF+text；主张只到可核验 landing page/metadata。 Topics: Computability, Logic, AI Algorithms. |
| 2025-04-02 <code>web-a58fadf0d59c</code> | PaperBench: Evaluating AI’s Ability to Replicate AI Research<br><https://openai.com/index/paperbench> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: evaluation_benchmarks, publication, release. |
| 2025-03-27 <code>doi-a9d3b5473075</code> | Relax: Composable Abstractions for End-to-End Dynamic Machine Learning<br><https://doi.org/10.1145/3676641.3716249> | <code>direct</code> / <code>research_paper</code> | [D/C] direct record；local PDF+text；论文/卡片结果未在本章复跑 [R 未建立]。 Topics: Machine Learning and Algorithms, Parallel Computing and Optimization Techniques, Algorithms and Data Compression. |
| 2025-03-26 <code>rss-e5cf09edb190</code> | Security on the path to AGI<br><https://openai.com/index/security-on-the-path-to-agi> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: security, Security. |
| 2025-03-25 <code>web-389ab448b190</code> | Introducing 4o Image Generation<br><https://openai.com/index/introducing-4o-image-generation> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: multimodal, release. |
| 2025-03-21 <code>web-55904b2e2ce7</code> | Early methods for studying affective use and emotional well-being on ChatGPT<br><https://openai.com/index/affective-use-study> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: foundation_models, publication, research. |
| 2025-03-20 <code>web-ef935fda2131</code> | Introducing next-generation audio models in the API<br><https://openai.com/index/introducing-our-next-generation-audio-models> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: multimodal, release. |
| 2025-03-10 <code>web-62125fff56fe</code> | Detecting misbehavior in frontier reasoning models<br><https://openai.com/index/chain-of-thought-monitoring> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: publication. |
| 2025-02-21 <code>rss-1098bacb270e</code> | Disrupting malicious uses of AI<br><https://openai.com/global-affairs/disrupting-malicious-uses-of-ai> | <code>direct</code> / <code>blog_with_report</code> | [D/C] direct record；local PDF+text；论文/卡片结果未在本章复跑 [R 未建立]。 Topics: security, Global Affairs. |
| 2025-02-18 <code>web-e59eeac6f005</code> | Introducing the SWE-Lancer benchmark<br><https://openai.com/index/swe-lancer> | <code>direct</code> / <code>benchmark</code> | [D/C] direct record；local PDF+text；论文/卡片结果未在本章复跑 [R 未建立]。 Topics: evaluation_benchmarks, publication. |
| 2025-02-12 <code>web-0385347048a6</code> | Sharing the latest Model Spec<br><https://openai.com/index/sharing-the-latest-model-spec> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: evaluation_benchmarks, milestone, release. |
| 2025-02-02 <code>rss-a30b39b771cc</code> | Understanding complex trends with deep research<br><https://openai.com/index/deep-research> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: Story. |
| 2025-02-02 <code>web-e0a00b60e015</code> | Introducing deep research<br><https://openai.com/index/introducing-deep-research> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: release. |
| 2025-01-22 <code>web-f4b16ea1ae46</code> | Trading inference-time compute for adversarial robustness<br><https://openai.com/index/trading-inference-time-compute-for-adversarial-robustness> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: safety_alignment, efficiency_systems, publication. |
| 2025-01-01 <code>doi-658d8cfe2cea</code> | How People Use ChatGPT<br><https://doi.org/10.2139/ssrn.5487080> | <code>direct</code> / <code>research_paper</code> | [D/U] direct record；无 local PDF+text；主张只到可核验 landing page/metadata。 Topics: foundation_models, AI in Service Interactions, Artificial Intelligence in Healthcare and Education. |
| 2024-12-20 <code>web-9cf289b414fd</code> | Deliberative alignment: reasoning enables safer language models<br><https://openai.com/index/deliberative-alignment> | <code>direct</code> / <code>research_paper</code> | [D/C] direct record；local PDF+text；论文/卡片结果未在本章复跑 [R 未建立]。 Topics: safety_alignment, publication, release. |
| 2024-11-21 <code>web-a93dc3ad8dab</code> | Advancing red teaming with people and AI<br><https://openai.com/index/advancing-red-teaming-with-people-and-ai> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: safety_alignment, publication. |
| 2024-11-02 <code>doi-f7c7fbf1a44f</code> | ArtVLM: Attribute Recognition Through Vision-Based Prefix Language Modeling<br><https://doi.org/10.1007/978-3-031-73383-3_8> | <code>direct</code> / <code>research_paper</code> | [D/C] direct record；local PDF+text；论文/卡片结果未在本章复跑 [R 未建立]。 Topics: Multimodal Machine Learning Applications, Natural Language Processing Techniques, Advanced Image and Video Retrieval Techniques. |
| 2024-10-30 <code>web-e92e95daf8e0</code> | Introducing SimpleQA<br><https://openai.com/index/introducing-simpleqa> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: publication. |
| 2024-10-23 <code>web-48d4ba4c85a1</code> | Simplifying, stabilizing, and scaling continuous-time consistency models<br><https://openai.com/index/simplifying-stabilizing-and-scaling-continuous-time-consistency-models> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: efficiency_systems, milestone. |
| 2024-10-15 <code>web-35471ebc6894</code> | Evaluating fairness in ChatGPT<br><https://openai.com/index/evaluating-fairness-in-chatgpt> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: foundation_models, safety_alignment, evaluation_benchmarks. |
| 2024-10-10 <code>web-c3a08c57269d</code> | MLE-bench: Evaluating Machine Learning Agents on Machine Learning Engineering<br><https://openai.com/index/mle-bench> | <code>direct</code> / <code>benchmark</code> | [D/C] direct record；local PDF+text；论文/卡片结果未在本章复跑 [R 未建立]。 Topics: agents_tool_use, evaluation_benchmarks, publication. |
| 2024-09-30 <code>doi-5370d5cab598</code> | Parrot: Pareto-Optimal Multi-reward Reinforcement Learning Framework for Text-to-Image Generation<br><https://doi.org/10.1007/978-3-031-72920-1_26> | <code>direct</code> / <code>research_paper</code> | [D/C] direct record；local PDF+text；论文/卡片结果未在本章复跑 [R 未建立]。 Topics: reinforcement_learning, multimodal, Generative Adversarial Networks and Image Synthesis. |
| 2024-09-16 <code>web-6852ddf2a8e2</code> | An update on our safety & security practices<br><https://openai.com/index/update-on-safety-and-security-practices> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: safety_alignment, security, safety. |
| 2024-09-12 <code>web-d1359f03b307</code> | Learning to reason with LLMs<br><https://openai.com/index/learning-to-reason-with-llms> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: release. |
| 2024-08-16 <code>web-66e5f8f34cff</code> | Disrupting a covert Iranian influence operation<br><https://openai.com/index/disrupting-a-covert-iranian-influence-operation> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: safety. |
| 2024-08-13 <code>web-10b59b402b80</code> | Introducing SWE-bench Verified<br><https://openai.com/index/introducing-swe-bench-verified> | <code>direct</code> / <code>benchmark</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: milestone. |
| 2024-07-31 <code>doi-a5f85653f59d</code> | Rethinking Machine Learning Collective Communication as a Multi-Commodity Flow Problem<br><https://doi.org/10.1145/3651890.3672249> | <code>direct</code> / <code>research_paper</code> | [D/C] direct record；local PDF+text；论文/卡片结果未在本章复跑 [R 未建立]。 Topics: Software-Defined Networks and 5G, Network Security and Intrusion Detection, Advanced Memory and Neural Computing. |
| 2024-07-24 <code>web-dcb04574301f</code> | Improving Model Safety Behavior with Rule-Based Rewards<br><https://openai.com/index/improving-model-safety-behavior-with-rule-based-rewards> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: safety_alignment, reinforcement_learning, publication. |
| 2024-07-17 <code>web-7d4217e43b69</code> | Prover-Verifier Games improve legibility of language model outputs<br><https://openai.com/index/prover-verifier-games-improve-legibility> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: foundation_models, conclusion. |
| 2024-07-10 <code>web-4fb73cf59fb2</code> | OpenAI and Los Alamos National Laboratory announce research partnership<br><https://openai.com/index/openai-and-los-alamos-national-laboratory-work-together> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: society_governance, milestone. |
| 2024-06-27 <code>web-575cb0a6069c</code> | Finding GPT-4’s mistakes with GPT-4<br><https://openai.com/index/finding-gpt4s-mistakes-with-gpt-4> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: foundation_models, publication. |
| 2024-06-20 <code>web-9f11083efd0a</code> | Improved Techniques for Training Consistency Models<br><https://openai.com/index/improved-techniques-for-training-consistency-models> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: data_training, publication. |
| 2024-06-20 <code>doi-044d93d342be</code> | GPTs are GPTs: Labor market impact potential of LLMs<br><https://doi.org/10.1126/science.adj0998> | <code>direct</code> / <code>research_paper</code> | [D/U] direct record；无 local PDF+text；主张只到可核验 landing page/metadata。 Topics: society_governance, Labor market dynamics and wage inequality, ICT Impact and Policies. |
| 2024-06-20 <code>web-c24d318bbbd8</code> | Consistency Models<br><https://openai.com/index/consistency-models> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: publication. |
| 2024-06-20 <code>web-6ba9a13e7150</code> | A Holistic Approach to Undesired Content Detection in the Real World<br><https://openai.com/index/a-holistic-approach-to-undesired-content-detection-in-the-real-world> | <code>direct</code> / <code>research_paper</code> | [D/C] direct record；local PDF+text；论文/卡片结果未在本章复跑 [R 未建立]。 Topics: publication. |
| 2024-06-07 <code>web-931159437ec9</code> | Expanding on how Voice Engine works and our safety research<br><https://openai.com/index/expanding-on-how-voice-engine-works-and-our-safety-research> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: safety_alignment, multimodal, safety. |
| 2024-06-06 <code>web-d58703e87ddb</code> | Extracting Concepts from GPT-4<br><https://openai.com/index/extracting-concepts-from-gpt-4> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: foundation_models, interpretability, publication. |
| 2024-06-03 <code>doi-980c45f8d4c6</code> | Generalized People Diversity: Learning a Human Perception-Aligned Diversity Representation for People Images<br><https://doi.org/10.1145/3630106.3658940> | <code>direct</code> / <code>research_paper</code> | [D/C] direct record；local PDF+text；论文/卡片结果未在本章复跑 [R 未建立]。 Topics: safety_alignment, multimodal, Multimodal Machine Learning Applications. |
| 2024-05-30 <code>rss-79aae2d11d48</code> | Disrupting deceptive uses of AI by covert influence operations<br><https://openai.com/index/disrupting-deceptive-uses-of-ai-by-covert-influence-operations> | <code>direct</code> / <code>blog_with_report</code> | [D/C] direct record；local PDF+text；论文/卡片结果未在本章复跑 [R 未建立]。 Topics: safety_alignment, Security. |
| 2024-05-21 <code>web-55d019e35eea</code> | OpenAI safety practices<br><https://openai.com/index/openai-safety-update> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: safety_alignment, safety. |
| 2024-05-13 <code>web-2d3a533e91db</code> | Hello GPT-4o<br><https://openai.com/index/hello-gpt-4o> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: foundation_models, milestone. |
| 2024-05-08 <code>web-d1406200c99e</code> | Introducing the Model Spec<br><https://openai.com/index/introducing-the-model-spec> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: research, safety. |
| 2024-05-07 <code>web-50e210ee0288</code> | Understanding the source of what we see and hear online<br><https://openai.com/index/understanding-the-source-of-what-we-see-and-hear-online> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: publication. |
| 2024-05-07 <code>rss-7447a7f27a11</code> | Our approach to data and AI<br><https://openai.com/index/approach-to-data-and-ai> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: data_training, Safety. |
| 2024-04-23 <code>web-e66e11aa417b</code> | OpenAI’s commitment to child safety: adopting safety by design principles<br><https://openai.com/index/child-safety-adopting-sbd-principles> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: safety_alignment, safety. |
| 2024-04-22 <code>doi-a26b81631a77</code> | PyTorch 2: Faster Machine Learning Through Dynamic Python Bytecode Transformation and Graph Compilation<br><https://doi.org/10.1145/3620665.3640366> | <code>direct</code> / <code>research_paper</code> | [D/U] direct record；无 local PDF+text；主张只到可核验 landing page/metadata。 Topics: coding_software, Parallel Computing and Optimization Techniques, Software Testing and Debugging Techniques. |
| 2024-04-19 <code>web-7d089e2d1180</code> | The Instruction Hierarchy: Training LLMs to Prioritize Privileged Instructions<br><https://openai.com/index/the-instruction-hierarchy> | <code>direct</code> / <code>blog_with_report</code> | [D/C] direct record；local PDF+text；论文/卡片结果未在本章复跑 [R 未建立]。 Topics: data_training, publication. |
| 2024-02-14 <code>rss-b3d8a86c92be</code> | Disrupting malicious uses of AI by state-affiliated threat actors<br><https://openai.com/index/disrupting-malicious-uses-of-ai-by-state-affiliated-threat-actors> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: security, Safety. |
| 2024-01-31 <code>web-65ffab8a128a</code> | Building an early warning system for LLM-aided biological threat creation<br><https://openai.com/index/building-an-early-warning-system-for-llm-aided-biological-threat-creation> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: foundation_models, publication. |
| 2024-01-16 <code>web-6689a9c7c9b2</code> | Democratic inputs to AI grant program: lessons learned and implementation plans<br><https://openai.com/index/democratic-inputs-to-ai-grant-program-update> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: multimodal, coding_software, society_governance. |
| 2024-01-15 <code>rss-2b20a8623e33</code> | How OpenAI is approaching 2024 worldwide elections<br><https://openai.com/index/how-openai-is-approaching-2024-worldwide-elections> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: society_governance, Safety & Alignment. |
| 2023-12-14 <code>web-b276b4af9d56</code> | Weak-to-strong generalization<br><https://openai.com/index/weak-to-strong-generalization> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: safety. |
| 2023-12-14 <code>web-bbc0fc0c1679</code> | Superalignment Fast Grants<br><https://openai.com/index/superalignment-fast-grants> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: safety_alignment, safety. |
| 2023-12-14 <code>web-25adcaa1c831</code> | Practices for Governing Agentic AI Systems<br><https://openai.com/index/practices-for-governing-agentic-ai-systems> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: agents_tool_use, society_governance, publication. |
| 2023-10-26 <code>web-a1c8df89a3c9</code> | Frontier risk and preparedness<br><https://openai.com/index/frontier-risk-and-preparedness> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: safety_alignment, safety. |
| 2023-10-01 <code>doi-41eec428025e</code> | VQ3D: Learning a 3D-Aware Generative Model on ImageNet<br><https://doi.org/10.1109/iccv51070.2023.00391> | <code>direct</code> / <code>research_paper</code> | [D/C] direct record；local PDF+text；论文/卡片结果未在本章复跑 [R 未建立]。 Topics: multimodal, Generative Adversarial Networks and Image Synthesis, Advanced Vision and Imaging. |
| 2023-09-30 <code>doi-d4caa6c7b84f</code> | Diffusion Models: A Comprehensive Survey of Methods and Applications<br><https://doi.org/10.1145/3626235> | <code>direct</code> / <code>research_paper</code> | [D/C] direct record；local PDF+text；论文/卡片结果未在本章复跑 [R 未建立]。 Topics: Generative Adversarial Networks and Image Synthesis, Mathematical Biology Tumor Growth, Advanced Mathematical Modeling in Engineering. |
| 2023-09-28 <code>doi-3d3e532a79c0</code> | Artificial Intelligence, Academic Misconduct, and the Borg: Why GPT-3 Text Generation in the Higher Education Classroom is Becoming Scary<br><https://doi.org/10.18357/anthropologica65120232166> | <code>direct</code> / <code>research_paper</code> | [D/C] direct record；local PDF+text；论文/卡片结果未在本章复跑 [R 未建立]。 Topics: foundation_models, Artificial Intelligence in Healthcare and Education, Topic Modeling. |
| 2023-09-19 <code>web-21066da02501</code> | OpenAI Red Teaming Network<br><https://openai.com/index/red-teaming-network> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: safety_alignment, efficiency_systems, safety. |
| 2023-08-15 <code>web-08af580a2cf0</code> | Using GPT-4 for content moderation<br><https://openai.com/index/using-gpt-4-for-content-moderation> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: foundation_models, safety. |
| 2023-08-01 <code>web-33ceebccb054</code> | Confidence-Building Measures for Artificial Intelligence: Workshop proceedings<br><https://openai.com/index/confidence-building-measures-for-artificial-intelligence> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: conclusion, safety. |
| 2023-07-26 <code>web-e05c394cd849</code> | Frontier Model Forum<br><https://openai.com/index/frontier-model-forum> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: safety. |
| 2023-07-21 <code>web-544d184728d7</code> | Moving AI governance forward<br><https://openai.com/index/moving-ai-governance-forward> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: society_governance, safety. |
| 2023-07-19 <code>doi-fc9fccb28838</code> | Channeling Creativity Through a Deeper Understanding of AI Image Generation<br><https://doi.org/10.1145/3588029.3599743> | <code>direct</code> / <code>research_paper</code> | [D/U] direct record；无 local PDF+text；主张只到可核验 landing page/metadata。 Topics: multimodal, 3D Surveying and Cultural Heritage. |
| 2023-07-06 <code>web-dccab78d5391</code> | Frontier AI regulation: Managing emerging risks to public safety<br><https://openai.com/index/frontier-ai-regulation> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: safety_alignment, society_governance, publication. |
| 2023-06-29 <code>web-4ceca03d184b</code> | Insights from global conversations<br><https://openai.com/index/insights-from-global-conversations> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: safety. |
| 2023-06-21 <code>doi-34e6121efc78</code> | Evaluation of GPT-3 AI Language Model in Research Paper Writing<br><https://doi.org/10.55525/tjst.1272369> | <code>direct</code> / <code>research_paper</code> | [D/C] direct record；local PDF+text；论文/卡片结果未在本章复跑 [R 未建立]。 Topics: foundation_models, evaluation_benchmarks, Topic Modeling. |
| 2023-06-01 <code>doi-417cd69b6353</code> | Fusing Pre-Trained Language Models with Multimodal Prompts through Reinforcement Learning<br><https://doi.org/10.1109/cvpr52729.2023.01044> | <code>direct</code> / <code>research_paper</code> | [D/U] direct record；无 local PDF+text；主张只到可核验 landing page/metadata。 Topics: reinforcement_learning, multimodal, Multimodal Machine Learning Applications. |
| 2023-05-31 <code>web-b6a5fdb87e9f</code> | Improving mathematical reasoning with process supervision<br><https://openai.com/index/improving-mathematical-reasoning-with-process-supervision> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: science_biology, publication. |
| 2023-05-25 <code>web-64a543805735</code> | Democratic inputs to AI<br><https://openai.com/index/democratic-inputs-to-ai> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: multimodal, society_governance, publication. |
| 2023-05-22 <code>web-238dd9d30994</code> | Governance of superintelligence<br><https://openai.com/index/governance-of-superintelligence> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: society_governance, safety. |
| 2023-05-10 <code>doi-bfe98f429411</code> | Judging facts, judging norms: Training machine learning models to judge humans requires a modified approach to labeling data<br><https://doi.org/10.1126/sciadv.abq0701> | <code>direct</code> / <code>research_paper</code> | [D/C] direct record；local PDF+text；论文/卡片结果未在本章复跑 [R 未建立]。 Topics: data_training, Explainable Artificial Intelligence (XAI), Machine Learning and Data Classification. |
| 2023-05-09 <code>web-784c8fdf9a87</code> | Language models can explain neurons in language models<br><https://openai.com/index/language-models-can-explain-neurons-in-language-models> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: interpretability, publication. |
| 2023-04-11 <code>rss-7b7c824de800</code> | Announcing OpenAI’s Bug Bounty Program<br><https://openai.com/index/bug-bounty-program> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: coding_software, Safety. |
| 2023-04-05 <code>web-770b56513215</code> | Our approach to AI safety<br><https://openai.com/index/our-approach-to-ai-safety> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: safety_alignment, safety. |
| 2023-03-24 <code>rss-1811bb452cd9</code> | March 20 ChatGPT outage: Here’s what happened<br><https://openai.com/index/march-20-chatgpt-outage> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: foundation_models, Company. |
| 2023-03-17 <code>web-b66b937dc53f</code> | GPTs are GPTs: An early look at the labor market impact potential of large language models<br><https://openai.com/index/gpts-are-gpts> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: society_governance, publication. |
| 2023-02-24 <code>web-cab0c2e06ee8</code> | Planning for AGI and beyond<br><https://openai.com/index/planning-for-agi-and-beyond> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: safety. |
| 2023-02-16 <code>web-b34c98233be7</code> | How should AI systems behave, and who should decide?<br><https://openai.com/index/how-should-ai-systems-behave> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: safety. |
| 2023-01-20 <code>doi-2837c9043516</code> | SPT-NRTL: A physics-guided machine learning model to predict thermodynamically consistent activity coefficients<br><https://doi.org/10.1016/j.fluid.2023.113731> | <code>direct</code> / <code>research_paper</code> | [D/C] direct record；local PDF+text；论文/卡片结果未在本章复跑 [R 未建立]。 Topics: efficiency_systems, science_biology, Machine Learning in Materials Science. |
| 2023-01-11 <code>web-31f4e39670c8</code> | Forecasting potential misuses of language models for disinformation campaigns and how to reduce risk<br><https://openai.com/index/forecasting-misuse> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: safety_alignment, publication. |
| 2023-01-10 <code>doi-880bd32341c4</code> | Generative Language Models and Automated Influence Operations: Emerging Threats and Potential Mitigations<br><https://doi.org/10.48550/arxiv.2301.04246> | <code>direct</code> / <code>research_paper</code> | [D/C] direct record；local PDF+text；论文/卡片结果未在本章复跑 [R 未建立]。 Topics: Topic Modeling. |
| 2023-01-01 <code>doi-8c25accc0c1f</code> | Toward Joint Language Modeling for Speech Units and Text<br><https://doi.org/10.18653/v1/2023.findings-emnlp.438> | <code>direct</code> / <code>research_paper</code> | [D/C] direct record；local PDF+text；论文/卡片结果未在本章复跑 [R 未建立]。 Topics: multimodal, Speech Recognition and Synthesis, Topic Modeling. |
| 2022-12-28 <code>doi-9aea6b51260e</code> | Does GPT-3 qualify as a co-author of a scientific paper publishable in peer-review journals according to the ICMJE criteria? - A Case Study.<br><https://doi.org/10.21203/rs.3.rs-2404314/v1> | <code>direct</code> / <code>research_paper</code> | [D/C] direct record；local PDF+text；论文/卡片结果未在本章复跑 [R 未建立]。 Topics: foundation_models, Artificial Intelligence in Healthcare and Education. |
| 2022-12-21 <code>doi-5954c84b6133</code> | Rapamycin in the context of Pascal’s Wager: generative pre-trained transformer perspective<br><https://doi.org/10.18632/oncoscience.571> | <code>direct</code> / <code>research_paper</code> | [D/C] direct record；local PDF+text；论文/卡片结果未在本章复跑 [R 未建立]。 Topics: foundation_models, efficiency_systems, Dietary Effects on Health. |
| 2022-10-19 <code>web-1a032a768300</code> | Scaling laws for reward model overoptimization<br><https://openai.com/index/scaling-laws-for-reward-model-overoptimization> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: reinforcement_learning, efficiency_systems, publication. |
| 2022-08-24 <code>web-d7f602cae828</code> | Our approach to alignment research<br><https://openai.com/index/our-approach-to-alignment-research> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: safety_alignment, safety. |
| 2022-07-28 <code>web-54ba9a907ae8</code> | Efficient training of language models to fill in the middle<br><https://openai.com/index/efficient-training-of-language-models-to-fill-in-the-middle> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: data_training, efficiency_systems, publication. |
| 2022-07-18 <code>doi-b3959ac132fd</code> | Q-Value Weighted Regression: Reinforcement Learning with Limited Data<br><https://doi.org/10.1109/ijcnn55064.2022.9892633> | <code>direct</code> / <code>research_paper</code> | [D/C] direct record；local PDF+text；论文/卡片结果未在本章复跑 [R 未建立]。 Topics: reinforcement_learning, data_training, Reinforcement Learning in Robotics. |
| 2022-06-23 <code>web-4e33c3e8216e</code> | Learning to play Minecraft with Video PreTraining<br><https://openai.com/index/vpt> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: multimodal, data_training, conclusion. |
| 2022-06-20 <code>doi-908824878c6d</code> | Disentangling the Components of Ethical Research in Machine Learning<br><https://doi.org/10.1145/3531146.3533781> | <code>direct</code> / <code>research_paper</code> | [D/C] direct record；local PDF+text；论文/卡片结果未在本章复跑 [R 未建立]。 Topics: Ethics and Social Impacts of AI, Explainable Artificial Intelligence (XAI), Adversarial Robustness in Machine Learning. |
| 2022-06-17 <code>web-b6096c30c708</code> | Evolution through large models<br><https://openai.com/index/evolution-through-large-models> | <code>direct</code> / <code>research_paper</code> | [D/C] direct record；local PDF+text；论文/卡片结果未在本章复跑 [R 未建立]。 Topics: publication. |
| 2022-06-13 <code>web-6c478793ad50</code> | AI-written critiques help humans notice flaws<br><https://openai.com/index/critiques> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: publication. |
| 2022-06-09 <code>web-4f4c5c964bce</code> | Techniques for training large neural networks<br><https://openai.com/index/techniques-for-training-large-neural-networks> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: data_training, efficiency_systems, publication. |
| 2022-06-02 <code>web-f5110c1c36ae</code> | Best practices for deploying language models<br><https://openai.com/index/best-practices-for-deploying-language-models> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: safety. |
| 2022-05-28 <code>web-a4a912d18824</code> | Teaching models to express their uncertainty in words<br><https://openai.com/index/teaching-models-to-express-their-uncertainty-in-words> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: publication. |
| 2022-04-13 <code>web-38a77e414481</code> | Measuring Goodhart’s law<br><https://openai.com/index/measuring-goodharts-law> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: publication. |
| 2022-04-13 <code>web-2c38d2f354fa</code> | Hierarchical text-conditional image generation with CLIP latents<br><https://openai.com/index/hierarchical-text-conditional-image-generation-with-clip-latents> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: multimodal, publication. |
| 2022-03-31 <code>doi-e3145de7d03f</code> | Exploration in deep reinforcement learning: A survey<br><https://doi.org/10.1016/j.inffus.2022.03.003> | <code>direct</code> / <code>research_paper</code> | [D/C] direct record；local PDF+text；论文/卡片结果未在本章复跑 [R 未建立]。 Topics: reinforcement_learning, Reinforcement Learning in Robotics, Supply Chain and Inventory Management. |
| 2022-03-03 <code>web-40010aa33c6b</code> | Lessons learned on language model safety and misuse<br><https://openai.com/index/language-model-safety-and-misuse> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: foundation_models, safety_alignment, conclusion. |
| 2022-03-03 <code>rss-6571f3ac4380</code> | Economic impacts research at OpenAI<br><https://openai.com/index/economic-impacts> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: society_governance, Safety & Alignment. |
| 2022-03-03 <code>web-212f765399d5</code> | A research agenda for assessing the economic impacts of code generation models<br><https://openai.com/index/economic-impacts-research> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: coding_software, society_governance, publication. |
| 2022-02-02 <code>web-89ed49981586</code> | Solving (some) formal math olympiad problems<br><https://openai.com/index/formal-math> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: science_biology, milestone. |
| 2022-01-27 <code>web-87477ad8b476</code> | Aligning language models to follow instructions<br><https://openai.com/index/instruction-following> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: safety_alignment, publication. |
| 2022-01-24 <code>web-d3cbea351a46</code> | Text and code embeddings by contrastive pre-training<br><https://openai.com/index/text-and-code-embeddings-by-contrastive-pre-training> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: coding_software, data_training, publication. |
| 2022-01-01 <code>doi-214e9dc598bc</code> | Hierarchical Transformers Are More Efficient Language Models<br><https://doi.org/10.18653/v1/2022.findings-naacl.117> | <code>direct</code> / <code>research_paper</code> | [D/C] direct record；local PDF+text；论文/卡片结果未在本章复跑 [R 未建立]。 Topics: efficiency_systems, Topic Modeling, Multimodal Machine Learning Applications. |
| 2021-12-16 <code>web-bb1464aabad9</code> | WebGPT: Improving the factual accuracy of language models through web browsing<br><https://openai.com/index/webgpt> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: publication. |
| 2021-11-24 <code>doi-0c33fac8aa24</code> | Evolving Multimodal Robot Behavior via Many Stepping Stones with the Combinatorial Multiobjective Evolutionary Algorithm<br><https://doi.org/10.1162/evco_a_00301> | <code>direct</code> / <code>research_paper</code> | [D/C] direct record；local PDF+text；论文/卡片结果未在本章复跑 [R 未建立]。 Topics: multimodal, robotics_embodied, Advanced Multi-Objective Optimization Algorithms. |
| 2021-10-29 <code>web-34f845c1e710</code> | Solving math word problems<br><https://openai.com/index/solving-math-word-problems> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: science_biology, publication. |
| 2021-09-23 <code>web-7f768525daf5</code> | Summarizing books with human feedback<br><https://openai.com/index/summarizing-books> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: publication. |
| 2021-09-08 <code>web-c16f4ae139bf</code> | TruthfulQA: Measuring how models mimic human falsehoods<br><https://openai.com/index/truthfulqa> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: publication. |
| 2021-08-31 <code>doi-bddb32d0dc59</code> | MiniF2F: a cross-system benchmark for formal Olympiad-level mathematics<br><https://doi.org/10.48550/arxiv.2109.00110> | <code>direct</code> / <code>research_paper</code> | [D/C] direct record；local PDF+text；论文/卡片结果未在本章复跑 [R 未建立]。 Topics: evaluation_benchmarks, science_biology, Topic Modeling. |
| 2021-07-28 <code>web-03e8feb802d3</code> | Introducing Triton: Open-source GPU programming for neural networks<br><https://openai.com/index/triton> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: coding_software, efficiency_systems, release. |
| 2021-07-07 <code>rss-47d0c87b08f0</code> | Evaluating large language models trained on code<br><https://openai.com/index/evaluating-large-language-models-trained-on-code> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: coding_software, evaluation_benchmarks, Research. |
| 2021-06-18 <code>doi-67245cbcdf25</code> | Process for Adapting Language Models to Society (PALMS) with Values-Targeted Datasets<br><https://doi.org/10.48550/arxiv.2106.10328> | <code>direct</code> / <code>research_paper</code> | [D/C] direct record；local PDF+text；论文/卡片结果未在本章复跑 [R 未建立]。 Topics: data_training, society_governance, Topic Modeling. |
| 2021-06-10 <code>web-49090b5da7f8</code> | Improving language model behavior by training on a curated dataset<br><https://openai.com/index/improving-language-model-behavior> | <code>direct</code> / <code>dataset</code> | [D/C] direct record；local PDF+text；论文/卡片结果未在本章复跑 [R 未建立]。 Topics: foundation_models, data_training, publication. |
| 2021-06-07 <code>doi-7905f1f8106c</code> | Towards robust and domain agnostic reinforcement learning competitions<br><https://doi.org/10.48550/arxiv.2106.03748> | <code>direct</code> / <code>research_paper</code> | [D/C] direct record；local PDF+text；论文/卡片结果未在本章复跑 [R 未建立]。 Topics: safety_alignment, reinforcement_learning, Reinforcement Learning in Robotics. |
| 2021-05-11 <code>doi-33e0a55739d5</code> | Diffusion Models Beat GANs on Image Synthesis<br><https://doi.org/10.48550/arxiv.2105.05233> | <code>direct</code> / <code>research_paper</code> | [D/C] direct record；local PDF+text；论文/卡片结果未在本章复跑 [R 未建立]。 Topics: multimodal, Generative Adversarial Networks and Image Synthesis, Model Reduction and Neural Networks. |
| 2021-04-01 <code>doi-502654249c2f</code> | Sim2Real in Robotics and Automation: Applications and Challenges<br><https://doi.org/10.1109/tase.2021.3064065> | <code>direct</code> / <code>research_paper</code> | [D/C] direct record；local PDF+text；论文/卡片结果未在本章复跑 [R 未建立]。 Topics: robotics_embodied, Manufacturing Process and Optimization, Robotic Mechanisms and Dynamics. |
| 2021-03-04 <code>web-93d6498f6233</code> | Multimodal neurons in artificial neural networks<br><https://openai.com/index/multimodal-neurons> | <code>direct</code> / <code>research_paper</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: multimodal, interpretability, efficiency_systems. |
| 2021-02-26 <code>doi-98a139038f2b</code> | Learning Transferable Visual Models From Natural Language Supervision<br><https://doi.org/10.48550/arxiv.2103.00020> | <code>direct</code> / <code>research_paper</code> | [D/C] direct record；local PDF+text；论文/卡片结果未在本章复跑 [R 未建立]。 Topics: Multimodal Machine Learning Applications, Domain Adaptation and Few-Shot Learning, Human Pose and Action Recognition. |
| 2021-02-24 <code>doi-4e082ecb573e</code> | Zero-Shot Text-to-Image Generation<br><https://doi.org/10.48550/arxiv.2102.12092> | <code>direct</code> / <code>research_paper</code> | [D/C] direct record；local PDF+text；论文/卡片结果未在本章复跑 [R 未建立]。 Topics: multimodal, Multimodal Machine Learning Applications, Generative Adversarial Networks and Image Synthesis. |
| 2021-02-18 <code>doi-a0200ddffdb4</code> | Improved Denoising Diffusion Probabilistic Models<br><https://doi.org/10.48550/arxiv.2102.09672> | <code>direct</code> / <code>research_paper</code> | [D/C] direct record；local PDF+text；论文/卡片结果未在本章复跑 [R 未建立]。 Topics: Generative Adversarial Networks and Image Synthesis, Music and Audio Processing, Bayesian Methods and Mixture Models. |
| 2021-02-04 <code>web-4748993efd3b</code> | Understanding the capabilities, limitations, and societal impact of large language models<br><https://openai.com/index/understanding-the-capabilities-limitations-and-societal-impact-of-large-language-models> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: society_governance, publication. |
| 2021-01-25 <code>web-0b8a27d6cb1c</code> | Scaling Kubernetes to 7,500 nodes<br><https://openai.com/index/scaling-kubernetes-to-7500-nodes> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: coding_software, efficiency_systems, conclusion. |
| 2021-01-13 <code>doi-d5e18f78084b</code> | Asymmetric self-play for automatic goal discovery in robotic manipulation<br><https://doi.org/10.48550/arxiv.2101.04882> | <code>direct</code> / <code>research_paper</code> | [D/C] direct record；local PDF+text；论文/卡片结果未在本章复跑 [R 未建立]。 Topics: reinforcement_learning, robotics_embodied, Reinforcement Learning in Robotics. |
| 2020-12-30 <code>doi-8d0ec0197004</code> | Artificial intelligence and complex statistical modeling in glaucoma diagnosis and management<br><https://doi.org/10.1097/icu.0000000000000741> | <code>direct</code> / <code>research_paper</code> | [D/U] direct record；无 local PDF+text；主张只到可核验 landing page/metadata。 Topics: Retinal Imaging and Analysis, Glaucoma and retinal disorders, Artificial Intelligence in Healthcare. |
| 2020-12-14 <code>doi-b9ad77199cc3</code> | Extracting Training Data from Large Language Models<br><https://doi.org/10.48550/arxiv.2012.07805> | <code>direct</code> / <code>research_paper</code> | [D/C] direct record；local PDF+text；论文/卡片结果未在本章复跑 [R 未建立]。 Topics: data_training, Privacy-Preserving Technologies in Data, Adversarial Robustness in Machine Learning. |
| 2020-12-08 <code>doi-fb8e080b3251</code> | Naturally Occurring Equivariance in Neural Networks<br><https://doi.org/10.23915/distill.00024.004> | <code>direct</code> / <code>research_paper</code> | [D/U] direct record；无 local PDF+text；主张只到可核验 landing page/metadata。 Topics: efficiency_systems, Neural Networks and Applications, Advanced Neural Network Applications. |
| 2020-11-20 <code>doi-83e2cb3228ea</code> | Very Deep VAEs Generalize Autoregressive Models and Can Outperform Them on Images<br><https://doi.org/10.48550/arxiv.2011.10650> | <code>direct</code> / <code>research_paper</code> | [D/C] direct record；local PDF+text；论文/卡片结果未在本章复跑 [R 未建立]。 Topics: multimodal, Advanced Neural Network Applications, Generative Adversarial Networks and Image Synthesis. |
| 2020-11-20 <code>doi-a118f6b66ef8</code> | All the News That’s Fit to Fabricate: AI-Generated Text as a Tool of Media Misinformation<br><https://doi.org/10.1017/xps.2020.37> | <code>direct</code> / <code>research_paper</code> | [D/U] direct record；无 local PDF+text；主张只到可核验 landing page/metadata。 Topics: Misinformation and Its Impacts, Social Media and Politics, Media Influence and Politics. |
| 2020-11-17 <code>doi-e932c0cabe1e</code> | Understanding RL vision<br><https://doi.org/10.23915/distill.00029> | <code>direct</code> / <code>research_paper</code> | [D/U] direct record；无 local PDF+text；主张只到可核验 landing page/metadata。 Topics: reinforcement_learning, Industrial Vision Systems and Defect Detection, Image Processing Techniques and Applications. |
| 2020-10-27 <code>doi-c6d75fd469f0</code> | Behavior Priors for Efficient Reinforcement Learning<br><https://doi.org/10.48550/arxiv.2010.14274> | <code>direct</code> / <code>research_paper</code> | [D/C] direct record；local PDF+text；论文/卡片结果未在本章复跑 [R 未建立]。 Topics: reinforcement_learning, efficiency_systems, Reinforcement Learning in Robotics. |
| 2020-10-14 <code>doi-c2aefa010a13</code> | A deep active learning system for species identification and counting in camera trap images<br><https://doi.org/10.1111/2041-210x.13504> | <code>direct</code> / <code>research_paper</code> | [D/C] direct record；local PDF+text；论文/卡片结果未在本章复跑 [R 未建立]。 Topics: multimodal, Machine Learning and Algorithms, Domain Adaptation and Few-Shot Learning. |
| 2020-10-01 <code>doi-18c8532754ab</code> | Emergent Social Learning via Multi-agent Reinforcement Learning<br><https://doi.org/10.48550/arxiv.2010.00581> | <code>direct</code> / <code>research_paper</code> | [D/C] direct record；local PDF+text；论文/卡片结果未在本章复跑 [R 未建立]。 Topics: agents_tool_use, reinforcement_learning, Reinforcement Learning in Robotics. |
| 2020-09-16 <code>doi-4ef4b29cc7a8</code> | Improving the accessibility and transferability of machine learning algorithms for identification of animals in camera trap images: MLWIC2<br><https://doi.org/10.1002/ece3.6692> | <code>direct</code> / <code>research_paper</code> | [D/C] direct record；local PDF+text；论文/卡片结果未在本章复跑 [R 未建立]。 Topics: multimodal, Species Distribution and Climate Change, Wildlife Ecology and Conservation. |
| 2020-09-07 <code>web-88e99ce4204e</code> | Generative language modeling for automated theorem proving<br><https://openai.com/index/generative-language-modeling-for-automated-theorem-proving> | <code>direct</code> / <code>research_paper</code> | [D/C] direct record；local PDF+text；论文/卡片结果未在本章复跑 [R 未建立]。 Topics: publication. |
| 2020-09-04 <code>web-e5838d162a66</code> | Learning to summarize with human feedback<br><https://openai.com/index/learning-to-summarize-with-human-feedback> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: publication. |
| 2020-07-12 <code>openalex-w3034296948</code> | Responsive Safety in Reinforcement Learning<br><https://openalex.org/W3034296948> | <code>direct</code> / <code>research_paper</code> | [D/C] direct record；local PDF+text；论文/卡片结果未在本章复跑 [R 未建立]。 Topics: safety_alignment, reinforcement_learning, Safety Systems Engineering in Autonomy. |
| 2020-07-12 <code>openalex-w3034909681</code> | A Game Theoretic Perspective on Model-Based Reinforcement Learning<br><https://icml.cc/Conferences/2020/AcceptedPapersInitial#550> | <code>direct</code> / <code>research_paper</code> | [D/U] direct record；无 local PDF+text；主张只到可核验 landing page/metadata。 Topics: reinforcement_learning, Reinforcement Learning in Robotics, Advanced Software Engineering Methodologies. |
| 2020-06-17 <code>web-d46ff01d5c8d</code> | Image GPT<br><https://openai.com/index/image-gpt> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: foundation_models, multimodal, publication. |
| 2020-06-08 <code>doi-27a8bbf41f27</code> | Reinforcement Learning Under Moral Uncertainty<br><https://doi.org/10.48550/arxiv.2006.04734> | <code>direct</code> / <code>research_paper</code> | [D/C] direct record；local PDF+text；论文/卡片结果未在本章复跑 [R 未建立]。 Topics: reinforcement_learning, Reinforcement Learning in Robotics, Experimental Behavioral Economics Studies. |
| 2020-05-28 <code>web-8f27ffb5bc66</code> | Language models are few-shot learners<br><https://openai.com/index/language-models-are-few-shot-learners> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: milestone. |
| 2020-05-08 <code>doi-395461a5e539</code> | Measuring the Algorithmic Efficiency of Neural Networks<br><https://doi.org/10.48550/arxiv.2005.04305> | <code>direct</code> / <code>research_paper</code> | [D/C] direct record；local PDF+text；论文/卡片结果未在本章复跑 [R 未建立]。 Topics: efficiency_systems, Advanced Neural Network Applications, Explainable Artificial Intelligence (XAI). |
| 2020-05-05 <code>doi-f46506c98bc3</code> | Silly Rules Improve the Capacity of Agents to Learn Stable Enforcement and Compliance Behaviors<br><https://doi.org/10.65109/qhud6811> | <code>direct</code> / <code>research_paper</code> | [D/C] direct record；local PDF+text；论文/卡片结果未在本章复跑 [R 未建立]。 Topics: agents_tool_use, Evolutionary Game Theory and Cooperation, Experimental Behavioral Economics Studies. |
| 2020-05-05 <code>web-3e1e4e4eb6ba</code> | AI and efficiency<br><https://openai.com/index/ai-and-efficiency> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: efficiency_systems, publication. |
| 2020-04-16 <code>web-a77a394cc7a9</code> | Improving verifiability in AI development<br><https://openai.com/index/improving-verifiability> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: publication. |
| 2020-04-14 <code>web-6e6e5524f81a</code> | OpenAI Microscope<br><https://openai.com/index/microscope> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: release. |
| 2020-04-01 <code>doi-f3a882b06762</code> | An Overview of Early Vision in InceptionV1<br><https://doi.org/10.23915/distill.00024.002> | <code>direct</code> / <code>research_paper</code> | [D/U] direct record；无 local PDF+text；主张只到可核验 landing page/metadata。 Topics: Adversarial Robustness in Machine Learning, Generative Adversarial Networks and Image Synthesis, Reinforcement Learning in Robotics. |
| 2020-03-25 <code>doi-4c4a270fbe8f</code> | Fiber: A Platform for Efficient Development and Distributed Training for Reinforcement Learning and Population-Based Methods<br><https://doi.org/10.48550/arxiv.2003.11164> | <code>direct</code> / <code>research_paper</code> | [D/C] direct record；local PDF+text；论文/卡片结果未在本章复跑 [R 未建立]。 Topics: reinforcement_learning, data_training, efficiency_systems. |
| 2020-03-23 <code>doi-5eda78a3d0bf</code> | Evolutionary Population Curriculum for Scaling Multi-Agent Reinforcement Learning<br><https://doi.org/10.48550/arxiv.2003.10423> | <code>direct</code> / <code>research_paper</code> | [D/C] direct record；local PDF+text；论文/卡片结果未在本章复跑 [R 未建立]。 Topics: agents_tool_use, reinforcement_learning, data_training. |
| 2020-03-10 <code>doi-981f179e5e64</code> | Retrospective Analysis of the 2019 MineRL Competition on Sample Efficient Reinforcement Learning<br><https://doi.org/10.48550/arxiv.2003.05012> | <code>direct</code> / <code>research_paper</code> | [D/C] direct record；local PDF+text；论文/卡片结果未在本章复跑 [R 未建立]。 Topics: reinforcement_learning, efficiency_systems, Reinforcement Learning in Robotics. |
| 2020-01-30 <code>web-1aad186b08fb</code> | OpenAI standardizes on PyTorch<br><https://openai.com/index/openai-pytorch> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: safety. |
| 2020-01-25 <code>doi-72dde12af2bb</code> | Silly rules improve the capacity of agents to learn stable enforcement and compliance behaviors<br><https://doi.org/10.48550/arxiv.2001.09318> | <code>direct</code> / <code>research_paper</code> | [D/C] direct record；local PDF+text；论文/卡片结果未在本章复跑 [R 未建立]。 Topics: agents_tool_use, Crime, Illicit Activities, and Governance, Crime Patterns and Interventions. |
| 2020-01-23 <code>web-bd2a7f12ec03</code> | Scaling laws for neural language models<br><https://openai.com/index/scaling-laws-for-neural-language-models> | <code>direct</code> / <code>research_paper</code> | [D/C] direct record；local PDF+text；论文/卡片结果未在本章复跑 [R 未建立]。 Topics: efficiency_systems, publication. |
| 2019-12-13 <code>doi-72de6a38bcf8</code> | Neural Network Surgery with Sets<br><https://doi.org/10.48550/arxiv.1912.06719> | <code>direct</code> / <code>research_paper</code> | [D/C] direct record；local PDF+text；论文/卡片结果未在本章复跑 [R 未建立]。 Topics: efficiency_systems, Neural Networks and Applications, Explainable Artificial Intelligence (XAI). |
| 2019-12-13 <code>web-2c89cede3d83</code> | Dota 2 with large scale deep reinforcement learning<br><https://openai.com/index/dota-2-with-large-scale-deep-reinforcement-learning> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: reinforcement_learning, efficiency_systems, publication. |
| 2019-12-11 <code>doi-2ff67821a939</code> | Regulatory Markets for AI Safety<br><https://doi.org/10.48550/arxiv.2001.00078> | <code>direct</code> / <code>research_paper</code> | [D/C] direct record；local PDF+text；论文/卡片结果未在本章复跑 [R 未建立]。 Topics: safety_alignment, society_governance, Adversarial Robustness in Machine Learning. |
| 2019-12-05 <code>web-19b78ea11083</code> | Deep double descent<br><https://openai.com/index/deep-double-descent> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: publication. |
| 2019-12-03 <code>web-2f2d250f7b2e</code> | Procgen Benchmark<br><https://openai.com/index/procgen-benchmark> | <code>direct</code> / <code>benchmark</code> | [D/C] direct record；local PDF+text；论文/卡片结果未在本章复跑 [R 未建立]。 Topics: evaluation_benchmarks, release. |
| 2019-12-03 <code>doi-a0f4dd5c6a07</code> | Leveraging Procedural Generation to Benchmark Reinforcement Learning<br><https://doi.org/10.48550/arxiv.1912.01588> | <code>direct</code> / <code>research_paper</code> | [D/C] direct record；local PDF+text；论文/卡片结果未在本章复跑 [R 未建立]。 Topics: reinforcement_learning, evaluation_benchmarks, Reinforcement Learning in Robotics. |
| 2019-11-21 <code>web-c23fd24e1cf8</code> | Safety Gym<br><https://openai.com/index/safety-gym> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: safety_alignment, release. |
| 2019-11-21 <code>web-20e4089b4976</code> | Benchmarking safe exploration in deep reinforcement learning<br><https://openai.com/index/benchmarking-safe-exploration-in-deep-reinforcement-learning> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: reinforcement_learning, evaluation_benchmarks, publication. |
| 2019-10-15 <code>web-69dcd4141a73</code> | Solving Rubik’s Cube with a robot hand<br><https://openai.com/index/solving-rubiks-cube> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: robotics_embodied, milestone. |
| 2019-09-19 <code>web-dd954fe97ea9</code> | Fine-tuning GPT-2 from human preferences<br><https://openai.com/index/fine-tuning-gpt-2> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: foundation_models, data_training, publication. |
| 2019-09-17 <code>web-bb4559fb85ee</code> | Emergent tool use from multi-agent interaction<br><https://openai.com/index/emergent-tool-use> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: agents_tool_use, milestone. |
| 2019-09-17 <code>doi-ce7617c3d37b</code> | Emergent Tool Use From Multi-Agent Autocurricula<br><https://doi.org/10.48550/arxiv.1909.07528> | <code>direct</code> / <code>research_paper</code> | [D/C] direct record；local PDF+text；论文/卡片结果未在本章复跑 [R 未建立]。 Topics: agents_tool_use, Reinforcement Learning in Robotics, Evolutionary Algorithms and Applications. |
| 2019-08-22 <code>web-90184b0d09ce</code> | Testing robustness against unforeseen adversaries<br><https://openai.com/index/testing-robustness> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: safety_alignment, evaluation_benchmarks, publication. |
| 2019-07-28 <code>doi-78d1b6b26e7f</code> | An Atari Model Zoo for Analyzing, Visualizing, and Comparing Deep Reinforcement Learning Agents<br><https://doi.org/10.24963/ijcai.2019/452> | <code>direct</code> / <code>research_paper</code> | [D/C] direct record；local PDF+text；论文/卡片结果未在本章复跑 [R 未建立]。 Topics: agents_tool_use, reinforcement_learning, Reinforcement Learning in Robotics. |
| 2019-07-10 <code>web-c411875eb53e</code> | Why responsible AI development needs cooperation on safety<br><https://openai.com/index/cooperation-on-safety> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: safety_alignment, publication. |
| 2019-07-10 <code>doi-b86a7c23c742</code> | Skill emergence and transfer in multi-agent environments<br><https://doi.org/10.1145/3319619.3326794> | <code>direct</code> / <code>research_paper</code> | [D/U] direct record；无 local PDF+text；主张只到可核验 landing page/metadata。 Topics: agents_tool_use, Reinforcement Learning in Robotics, Optimization and Search Problems. |
| 2019-05-03 <code>web-7444293e8aab</code> | Transfer of adversarial robustness between perturbation types<br><https://openai.com/index/transfer-of-adversarial-robustness-between-perturbation-types> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: safety_alignment, publication. |
| 2019-04-25 <code>web-e603f4950615</code> | MuseNet<br><https://openai.com/index/musenet> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: milestone. |
| 2019-04-23 <code>web-288dc1f2b785</code> | Generative modeling with sparse transformers<br><https://openai.com/index/sparse-transformer> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: efficiency_systems, publication. |
| 2019-04-15 <code>web-ee157763b4c2</code> | OpenAI Five defeats Dota 2 world champions<br><https://openai.com/index/openai-five-defeats-dota-2-world-champions> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: milestone. |
| 2019-03-21 <code>web-a193572aa133</code> | Implicit generation and generalization methods for energy-based models<br><https://openai.com/index/energy-based-models> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: publication. |
| 2019-03-06 <code>web-e4ccd46451d8</code> | Introducing Activation Atlases<br><https://openai.com/index/introducing-activation-atlases> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: interpretability, publication. |
| 2019-03-04 <code>web-5543eae6f599</code> | Neural MMO: A massively multiagent game environment<br><https://openai.com/index/neural-mmo> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: agents_tool_use, milestone. |
| 2019-02-19 <code>web-badf447fbf1d</code> | AI safety needs social scientists<br><https://openai.com/index/ai-safety-needs-social-scientists> | <code>direct</code> / <code>research_paper</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: safety_alignment, publication. |
| 2019-02-14 <code>web-1b5948ab0241</code> | Better language models and their implications<br><https://openai.com/index/better-language-models> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: milestone. |
| 2019-02-04 <code>web-8eb7839a3c9c</code> | Computational limitations in robust classification and win-win results<br><https://openai.com/index/computational-limitations-in-robust-classification-and-win-win-results> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: safety_alignment, publication. |
| 2019-01-27 <code>doi-dda2dbb909d2</code> | Legible Normativity for AI Alignment<br><https://doi.org/10.1145/3306618.3314258> | <code>direct</code> / <code>research_paper</code> | [D/C] direct record；local PDF+text；论文/卡片结果未在本章复跑 [R 未建立]。 Topics: safety_alignment, Experimental Behavioral Economics Studies, Evolutionary Game Theory and Cooperation. |
| 2019-01-27 <code>doi-795a1bf31797</code> | Incomplete Contracting and AI Alignment<br><https://doi.org/10.1145/3306618.3314250> | <code>direct</code> / <code>research_paper</code> | [D/C] direct record；local PDF+text；论文/卡片结果未在本章复跑 [R 未建立]。 Topics: safety_alignment, Law, Economics, and Judicial Systems, Auction Theory and Applications. |
| 2018-12-14 <code>web-55cf833bc7db</code> | How AI training scales<br><https://openai.com/index/how-ai-training-scales> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: data_training, efficiency_systems, milestone. |
| 2018-12-06 <code>web-8c8b67d910ad</code> | Quantifying generalization in reinforcement learning<br><https://openai.com/index/quantifying-generalization-in-reinforcement-learning> | <code>direct</code> / <code>research_paper</code> | [D/C] direct record；local PDF+text；论文/卡片结果未在本章复跑 [R 未建立]。 Topics: reinforcement_learning, publication. |
| 2018-11-08 <code>web-da2451f0c1b3</code> | Spinning Up in Deep RL<br><https://openai.com/index/spinning-up-in-deep-rl> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: reinforcement_learning, release. |
| 2018-11-07 <code>web-32d654a60eb0</code> | Learning concepts with energy functions<br><https://openai.com/index/learning-concepts-with-energy-functions> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: interpretability, publication. |
| 2018-11-05 <code>web-766b5a24063f</code> | Plan online, learn offline: Efficient learning and exploration via model-based control<br><https://openai.com/index/plan-online-learn-offline> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: efficiency_systems, publication. |
| 2018-11-03 <code>doi-0e1975aadac7</code> | Legible Normativity for AI Alignment: The Value of Silly Rules<br><https://doi.org/10.48550/arxiv.1811.01267> | <code>direct</code> / <code>research_paper</code> | [D/C] direct record；local PDF+text；论文/卡片结果未在本章复跑 [R 未建立]。 Topics: safety_alignment, Experimental Behavioral Economics Studies, Evolutionary Game Theory and Cooperation. |
| 2018-10-31 <code>web-586f20cf7e17</code> | Reinforcement learning with prediction-based rewards<br><https://openai.com/index/reinforcement-learning-with-prediction-based-rewards> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: reinforcement_learning, milestone. |
| 2018-10-22 <code>web-650ead310c3d</code> | Learning complex goals with iterated amplification<br><https://openai.com/index/learning-complex-goals-with-iterated-amplification> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: publication. |
| 2018-10-02 <code>web-bf0161f536fd</code> | FFJORD: Free-form continuous dynamics for scalable reversible generative models<br><https://openai.com/index/ffjord> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: efficiency_systems, publication. |
| 2018-09-14 <code>doi-72999139bf4d</code> | Inferring single-trial neural population dynamics using sequential auto-encoders<br><https://doi.org/10.1038/s41592-018-0109-9> | <code>direct</code> / <code>research_paper</code> | [D/U] direct record；无 local PDF+text；主张只到可核验 landing page/metadata。 Topics: coding_software, Neural dynamics and brain function, Functional Brain Connectivity Studies. |
| 2018-08-23 <code>web-a626622dc4dd</code> | The International 2018: Results<br><https://openai.com/index/the-international-2018-results> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: milestone. |
| 2018-08-13 <code>web-51bd36a671fe</code> | Large-scale study of curiosity-driven learning<br><https://openai.com/index/large-scale-study-of-curiosity-driven-learning> | <code>direct</code> / <code>research_paper</code> | [D/C] direct record；local PDF+text；论文/卡片结果未在本章复跑 [R 未建立]。 Topics: efficiency_systems, publication. |
| 2018-08-06 <code>web-f7475741716d</code> | OpenAI Five Benchmark: Results<br><https://openai.com/index/openai-five-benchmark-results> | <code>direct</code> / <code>benchmark</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: evaluation_benchmarks, milestone. |
| 2018-07-30 <code>web-afb3f994b65f</code> | Learning dexterity<br><https://openai.com/index/learning-dexterity> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: milestone. |
| 2018-07-26 <code>web-edebd9bf64b1</code> | Variational option discovery algorithms<br><https://openai.com/index/variational-option-discovery-algorithms> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: publication. |
| 2018-07-18 <code>rss-c2f416b49d98</code> | OpenAI Five Benchmark<br><https://openai.com/index/openai-five-benchmark> | <code>direct</code> / <code>benchmark</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: evaluation_benchmarks, Company. |
| 2018-07-09 <code>web-745cade4a69f</code> | Glow: Better reversible generative models<br><https://openai.com/index/glow> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: milestone. |
| 2018-07-09 <code>doi-c4f7ab712538</code> | Evaluating Generalization in Multiagent Systems using Agent-Interaction Graphs<br><https://doi.org/10.65109/pnky6321> | <code>direct</code> / <code>research_paper</code> | [D/U] direct record；无 local PDF+text；主张只到可核验 landing page/metadata。 Topics: agents_tool_use, evaluation_benchmarks, Advanced Graph Neural Networks. |
| 2018-07-04 <code>web-f2da50172fdb</code> | Learning Montezuma’s Revenge from a single demonstration<br><https://openai.com/index/learning-montezumas-revenge-from-a-single-demonstration> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: conclusion. |
| 2018-06-26 <code>doi-01d7b7992345</code> | Learning Complex Dexterous Manipulation with Deep Reinforcement Learning and Demonstrations<br><https://doi.org/10.15607/rss.2018.xiv.049> | <code>direct</code> / <code>research_paper</code> | [D/C] direct record；local PDF+text；论文/卡片结果未在本章复跑 [R 未建立]。 Topics: reinforcement_learning, robotics_embodied, Robot Manipulation and Learning. |
| 2018-06-25 <code>web-881e79a7db84</code> | OpenAI Five<br><https://openai.com/index/openai-five> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: milestone. |
| 2018-06-22 <code>web-04a283baef83</code> | Retro Contest: Results<br><https://openai.com/index/retro-contest-results> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: evaluation_benchmarks, conclusion. |
| 2018-06-17 <code>web-ecdd1d648f31</code> | Learning policy representations in multiagent systems<br><https://openai.com/index/learning-policy-representations-in-multiagent-systems> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: agents_tool_use, reinforcement_learning, society_governance. |
| 2018-06-11 <code>web-91c4184aef84</code> | Improving language understanding with unsupervised learning<br><https://openai.com/index/language-unsupervised> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: milestone. |
| 2018-06-02 <code>web-6a26561219cf</code> | GamePad: A learning environment for theorem proving<br><https://openai.com/index/gamepad> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: publication. |
| 2018-05-25 <code>web-31ad9adfb05a</code> | Gym Retro<br><https://openai.com/index/gym-retro> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: release. |
| 2018-05-16 <code>web-6a0c4b2b4d23</code> | AI and compute<br><https://openai.com/index/ai-and-compute> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: conclusion. |
| 2018-05-03 <code>web-b9028523d1d6</code> | AI safety via debate<br><https://openai.com/index/debate> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: safety_alignment, publication. |
| 2018-05-01 <code>doi-56c59f553eee</code> | Overcoming Exploration in Reinforcement Learning with Demonstrations<br><https://doi.org/10.1109/icra.2018.8463162> | <code>direct</code> / <code>research_paper</code> | [D/C] direct record；local PDF+text；论文/卡片结果未在本章复跑 [R 未建立]。 Topics: reinforcement_learning, Reinforcement Learning in Robotics, Robot Manipulation and Learning. |
| 2018-04-27 <code>doi-a1c2b563b3cf</code> | DeepType: Multilingual Entity Linking by Neural Type System Evolution<br><https://doi.org/10.1609/aaai.v32i1.12008> | <code>direct</code> / <code>research_paper</code> | [D/C] direct record；local PDF+text；论文/卡片结果未在本章复跑 [R 未建立]。 Topics: Topic Modeling, Natural Language Processing Techniques, Semantic Web and Ontologies. |
| 2018-04-18 <code>web-49bd957aa3cd</code> | Evolved Policy Gradients<br><https://openai.com/index/evolved-policy-gradients> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: reinforcement_learning, society_governance, milestone. |
| 2018-04-10 <code>web-4ec028b44ef3</code> | Gotta Learn Fast: A new benchmark for generalization in RL<br><https://openai.com/index/gotta-learn-fast> | <code>direct</code> / <code>benchmark</code> | [D/C] direct record；local PDF+text；论文/卡片结果未在本章复跑 [R 未建立]。 Topics: reinforcement_learning, evaluation_benchmarks, publication. |
| 2018-04-05 <code>web-acd6abaea30b</code> | Retro Contest<br><https://openai.com/index/retro-contest> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: evaluation_benchmarks, milestone. |
| 2018-03-20 <code>web-f8c36640d772</code> | Variance reduction for policy gradient with action-dependent factorized baselines<br><https://openai.com/index/variance-reduction-for-policy-gradient-with-action-dependent-factorized-baselines> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: reinforcement_learning, society_governance, publication. |
| 2018-03-15 <code>web-57da868082bd</code> | Improving GANs using optimal transport<br><https://openai.com/index/improving-gans-using-optimal-transport> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: publication. |
| 2018-03-08 <code>web-d505c1fa06f7</code> | On first-order meta-learning algorithms<br><https://openai.com/index/on-first-order-meta-learning-algorithms> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: publication. |
| 2018-03-07 <code>web-056a847a80e1</code> | Reptile: A scalable meta-learning algorithm<br><https://openai.com/index/reptile> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: efficiency_systems, publication. |
| 2018-03-03 <code>web-d0e669004c4c</code> | Some considerations on learning to explore via meta-reinforcement learning<br><https://openai.com/index/some-considerations-on-learning-to-explore-via-meta-reinforcement-learning> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: reinforcement_learning, publication. |
| 2018-02-26 <code>web-00023fa6b69f</code> | Multi-Goal Reinforcement Learning: Challenging robotics environments and request for research<br><https://openai.com/index/multi-goal-reinforcement-learning> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: reinforcement_learning, robotics_embodied, publication. |
| 2018-02-26 <code>web-d82bfc8fbf3b</code> | Ingredients for robotics research<br><https://openai.com/index/ingredients-for-robotics-research> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: robotics_embodied, release. |
| 2018-02-20 <code>web-e2f2677a18ef</code> | Preparing for malicious uses of AI<br><https://openai.com/index/preparing-for-malicious-uses-of-ai> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: security, publication. |
| 2018-02-15 <code>web-74460ec29c9b</code> | Interpretable machine learning through teaching<br><https://openai.com/index/interpretable-machine-learning-through-teaching> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: interpretability, publication. |
| 2018-02-15 <code>openalex-w2786926540</code> | Generative Models for Alignment and Data Efficiency in Language<br><https://openreview.net/pdf?id=rJ7RBNe0-> | <code>direct</code> / <code>research_paper</code> | [D/U] direct record；无 local PDF+text；主张只到可核验 landing page/metadata。 Topics: safety_alignment, data_training, efficiency_systems. |
| 2018-02-07 <code>web-7f5353232f4b</code> | Discovering types for entity disambiguation<br><https://openai.com/index/discovering-types-for-entity-disambiguation> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: publication. |
| 2018-01-31 <code>web-c105ed6e84d9</code> | Requests for Research 2.0<br><https://openai.com/index/requests-for-research-2> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: release. |
| 2018-01-18 <code>web-4c9c2f788cfe</code> | Scaling Kubernetes to 2,500 nodes<br><https://openai.com/index/scaling-kubernetes-to-2500-nodes> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: coding_software, efficiency_systems, conclusion. |
| 2017-12-06 <code>web-f0d270fbca2e</code> | Block-sparse GPU kernels<br><https://openai.com/index/block-sparse-gpu-kernels> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: coding_software, efficiency_systems, release. |
| 2017-12-04 <code>web-642755945311</code> | Learning sparse neural networks through L₀ regularization<br><https://openai.com/index/learning-sparse-neural-networks-through-l0-regularization> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: efficiency_systems, publication. |
| 2017-11-02 <code>web-be84f3d01add</code> | Interpretable and pedagogical examples<br><https://openai.com/index/interpretable-and-pedagogical-examples> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: interpretability, publication. |
| 2017-10-26 <code>web-3c2715ba8c64</code> | Learning a hierarchy<br><https://openai.com/index/learning-a-hierarchy> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: publication. |
| 2017-10-19 <code>web-2d6c85bd3e86</code> | Generalizing from simulation<br><https://openai.com/index/generalizing-from-simulation> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: publication. |
| 2017-10-18 <code>web-cafc87444fd3</code> | Sim-to-real transfer of robotic control with dynamics randomization<br><https://openai.com/index/sim-to-real-transfer-of-robotic-control-with-dynamics-randomization> | <code>direct</code> / <code>research_paper</code> | [D/C] direct record；local PDF+text；论文/卡片结果未在本章复跑 [R 未建立]。 Topics: robotics_embodied, publication. |
| 2017-10-18 <code>web-81a665efcb32</code> | Asymmetric actor critic for image-based robot learning<br><https://openai.com/index/asymmetric-actor-critic-for-image-based-robot-learning> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: reinforcement_learning, multimodal, robotics_embodied. |
| 2017-10-17 <code>web-740d039f3421</code> | Domain randomization and generative models for robotic grasping<br><https://openai.com/index/domain-randomization-and-generative-models-for-robotic-grasping> | <code>direct</code> / <code>research_paper</code> | [D/C] direct record；local PDF+text；论文/卡片结果未在本章复跑 [R 未建立]。 Topics: robotics_embodied, publication. |
| 2017-10-11 <code>web-6b89bea3012e</code> | Meta-learning for wrestling<br><https://openai.com/index/meta-learning-for-wrestling> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: publication. |
| 2017-10-11 <code>web-581dd87d0e73</code> | Competitive self-play<br><https://openai.com/index/competitive-self-play> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: reinforcement_learning, milestone. |
| 2017-09-29 <code>web-994b9539f355</code> | Nonlinear computation in deep linear networks<br><https://openai.com/index/nonlinear-computation-in-deep-linear-networks> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: efficiency_systems, conclusion. |
| 2017-09-14 <code>web-22f524cf064d</code> | Learning to model other minds<br><https://openai.com/index/learning-to-model-other-minds> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: publication. |
| 2017-09-13 <code>web-df4bf285848c</code> | Learning with opponent-learning awareness<br><https://openai.com/index/learning-with-opponent-learning-awareness> | <code>direct</code> / <code>research_paper</code> | [D/C] direct record；local PDF+text；论文/卡片结果未在本章复跑 [R 未建立]。 Topics: publication. |
| 2017-09-01 <code>doi-1073bf5fca41</code> | Domain randomization for transferring deep neural networks from simulation to the real world<br><https://doi.org/10.1109/iros.2017.8202133> | <code>direct</code> / <code>research_paper</code> | [D/C] direct record；local PDF+text；论文/卡片结果未在本章复跑 [R 未建立]。 Topics: efficiency_systems, Domain Adaptation and Few-Shot Learning, Advanced Neural Network Applications. |
| 2017-08-18 <code>web-7d4417b6f1fa</code> | OpenAI Baselines: ACKTR & A2C<br><https://openai.com/index/openai-baselines-acktr-a2c> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: release. |
| 2017-08-16 <code>web-7e4bda609cfe</code> | More on Dota 2<br><https://openai.com/index/more-on-dota-2> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: milestone. |
| 2017-08-11 <code>web-e636241aa383</code> | Dota 2<br><https://openai.com/index/dota-2> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: milestone. |
| 2017-08-03 <code>web-698b1890fdd1</code> | Gathering human feedback<br><https://openai.com/index/gathering-human-feedback> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: release. |
| 2017-07-27 <code>web-96322b294d06</code> | Better exploration with parameter noise<br><https://openai.com/index/better-exploration-with-parameter-noise> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: publication. |
| 2017-07-20 <code>web-9ab874ca8944</code> | Proximal Policy Optimization<br><https://openai.com/index/openai-baselines-ppo> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: reinforcement_learning, society_governance, release. |
| 2017-07-17 <code>web-0ea53cd481f0</code> | Robust adversarial inputs<br><https://openai.com/index/robust-adversarial-inputs> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: safety_alignment, publication. |
| 2017-07-05 <code>web-c1af6b88bb0c</code> | Hindsight Experience Replay<br><https://openai.com/index/hindsight-experience-replay> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: publication. |
| 2017-07-01 <code>web-27094283cc50</code> | Teacher–student curriculum learning<br><https://openai.com/index/teacher-student-curriculum-learning> | <code>direct</code> / <code>research_paper</code> | [D/C] direct record；local PDF+text；论文/卡片结果未在本章复跑 [R 未建立]。 Topics: data_training, publication. |
| 2017-06-28 <code>web-1f51b604f32d</code> | Faster physics in Python<br><https://openai.com/index/faster-physics-in-python> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: science_biology, release. |
| 2017-06-13 <code>web-98fa91b77ece</code> | Learning from human preferences<br><https://openai.com/index/learning-from-human-preferences> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: release. |
| 2017-06-08 <code>web-859febdd36b4</code> | Learning to cooperate, compete, and communicate<br><https://openai.com/index/learning-to-cooperate-compete-and-communicate> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: publication. |
| 2017-06-05 <code>web-b4a0b823929e</code> | UCB exploration via Q-ensembles<br><https://openai.com/index/ucb-exploration-via-q-ensembles> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: publication. |
| 2017-05-24 <code>web-e9f7d2174d9e</code> | OpenAI Baselines: DQN<br><https://openai.com/index/openai-baselines-dqn> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: release. |
| 2017-05-24 <code>doi-c20dba61f182</code> | ImageNet classification with deep convolutional neural networks<br><https://doi.org/10.1145/3065386> | <code>direct</code> / <code>research_paper</code> | [D/C] direct record；local PDF+text；论文/卡片结果未在本章复跑 [R 未建立]。 Topics: multimodal, efficiency_systems, Advanced Neural Network Applications. |
| 2017-05-16 <code>web-2cdfd9c283a5</code> | Robots that learn<br><https://openai.com/index/robots-that-learn> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: robotics_embodied, milestone. |
| 2017-05-15 <code>web-d47075c8df33</code> | Roboschool<br><https://openai.com/index/roboschool> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: release. |
| 2017-05-01 <code>doi-9fba36d36bf7</code> | Deep reinforcement learning for tensegrity robot locomotion<br><https://doi.org/10.1109/icra.2017.7989079> | <code>direct</code> / <code>research_paper</code> | [D/C] direct record；local PDF+text；论文/卡片结果未在本章复跑 [R 未建立]。 Topics: reinforcement_learning, robotics_embodied, Structural Analysis and Optimization. |
| 2017-04-21 <code>web-0ec1aa4d25c6</code> | Equivalence between policy gradients and soft Q-learning<br><https://openai.com/index/equivalence-between-policy-gradients-and-soft-q-learning> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: reinforcement_learning, society_governance, publication. |
| 2017-04-10 <code>web-78ee2f7e35fb</code> | Stochastic Neural Networks for hierarchical reinforcement learning<br><https://openai.com/index/stochastic-neural-networks-for-hierarchical-reinforcement-learning> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: reinforcement_learning, efficiency_systems, publication. |
| 2017-04-06 <code>web-5dc883adf998</code> | Unsupervised sentiment neuron<br><https://openai.com/index/unsupervised-sentiment-neuron> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: interpretability, publication. |
| 2017-04-01 <code>web-99e102be9e17</code> | Spam detection in the physical world<br><https://openai.com/index/spam-detection-in-the-physical-world> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: conclusion. |
| 2017-03-31 <code>doi-23378bde8a15</code> | Practical Black-Box Attacks against Machine Learning<br><https://doi.org/10.1145/3052973.3053009> | <code>direct</code> / <code>research_paper</code> | [D/C] direct record；local PDF+text；论文/卡片结果未在本章复跑 [R 未建立]。 Topics: security, Adversarial Robustness in Machine Learning, Advanced Malware Detection Techniques. |
| 2017-03-24 <code>web-d5d36adbb328</code> | Evolution strategies as a scalable alternative to reinforcement learning<br><https://openai.com/index/evolution-strategies> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: reinforcement_learning, efficiency_systems, publication. |
| 2017-03-21 <code>web-cb1bd737766f</code> | One-shot imitation learning<br><https://openai.com/index/one-shot-imitation-learning> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: publication. |
| 2017-03-16 <code>web-fc3184f92101</code> | Learning to communicate<br><https://openai.com/index/learning-to-communicate> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: conclusion. |
| 2017-03-15 <code>web-7f42d0024562</code> | Emergence of grounded compositional language in multi-agent populations<br><https://openai.com/index/emergence-of-grounded-compositional-language-in-multi-agent-populations> | <code>direct</code> / <code>research_paper</code> | [D/C] direct record；local PDF+text；论文/卡片结果未在本章复跑 [R 未建立]。 Topics: agents_tool_use, publication. |
| 2017-03-12 <code>web-5edfbdd1919b</code> | Prediction and control with temporal segment models<br><https://openai.com/index/prediction-and-control-with-temporal-segment-models> | <code>direct</code> / <code>research_paper</code> | [D/C] direct record；local PDF+text；论文/卡片结果未在本章复跑 [R 未建立]。 Topics: publication. |
| 2017-03-06 <code>web-f2c27a92d0c0</code> | Third-person imitation learning<br><https://openai.com/index/third-person-imitation-learning> | <code>direct</code> / <code>research_paper</code> | [D/C] direct record；local PDF+text；论文/卡片结果未在本章复跑 [R 未建立]。 Topics: publication. |
| 2017-02-27 <code>doi-2e5ddafc1263</code> | Reinforcement Learning with Deep Energy-Based Policies<br><https://doi.org/10.48550/arxiv.1702.08165> | <code>direct</code> / <code>research_paper</code> | [D/C] direct record；local PDF+text；论文/卡片结果未在本章复跑 [R 未建立]。 Topics: reinforcement_learning, Reinforcement Learning in Robotics, Adversarial Robustness in Machine Learning. |
| 2017-02-24 <code>web-aa725438d590</code> | Attacking machine learning with adversarial examples<br><https://openai.com/index/attacking-machine-learning-with-adversarial-examples> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: security, conclusion. |
| 2017-02-08 <code>web-b61e34ff0c8b</code> | Adversarial attacks on neural network policies<br><https://openai.com/index/adversarial-attacks-on-neural-network-policies> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: efficiency_systems, security, publication. |
| 2017-01-19 <code>web-54f60ca8fca3</code> | PixelCNN++: Improving the PixelCNN with discretized logistic mixture likelihood and other modifications<br><https://openai.com/index/pixelcnn-plus-plus> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: publication. |
| 2016-12-21 <code>web-02a375c53f6d</code> | Faulty reward functions in the wild<br><https://openai.com/index/faulty-reward-functions> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: reinforcement_learning, conclusion. |
| 2016-12-05 <code>web-d346a1aa6260</code> | Universe<br><https://openai.com/index/universe> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: release. |
| 2016-11-15 <code>web-5185c2677d36</code> | #Exploration: A study of count-based exploration for deep reinforcement learning<br><https://openai.com/index/exploration> | <code>direct</code> / <code>research_paper</code> | [D/C] direct record；local PDF+text；论文/卡片结果未在本章复跑 [R 未建立]。 Topics: reinforcement_learning, publication. |
| 2016-11-14 <code>web-448ab2e5dc1c</code> | On the quantitative analysis of decoder-based generative models<br><https://openai.com/index/on-the-quantitative-analysis-of-decoder-based-generative-models> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: coding_software, publication. |
| 2016-11-11 <code>web-75f26884ae62</code> | A connection between generative adversarial networks, inverse reinforcement learning, and energy-based models<br><https://openai.com/index/a-connection-between-generative-adversarial-networks-inverse-reinforcement-learning-and-energy-based-models> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: reinforcement_learning, efficiency_systems, publication. |
| 2016-11-09 <code>web-ee30ed958850</code> | RL²: Fast reinforcement learning via slow reinforcement learning<br><https://openai.com/index/rl2> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: reinforcement_learning, publication. |
| 2016-11-08 <code>web-72d654a4e6e8</code> | Variational lossy autoencoder<br><https://openai.com/index/variational-lossy-autoencoder> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: coding_software, publication. |
| 2016-11-02 <code>web-9234474c676c</code> | Extensions and limitations of the neural GPU<br><https://openai.com/index/extensions-and-limitations-of-the-neural-gpu> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: coding_software, publication. |
| 2016-10-18 <code>web-7ff7a8c4064a</code> | Semi-supervised knowledge transfer for deep learning from private training data<br><https://openai.com/index/semi-supervised-knowledge-transfer-for-deep-learning-from-private-training-data> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: data_training, publication. |
| 2016-10-11 <code>web-e0cf1cef8809</code> | Transfer from simulation to real world through learning deep inverse dynamics model<br><https://openai.com/index/transfer-from-simulation-to-real-world-through-learning-deep-inverse-dynamics-model> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: publication. |
| 2016-08-29 <code>web-5eec2d002d2e</code> | Infrastructure for deep learning<br><https://openai.com/index/infrastructure-for-deep-learning> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: conclusion. |
| 2016-06-21 <code>web-99799942438c</code> | Concrete AI safety problems<br><https://openai.com/index/concrete-ai-safety-problems> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: safety_alignment, publication. |
| 2016-06-20 <code>web-41c470b2e565</code> | OpenAI technical goals<br><https://openai.com/index/openai-technical-goals> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: safety. |
| 2016-06-19 <code>openalex-w2963641140</code> | Benchmarking deep reinforcement learning for continuous control<br><https://openalex.org/W2963641140> | <code>direct</code> / <code>research_paper</code> | [D/C] direct record；local PDF+text；论文/卡片结果未在本章复跑 [R 未建立]。 Topics: reinforcement_learning, evaluation_benchmarks, Reinforcement Learning in Robotics. |
| 2016-06-16 <code>web-92d63a138081</code> | Generative models<br><https://openai.com/index/generative-models> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: publication. |
| 2016-06-12 <code>doi-3ca1a4531296</code> | InfoGAN: Interpretable Representation Learning by Information Maximizing Generative Adversarial Nets<br><https://doi.org/10.48550/arxiv.1606.03657> | <code>direct</code> / <code>research_paper</code> | [D/C] direct record；local PDF+text；论文/卡片结果未在本章复跑 [R 未建立]。 Topics: interpretability, Generative Adversarial Networks and Image Synthesis, Face recognition and analysis. |
| 2016-05-25 <code>web-fbe426d73ec2</code> | Adversarial training methods for semi-supervised text classification<br><https://openai.com/index/adversarial-training-methods-for-semi-supervised-text-classification> | <code>direct</code> / <code>research_paper</code> | [D/C] direct record；local PDF+text；论文/卡片结果未在本章复跑 [R 未建立]。 Topics: data_training, publication. |
| 2016-04-27 <code>web-c33fb9b7adb1</code> | OpenAI Gym Beta<br><https://openai.com/index/openai-gym-beta> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: release. |
| 2016-02-25 <code>web-1faeb6d01263</code> | Weight normalization: A simple reparameterization to accelerate training of deep neural networks<br><https://openai.com/index/weight-normalization> | <code>direct</code> / <code>research_paper</code> | [D/C] direct record；local PDF+text；论文/卡片结果未在本章复跑 [R 未建立]。 Topics: data_training, efficiency_systems, publication. |
| 未标日期 <code>web-dbb9bd278cda</code> | OpenAI's approach to AI and national security<br><https://openai.com/global-affairs/openais-approach-to-ai-and-national-security> | <code>direct</code> / <code>blog_with_report</code> | [D-meta/U] official title/date/route；正文强度服从 page-evidence，不能由元数据补写方法或数字。 Topics: security, safety. |
| 2024-10-09 <code>web-13ac2c97a978</code> | An update on disrupting deceptive uses of AI<br><https://openai.com/global-affairs/an-update-on-disrupting-deceptive-uses-of-ai> | <code>direct</code> / <code>blog_with_report</code> | [D/C] direct record；local PDF+text；论文/卡片结果未在本章复跑 [R 未建立]。 Topics: safety_alignment, safety. |
| 2023-12-18 <code>preparedness-framework-beta</code> | Preparedness Framework (Beta)<br><https://openai.com/index/frontier-risk-and-preparedness> | <code>direct</code> / <code>technical_report</code> | [D/C] direct record；local PDF+text；论文/卡片结果未在本章复跑 [R 未建立]。 Topics: safety, governance, preparedness. |

### 16.3 Affiliated：明确 OpenAI 机构署名的广义研究（122）

| 日期 / 记录 | 论文与 primary URL | 类型 | work-level 边界 |
|---|---|---|---|
| 2026-06-30 <code>doi-9f3dedaf3e1d</code> | GeneBench-Pro: Evaluating Multistage Statistical Reasoning in Genomics, Quantitative Biology, and Translational Biomedicine<br><https://doi.org/10.64898/2026.06.29.735386> | <code>affiliated</code> / <code>research_paper</code> | [D/U] work-level OpenAI affiliation；当前无 local PDF+text；不推断产品采用或模型 lineage。 Topics: evaluation_benchmarks, science_biology, Artificial Intelligence in Healthcare and Education. |
| 2026-06-18 <code>doi-f6896ae8cb61</code> | LLM-Assisted Reanalysis of Unsolved Rare Disease Genomes Increases Diagnostic Yield<br><https://doi.org/10.1056/aics2501343> | <code>affiliated</code> / <code>research_paper</code> | [D/U] work-level OpenAI affiliation；当前无 local PDF+text；不推断产品采用或模型 lineage。 Topics: foundation_models, science_biology, Genomics and Rare Diseases. |
| 2026-06-14 <code>doi-c7aeefaa636d</code> | Building the Engine of AI: From Foundational VLSI Technologies to System-Scale Impact<br><https://doi.org/10.1109/vlsitechnologyandcir65830.2026.11577572> | <code>affiliated</code> / <code>research_paper</code> | [D/U] work-level OpenAI affiliation；当前无 local PDF+text；不推断产品采用或模型 lineage。 Topics: efficiency_systems, Cybernetics and Technology in Society, History of Computing Technologies. |
| 2026-05-29 <code>doi-b8f3e6ba77e2</code> | The Industrial Revolution of the Intelligence Age<br><https://doi.org/10.1109/mc.2026.3678508> | <code>affiliated</code> / <code>research_paper</code> | [D/U] work-level OpenAI affiliation；当前无 local PDF+text；不推断产品采用或模型 lineage。 Topics: Intelligence, Security, War Strategy, Competitive and Knowledge Intelligence, History of Computing Technologies. |
| 2026-04-30 <code>openalex-w7159891676</code> | The proportion of permutations fixing a $k$-set<br><https://arxiv.org/abs/2604.28116> | <code>affiliated</code> / <code>research_paper</code> | [D/C] work-level OpenAI affiliation；local PDF+text；不推断产品采用或模型 lineage。 Topics: Bayesian Methods and Mixture Models, Limits and Structures in Graph Theory, Random Matrices and Applications. |
| 2026-04-27 <code>doi-fa86d62168ff</code> | How AI and Human Behaviors Shape Psychosocial Effects of Extended Chatbot Use: A Longitudinal Randomized Controlled Study<br><https://doi.org/10.21203/rs.3.rs-8148142/v1> | <code>affiliated</code> / <code>research_paper</code> | [D/C] work-level OpenAI affiliation；local PDF+text；不推断产品采用或模型 lineage。 Topics: AI in Service Interactions, Digital Mental Health Interventions. |
| 2026-03-25 <code>doi-21909217dc1a</code> | Generalized Couch-Torrence inversions<br><https://doi.org/10.1007/jhep03(2026)239> | <code>affiliated</code> / <code>research_paper</code> | [D/C] work-level OpenAI affiliation；local PDF+text；不推断产品采用或模型 lineage。 Topics: Black Holes and Theoretical Physics, Astrophysical Phenomena and Observations, Noncommutative and Quantum Gravity Theories. |
| 2026-03-20 <code>doi-cf857b27447a</code> | Rigidity Matroids and Linear Algebraic Matroids with Applications to Matrix Completion and Tensor Codes<br><https://doi.org/10.1007/s00493-026-00208-z> | <code>affiliated</code> / <code>research_paper</code> | [D/C] work-level OpenAI affiliation；local PDF+text；不推断产品采用或模型 lineage。 Topics: coding_software, Matrix Theory and Algorithms, Advanced Topics in Algebra. |
| 2026-03-14 <code>doi-7c7db3a53d76</code> | HALO: Hardware-Aware Quantization with Low Critical-Path-Delay Weights for LLM Acceleration<br><https://doi.org/10.1609/aaai.v40i27.39406> | <code>affiliated</code> / <code>research_paper</code> | [D/C] work-level OpenAI affiliation；local PDF+text；不推断产品采用或模型 lineage。 Topics: foundation_models, efficiency_systems, Advanced Neural Network Applications. |
| 2026-03-04 <code>doi-9cf27e1ee902</code> | Genome modelling and design across all domains of life with Evo 2<br><https://doi.org/10.1038/s41586-026-10176-5> | <code>affiliated</code> / <code>research_paper</code> | [D/C] work-level OpenAI affiliation；local PDF+text；不推断产品采用或模型 lineage。 Topics: science_biology, Developmental Biology and Gene Regulation, Evolutionary Algorithms and Applications. |
| 2026-02-23 <code>doi-6fdb04e154e1</code> | Resonances in binary extreme-mass-ratio inspirals<br><https://doi.org/10.1103/7qjn-jl4n> | <code>affiliated</code> / <code>research_paper</code> | [D/C] work-level OpenAI affiliation；local PDF+text；不推断产品采用或模型 lineage。 Topics: Pulsars and Gravitational Waves Research, High-Energy Particle Collisions Research, Superconducting and THz Device Technology. |
| 2026-02-07 <code>doi-eddcb9c2a4f6</code> | Reducing Isotropy and Volume to KLS: Faster Rounding and Volume Algorithms<br><https://doi.org/10.1145/3795687> | <code>affiliated</code> / <code>research_paper</code> | [D/C] work-level OpenAI affiliation；local PDF+text；不推断产品采用或模型 lineage。 Topics: Markov Chains and Monte Carlo Methods, Data Management and Algorithms, Complexity and Algorithms in Graphs. |
| 2026-01-21 <code>doi-0a3555948b27</code> | GDPVAL: Evaluating AI Model Performance on Real-World Economically Valuable Tasks<br><https://doi.org/10.70777/si.v2i4.17197> | <code>affiliated</code> / <code>research_paper</code> | [D/C] work-level OpenAI affiliation；local PDF+text；不推断产品采用或模型 lineage。 Topics: evaluation_benchmarks, society_governance, Explainable Artificial Intelligence (XAI). |
| 2026-01-19 <code>doi-4571098d1633</code> | A Literature Review on Data Democratization, Self-Service Business Intelligence, and Data Literacy<br><https://doi.org/10.4018/ijban.398838> | <code>affiliated</code> / <code>research_paper</code> | [D/C] work-level OpenAI affiliation；local PDF+text；不推断产品采用或模型 lineage。 Topics: multimodal, data_training, society_governance. |
| 2026-01-01 <code>doi-dbca775931b1</code> | EcoCell: Energy Conservation Through Traffic Shaping in Cellular Radio Access Networks<br><https://doi.org/10.4230/oasics.nines.2026.6> | <code>affiliated</code> / <code>research_paper</code> | [D/C] work-level OpenAI affiliation；local PDF+text；不推断产品采用或模型 lineage。 Topics: efficiency_systems, Green IT and Sustainability, Advanced MIMO Systems Optimization. |
| 2026-01-01 <code>doi-3d8ae6c04b39</code> | A Resource-Rational Account of Human Eye Movements During Immersive Visual Search<br><https://doi.org/10.1162/opmi.a.322> | <code>affiliated</code> / <code>research_paper</code> | [D/U] work-level OpenAI affiliation；当前无 local PDF+text；不推断产品采用或模型 lineage。 Topics: Visual Attention and Saliency Detection, Neural and Behavioral Psychology Studies, Gaze Tracking and Assistive Technology. |
| 2025-12-18 <code>doi-1b567d9f1018</code> | Evaluating the Social Impact of Generative AI Systems<br><https://doi.org/10.1093/oxfordhb/9780198940272.013.0025> | <code>affiliated</code> / <code>research_paper</code> | [D/U] work-level OpenAI affiliation；当前无 local PDF+text；不推断产品采用或模型 lineage。 Topics: evaluation_benchmarks, Ethics and Social Impacts of AI, Explainable Artificial Intelligence (XAI). |
| 2025-12-11 <code>doi-884a4f3fb10c</code> | Linear Layouts: Robust Code Generation of Efficient Tensor Computation Using F_2<br><https://doi.org/10.1145/3760250.3762221> | <code>affiliated</code> / <code>research_paper</code> | [D/U] work-level OpenAI affiliation；当前无 local PDF+text；不推断产品采用或模型 lineage。 Topics: safety_alignment, coding_software, efficiency_systems. |
| 2025-12-06 <code>doi-d7f4331e0450</code> | Improving Streaming ASR via Differentially Private Fusion of Data from Multiple Sources<br><https://doi.org/10.1109/asru65441.2025.11434648> | <code>affiliated</code> / <code>research_paper</code> | [D/U] work-level OpenAI affiliation；当前无 local PDF+text；不推断产品采用或模型 lineage。 Topics: data_training, Privacy-Preserving Technologies in Data, Cryptography and Data Security. |
| 2025-11-17 <code>doi-1faefb71d312</code> | Canadian ophthalmology workforce trends from 1971 to 2022: longitudinal analysis of age, sex, and distribution compared to other surgical specialties<br><https://doi.org/10.1016/j.jcjo.2025.09.013> | <code>affiliated</code> / <code>research_paper</code> | [D/U] work-level OpenAI affiliation；当前无 local PDF+text；不推断产品采用或模型 lineage。 Topics: Diversity and Career in Medicine, Intraocular Surgery and Lenses, Surgical Simulation and Training. |
| 2025-10-19 <code>doi-73b4d20a9248</code> | Trust but Verify: Programmatic VLM Evaluation in the Wild<br><https://doi.org/10.1109/iccv51701.2025.00312> | <code>affiliated</code> / <code>research_paper</code> | [D/C] work-level OpenAI affiliation；local PDF+text；不推断产品采用或模型 lineage。 Topics: coding_software, evaluation_benchmarks, Stock Market Forecasting Methods. |
| 2025-10-15 <code>doi-e996a0621dbe</code> | Towards Interactive Evaluations for Interaction Harms in Human-AI Systems<br><https://doi.org/10.1609/aies.v8i2.36631> | <code>affiliated</code> / <code>research_paper</code> | [D/C] work-level OpenAI affiliation；local PDF+text；不推断产品采用或模型 lineage。 Topics: safety_alignment, evaluation_benchmarks, Adversarial Robustness in Machine Learning. |
| 2025-10-09 <code>doi-c9cf972d9f4d</code> | Mind the Abstraction Gap: Bringing Equality Saturation to Real-World ML Compilers<br><https://doi.org/10.1145/3763062> | <code>affiliated</code> / <code>research_paper</code> | [D/U] work-level OpenAI affiliation；当前无 local PDF+text；不推断产品采用或模型 lineage。 Topics: coding_software, Parallel Computing and Optimization Techniques, Reinforcement Learning in Robotics. |
| 2025-10-03 <code>doi-e52517dbb787</code> | Flow Autoencoders are Effective Protein Tokenizers<br><https://doi.org/10.1101/2025.10.01.679645> | <code>affiliated</code> / <code>research_paper</code> | [D/C] work-level OpenAI affiliation；local PDF+text；不推断产品采用或模型 lineage。 Topics: coding_software, science_biology, Protein Structure and Dynamics. |
| 2025-10-01 <code>doi-b1f0d10f19ed</code> | The impact of advanced AI systems on democracy<br><https://doi.org/10.1038/s41562-025-02309-z> | <code>affiliated</code> / <code>research_paper</code> | [D/U] work-level OpenAI affiliation；当前无 local PDF+text；不推断产品采用或模型 lineage。 Topics: multimodal, society_governance, Ethics and Social Impacts of AI. |
| 2025-10-01 <code>doi-420f3658e493</code> | Mercury: Unlocking Multi-GPU Operator Optimization for LLMs via Remote Memory Scheduling<br><https://doi.org/10.1145/3731569.3764798> | <code>affiliated</code> / <code>research_paper</code> | [D/U] work-level OpenAI affiliation；当前无 local PDF+text；不推断产品采用或模型 lineage。 Topics: agents_tool_use, coding_software, efficiency_systems. |
| 2025-08-20 <code>doi-e9f660c424bd</code> | Ranking by engagement and non‐engagement signals: Learnings from industry<br><https://doi.org/10.1111/nyas.15399> | <code>affiliated</code> / <code>research_paper</code> | [D/U] work-level OpenAI affiliation；当前无 local PDF+text；不推断产品采用或模型 lineage。 Topics: Digital Marketing and Social Media, Misinformation and Its Impacts, Complex Network Analysis Techniques. |
| 2025-08-03 <code>doi-8cfd8a27ef49</code> | DV365: Extremely Long User History Modeling at Instagram<br><https://doi.org/10.1145/3711896.3737209> | <code>affiliated</code> / <code>research_paper</code> | [D/C] work-level OpenAI affiliation；local PDF+text；不推断产品采用或模型 lineage。 Topics: Recommender Systems and Techniques, Caching and Content Delivery, Human Mobility and Location-Based Analysis. |
| 2025-06-05 <code>doi-17e65d0d6ee0</code> | Force-Field Optimization by End-to-End Differentiable Atomistic Simulation<br><https://doi.org/10.1021/acs.jctc.4c01784> | <code>affiliated</code> / <code>research_paper</code> | [D/C] work-level OpenAI affiliation；local PDF+text；不推断产品采用或模型 lineage。 Topics: Machine Learning in Materials Science, Nuclear Materials and Properties, Ion-surface interactions and analysis. |
| 2025-05-24 <code>doi-b5e93ba61ccf</code> | RL-Finetuning of OpenAI o1-mini to Enhance Biomedical Reasoning<br><https://doi.org/10.1101/2025.05.19.654988> | <code>affiliated</code> / <code>research_paper</code> | [D/C] work-level OpenAI affiliation；local PDF+text；不推断产品采用或模型 lineage。 Topics: reinforcement_learning, science_biology, Topic Modeling. |
| 2025-05-01 <code>doi-a5d3f457e759</code> | The LAW Theorem: Local Reads and Linearizable Asynchronous Replication<br><https://doi.org/10.14778/3746405.3746411> | <code>affiliated</code> / <code>research_paper</code> | [D/U] work-level OpenAI affiliation；当前无 local PDF+text；不推断产品采用或模型 lineage。 Topics: Distributed systems and fault tolerance, Advanced Data Storage Technologies, Parallel Computing and Optimization Techniques. |
| 2025-04-22 <code>doi-45e876e7409d</code> | Inferring interaction potentials from stochastic particle trajectories<br><https://doi.org/10.1103/physrevresearch.7.023075> | <code>affiliated</code> / <code>research_paper</code> | [D/C] work-level OpenAI affiliation；local PDF+text；不推断产品采用或模型 lineage。 Topics: Material Dynamics and Properties, Electrostatics and Colloid Interactions, Protein Structure and Dynamics. |
| 2025-04-22 <code>doi-4adaa858ff6e</code> | Evaluating Robustness of LLMs on Crisis-Related Microblogs across Events, Information Types, and Linguistic Features<br><https://doi.org/10.1145/3696410.3714511> | <code>affiliated</code> / <code>research_paper</code> | [D/C] work-level OpenAI affiliation；local PDF+text；不推断产品采用或模型 lineage。 Topics: safety_alignment, evaluation_benchmarks, interpretability. |
| 2025-03-07 <code>doi-52ae17f9c5db</code> | Recycling Scraps: Improving Private Learning by Leveraging Checkpoints<br><https://doi.org/10.56553/popets-2025-0079> | <code>affiliated</code> / <code>research_paper</code> | [D/C] work-level OpenAI affiliation；local PDF+text；不推断产品采用或模型 lineage。 Topics: Open Education and E-Learning, Higher Education Learning Practices. |
| 2025-01-01 <code>doi-3079af4487b3</code> | Automating the Search for Artificial Life With Foundation Models<br><https://doi.org/10.1162/artl.a.8> | <code>affiliated</code> / <code>research_paper</code> | [D/C] work-level OpenAI affiliation；local PDF+text；不推断产品采用或模型 lineage。 Topics: Cellular Automata and Applications, Computability, Logic, AI Algorithms, Modular Robots and Swarm Intelligence. |
| 2024-11-11 <code>doi-5f9fe9804290</code> | The Human Factor in AI Red Teaming: Perspectives from Social and Collaborative Computing<br><https://doi.org/10.1145/3678884.3687147> | <code>affiliated</code> / <code>research_paper</code> | [D/C] work-level OpenAI affiliation；local PDF+text；不推断产品采用或模型 lineage。 Topics: safety_alignment, society_governance, Big Data and Business Intelligence. |
| 2024-10-16 <code>doi-7225c5eb6e76</code> | The PPOu Framework: A Structured Approach for Assessing the Likelihood of Malicious Use of Advanced AI Systems<br><https://doi.org/10.1609/aies.v7i1.31653> | <code>affiliated</code> / <code>research_paper</code> | [D/C] work-level OpenAI affiliation；local PDF+text；不推断产品采用或模型 lineage。 Topics: security, Terrorism, Counterterrorism, and Political Violence, Network Security and Intrusion Detection. |
| 2024-08-25 <code>doi-7519c5aaba75</code> | ACF-S: An 8-Terabit / Sec SuperNIC for High-Performance Data Movement in AI &amp; Accelerated Compute Networks<br><https://doi.org/10.1109/hcs61935.2024.10664686> | <code>affiliated</code> / <code>research_paper</code> | [D/U] work-level OpenAI affiliation；当前无 local PDF+text；不推断产品采用或模型 lineage。 Topics: data_training, efficiency_systems, Parallel Computing and Optimization Techniques. |
| 2024-07-18 <code>doi-13ec09c47ebb</code> | CryoDRGN-ET: deep reconstructing generative networks for visualizing dynamic biomolecules inside cells<br><https://doi.org/10.1038/s41592-024-02340-4> | <code>affiliated</code> / <code>research_paper</code> | [D/U] work-level OpenAI affiliation；当前无 local PDF+text；不推断产品采用或模型 lineage。 Topics: efficiency_systems, Cell Image Analysis Techniques, Advanced Electron Microscopy Techniques and Applications. |
| 2024-06-24 <code>doi-5e5dfa3f51aa</code> | Programming patchy particles for materials assembly design<br><https://doi.org/10.1073/pnas.2311891121> | <code>affiliated</code> / <code>research_paper</code> | [D/C] work-level OpenAI affiliation；local PDF+text；不推断产品采用或模型 lineage。 Topics: coding_software, Pickering emulsions and particle stabilization, Advanced Materials and Mechanics. |
| 2024-05-13 <code>doi-70951b4faef7</code> | Topological Embedding of Human Brain Networks with Applications to Dynamics of Temporal Lobe Epilepsy<br><https://doi.org/10.48550/arxiv.2405.07835> | <code>affiliated</code> / <code>research_paper</code> | [D/C] work-level OpenAI affiliation；local PDF+text；不推断产品采用或模型 lineage。 Topics: efficiency_systems, Functional Brain Connectivity Studies, Topological and Geometric Data Analysis. |
| 2024-04-02 <code>doi-d32bace80a3d</code> | AI is a viable alternative to high throughput screening: a 318-target study<br><https://doi.org/10.1038/s41598-024-54655-z> | <code>affiliated</code> / <code>research_paper</code> | [D/U] work-level OpenAI affiliation；当前无 local PDF+text；不推断产品采用或模型 lineage。 Topics: Computational Drug Discovery Methods, Machine Learning in Materials Science, Microbial Natural Products and Biosynthesis. |
| 2024-01-01 <code>doi-8af86d7b79a3</code> | Multi-Group Fairness Evaluation via Conditional Value-at-Risk Testing<br><https://doi.org/10.1109/jsait.2024.3397741> | <code>affiliated</code> / <code>research_paper</code> | [D/C] work-level OpenAI affiliation；local PDF+text；不推断产品采用或模型 lineage。 Topics: safety_alignment, evaluation_benchmarks, Adversarial Robustness in Machine Learning. |
| 2024-01-01 <code>doi-eaa43564dc05</code> | AI Regulations<br><https://doi.org/10.1007/978-3-031-54252-7_3> | <code>affiliated</code> / <code>research_paper</code> | [D/U] work-level OpenAI affiliation；当前无 local PDF+text；不推断产品采用或模型 lineage。 Topics: society_governance, Ethics and Social Impacts of AI, Impact of AI and Big Data on Business and Society. |
| 2024-01-01 <code>doi-a7f0d1da3dad</code> | AI Can Help Workers Gain New Technical Capabilities Without Gaining Knowledge<br><https://doi.org/10.2139/ssrn.4944588> | <code>affiliated</code> / <code>research_paper</code> | [D/U] work-level OpenAI affiliation；当前无 local PDF+text；不推断产品采用或模型 lineage。 Topics: Labor market dynamics and wage inequality, AI and HR Technologies. |
| 2023-12-20 <code>doi-9b98967a5e35</code> | Honeycomb: Ordered Key-Value Store Acceleration on an FPGA-Based SmartNIC<br><https://doi.org/10.1109/tc.2023.3345173> | <code>affiliated</code> / <code>research_paper</code> | [D/C] work-level OpenAI affiliation；local PDF+text；不推断产品采用或模型 lineage。 Topics: Distributed systems and fault tolerance, Parallel Computing and Optimization Techniques, Advanced Data Storage Technologies. |
| 2023-07-01 <code>doi-a5b2824c4799</code> | Enhancing the reliability and accuracy of AI-enabled diagnosis via complementarity-driven deferral to clinicians<br><https://doi.org/10.1038/s41591-023-02437-x> | <code>affiliated</code> / <code>research_paper</code> | [D/U] work-level OpenAI affiliation；当前无 local PDF+text；不推断产品采用或模型 lineage。 Topics: Artificial Intelligence in Healthcare and Education, AI in cancer detection, COVID-19 diagnosis using AI. |
| 2023-03-20 <code>doi-1bcbb2440b99</code> | DrGPUM: Guiding Memory Optimization for GPU-Accelerated Applications<br><https://doi.org/10.1145/3582016.3582044> | <code>affiliated</code> / <code>research_paper</code> | [D/C] work-level OpenAI affiliation；local PDF+text；不推断产品采用或模型 lineage。 Topics: coding_software, efficiency_systems, Parallel Computing and Optimization Techniques. |
| 2023-02-20 <code>doi-07bae65d348c</code> | A Qubit, a Coin, and an Advice String Walk Into a Relational Problem<br><https://doi.org/10.4230/lipics.itcs.2024.1> | <code>affiliated</code> / <code>research_paper</code> | [D/C] work-level OpenAI affiliation；local PDF+text；不推断产品采用或模型 lineage。 Topics: Quantum Computing Algorithms and Architecture, Complexity and Algorithms in Graphs, Computability, Logic, AI Algorithms. |
| 2023-01-01 <code>doi-6b32df8d1cdb</code> | Report of the 1st Workshop on Generative AI and Law<br><https://doi.org/10.2139/ssrn.4634513> | <code>affiliated</code> / <code>research_paper</code> | [D/C] work-level OpenAI affiliation；local PDF+text；不推断产品采用或模型 lineage。 Topics: Law, AI, and Intellectual Property, Ethics and Social Impacts of AI, Artificial Intelligence in Law. |
| 2023-01-01 <code>doi-96795eb88615</code> | End-to-End Differentiable Reactive Molecular Dynamics Simulations Using JAX<br><https://doi.org/10.1007/978-3-031-32041-5_11> | <code>affiliated</code> / <code>research_paper</code> | [D/U] work-level OpenAI affiliation；当前无 local PDF+text；不推断产品采用或模型 lineage。 Topics: Machine Learning in Materials Science, Fuel Cells and Related Materials, Computational Drug Discovery Methods. |
| 2023-01-01 <code>doi-3f54efa2d807</code> | Bit Complexity of Jordan Normal Form and Polynomial Spectral Factorization<br><https://doi.org/10.4230/lipics.itcs.2023.42> | <code>affiliated</code> / <code>research_paper</code> | [D/C] work-level OpenAI affiliation；local PDF+text；不推断产品采用或模型 lineage。 Topics: Matrix Theory and Algorithms, Polynomial and algebraic computation, Numerical Methods and Algorithms. |
| 2022-12-14 <code>doi-5181cdd92f3f</code> | Advancing ethics review practices in AI research<br><https://doi.org/10.1038/s42256-022-00585-2> | <code>affiliated</code> / <code>research_paper</code> | [D/C] work-level OpenAI affiliation；local PDF+text；不推断产品采用或模型 lineage。 Topics: Ethics in Clinical Research, Ethics and Social Impacts of AI, Psychology of Moral and Emotional Judgment. |
| 2022-11-14 <code>doi-267638d5ce31</code> | Enhancing the reliability and accuracy of AI-enabled diagnosis via complementarity-driven deferral to clinicians (CoDoC)<br><https://doi.org/10.21203/rs.3.rs-2231672/v1> | <code>affiliated</code> / <code>research_paper</code> | [D/C] work-level OpenAI affiliation；local PDF+text；不推断产品采用或模型 lineage。 Topics: Machine Learning in Healthcare, Artificial Intelligence in Healthcare and Education. |
| 2022-06-15 <code>doi-5c06485e9e08</code> | Dataset and Code for the Research Paper: "A Smile is All You Need: Predicting Limiting Activity Coefficients from SMILES with Natural Language Processing."<br><https://doi.org/10.5281/zenodo.8271713> | <code>affiliated</code> / <code>dataset</code> | [D/U] work-level OpenAI affiliation；当前无 local PDF+text；不推断产品采用或模型 lineage。 Topics: coding_software, data_training, efficiency_systems. |
| 2022-06-01 <code>doi-bacca5086396</code> | Robust fine-tuning of zero-shot models<br><https://doi.org/10.1109/cvpr52688.2022.00780> | <code>affiliated</code> / <code>research_paper</code> | [D/C] work-level OpenAI affiliation；local PDF+text；不推断产品采用或模型 lineage。 Topics: safety_alignment, data_training, Domain Adaptation and Few-Shot Learning. |
| 2022-01-01 <code>doi-f910374bf05c</code> | A smile is all you need: predicting limiting activity coefficients from SMILES with natural language processing<br><https://doi.org/10.1039/d2dd00058j> | <code>affiliated</code> / <code>research_paper</code> | [D/C] work-level OpenAI affiliation；local PDF+text；不推断产品采用或模型 lineage。 Topics: efficiency_systems, Process Optimization and Integration, Business Process Modeling and Analysis. |
| 2021-12-09 <code>doi-5fe1769759cd</code> | Filling gaps in trustworthy development of AI<br><https://doi.org/10.1126/science.abi7176> | <code>affiliated</code> / <code>research_paper</code> | [D/C] work-level OpenAI affiliation；local PDF+text；不推断产品采用或模型 lineage。 Topics: Ethics and Social Impacts of AI, Artificial Intelligence in Healthcare and Education, Explainable Artificial Intelligence (XAI). |
| 2021-12-01 <code>doi-82af25041c5d</code> | Deep double descent: where bigger models and more data hurt*<br><https://doi.org/10.1088/1742-5468/ac3a74> | <code>affiliated</code> / <code>research_paper</code> | [D/C] work-level OpenAI affiliation；local PDF+text；不推断产品采用或模型 lineage。 Topics: data_training, Adversarial Robustness in Machine Learning, Machine Learning and Algorithms. |
| 2021-10-01 <code>doi-8892e63a02e6</code> | Batch size-invariance for policy optimization<br><https://doi.org/10.48550/arxiv.2110.00641> | <code>affiliated</code> / <code>research_paper</code> | [D/C] work-level OpenAI affiliation；local PDF+text；不推断产品采用或模型 lineage。 Topics: reinforcement_learning, society_governance, Advanced Bandit Algorithms Research. |
| 2021-09-22 <code>doi-cfb93bf896c3</code> | Recursively Summarizing Books with Human Feedback<br><https://doi.org/10.48550/arxiv.2109.10862> | <code>affiliated</code> / <code>research_paper</code> | [D/C] work-level OpenAI affiliation；local PDF+text；不推断产品采用或模型 lineage。 Topics: Topic Modeling, Advanced Text Analysis Techniques, Natural Language Processing Techniques. |
| 2021-08-12 <code>doi-84af9c2dc863</code> | A Cryptographic Test of Quantumness and Certifiable Randomness from a Single Quantum Device<br><https://doi.org/10.1145/3441309> | <code>affiliated</code> / <code>research_paper</code> | [D/C] work-level OpenAI affiliation；local PDF+text；不推断产品采用或模型 lineage。 Topics: evaluation_benchmarks, Cryptography and Data Security, Quantum Computing Algorithms and Architecture. |
| 2021-06-29 <code>doi-a84ad82e8ede</code> | Using Tactile Sensing to Improve the Sample Efficiency and Performance of Deep Deterministic Policy Gradients for Simulated In-Hand Manipulation Tasks<br><https://doi.org/10.3389/frobt.2021.538773> | <code>affiliated</code> / <code>research_paper</code> | [D/C] work-level OpenAI affiliation；local PDF+text；不推断产品采用或模型 lineage。 Topics: reinforcement_learning, efficiency_systems, robotics_embodied. |
| 2021-04-08 <code>doi-eb4d1cd68db6</code> | Weight Banding<br><https://doi.org/10.23915/distill.00024.009> | <code>affiliated</code> / <code>research_paper</code> | [D/U] work-level OpenAI affiliation；当前无 local PDF+text；不推断产品采用或模型 lineage。 Topics: Diet and metabolism studies, Mobile Health and mHealth Applications, Nutritional Studies and Diet. |
| 2021-04-05 <code>doi-64dc5e6ed35d</code> | Branch Specialization<br><https://doi.org/10.23915/distill.00024.008> | <code>affiliated</code> / <code>research_paper</code> | [D/U] work-level OpenAI affiliation；当前无 local PDF+text；不推断产品采用或模型 lineage。 Topics: Visual perception and processing mechanisms, Remote Sensing and LiDAR Applications, Industrial Vision Systems and Defect Detection. |
| 2021-02-24 <code>doi-a3729e949b5b</code> | First return, then explore<br><https://doi.org/10.1038/s41586-020-03157-9> | <code>affiliated</code> / <code>research_paper</code> | [D/C] work-level OpenAI affiliation；local PDF+text；不推断产品采用或模型 lineage。 Topics: Reinforcement Learning in Robotics, Advanced Bandit Algorithms Research, Artificial Intelligence in Games. |
| 2021-02-04 <code>doi-c4d5fc0d0c12</code> | Visualizing Weights<br><https://doi.org/10.23915/distill.00024.007> | <code>affiliated</code> / <code>research_paper</code> | [D/U] work-level OpenAI affiliation；当前无 local PDF+text；不推断产品采用或模型 lineage。 Topics: Explainable Artificial Intelligence (XAI). |
| 2021-01-27 <code>doi-a522f79dda67</code> | High/Low frequency detectors<br><https://doi.org/10.23915/distill.00024.005> | <code>affiliated</code> / <code>research_paper</code> | [D/U] work-level OpenAI affiliation；当前无 local PDF+text；不推断产品采用或模型 lineage。 Topics: CCD and CMOS Imaging Sensors, Radiation Detection and Scintillator Technologies, Adversarial Robustness in Machine Learning. |
| 2021-01-11 <code>doi-431b08897417</code> | An evidence review of face masks against COVID-19<br><https://doi.org/10.1073/pnas.2014564118> | <code>affiliated</code> / <code>research_paper</code> | [D/C] work-level OpenAI affiliation；local PDF+text；不推断产品采用或模型 lineage。 Topics: Infection Control and Ventilation, COVID-19 and healthcare impacts, COVID-19 epidemiological studies. |
| 2021-01-01 <code>doi-bc72ffa3c3ea</code> | Legal Priorities Research: A Research Agenda<br><https://doi.org/10.2139/ssrn.3931256> | <code>affiliated</code> / <code>research_paper</code> | [D/U] work-level OpenAI affiliation；当前无 local PDF+text；不推断产品采用或模型 lineage。 Topics: society_governance, Ethics and Social Impacts of AI, Law, AI, and Intellectual Property. |
| 2021-01-01 <code>doi-512cfebb5ac6</code> | Antitrust-Compliant AI Industry Self-Regulation<br><https://doi.org/10.2139/ssrn.3933677> | <code>affiliated</code> / <code>research_paper</code> | [D/U] work-level OpenAI affiliation；当前无 local PDF+text；不推断产品采用或模型 lineage。 Topics: society_governance, Merger and Competition Analysis. |
| 2020-11-24 <code>doi-89e4936d5259</code> | Variation in reversal learning by three generalist mesocarnivores<br><https://doi.org/10.1007/s10071-020-01438-4> | <code>affiliated</code> / <code>research_paper</code> | [D/C] work-level OpenAI affiliation；local PDF+text；不推断产品采用或模型 lineage。 Topics: Wildlife Ecology and Conservation, Primate Behavior and Ecology, Bat Biology and Ecology Studies. |
| 2020-11-02 <code>doi-9f90e9ed60b8</code> | Face Masks Against COVID-19: An Evidence Review<br><https://doi.org/10.20944/preprints202004.0203.v4> | <code>affiliated</code> / <code>research_paper</code> | [D/U] work-level OpenAI affiliation；当前无 local PDF+text；不推断产品采用或模型 lineage。 Topics: Infection Control and Ventilation, COVID-19 and healthcare impacts, COVID-19 epidemiological studies. |
| 2020-09-15 <code>doi-351bfdadb4e4</code> | The Importance of Pessimism in Fixed-Dataset Policy Optimization<br><https://doi.org/10.48550/arxiv.2009.06799> | <code>affiliated</code> / <code>research_paper</code> | [D/C] work-level OpenAI affiliation；local PDF+text；不推断产品采用或模型 lineage。 Topics: reinforcement_learning, data_training, society_governance. |
| 2020-09-10 <code>doi-17545e6c66b1</code> | Measurement in AI Policy: Opportunities and Challenges<br><https://doi.org/10.48550/arxiv.2009.09071> | <code>affiliated</code> / <code>research_paper</code> | [D/C] work-level OpenAI affiliation；local PDF+text；不推断产品采用或模型 lineage。 Topics: reinforcement_learning, evaluation_benchmarks, society_governance. |
| 2020-09-09 <code>doi-3e4ea822ac9d</code> | Phasic Policy Gradient<br><https://doi.org/10.48550/arxiv.2009.04416> | <code>affiliated</code> / <code>research_paper</code> | [D/C] work-level OpenAI affiliation；local PDF+text；不推断产品采用或模型 lineage。 Topics: reinforcement_learning, society_governance, Reinforcement Learning in Robotics. |
| 2020-07-12 <code>openalex-w3035754002</code> | Learning General-Purpose Controllers via Locally Communicating Sensorimotor Modules<br><https://icml.cc/Conferences/2020/AcceptedPapersInitial#784> | <code>affiliated</code> / <code>research_paper</code> | [D/U] work-level OpenAI affiliation；当前无 local PDF+text；不推断产品采用或模型 lineage。 Topics: Robotic Mechanisms and Dynamics, Robot Manipulation and Learning, Neural Networks and Applications. |
| 2020-07-12 <code>openalex-w3034445277</code> | Generative Pretraining From Pixels<br><http://proceedings.mlr.press/v119/chen20s/chen20s.pdf> | <code>affiliated</code> / <code>research_paper</code> | [D/C] work-level OpenAI affiliation；local PDF+text；不推断产品采用或模型 lineage。 Topics: data_training, Generative Adversarial Networks and Image Synthesis, Advanced Image and Video Retrieval Techniques. |
| 2020-07-12 <code>openalex-w3034318616</code> | Estimating Q(s,s') with Deterministic Dynamics Gradients<br><https://openalex.org/W3034318616> | <code>affiliated</code> / <code>research_paper</code> | [D/U] work-level OpenAI affiliation；当前无 local PDF+text；不推断产品采用或模型 lineage。 Topics: Gas Dynamics and Kinetic Theory, Hydrocarbon exploration and reservoir analysis, Field-Flow Fractionation Techniques. |
| 2020-07-12 <code>openalex-w3035610106</code> | Conditional Augmentation for Generative Modeling<br><https://icml.cc/Conferences/2020/AcceptedPapersInitial#983> | <code>affiliated</code> / <code>research_paper</code> | [D/U] work-level OpenAI affiliation；当前无 local PDF+text；不推断产品采用或模型 lineage。 Topics: Model-Driven Software Engineering Techniques, Simulation Techniques and Applications. |
| 2020-07-01 <code>doi-0581babed13c</code> | Automatic Curriculum Learning For Deep RL: A Short Survey<br><https://doi.org/10.24963/ijcai.2020/671> | <code>affiliated</code> / <code>research_paper</code> | [D/C] work-level OpenAI affiliation；local PDF+text；不推断产品采用或模型 lineage。 Topics: reinforcement_learning, data_training, Hermeneutics and Narrative Identity. |
| 2020-06-25 <code>doi-6d4ec745d574</code> | Scaling MAP-Elites to deep neuroevolution<br><https://doi.org/10.1145/3377930.3390217> | <code>affiliated</code> / <code>research_paper</code> | [D/C] work-level OpenAI affiliation；local PDF+text；不推断产品采用或模型 lineage。 Topics: efficiency_systems, Reinforcement Learning in Robotics, Adversarial Robustness in Machine Learning. |
| 2020-06-17 <code>doi-98af67474d6d</code> | Curve Detectors<br><https://doi.org/10.23915/distill.00024.003> | <code>affiliated</code> / <code>research_paper</code> | [D/U] work-level OpenAI affiliation；当前无 local PDF+text；不推断产品采用或模型 lineage。 Topics: Anomaly Detection Techniques and Applications, Adversarial Robustness in Machine Learning, Cell Image Analysis Techniques. |
| 2020-05-01 <code>doi-7ff82a17afcb</code> | Dynamic interventions to control COVID-19 pandemic: a multivariate prediction modelling study comparing 16 worldwide countries<br><https://doi.org/10.1007/s10654-020-00649-w> | <code>affiliated</code> / <code>research_paper</code> | [D/C] work-level OpenAI affiliation；local PDF+text；不推断产品采用或模型 lineage。 Topics: COVID-19 epidemiological studies, COVID-19 Pandemic Impacts, Vaccine Coverage and Hesitancy. |
| 2020-04-30 <code>doi-cfe17a4d8015</code> | Jukebox: A Generative Model for Music<br><https://doi.org/10.48550/arxiv.2005.00341> | <code>affiliated</code> / <code>research_paper</code> | [D/C] work-level OpenAI affiliation；local PDF+text；不推断产品采用或模型 lineage。 Topics: Music and Audio Processing, Music Technology and Sound Studies, Speech and Audio Processing. |
| 2020-04-15 <code>doi-2d90df96160d</code> | Toward Trustworthy AI Development: Mechanisms for Supporting Verifiable Claims<br><https://doi.org/10.48550/arxiv.2004.07213> | <code>affiliated</code> / <code>research_paper</code> | [D/C] work-level OpenAI affiliation；local PDF+text；不推断产品采用或模型 lineage。 Topics: Adversarial Robustness in Machine Learning, Ethics and Social Impacts of AI, Law, AI, and Intellectual Property. |
| 2020-04-09 <code>doi-ee08fb52280d</code> | The Surprising Creativity of Digital Evolution: A Collection of Anecdotes from the Evolutionary Computation and Artificial Life Research Communities<br><https://doi.org/10.1162/artl_a_00319> | <code>affiliated</code> / <code>research_paper</code> | [D/C] work-level OpenAI affiliation；local PDF+text；不推断产品采用或模型 lineage。 Topics: Evolutionary Game Theory and Cooperation, Evolution and Genetic Dynamics, Evolutionary Algorithms and Applications. |
| 2020-03-10 <code>doi-da8a9f9b3311</code> | Zoom In: An Introduction to Circuits<br><https://doi.org/10.23915/distill.00024.001> | <code>affiliated</code> / <code>research_paper</code> | [D/U] work-level OpenAI affiliation；当前无 local PDF+text；不推断产品采用或模型 lineage。 Topics: interpretability, Low-power high-performance VLSI design, Advancements in Semiconductor Devices and Circuit Design. |
| 2020-03-10 <code>doi-3f4601bcac3a</code> | Thread: Circuits<br><https://doi.org/10.23915/distill.00024> | <code>affiliated</code> / <code>research_paper</code> | [D/U] work-level OpenAI affiliation；当前无 local PDF+text；不推断产品采用或模型 lineage。 Topics: interpretability, Evolutionary Algorithms and Applications, Industrial Automation and Control Systems. |
| 2020-01-01 <code>doi-b6cc6561afbe</code> | Novel Dense Subgraph Discovery Primitives: Risk Aversion and Exclusion Queries<br><https://doi.org/10.1007/978-3-030-46150-8_23> | <code>affiliated</code> / <code>research_paper</code> | [D/C] work-level OpenAI affiliation；local PDF+text；不推断产品采用或模型 lineage。 Topics: safety_alignment, Data Management and Algorithms, Complexity and Algorithms in Graphs. |
| 2020-01-01 <code>doi-2f55f41b6f9d</code> | Making Gender Visible in Digital ICTs and International Security<br><https://doi.org/10.2139/ssrn.4170993> | <code>affiliated</code> / <code>research_paper</code> | [D/U] work-level OpenAI affiliation；当前无 local PDF+text；不推断产品采用或模型 lineage。 Topics: security, Gender, Security, and Conflict, Global Security and Public Health. |
| 2019-12-13 <code>doi-b4b263a4548f</code> | Long-Term Planning and Situational Awareness in OpenAI Five<br><https://doi.org/10.48550/arxiv.1912.06721> | <code>affiliated</code> / <code>research_paper</code> | [D/C] work-level OpenAI affiliation；local PDF+text；不推断产品采用或模型 lineage。 Topics: Reinforcement Learning in Robotics, Explainable Artificial Intelligence (XAI), Generative Adversarial Networks and Image Synthesis. |
| 2019-11-18 <code>doi-6f84d440fb92</code> | Learning dexterous in-hand manipulation<br><https://doi.org/10.1177/0278364919887447> | <code>affiliated</code> / <code>research_paper</code> | [D/C] work-level OpenAI affiliation；local PDF+text；不推断产品采用或模型 lineage。 Topics: robotics_embodied, Robot Manipulation and Learning, Reinforcement Learning in Robotics. |
| 2019-10-01 <code>doi-601f4db067cf</code> | Bayesian Relational Memory for Semantic Visual Navigation<br><https://doi.org/10.1109/iccv.2019.00286> | <code>affiliated</code> / <code>research_paper</code> | [D/C] work-level OpenAI affiliation；local PDF+text；不推断产品采用或模型 lineage。 Topics: efficiency_systems, Multimodal Machine Learning Applications, Human Pose and Action Recognition. |
| 2019-09-30 <code>doi-edf4d8c26173</code> | The Paths Perspective on Value Learning<br><https://doi.org/10.23915/distill.00020> | <code>affiliated</code> / <code>research_paper</code> | [D/U] work-level OpenAI affiliation；当前无 local PDF+text；不推断产品采用或模型 lineage。 Topics: Education and Critical Thinking Development, Ethics in Business and Education. |
| 2019-08-06 <code>doi-01086665ccce</code> | A Discussion of 'Adversarial Examples Are Not Bugs, They Are Features': Two Examples of Useful, Non-Robust Features<br><https://doi.org/10.23915/distill.00019.3> | <code>affiliated</code> / <code>research_paper</code> | [D/U] work-level OpenAI affiliation；当前无 local PDF+text；不推断产品采用或模型 lineage。 Topics: safety_alignment, interpretability, Adversarial Robustness in Machine Learning. |
| 2019-08-06 <code>doi-738a995f35a3</code> | A Discussion of 'Adversarial Examples Are Not Bugs, They Are Features': Robust Feature Leakage<br><https://doi.org/10.23915/distill.00019.2> | <code>affiliated</code> / <code>research_paper</code> | [D/U] work-level OpenAI affiliation；当前无 local PDF+text；不推断产品采用或模型 lineage。 Topics: safety_alignment, interpretability, Adversarial Robustness in Machine Learning. |
| 2019-08-06 <code>doi-cf36b5e1d868</code> | A Discussion of 'Adversarial Examples Are Not Bugs, They Are Features'<br><https://doi.org/10.23915/distill.00019> | <code>affiliated</code> / <code>research_paper</code> | [D/U] work-level OpenAI affiliation；当前无 local PDF+text；不推断产品采用或模型 lineage。 Topics: interpretability, Adversarial Robustness in Machine Learning, Advanced Malware Detection Techniques. |
| 2019-07-10 <code>doi-1fd46605a986</code> | The Role of Cooperation in Responsible AI Development<br><https://doi.org/10.48550/arxiv.1907.04534> | <code>affiliated</code> / <code>research_paper</code> | [D/C] work-level OpenAI affiliation；local PDF+text；不推断产品采用或模型 lineage。 Topics: Innovation, Sustainability, Human-Machine Systems, Open Source Software Innovations, Experimental Behavioral Economics Studies. |
| 2019-06-01 <code>doi-d927959b046e</code> | Transfer Learning via Unsupervised Task Discovery for Visual Question Answering<br><https://doi.org/10.1109/cvpr.2019.00858> | <code>affiliated</code> / <code>research_paper</code> | [D/C] work-level OpenAI affiliation；local PDF+text；不推断产品采用或模型 lineage。 Topics: Multimodal Machine Learning Applications, Domain Adaptation and Few-Shot Learning, Advanced Image and Video Retrieval Techniques. |
| 2019-04-18 <code>openalex-w2938251438</code> | From GAN to WGAN<br><https://arxiv.org/pdf/1904.08994.pdf> | <code>affiliated</code> / <code>research_paper</code> | [D/C] work-level OpenAI affiliation；local PDF+text；不推断产品采用或模型 lineage。 Topics: Generative Adversarial Networks and Image Synthesis, Model Reduction and Neural Networks, Digital Media Forensic Detection. |
| 2019-03-06 <code>doi-8afa2625ac93</code> | Activation Atlas<br><https://doi.org/10.23915/distill.00015> | <code>affiliated</code> / <code>research_paper</code> | [D/U] work-level OpenAI affiliation；当前无 local PDF+text；不推断产品采用或模型 lineage。 Topics: interpretability, Adversarial Robustness in Machine Learning, Domain Adaptation and Few-Shot Learning. |
| 2018-12-04 <code>doi-ea0f2b1a9835</code> | A Spectral Regularizer for Unsupervised Disentanglement<br><https://doi.org/10.48550/arxiv.1812.01161> | <code>affiliated</code> / <code>research_paper</code> | [D/C] work-level OpenAI affiliation；local PDF+text；不推断产品采用或模型 lineage。 Topics: Model Reduction and Neural Networks, Generative Adversarial Networks and Image Synthesis, Computational Physics and Python Applications. |
| 2018-10-30 <code>doi-ecf3d457cf00</code> | Exploration by Random Network Distillation<br><https://doi.org/10.48550/arxiv.1810.12894> | <code>affiliated</code> / <code>research_paper</code> | [D/C] work-level OpenAI affiliation；local PDF+text；不推断产品采用或模型 lineage。 Topics: data_training, efficiency_systems, Reinforcement Learning in Robotics. |
| 2018-07-25 <code>doi-3ab0e0e7d84f</code> | 3D Sketching using Multi-View Deep Volumetric Prediction<br><https://doi.org/10.1145/3203197> | <code>affiliated</code> / <code>research_paper</code> | [D/C] work-level OpenAI affiliation；local PDF+text；不推断产品采用或模型 lineage。 Topics: Computer Graphics and Visualization Techniques, 3D Shape Modeling and Analysis, Advanced Vision and Imaging. |
| 2018-01-01 <code>doi-ae057a0963da</code> | Liberal Radicalism: Formal Rules for a Society Neutral Among Communities<br><https://doi.org/10.2139/ssrn.3243656> | <code>affiliated</code> / <code>research_paper</code> | [D/U] work-level OpenAI affiliation；当前无 local PDF+text；不推断产品采用或模型 lineage。 Topics: society_governance, Auction Theory and Applications, Game Theory and Applications. |
| 2018-01-01 <code>doi-bbe78f5cebbe</code> | Approximating Cycles in Directed Graphs: Fast Algorithms for Girth and Roundtrip Spanners<br><https://doi.org/10.1137/1.9781611975031.91> | <code>affiliated</code> / <code>research_paper</code> | [D/C] work-level OpenAI affiliation；local PDF+text；不推断产品采用或模型 lineage。 Topics: Complexity and Algorithms in Graphs, Interconnection Networks and Systems, Advanced Graph Theory Research. |
| 2018-01-01 <code>doi-0a432c033cf1</code> | A Crossroads, not an Island: A Response to Hanoch Dagan<br><https://doi.org/10.2139/ssrn.3306738> | <code>affiliated</code> / <code>research_paper</code> | [D/U] work-level OpenAI affiliation；当前无 local PDF+text；不推断产品采用或模型 lineage。 Topics: Political Economy and Marxism. |
| 2017-10-01 <code>doi-2539332d3cf2</code> | Optimal Lower Bounds for Universal Relation, and for Samplers and Finding Duplicates in Streams<br><https://doi.org/10.1109/focs.2017.50> | <code>affiliated</code> / <code>research_paper</code> | [D/C] work-level OpenAI affiliation；local PDF+text；不推断产品采用或模型 lineage。 Topics: Complexity and Algorithms in Graphs, Cryptography and Data Security, Privacy-Preserving Technologies in Data. |
| 2017-09-01 <code>doi-c3b979fb3ee9</code> | Policy transfer via modularity and reward guiding<br><https://doi.org/10.1109/iros.2017.8205959> | <code>affiliated</code> / <code>research_paper</code> | [D/U] work-level OpenAI affiliation；当前无 local PDF+text；不推断产品采用或模型 lineage。 Topics: reinforcement_learning, society_governance, Robot Manipulation and Learning. |
| 2017-07-28 <code>doi-38b4746c1c64</code> | Value Iteration Networks<br><https://doi.org/10.24963/ijcai.2017/700> | <code>affiliated</code> / <code>research_paper</code> | [D/C] work-level OpenAI affiliation；local PDF+text；不推断产品采用或模型 lineage。 Topics: efficiency_systems, Reinforcement Learning in Robotics, AI-based Problem Solving and Planning. |
| 2017-07-28 <code>doi-4c21927906c9</code> | The Off-Switch Game<br><https://doi.org/10.24963/ijcai.2017/32> | <code>affiliated</code> / <code>research_paper</code> | [D/C] work-level OpenAI affiliation；local PDF+text；不推断产品采用或模型 lineage。 Topics: Computability, Logic, AI Algorithms, Complex Systems and Decision Making, Game Theory and Applications. |
| 2017-05-30 <code>doi-0d4d4a16d1a2</code> | Constrained Policy Optimization<br><https://doi.org/10.48550/arxiv.1705.10528> | <code>affiliated</code> / <code>research_paper</code> | [D/C] work-level OpenAI affiliation；local PDF+text；不推断产品采用或模型 lineage。 Topics: reinforcement_learning, society_governance, Reinforcement Learning in Robotics. |
| 2017-05-01 <code>doi-de7104b25a2f</code> | PLATO: Policy learning using adaptive trajectory optimization<br><https://doi.org/10.1109/icra.2017.7989379> | <code>affiliated</code> / <code>research_paper</code> | [D/C] work-level OpenAI affiliation；local PDF+text；不推断产品采用或模型 lineage。 Topics: reinforcement_learning, society_governance, Reinforcement Learning in Robotics. |
| 2017-05-01 <code>doi-90dd453a55cf</code> | Learning from the hindsight plan — Episodic MPC improvement<br><https://doi.org/10.1109/icra.2017.7989043> | <code>affiliated</code> / <code>research_paper</code> | [D/C] work-level OpenAI affiliation；local PDF+text；不推断产品采用或模型 lineage。 Topics: Advanced Control Systems Optimization, Iterative Learning Control Systems, Cardiac Valve Diseases and Treatments. |
| 2017-03-24 <code>doi-41db18164f17</code> | On the string consensus problem and the Manhattan sequence consensus problem<br><https://doi.org/10.1016/j.tcs.2017.03.022> | <code>affiliated</code> / <code>research_paper</code> | [D/C] work-level OpenAI affiliation；local PDF+text；不推断产品采用或模型 lineage。 Topics: Algorithms and Data Compression, semigroups and automata theory, DNA and Biological Computing. |
| 2017-03-09 <code>doi-d324b2484354</code> | Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks<br><https://doi.org/10.48550/arxiv.1703.03400> | <code>affiliated</code> / <code>research_paper</code> | [D/C] work-level OpenAI affiliation；local PDF+text；不推断产品采用或模型 lineage。 Topics: efficiency_systems, Domain Adaptation and Few-Shot Learning, Advanced Neural Network Applications. |
| 2017-01-01 <code>doi-0310513d8c94</code> | Privatizing Law: Is Rule of Law an Equilibrium without Private Ordering?<br><https://doi.org/10.2139/ssrn.3057093> | <code>affiliated</code> / <code>research_paper</code> | [D/U] work-level OpenAI affiliation；当前无 local PDF+text；不推断产品采用或模型 lineage。 Topics: Law, Economics, and Judicial Systems, Legal and Constitutional Studies, Corruption and Economic Development. |
| 2016-06-10 <code>doi-b6016bd48177</code> | Generative Adversarial Imitation Learning<br><https://doi.org/10.48550/arxiv.1606.03476> | <code>affiliated</code> / <code>research_paper</code> | [D/C] work-level OpenAI affiliation；local PDF+text；不推断产品采用或模型 lineage。 Topics: Reinforcement Learning in Robotics, Model Reduction and Neural Networks, Generative Adversarial Networks and Image Synthesis. |
| 2016-01-01 <code>doi-c8b5aaf10a15</code> | Is Rule of Law an Equilibrium Without Private Ordering?<br><https://doi.org/10.2139/ssrn.2785017> | <code>affiliated</code> / <code>research_paper</code> | [D/U] work-level OpenAI affiliation；当前无 local PDF+text；不推断产品采用或模型 lineage。 Topics: Judicial and Constitutional Studies, Legal and Constitutional Studies, Law, Economics, and Judicial Systems. |
| 2015-01-01 <code>doi-627f8e36017c</code> | Life in the Law-Thick World: The Legal Resource Landscape for Ordinary Americans<br><https://doi.org/10.2139/ssrn.2547664> | <code>affiliated</code> / <code>research_paper</code> | [D/U] work-level OpenAI affiliation；当前无 local PDF+text；不推断产品采用或模型 lineage。 Topics: society_governance, Legal Education and Practice Innovations, Law, Economics, and Judicial Systems. |
| 2015-01-01 <code>doi-7e8330cddfb6</code> | Building Legal Order in Ancient Athens<br><https://doi.org/10.2139/ssrn.2547107> | <code>affiliated</code> / <code>research_paper</code> | [D/U] work-level OpenAI affiliation；当前无 local PDF+text；不推断产品采用或模型 lineage。 Topics: society_governance, Classical Antiquity Studies, Historical Economic and Legal Thought. |

## 17. 研究议程：从现有证据还能严格提出什么问题

以下问题来自全谱系综合，属于 **[I]**；当前公开来源中的答案仍为 **[U]**：

1. **pretraining-post-training 联合 scaling：** 当预训练、SFT、preference/process data、RL
   rollouts 与 inference compute 同时可增加时，固定总成本下的最优分配是什么？现有 scaling
   laws、o1 curves 与产品价格不能拼成统一定律。
2. **reasoning trace 的双重用途：** 如何在不直接优化“看起来诚实”的 trace、从而诱发 gaming
   的情况下，保留 CoT 对训练 credit assignment 与 deployment monitoring 的价值？
3. **verifier correlation：** policy、reward model、judge、monitor、classifier 若共享 backbone/
   data/spec，如何估计 correlated blind spots？需要跨家族、human/executable、adversarially
   generated judges 的 factorial design。
4. **agent horizon 与 state：** repository、browser、OS、science/lab 任务中，memory/compaction、
   retry、tool failure、human takeover 如何改变长期 credit assignment 与 apparent success？
5. **环境与权限的因果贡献：** 在 frozen weights 下，对 tool schema、sandbox、network、write
   permission、confirmation、rollback 做消融，能否解释不同产品 card 的安全差异？
6. **多模态共同表示是否真实共享：** image/audio/video/text 在 GPT-4o、Images、Sora、Live 等
   系列中共享哪些 encoder/tokenizer/latent/optimizer？需要 architecture 或 controlled transfer
   evidence，而不是 UI/品牌推断。
7. **science agent 的证据闭环：** literature hypothesis、code analysis、simulation、wet-lab action、
   measurement、replication 各自的错误如何传播？实验室成功不能只报最佳一次 run。
8. **evaluation lifecycle：** benchmark 的污染、饱和、依赖腐化、judge 漂移达到何种阈值时应
   version、重构或退役？SWE-bench Verified 的生命周期应转成通用治理协议。
9. **safe-completion 的长期均衡：** 输出级 reward 在 adaptive users、multiturn/tool use、不同
   languages/domains 中是否保持 helpfulness-safety frontier，还是把危害移到更长轨迹？
10. **Preparedness/FGF 的可审计性：** 外部研究者最少需要哪些 anonymized eval、threat-model、
    safeguard efficacy、exception 与 incident artifacts 才能独立检验“sufficiently minimized”？
11. **开放权重的净风险：** capability uplift、finetuning accessibility、safeguard removal、defensive
    research 和 ecosystem substitution 如何进入同一 counterfactual，而不是只看单一 jailbreak？
12. **总成本和社会影响 denominator：** training/serving/monitoring 人力与能源、组织 adoption、
    task displacement/complementarity、distributional effects 如何连接，而不把 benchmark 得分
    直接换算成 GDP 或就业？

## 18. 使用本章时应避免的具体误述

- “657 条都是 OpenAI 的模型技术报告。”——79 core、456 direct、122 affiliated；其中 366 条是
  HTML/官方页面型记录，只有 9 条被 inventory 分类为 technical report。
- “220 份本地 PDF 说明其余 437 条不存在正文。”——大量研究是 HTML-native；另有 2 个
  publisher PDF route 因访问控制未归档，435 条本来就没有公开 PDF route。
- “所有 affiliated 论文都用于训练 GPT。”——它只证明 work-level 机构关系；模型采用 [U]。
- “GPT-4 有 1.76T 参数/某个精确 MoE 配置。”——technical report 明确不披露 architecture、size、
  hardware、training compute、dataset construction 与完整 method。
- “GPT-3 训练集就是 300B unique tokens。”——约 300B 是 weighted sampling exposure。
- “Scaling Laws 的指数适用于所有后续模型。”——它们是特定模型族、数据、tokenizer 与区间的
  empirical fit，不是自然常数。
- “o1 在 AIME 得 93% pass@1。”——93% 来自 1,000 samples 的 learned reranker；pass@1 是 74.4%。
- “WebGPT 的主结论证明 PPO 一定优于 selection。”——其关键 175B 结果包含 BC+RM best-of-64，
  训练/选择机制必须分开。
- “PRM800K 表明 solver 用 policy-gradient RL 训练。”——公开论文重点是 process supervision/
  verifier 与 candidate selection；不能补写未报告的 solver RL。
- “GPT-4o、DALL·E、Sora 和 GPT-Live 共享同一生成架构。”——品牌/产品接口不是 weight lineage。
- “context window 大就等于 agent 能稳定完成同长度任务。”——retrieval、planning、state update、
  permission 和 tool failure 是独立维度。
- “Operator 的 OSWorld 38.1 可直接与任意 GUI agent 横排。”——checkpoint、harness、site snapshot、
  action schema、prompt、budget 与 judge 必须一致。
- “system card 是独立安全认证。”——它是第一方 release evidence；独立有效性仍需外部可重建审计。
- “Preparedness High 等于危险概率很高/很低。”——High 是框架内 capability threshold，不是概率。
- “700,000 A100e GPU-hours 是 GPT-5.6 训练成本。”——这是 universal-jailbreak 自动搜索投入。
- “威胁报告没看到 breakout，所以 AI 不会放大影响行动。”——已发现/已调查样本有选择偏差，未来
  capability、distribution 与未发现行动仍为 [U]。
- “CoT 可监控说明 CoT 完全忠实。”——monitorability 只要求 monitor 能提取目标 property；
  faithfulness 是更强、不同的主张。
- “公开 gpt-oss weights 揭示闭源 GPT-5.x 配方。”——没有明确共享 data/optimizer/reward/runtime
  evidence 时不能作该推断。
- “官方停止用某 benchmark 就证明原因一定是污染。”——若正文未归档，标题只确认 lifecycle
  action；原因需回到一手报告。
- “这 657 条证明互联网再无遗漏。”——它只逐条覆盖截止日、按公开发现规则冻结的 universe。

## 19. 完整性声明、专项审计边界与更新规则

**当前可证明的完整性：** 本章对冻结 `openai.jsonl` 的 **657/657** 条记录都给出 inventory ID、
primary URL、tier/type 与证据边界；计数为 core 79、direct 456、affiliated 122，type 为
blog/report page 366、research paper 236、system card 34、technical report 9、benchmark 9、
dataset 2、model card 1。正文覆盖基础模型/data/scaling、post-training/reasoning、multimodal、
code/math/science、agents/tools、systems、evals、interpretability、safety/governance/threats/incidents
和广义机构研究。

**64 卡专项审计的边界：** `openai-cards-audit.md` 是对 official cards/deployment-safety/
governance/threat/incident 候选的 focused reconciliation，不是“OpenAI 一共只有 64 份安全材料”。
它修复 27 个缺 PDF route、2 个错误 route，并补入 6 个缺失记录；冻结 inventory 随后包含 34 个
`system_card` type 和更多 technical/HTML safety records。64 是审计候选数，34 是 schema type
计数，657 是本章总 universe，三者不可混用。

**本章不声称的完整性：** 私有/撤下/robots-blocked 页面、未索引论文、未结构化 affiliation、
未来发布与内部资料仍可能遗漏；RSS/sitemap 只证明元数据的记录没有被升级成全文 claim；
2 个 publisher PDF route 尚未形成本地副本；最新闭源旗舰无法端到端复现。故“all”严格解释为
**截止 2026-07-19、按已公开发现与归属规则冻结并可逐条审计的 public source universe**。

后续维护必须：

1. 保留 verified-through date，先更新 discovery snapshot、page evidence 和 inventory，再更新正文；
2. 同一 work 的 arXiv/conference/journal 按内容去重，实质不同 report/card/addendum 保留版本；
3. official page、deployment-safety context、CDN PDF 与 DOI/arXiv 分别保存 provenance，不用一个
   landing page 冒充全文；
4. 每次下载重新验证 HTTP/content type、PDF magic、`pdfinfo`、SHA-256，并保留失败记录；
5. 每个重大数字保留 checkpoint、prompt、sampling、tools、context、budget、judge、date、
   section/page/table/figure 与 denominator；
6. affiliated 新增项必须有原始 author affiliation/权威元数据，不因“使用 GPT”或把模型列为
   coauthor 而纳入；
7. HTML/RSS 证据状态变化时，只升级被真正归档/读取的 claim；access challenge 不能由标题脑补；
8. 只有保存 code/model revision、environment、data hash、命令、seed、raw logs 和 artifacts 的
   实测才标 [R]；
9. card/framework 更新不得静默覆盖旧版，要保留 evaluation object、threshold 与 safeguard 的
   时间差；
10. 每次发布前重跑 657 ID/URL coverage、tier/type/count、PDF/text/hash、数学分隔符、内部链接
    与 MkDocs strict build。

本章追求的“全面”不是链接堆积，而是让任何读者都能逐层回答：**哪一条来源披露了什么 [D]，
哪一个制品真的在本地可检查 [C]，哪些结果被独立复现 [R]，综合判断依赖什么 [I]，以及证据
要求我们在哪些地方停在未知 [U]。**
