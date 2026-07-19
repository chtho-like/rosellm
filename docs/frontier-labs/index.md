# 六家前沿实验室公开研究全景：入口、口径与复现路径

本专题把 **DeepSeek、Moonshot AI / Kimi、Zhipu AI / Z.ai（GLM）、OpenAI、
Anthropic、Google DeepMind / Gemini** 的模型技术报告、系统卡、模型卡、直接
相关研究和工作级机构署名论文放进同一套可审计语料库。范围不以 agentic RL
为边界；预训练、架构、优化、系统、推理、多模态、代码、数学、科学、机器人、
评测、安全、可解释性、治理与社会影响都在范围内。

> **快照日期：2026-07-19。** “全量”指[覆盖口径与方法](coverage.md)
> 下的**可审计公开来源并集**，不是对未公开、已删除、私有或没有写明机构归属
> 的工作的不可证伪声称。

下表逐项取自同日生成的 `research/literature/coverage.json`；`archive/` 是可继续下载、
修复和抽取的工作目录，只有重新生成 coverage 快照后，其中新增制品才进入这里的计数。

## 当前成果

| 机构 | 可审计记录 | 有公开 PDF / arXiv | 本地有效 PDF | 可检索全文 | 深度报告 |
|---|---:|---:|---:|---:|---|
| Anthropic | 244 | 126 | 125 | 125 | [专题](anthropic.md) |
| DeepSeek | 37 | 35 | 35 | 35 | [与 Kimi 联合专题](deepseek-kimi.md) |
| Google DeepMind / Gemini | 1,985 | 1,625 | 1,543 | 1,543 | [专题](gemini.md) |
| Moonshot AI / Kimi | 32 | 24 | 24 | 24 | [与 DeepSeek 联合专题](deepseek-kimi.md) |
| OpenAI | 657 | 222 | 220 | 220 | [专题](openai.md) |
| Zhipu AI / Z.ai | 95 | 58 | 57 | 57 | [GLM 专题](glm.md) |
| **总计** | **3,050** | **2,090** | **2,004** | **2,004** | [跨实验室综合](cross-lab.md) |

该 coverage 快照另有 960 条记录没有可确认的独立 PDF route；其公开形态包括 HTML、代码仓、
数据集页、DOI/出版社落地页。它们没有因为“下不到 PDF”而从历史中消失；
[覆盖率账本](coverage.md)逐条区分“网页原生”“页面已归档”“公开 PDF 路线失败”
和“本地全文已提取”。完整的 3,050 条紧凑书目记录在[总书目](bibliography.md)。
这里的 3,050 是 inventory / bibliographic record 数，不是去重后的独立作品数；
`research/literature/candidates/document-equivalences.jsonl` 另行跟踪同一作品的
预印本、正式发表版或官方介绍页等记录关系。

## 研究制品的五层结构

1. **权威目录。** `research/literature/inventory/*.jsonl` 一行一项，保存稳定 ID、
   作者、日期、类型、归属层级、DOI/arXiv、主页面、PDF 路径、来源页和归属证据。
2. **发现与排除账本。** `research/literature/inventory/*-discovery.md` 记录官方
   索引计数、分页边界、查询、去重、跨机构重合与尚未解决的入口。
3. **来源快照与候选证据。** `research/literature/candidates/` 保存官方 sitemap、
   RSS、页面证据、开放获取恢复尝试和官方报告版本关系；这些材料解释“为什么
   纳入”和“为什么没有 PDF”。
4. **可重建本地档案。** Git 忽略的 `research/literature/archive/` 保存通过
   PDF 魔数和 `pdfinfo` 验证的原件、SHA-256 manifest、全文文本与 HTML 快照。
5. **分析层。** 本目录的专题报告把事实、复现、推断和未知分开，并把每个目录
   ID 与至少一个来源 URL 重新连回叙述或可审计附录。

这个分层很重要：机器目录负责“不漏项和不混同”，长文负责“解释技术关系”，
原件与快照负责“让结论可以被复核”。

## 怎么读这些报告

报告统一使用[研究证据标准](../research-method.md)：

- **[D] disclosed**：一手材料明确披露；
- **[C] confirmed**：公开制品、代码、权重、数据或可观察接口可独立确认；
- **[R] reproduced**：仓库内或公开复现实验实际复现；
- **[I] inferred**：由多个已披露事实推得，但来源没有原句如此声称；
- **[U] unknown**：公开证据不足，必须停止推断。

推荐阅读顺序是：

1. 先读各专题的“证据边界”和时间轴，建立模型、方法与机构研究的不同层级；
2. 再读架构—数据—优化—后训练—系统—评测这条因果链；
3. 对照[跨实验室综合](cross-lab.md)，只比较协议可比的量，不横排营销数字；
4. 最后用各专题的逐条记录附录、[覆盖率账本](coverage.md)和[总书目](bibliography.md)回查
   单条记录。

## 三种归属不能混为一谈

- `core`：旗舰或专用模型技术报告、系统卡、模型卡、正式风险报告；
- `direct`：直接支撑或分析这些模型的架构、训练、推理、agent、评测、安全、
  可解释性与应用研究；
- `affiliated`：论文自身显示相关机构署名的更广义研究。

因此，“Google DeepMind 有 1,985 条”不等于“Gemini 有 1,985 篇技术报告”；
其中 1,943 条是严格保留来源边界的 affiliated 长尾。跨机构共同论文可以在两个
机构目录各出现一条书目记录，因为这里还要回答“这家机构参与了哪些研究”；
DOI、arXiv、规范化标题和 equivalence ledger 负责指出它们其实属于同一项作品，
但汇总数仍按 inventory record 计。

## 一键复现

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-research.txt

make research-validate
make research-download
make research-recover
make research-extract
python scripts/literature_corpus.py extract --repair-quality --workers 6
make research-report
```

正常下载失败后，恢复器只查询同一 OpenAlex/DOI 身份的替代开放获取位置；候选
必须再次通过 PDF 验证才会采用。抽取器优先使用 pypdf，只有文本出现编码替换或
主解析失败时才比较 Poppler `pdftotext`，且后备文本必须损坏更少并保留足够正文。

## 已知边界与更新规则

- OpenAI 站点对自动化正文读取有限制，所以官方 sitemap 与 RSS 快照承担不同
  证据角色；RSS 元数据不伪装成正文归档。
- 出版社 403、登录墙、失效 DOI 和没有独立 PDF 的网页报告会留在 gap ledger，
  不会被写成“已下载”。
- 少数 PDF 的嵌入字体让 pypdf 与 Poppler 在同一位置都输出 Unicode 替换字符；
  原 PDF、校验摘要与透明质量标记全部保留，分析时以原件为最终依据。
- 官方页面若暴露同一报告的多个字节不同版本，目录只保留一个规范记录，
  `candidates/document-versions.jsonl` 保存每个已验证版本的 URL、SHA-256 与页数。
- 新快照必须重新运行 schema、重复项、ID/URL 覆盖、下载、全文、数学与 MkDocs
  严格构建检查；不能只向长文追加一条链接。
