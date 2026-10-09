# CASE-EVID-001｜病例证据检索算法研究 SOP v1.0
生效：2026-10-09｜适用：所有 E083 及之后的算法实验和复现实验。变更只能通过新版本与 changelog，不允许覆盖既有冻结果。

## 0. 不可变基线与研究问题
研究目标为**诊断推理证据检索（reasoning-evidence retrieval）**，而非宣称诊断正确率提升。固定历史基线 N8-RRF-LEX-XMOD-STRONGMASK：词法病例检索 + query→历史病例的 masked reasoning 直接跨模态匹配，RRF c=60。其历史 E081 方法配置 SHA-256 为 `9f660f9b349a79f6a6c42c967fbaf4581930f1a4496345ced83d49dde1b8385c`。
MedCaseReasoning dataset revision `469a5365bc534b5b2b9cfbc52b2ef2a10f43a339`；train 13,092，val 500，test 897。既有 E081 MedR-Bench 840 来源不重叠病例外部分析已经看过，不得再次用作新模型选优。
**重要：**既有 N8 外部证据只支持该冻结版本，E083 后的新模型必须重新获得未触碰的来源不重叠外部队列或前瞻性专家研究后才能作确认性声明。897 T001 仅供历史冻结版本解释；禁止新算法用来选参、做消融或作为“首次独立验证”。

## 1. 数据生命周期
| 阶段 | 输入许可 | 输出与冻结点 | 禁止 |
|---|---|---|---|
| S0 假设注册 | 文献、历史已知的整体结果 | 一句话假设、固定基线与主终点 | 先看新评估结果再编写假设 |
| S1 原始数据核验 | 源数据和公开许可证 | revision、SHA-256、行数、字段、PMCID 冲突、近重复检查 | 更改原始镜像 |
| S2 划分与权限 | train；既有 val 标记 reused | PMCID-grouped deterministic split，seed，holdout SHA | 同一 PMCID 同时充当查询与证据库 |
| S3 代码/配置准备 | train-reference 子集与 train-meta 查询 prompt | 方法配置、预算、评价器版本、依赖锁、随机种子 | 用查询端诊断/推理作任何检索输入 |
| S4 开发训练 | train-reference 有标注，train-meta 在训练阶段用于监督但必须划分嵌套查询 | 模型与 feature audit；对照消融 | 读 test、E081 gold；把验证标签作为推理特征 |
| S5 盲排名 | 只读 query prompt + 历史库字段 | `rankings.jsonl` 与 `ranking_lock.json`（query + 配置 + SHA） | 排名前打开对应评估查询的 gold |
| S6 锁后评价 | 先验已锁排名，然后才能打开 held-out 目标推理 | 每例结果、paired 95% cluster bootstrap、95% CI、阴性结果 | 评估后原地修改排名并仍用同一 run_id |
| S7 开发决策 | train-internal 报告；reused val 只作探索辅助 | ACCEPT/REJECT/INCONCLUSIVE；多方法比较的限制 | 利用 reused val 或既有 MedR 外部评价自称确认性 |
| S8 新数据确认 | 预先未接触的新、来源不重叠病例 | 新 frozen 版本与一次性外部评估 | 探索性模型在外部集上反复试错 |
| S9 临床与发布 | 外部冻结方法、专家评分前的独立盲评包 | 医生效用、组间比较、数据许可/运行代码/方法卡 | 把人工证据偏好写成诊断准确率提升 |

## 2. 实验注册模板
每个实验先写 `configs/experiments/E###_v#.json`，包含 `experiment_id`, `hypothesis`, `code_revision`, `dataset_revision`, `query_split`, `query_fields`, `historical_index_fields`, `candidate_budget`, `primary_metric`, `comparator`, `n_queries`, `group_seed`, `hyperparameters`, `allowed_comparisons`, `analysis_criteria` 与 `status`。**不在 manifest 的运行不进入正式结果表**。新实验必须独立 ID，不覆盖旧版。测试集和已公开外部标签访问标志应为 false。
评估脚本必须分成 `rank`、`eval` 两阶段；第一阶段仅能产生盲排名和 SHA256 锁，第二阶段验证锁后再导入对应 gold，且排序绝不重算。方法若使用交叉编码器，也必须在同样的输入授权下运行。

## 3. 新算法评价固定口径
主终点：**diagnosis-masked symmetric reasoning-set Soft-F1@10**，观察 Top-1/3 与相对/绝对变化但不把它们与准确率混同。对照为已冻结 N8-RRF，并记录 TF-IDF 参照。
独立正交评价：char-wb ngrams 3–5，尽量同时使用非同词法尺度的医学 dense evaluator；必须披露检索器本身和主要 TF-IDF 评价器之间的同源词法偏好。
审计：exact diagnosis hit 作为次级现象；病例来源去重、诊断短语/缩写暴露、推理点数量/文本长度偏倚、预算公平性、train/meta PMCID 来源分离、group bootstrap。oracle 只能用于事后上限诊断，不能作部署评分。
关键开发门槛（方向性、不是统计确认）：train-internal 主终点增益为正，独立 char 评价增益也为正，无泄漏，候选预算预先相同；否则仅作为负结果归档。

## 4. 版本与目录
GitHub：`tools/` 源码，`configs/` 方法配置，`docs/` SOP 与研究日志，`tests/` 数据泄漏和接口测试；禁止删除失败运行的审计记录。
Google Drive：`00_Project/SOP`、`01_Data_Raw`（只读+SHA）、`03_Code`（发布快照）、`04_Experiments/E###`（每轮 manifest/lock/评分及日志）、`05_Results`（只收验证通过的主表）、`06_Manuscript`（claim→source 对照）。
文件命名 `{study}_{experiment}_{phase}_{YYYYMMDD}_{run_sha8}`。每个 bundle 同时保留代码提交、方法 YAML/JSON、Python 环境、失败日志、per-query、配对增益区间、权利许可和实验状态。Drive 当前上级 E080 目录：`https://drive.google.com/drive/folders/1nmV4OWp5SD-iVbQr6wgbRAxFdBUno4F8`。

## 5. 不允许的行为
- 任何 N8 冻结后输出、T001 test 或已看过的 MedR-Bench 数字作为 N10 参数选择依据。
- 查询时读取 `final_diagnosis`、`diagnostic_reasoning`、或用真实 `current_u`、oracle utility 排序。
- 训练和验证查询/库之间同 PMCID、同一病例近重复却不标注。
- 只报告性能提高的 seed/队列/指标，删掉阴性/失效模型。
- 合并置信区间不注明方法、复用多次验证却宣称一次性检验。
- 因分数提高而直接宣称临床诊断准确率提高。

## 6. Gate 与版本
`python tools/case_evid_policy_gate.py --policy configs/case_evid_policy_v1.json --experiment configs/experiments/E083_N10_v1.json --source tools/e083_n10_pointwise.py`
机器门禁只能发现部分违反契约行为，不能替代人工源码、数据同源性和指标定义审计。若策略改变必须产生 SOP-1.1 或更高版本，并列出受影响实验。现有 N8/E081 记录不因新 SOP 而追溯性重写。
