# v2 文档索引

本目录集中管理 v2 harness 的分析、调研与设计文档,解决文档散落问题。

## 目录结构

```
docs/
├── README.md              # 本索引
├── analysis/              # 实验结果分析、框架诊断、改进方向
│   └── framework-diagnosis-and-improvements.md
└── research/              # 外部项目/方法调研
    └── kernelfoundry-findings.md
```

## 文档清单

### analysis/ — 分析与诊断
- **[framework-diagnosis-and-improvements.md](analysis/framework-diagnosis-and-improvements.md)**
  — v2 效果分析(基于 events.jsonl 磁盘核实):正确性通过率低的根因、L3:21 调参无提升真因、
  novelty 不触发缺陷、五问答、改进路线(按收益排序)。**当前主分析文档,先读它。**
- **[improvement-implementation-plan.md](analysis/improvement-implementation-plan.md)**
  — 第一轮详细改进实施方案:7 项改进(A 双精度见证门 / B Triton 硬约束 prompt / C 静态门 /
  D 契约纠错+敏感性检查 / E novelty 名额 / F repair 预算 / G 参考库),每项含改动点(文件:行)、
  代码草图、风险、验收、实施顺序。已实施 A–F + H1/H2/H3(tensor-core 精度维度)。
- **[improvement-plan-round2.md](analysis/improvement-plan-round2.md)**
  — 第二轮改进实施方案(2026-09-04):针对第一轮落地后新暴露的三个问题——J(参考 train/eval
  运行模式探测并告知 agent,修 L3:21 MBConv 的 BatchNorm 语义缺口)/ K(boundary+空闲资源触发
  轻量参数空间扩展闭环)/ L(dtype 提为贯穿 cast+dot 的单一 knob,修 fp16 未被独立评测的 gap)。
  含每项的改动点、通用场景负面作用评估、风险、验收。**尚未实施,待批准。**

### research/ — 调研
- **[kernelfoundry-findings.md](research/kernelfoundry-findings.md)**
  — Intel ISL KernelFoundry(ICML 2026)源码调研:整体流程、prompt 策略、双精度见证正确性门、
  性能调优机制、v2 可借鉴的 9 个具体点(带文件证据)。

## 项目内其他关键文档(不在本目录,位置说明)

- `v2/README.md` — 项目入口 / 架构概览(保留在项目根,惯例)。
- `v2/HANDOFF.md` — 跨 session 交接文档,含里程碑状态、已修复 bug、坑与教训(保留在项目根)。
- `D:/ClaudeCode/data/plans/lively-cuddling-clarke.md` — 已批准的实施计划(架构/里程碑 M0–M5)。

## 前史研究文档(在工作路径根 `D:/Pyhon_projects/opop/`,设计参考)

- `gpu_kernel_repo_poc_research.md` — GPU kernel 仓库 PoC 前期调研。
- `gpu_kernel_structure_search_research_report.md` — 结构搜索方法调研报告。
- `v2-kernel-optimization-flow.md` — v2 数据对象与阶段语义设计。
- (根目录另有 14GB `.db` 文件与顶层 `HANDOFF.md`;后者是更早的根级交接,以 `v2/HANDOFF.md` 为准。)
