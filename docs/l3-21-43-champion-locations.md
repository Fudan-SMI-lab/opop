# L3:21 / L3:43 冠军候选的存放位置(2026-09-27 核验)

> 两个冠军的源码在**远端服务器与本地各有一份**,四处的关键文件 sha256 已逐位核对一致。
> box1 / box2 均为 autodl 无卡开机状态(关机 ≠ 释放,连续 15 天不开机才释放;
> 无卡模式下 `nvidia-smi` 报 Permission denied 是正常特征,不是故障)。

## L3:21 冠军 `cand-8a5dcef3`(3.4545 ms final re-eval,fp16)

| 位置 | 路径 | 内容 |
|---|---|---|
| **box1 原件** | ssh 别名 `autodl`(`connect.bjb3.seetacloud.com:49408`)<br>`/root/autodl-tmp/opop-workspace/opop-glm/runs-l3/run-l3-21-20260908-232211/` | 完整 run(46 MB):candidates 全量 / report / events / jobs / sandboxes |
| **本地镜像** | `external_files/box1-runs/runs-l3/run-l3-21-20260908-232211/` | 冠军候选目录(source.py + witness + trials)、`report/best_kernel.py`、`report/report.md`、`report/trials.csv`、`manifest.json`、`events.jsonl`、`state.json`(4.2 MB;未拉 jobs 原始输出与 sandboxes) |

- 冠军源码:`candidates/cand-8a5dcef3/source.py`(sha256 `800da774…b0e428adb`)
- 物化交付物:`report/best_kernel.py`(sha256 `e0644f80…c2b6f7b4`)——引用以此为准

## L3:43 冠军 `cand-173b12ae`(2.7628 ms final re-eval,bf16+fp32 acc)

| 位置 | 路径 | 内容 |
|---|---|---|
| **box2 原件** | ssh 别名 `autodl2`(`connect.bjb2.seetacloud.com:22010`,key `~/.ssh/autodl2_key`)<br>`/root/autodl-tmp/opop-workspace/opop-glm/runs-l3/run-l3-43-20260909-015247/` | 完整 run(35 MB);同目录另有四 run corpus(共 106 MB,含 3.01ms 那轮与 L3:21/L3:48 的 box2 run) |
| **本地镜像** | `external_files/box2-runs/runs-l3/run-l3-43-20260909-015247/` | **run 全量镜像**(88 MB):candidates 全部 20+ 候选 / report / events / jobs / manifest / sandboxes |

- 冠军源码:`candidates/cand-173b12ae/source.py`(sha256 `8ab5ac11…ead909d5`)
- 物化交付物:`report/best_kernel.py`(sha256 `a77876e1…9b79efcfe`)——引用以此为准

## 三条注意

1. **sha 一致性**:两个冠军的 source.py 与 best_kernel.py,远端 vs 本地 sha256 均逐位一致
   (2026-09-27 实测),引用任一处皆可;论文数字出处见
   `docs/l3-21-43-champion-sources-and-external-comparison.md`。
2. **source.py 的字节 sha ≠ events 里的注册 sha 是预期的**(candidates 下是参数化后版本,
   注册身份在 generator sandbox)——不要据此怀疑文件被改。
3. **释放风险与安全**:两台机器的 15 天倒计时以各自最后一次开机重置(box2 于 09-27 前后
   开机、box1 于 09-27 开机)。若长期不用,本地镜像即成唯一副本——两个 `external_files/box?-runs/`
   目录**不要清理**。远端 run 的 `sandboxes/` 内含 0644 明文 API key,**不要打包外发 runs 目录**,
   交付前轮换 key。
