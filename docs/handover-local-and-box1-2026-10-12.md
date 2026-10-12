# 交接文档:本地结果位置 · box1 现状与整理 · box1 代码更新(2026-10-12)

> 本文所有状态都是 **2026-10-12 实测**,不是推断。实测范围:Windows 本地 `D:\Pyhon_projects\opop`
> 加 box1(无卡开机)。box4 的同类事实附在 §2.6。
> 读者:接手开发或整理 box1 的人或 agent。

---

## 1. 本地:结果在哪,项目是什么状态

### 1.1 项目根 `D:\Pyhon_projects\opop\`

| 路径 | 是什么 |
|---|---|
| `v5\` | **当前开发主树**(git worktree,分支 `v5`,HEAD `0e39140` = GitHub v5 最新)。**有大量未提交改动**,见 §1.3 |
| `v2\` `v3\` `v4\` | 历史分支的 worktree(各自 HEAD:b1d8e04 / 391b472 / db71d73),只读参考 |
| `external_files\` | 外部框架对比包、各服务器备份、冠军镜像(§1.2) |
| `KernelBench\` | 本地 KernelBench 副本 |
| 其余(`paper\`、`runs\`、`linux-server\` 等) | 早期工作目录,与当前开发无直接关系 |

GitHub:`https://github.com/Fudan-SMI-lab/opop`,远端分支 `main v1 v2 v3 v4 v5`。
**还没有 `v5-C3`**(之前计划的 C3 推送没有完成)。

### 1.2 实验结果与备份(`external_files\`)

| 内容 | 位置 | 备注 |
|---|---|---|
| **L3:43 冠军** `cand-173b12ae`(2.7628 ms) | `box2-runs\runs-l3\run-l3-43-20260909-015247\` | run 全量镜像(88 MB) |
| **L3:21 冠军** `cand-8a5dcef3`(3.4545 ms) | `box1-runs\runs-l3\run-l3-21-20260908-232211\` | 冠军+报告+events(4.2 MB);run 全量在 `box1-backup-20260927\box1-oldcampaign-runs-20260927.tgz` |
| box1 全部 run 备份 | `box1-backup-20260927\`(+ 同级 `box1-box9-backup-20260927-MANIFEST.md`) | kb-4090 扩测(L2:16/L2:59 + GPT-5.6 两臂)、旧 campaign、C2/C3 curated 证据、pip freeze、wheel 哈希、opencode.db |
| box4 全部 run 备份 | `box4-backup-20260927\`(+ `box7-box4-backup-20260927-MANIFEST.md`) | kb-4090 扩测(L2:61/L3:48)、C2/C3 curated 证据、11 个分臂 opencode.db |
| box2 L3 四 run | `box2-runs\` + `box2-runs-l3\` + `box2-backup-20260927\` | |
| box6(A800)/ box7(4×4090)/ box9(5×5090) | `box6-backup-20260927\`、`box7-backup-20260927\`、`box9-backup-20260927\` | 各有 MANIFEST(`box6-box2-…`、`box7-box4-…`、`box1-box9-…`);总清单见 `SERVER_BACKUP_CHECKLIST.md` |
| 外部 SOTA 对比包 | `4090_52_experiments_results_and_best_sources\` 等 | 4 框架 × 3 基座 |
| A800 / 5090 六任务结果 | `A800_6tasks_glm53_results_and_best_sources\`、`A800_GLM_old12_new17_essential_20260921\`、`RTX5090_GLM53_MAX_27_COMPLETED\`、`a800-runs-*` | |

⚠ 备份里的 `opencode-box1.db.gz` 与 `opencode-box4.db.gz` **含 API key 明文**(MANIFEST 有标注)。只留在本地,不外发。

各备份的远端 tgz 与本地副本 sha256 已逐位核对一致(2026-10-12)。

### 1.3 `v5\` 工作树的未提交改动(**这是当前唯一的完整 C3 代码**)

- 38 个已跟踪文件有实质改动(+1948/−300 行,已排除换行差异),主要在 `examples/c3_qwen3/`;
- 未跟踪文件 67 个,其中包括:
  - **5 个 GitHub 上没有的 C3 模块**:`fast_sites.py`(站点排名)、`fixture_budget.py`(容量推导)、
    `fast_acceptance.py`、`fast_formal.py`、`fast_matrix.py`;
  - 约 18 个 C3 测试、`scripts/run_ttft_probe.sh`、`scripts/synthesize_c3_matrix.py`;
  - 大量 `docs/`,包括近期写的论文材料:`paper-contributions-and-pipeline.md`、
    `paper-contribution-evidence-map-2026-09-23.md`、`c3-site-classification-and-selection-explained.md`、
    `l3-21-43-champion-locations.md` 等。**这些都还没进 git。**
- `results/c2-helper-pilot/`(51 MB)也未跟踪,它属于 C2 原始数据,不应随 C3 推送。

**⇒ 本地是 C3 代码和论文文档的唯一完整副本,且没有版本控制保护。**
这是当前最大的风险,优先级高于 box1 整理(见 §3.1)。

---

## 2. box1:情况、文件与整理

### 2.1 基本情况

| 项 | 值 |
|---|---|
| 访问 | `ssh autodl`(`connect.bjb3.seetacloud.com:49408`) |
| 卡 | 2×RTX 4090(当前**无卡开机**,`nvidia-smi` 报 Permission denied 属正常) |
| CPU | Xeon 8358P @2.60GHz(比 box4 的 8352V 快,作业耗时少 26–32%) |
| 数据盘 `/root/autodl-tmp` | 250G,已用 74G,空闲 177G |
| 系统盘 `/` | 30G,已用 19G(主要是 `miniconda3` 9.4G、`.cache` 6.8G、`.triton` 3.8G) |
| 主 venv | `/root/autodl-tmp/kernel-opt-venv`:torch 2.13.0+cu129 / triton 3.7.1 / transformers 5.16.1 / ninja |
| 工具链 | login shell 下 `opencode`、`node`、`python3` 都在 `/root/miniconda3/bin`;ninja 在 `/usr/local/bin`;**nvcc 不在默认 PATH**(在 `/usr/local/cuda/bin`,启动脚本会自行注入) |
| KernelBench | `/root/KernelBench`(指向 `opop-workspace/KernelBench` 的符号链接)@423217d;另有 `kb-4090/KernelBench`@423217d |
| provider | zhipuai,`opop-workspace/opop-glm/.opencode/opencode.jsonc`(mode 600);没有胜算云 |
| 进程 | 没有残留实验进程 |

### 2.2 杂乱程度

- **`/root`**:170 个可见项,其中 **153 个零散文件**(`chk_*.py`、`probe_*.py`、`audit_*.py`、`*.log`、`*.json`、
  `v4-*.bundle` 等),外加 **30 个隐藏的 `.tmp-*.py`**,日期集中在 09-08 到 09-24;
- **`/root/autodl-tmp`**:224 项,其中 **172 个零散文件**(119 个 `.py`、22 个 `.log`、12 个 `.sh`)和 52 个目录。

这些都是历次会话的一次性分析脚本和日志,数量多,但单个都很小。

### 2.3 依赖扫描:哪些东西被引用

扫描范围:数据盘上全部 yaml/jsonc/sh/toml/ini(418 个文件)、`/root/*.sh`、rc 文件、
两个 venv 的 `pyvenv.cfg` / `.pth` / `direct_url.json`、`/root` 下的符号链接。另外查了 GitHub v5 代码里的硬编码路径。

**被引用的(一律原地不动)**:

| 路径 | 被谁依赖 | 动了会怎样 |
|---|---|---|
| `kernel-opt-venv/` | 97 个配置/脚本 | 全部实验无法启动 |
| `orch-venv/` | 225 个旧配置 | 旧 v2–v4 配置无法启动 |
| `kb-4090/` | **kernel-opt-venv 的 editable 安装**(`.pth` 指向 `kb-4090/opop/src`) | **venv 里 `import kernel_optimizer` 直接失败** |
| `opop-workspace/` | orch-venv 的 editable 安装 + `/root/KernelBench` 等 5 个符号链接 + 168 个配置 | KernelBench、旧 L3 生产树、provider 配置全部失联 |
| `nvwheels/`、`torch-2.13.0+cu129-…whl`、`triton-3.7.1-…whl` | kernel-opt-venv 的安装来源(`direct_url.json`) | 已装好的 venv 不受影响;**重建 venv 时需要它们**(另有 box1 wheel sha 记录,见 §1.2) |
| `nodejs/` | `kb-4090/launch_4090*.sh` 写进 PATH | kb-4090 启动脚本里的 node 路径失效(login shell 用的是 miniconda 的 opencode,不受影响) |
| `c2-local-experiment/` | `snapshot_code.sh` 的 `SRC`(C3 车道快照来源)+ 48 个配置 | C3 车道无法起快照 |
| `c3-fast-kernel-iteration/`、`c3-qwen3-4b-operator-goals/`、`c3-snapshots/` | C3 启动脚本与 Python 驱动 | C3 证据与权重失联(14G 的 `c3-qwen3-4b-operator-goals` 里有 Qwen3 权重) |
| **数据盘顶层 C3 启动脚本**:`run_lanes.sh` `run_dev_only.sh` `run_goal_site_probe.sh` `snapshot_code.sh` `merge_and_matrix.sh` `run_formal_campaign.py` `run_f3.py` `run_split_matrix.py` `render_matrix_table.py` `run_b26_gemm_floor.py` | 互相用**绝对路径**引用 | 移走任何一个,C3 车道就会断。**它们和 172 个零散文件混在一起,是整理时最容易误伤的一组** |
| `.triton-cache-*`(9 个,共约 28G) | 各自被一个实验配置的 `triton_cache_dir` 引用 | 删除不会出错,只会让对应配置第一次运行时重新编译。**但 `framework_id`/结果不受影响**,所以它们是**可回收空间**而非依赖 |
| `xdg-v4*`(4 个) | v4 时代配置的 `XDG_*` 目录 | 里面是分臂 opencode.db,本地已有备份 |
| `/root` 下的 6 个符号链接 | `opop`、`opop-glm`、`prep`、`KernelBench`、`KernelBench-l2-37-orig`、`autodl-pub` | 指向数据盘,**保留** |
| `/root/probe-clean/` | GitHub v5 代码里有 51 处硬编码 | 跑 C2 相关的 probe 脚本时找不到路径 |
| `/root/run_summary.py`、`/root/audit_ties.py`、`/root/orchestrator_running.sh`、`/root/current-run.logpath` | `linux-server/scripts/watch_run_completion.sh`(调用前三个)、`launch_l3_task.sh`(写最后一个);GitHub v5 也带这两个脚本 | 用这套旧 L3 启动/监视脚本时报错 |
| `/root/miniconda3/` | 两个 venv 的基础解释器(`pyvenv.cfg` 里 `home=`),以及 opencode/node | **所有 venv 失效** |
| `/root/.config/opencode`、`/root/.local/share/opencode` | opencode 全局配置与会话库 | agent 调用异常 |

**没有被任何东西引用的**:`/root` 下的 30 个 `.tmp-*.py`、`chk_*.py` 等一次性脚本、`*.log`、`*.json` 结果片段、
`v4-*.bundle`;数据盘顶层除上面那组启动脚本之外的零散 `.py/.log/.sh`。已确认没有任何脚本 source 或调用 `.tmp-*`。

另有一类**被"已经不跑的旧东西"引用**的路径:`_backup/`(被 `/root/backup*.sh` 引用)、`c2-helper-pilot/` 等
(被 `_backup/stage-*` 内的旧配置副本引用)。这些引用本身就在存档里,不构成运行依赖。

### 2.4 整理方案(推荐:**归档移动,不删除**)

原则:**只动"零引用"的零散文件,所有目录原地不动**;移动而不是删除;每步都能一键回滚。

```bash
# 在 box1 上执行。0) 先建归档目录并保存清单(用于回滚)
A=/root/autodl-tmp/_archive-20261012
mkdir -p $A/root-loose $A/datadisk-loose
cd /root && ls -la > $A/root-before.txt
cd /root/autodl-tmp && ls -la > $A/datadisk-before.txt

# 1) /root:只移动普通文件,保留下面这几个仍被引用的
KEEP_ROOT="run_summary.py audit_ties.py orchestrator_running.sh current-run.logpath .bashrc .profile .condarc .npmrc .bash_history"
cd /root
for f in $(find . -maxdepth 1 -type f -printf '%f\n'); do
  case " $KEEP_ROOT " in *" $f "*) continue;; esac
  mv -- "$f" $A/root-loose/
done
# /root 下的目录(probe-clean、miniconda3、Documents、fidwork、w1pack 等)和 6 个符号链接一律不动

# 2) 数据盘:只移动顶层普通文件,保留 C3 启动脚本、wheel、备份 tgz
KEEP_DT="run_lanes.sh run_dev_only.sh run_goal_site_probe.sh snapshot_code.sh merge_and_matrix.sh \
run_formal_campaign.py run_f3.py run_split_matrix.py render_matrix_table.py run_b26_gemm_floor.py \
torch-2.13.0+cu129-cp312-cp312-manylinux_2_28_x86_64.whl \
triton-3.7.1-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"
cd /root/autodl-tmp
for f in $(find . -maxdepth 1 -type f -printf '%f\n'); do
  case " $KEEP_DT " in *" $f "*) continue;; esac
  case "$f" in *.tgz|*.gz|*.bundle) continue;; esac   # 备份包另行决定,见 2.5
  mv -- "$f" $A/datadisk-loose/
done

# 3) 验证(见 2.6),通过后再考虑 2.5 的空间回收
```

回滚:`mv $A/root-loose/* $A/root-loose/.tmp-* /root/; mv $A/datadisk-loose/* /root/autodl-tmp/`。

注:`scripts/probe_resource_map.py` 等会把输出默认写到 `/root/probe_resource_*.json`,这是**输出**路径不是输入依赖,
移走旧文件不影响它们再次运行(会重新生成)。

整理后的效果:`/root` 从约 200 项减到约 20 项(目录、符号链接、4 个被引用文件、rc 文件);
数据盘顶层从 224 项减到约 75 项(52 个目录 + 约 10 个启动脚本 + 2 个 wheel + 备份包)。

### 2.5 可选:空间回收(**需要人工确认,不在上面的脚本里**)

| 项 | 大小 | 建议 |
|---|---:|---|
| 9 个 `.triton-cache-*` | 约 28G | 可删。代价只是对应旧配置首次运行时重新编译;新实验会用自己的 cache 目录 |
| `venv-bundle.tgz` | 4.1G | 可删。它是 venv 的打包,用于迁移到别的机器;venv 本身还在,wheel 也在 |
| `opencode-bundle.tgz` | 168M | 同上 |
| `box1-*-20260927.tgz`、`opencode-box1.db.gz` | 约 110M | 本地已有一致副本,可删;**`opencode-box1.db.gz` 含明文 key,建议删** |
| `/root/.cache`(系统盘 6.8G)、`/root/.triton`(3.8G) | 约 10G | 系统盘只有 30G,若吃紧可以清;是 pip/triton 缓存,可重建 |

目前数据盘还有 177G 空闲,**空间不是问题**,所以 2.5 不急。这一节要做的是"清洁",不是"省空间"。

### 2.6 整理会不会影响后续实验:**不会**(依据与验证步骤)

依据:所有被移动的文件都在 §2.3 的"零引用"集合里;目录、venv、符号链接、启动脚本、wheel 一个都不动。

整理后**必须**跑以下验证,任一项失败就回滚:

```bash
# venv 与 editable 安装
/root/autodl-tmp/kernel-opt-venv/bin/python -c "import kernel_optimizer, torch, triton; print(kernel_optimizer.__file__, torch.__version__, triton.__version__)"
# 期望:.../kb-4090/opop/src/kernel_optimizer/__init__.py 2.13.0+cu129 3.7.1
/root/autodl-tmp/orch-venv/bin/python -c "import kernel_optimizer; print(kernel_optimizer.__file__)"
# 符号链接与 KernelBench
ls /root/KernelBench/KernelBench/level3 | wc -l      # 期望 51 左右
# 工具链
bash -lc 'command -v opencode node python3 ninja'
# C3 启动脚本引用的绝对路径都还在
for f in run_lanes.sh run_dev_only.sh run_goal_site_probe.sh snapshot_code.sh merge_and_matrix.sh; do
  grep -oE '/root/autodl-tmp/[A-Za-z0-9._+-]+' /root/autodl-tmp/$f | sort -u | while read p; do [ -e "$p" ] || echo "MISSING $p (from $f)"; done
done
# KernelBench 侧 harness 自检(不需要 GPU)
cd /root/autodl-tmp/kb-4090/opop && /root/autodl-tmp/kernel-opt-venv/bin/python -m pytest tests -x -q -k "not gpu" 2>&1 | tail -3
```

**box4 的同类情况**:结构与 box1 相同(同样有 `kb-4090/`、`c2-local-experiment/`、零散文件),
空闲 110G。CPU 较慢,其余环境一致。若 box4 也要保留,可以套用同一流程,但两个 KEEP 列表要先在 box4 上重新扫一遍,不能直接照搬。

---

## 3. 整理后,box1 的代码是否要更新,怎么更新

### 3.1 结论:**要更新**,但顺序必须是"先推 GitHub,再上 box1"

box1 现有 4 棵代码树,**没有一棵是最新的**:

| 树 | HEAD | 状态 | 用途 |
|---|---|---|---|
| `kb-4090/opop` | 8baa008(v5,落后 GitHub 1 个提交) | 手工补了 `0e39140` 的大部分;**`configs/default.yaml` 仍是 1500s 超时** | **kernel-opt-venv 的 editable 安装就指向它** |
| `c2-local-experiment/code/v5` | af77534(detached,落后 30) | 85 处未提交改动 | C3 车道的快照来源 |
| `opop-workspace/opop` | 062f8e8(v2) | 36 处改动 | 9 月初 L3 冠军的生产树,orch-venv 指向它 |
| `work/opop` | bd65af0(v4) | 干净 | 历史 |

更大的问题在 GitHub 一侧:**GitHub v5 本身缺 C3 后期的工作**(§1.3 的 5 个模块和 38 个文件改动)。
如果现在在 box1 上 `git clone -b v5`,拿到的 C3 没有站点排名、floor 自校准和容量推导,等于退回 09-20 的状态。

### 3.2 步骤一(本地):把 C3 工作推到新分支

在 `D:\Pyhon_projects\opop\v5` 做,建议分支名 `v5-C3`:

1. `git switch -c v5-C3`(基于 `0e39140`);
2. 添加:`examples/c3_qwen3/` 全部改动与 5 个新模块、对应 `tests/test_c3_*`、`scripts/run_ttft_probe.sh`、
   `scripts/synthesize_c3_matrix.py`、相关 `docs/`;
3. **不要添加**:`results/c2-helper-pilot/`(51 MB 的 C2 原始数据)、任何含 key 的文件。
   提交前跑一遍 `v5/scripts/verify_run_backup.py` 或 `git diff --cached | grep -iE "api.?key|sk-|bearer"` 做密钥扫描;
4. 本地跑 `pytest tests -q -k c3` 确认通过,再 `git push -u origin v5-C3`;
5. 用 `git ls-remote --heads origin v5-C3` 确认远端确实有了。记忆里记过一次"push 显示成功但实际没上去"。

### 3.3 步骤二(box1):在新目录 clone,不碰旧树

```bash
source /etc/network_turbo                 # autodl 访问 GitHub 必须先开
mkdir -p /root/autodl-tmp/dev && cd /root/autodl-tmp/dev
git clone -b v5-C3 https://github.com/Fudan-SMI-lab/opop.git opop
cd opop && git rev-parse --short HEAD     # 记下 commit
```

**为什么不在旧树上 `git pull`**:

- `c2-local-experiment/code/v5` 是 detached HEAD,还有 85 处未提交改动,pull 会冲突,也可能毁掉没备份的改动;
- `kb-4090/opop` 是 kernel-opt-venv 的 editable 安装目标,改它等于改所有用这个 venv 的旧实验;
- C3 的 `framework_id` 会哈希约 24–42 个源文件,在旧树上改代码会让已有证据的身份对不上;
- 四棵旧树是论文已有结果的出处,保持原样才能复现。

### 3.4 步骤三:venv 的处理(二选一)

kernel-opt-venv 现在以 editable 方式指向 `kb-4090/opop/src`。新代码要生效,有两种做法:

- **(推荐)新建一个 venv 给新代码用,旧 venv 原样保留**:
  ```bash
  /root/miniconda3/bin/python3.12 -m venv /root/autodl-tmp/dev-venv
  /root/autodl-tmp/dev-venv/bin/pip install \
      /root/autodl-tmp/torch-2.13.0+cu129-cp312-cp312-manylinux_2_28_x86_64.whl \
      /root/autodl-tmp/triton-3.7.1-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl \
      /root/autodl-tmp/nvwheels/*.whl
  /root/autodl-tmp/dev-venv/bin/pip install -e /root/autodl-tmp/dev/opop ninja transformers==5.16.1
  ```
  好处:旧实验配置(都写死了 `kernel-opt-venv`)完全不受影响,两边隔离。
  代价:约 7–8G 磁盘和一次安装时间(wheel 都在本地,无需重新下载 torch)。
  安装后核对三个版本必须逐字符等于 C3 的硬钉值:torch `2.13.0+cu129` / triton `3.7.1` / transformers `5.16.1`,
  否则 C3 车道会直接 `BLOCKED`。
- **(省事)把现有 kernel-opt-venv 改指向新树**:`kernel-opt-venv/bin/pip install -e /root/autodl-tmp/dev/opop`。
  代价:所有引用 kernel-opt-venv 的旧配置从此跑的是新代码,旧结果不再能按原代码复现。
  只在确定不再需要复现旧实验时才这样做。

### 3.5 步骤四:新树的运行前配置

1. **超时**:确认新树 `configs/default.yaml` 与 `config.py` 的 `request_timeout_s` 都是 **3600**
   (GitHub `0e39140` 已改;1500 会把慢的 agent 调用截断);
2. **provider**:新建一个自己的 `.opencode/opencode.jsonc`(可以从 `opop-workspace/opop-glm/.opencode/` 复制),
   `chmod 600`。`launch_cwd` 和 `runs_dir` 都必须位于含 `.opencode/` 的目录下。
   若要换成胜算云,按 `docs/shengsuanyun-api-config.md` §7 操作,先备份原文件;
3. **C3 车道脚本**:数据盘顶层的 `snapshot_code.sh` 把 `SRC` 写死为 `c2-local-experiment/code/v5`。
   要让 C3 车道使用新代码,**复制一份**启动脚本到新目录,并把 `SRC` 改成 `/root/autodl-tmp/dev/opop`。
   不要直接改原脚本,它是旧车道证据的一部分;
4. **自检**:`pytest tests -q`(含 C3 测试;C3 测试需要 `results/c3-*` 两个符号链接,
   `snapshot_code.sh` 第 25–26 行有现成写法),然后跑一次 `kernel-opt --config <cfg> agent-smoke --task level1:19`
   (`--config` 必须写在子命令前)。

### 3.6 一页清单

1. **[本地] 先推 `v5-C3`**(最高优先级:本地是 C3 和论文文档的唯一副本)
2. [box1] 按 §2.4 归档零散文件,然后跑 §2.6 的验证
3. [box1] 在 `/root/autodl-tmp/dev/` clone `v5-C3`
4. [box1] 新建 `dev-venv`(推荐),或把 kernel-opt-venv 改指向新树
5. [box1] 检查超时 3600、配置 provider、复制并改写 C3 启动脚本、跑自检
6. 交付前:轮换 zhipuai key 和 GitHub PAT(runs 的 `sandboxes/` 和 opencode.db 里都有明文副本)

---

## 附:本文依据的实测记录

- 服务器代码与 GitHub 的差异:memory `server-code-trees-vs-github-v5-2026-10-12`
- box1/box4 环境与备份核对:memory `box1-and-box4-state-2026-10-12-dev-box-choice`
- 两个 L3 冠军的位置:`docs/l3-21-43-champion-locations.md`
- 历次服务器备份清单:`external_files/box1-box9-backup-20260927-MANIFEST.md`、`external_files/box7-box4-backup-20260927-MANIFEST.md`
