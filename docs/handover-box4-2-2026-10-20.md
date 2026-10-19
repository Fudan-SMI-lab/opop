# box4-2 交接文档(2026-10-20)

> 本文所有状态都是 **2026-10-20 在 box4-2 上实测**的结果,不是推断。
> 读者:接手 box4-2 开发或实验的人或 agent。
> box1 暂时不可用,后续开发改在 box4-2 上进行。box4-2 是原 box4 的克隆机,原 box4(端口 37752)已拒绝连接。

---

## 0. 一页速览

| 项 | 值 |
|---|---|
| 连接 | `ssh -p 30726 root@connect.bjb1.seetacloud.com`(见 §1) |
| 卡 | **2×RTX 4090,有卡开机**(driver 580.105.08,CUDA 12.8)。实测时两卡空闲 |
| 新开发用 | 代码 `/root/autodl-tmp/dev/opop`(GitHub `v5-C3`@`8de73b4`,干净);venv `/root/autodl-tmp/dev-venv` |
| 旧树 | 4 棵,**全部只读**,不要在上面改代码或 `git pull`(§4.3) |
| 整理 | 已完成。零散文件已移入 `/root/autodl-tmp/_archive-20261020/`,可一键回滚(§3) |
| 自检 | `doctor` 全绿;真实 agent 调用(zhipuai/glm-5.3)成功;全套 pytest 2743 passed,11 个失败都是环境或数据布局原因(§5) |
| 磁盘 | 数据盘(122G)**空闲 73G**,系统盘(30G)空闲 28G。清理前分别是 30G 和 25G(§6) |

---

## 1. 如何连接

```bash
ssh -p 30726 root@connect.bjb1.seetacloud.com
```

- **密钥登录已经可用**:克隆机继承了原 box4 的 `authorized_keys`。本地 `~/.ssh/id_ed25519` 实测可以登录,不需要密码。
- 密码由用户单独提供,**不写进任何会提交到 git 的文件**。这个仓库是公开的(`Fudan-SMI-lab/opop`)。
- 本地 `~/.ssh/config` 还没有这台机器的别名。如果需要,可以加:
  ```
  Host box4-2
    HostName connect.bjb1.seetacloud.com
    Port 30726
    User root
    ServerAliveInterval 30
  ```
- 单次 ssh 失败不代表故障(autodl 网关在负载高时会偶发拒绝),重试一次再判断。
- 访问 GitHub 前要先执行 `source /etc/network_turbo`。今天第一次 clone 返回了 503,第二次 7 秒成功。
- autodl 的生命周期:**关机不等于释放**,连续关机 15 天才会释放。无卡模式开机时 `nvidia-smi` 会报 Permission denied,这是正常的。

---

## 2. 硬件与环境(实测)

| 项 | 值 |
|---|---|
| 主机名 | `autodl-container-e8e84ea25b-cffd5c81` |
| CPU / 内存 | Xeon Gold 6430(128 核)/ 1007G |
| GPU | 0 号 `GPU-4bead555-…`,1 号 `GPU-69c59246-…`;各 24G |
| 数据盘 `/root/autodl-tmp` | 122G(xfs),清理后已用 50G,**空闲 73G** |
| 系统盘 `/` | 30G,清理后已用 2.3G |
| 基础解释器 | `/root/miniconda3/bin/python3.12` |
| 版本(两个 venv 相同) | torch `2.13.0+cu129` / triton `3.7.1` / transformers `5.16.1` / ninja |
| opencode / node | `/usr/local/bin/opencode`(1.18.29)和 `node` 都是符号链接,指向 `/root/autodl-tmp/node/bin/`,**所以 `node/` 目录是依赖,不能删** |
| nvcc | `/usr/local/cuda/bin/nvcc`,**不在默认 PATH 上**。启动脚本和 `doctor` 会自己找到它 |
| ncu | 有,但容器里读不到硬件计数器(ERR_NVGPUCTRPERM)。Tier 0/1 剖析可用 |
| KernelBench | `kb-4090/KernelBench` 与 `opop-workspace/KernelBench` 都是 git @`423217d`;`/root/KernelBench` 是普通目录(不是 git,也不是符号链接) |
| Qwen3-4B 权重 | `c3-qwen3-4b-operator-goals/range-resume/` 下 15 个 safetensors(`assets/` 共 14G) |
| provider | **zhipuai**(`opop-workspace/opop-glm/.opencode/opencode.jsonc`,mode 600);**胜算云**(`kb-4090/work/.opencode/opencode.jsonc`,mode 600) |
| L3:43 冠军 | `opop-workspace/opop-glm/runs-l3/run-l3-43-20260909-015247/report/best_kernel.py`,sha256 `a77876e1…9b79efcfe`,与本地镜像一致 |
| 常驻进程 | 只有 autodl 自带的 jupyter-lab 和 tensorboard,没有实验进程 |

---

## 3. 整理(已完成)

### 3.1 做了什么

**原则**:只移动顶层零散的普通文件,不碰任何目录、符号链接、venv 或被引用的脚本。只移动不删除,并且可以一键回滚。

| 位置 | 移走 | 保留 |
|---|---:|---|
| `/root` | 130 个文件(一次性 `chk_*/probe_*/audit_*.py`、`.tmp-*.py`、日志、json 片段) | 被引用的 `run_summary.py`、`orchestrator_running.sh`、`current-run.logpath`(旧 L3 监视脚本使用)、`s4-*/s7-*` 记录、rc 文件、`v4-*.bundle` |
| `/root/autodl-tmp` 顶层 | 117 个文件 | C3 启动脚本与驱动(§4.4 表)、`e1-box2.log`、所有 `.tgz/.gz/.bundle` |
| `/tmp` | `nt.py` 1 个 | 其余不动 |

整理后:`/root` 剩 26 个可见项(目录、符号链接、被引用的文件、`v4-*.bundle`),数据盘顶层剩 30 个文件。

注:旧的 `work/opop/linux-server/scripts/watch_run_completion.sh` 会调用 `/root/audit_ties.py`。box1 上有这个文件,**box4-2 上整理前就没有**(`root-before.txt` 里也找不到),所以与整理无关。只有在这台机器上使用旧的 L3 监视脚本时,才需要从 box1 的备份里取回它。

**为什么要单独移走 `/tmp/nt.py`**:这是一个旧的分析脚本,文件名和标准库的 `nt` 模块相同。只要在 `/tmp` 下启动 Python,`import pathlib` 就会去加载它,导致 torch 在 import 阶段崩溃。`dev/opop/scripts/sweep_resource_map.py` 的注释里提到过它。今天在 `/tmp` 下验证 dev-venv 时复现了这个崩溃。移走以后,在 `/tmp` 下 `import torch` 正常。
**经验**:不要在 `/tmp` 下启动 Python。`/tmp` 里还有 125 个旧 `.py`,目前没有其他与标准库同名的。

### 3.2 归档与回滚

```
/root/autodl-tmp/_archive-20261020/
  root-loose/  datadisk-loose/  tmp-loose/     被移走的文件
  root-moved.txt  datadisk-moved.txt  tmp-moved.txt   移动清单
  root-before.txt datadisk-before.txt          整理前的 ls -la
  first-attempt-manifests/                     第一次尝试的清单(已作废,仅留档)
  ROLLBACK.sh                                  一键回滚(mv -n,不覆盖)
```

回滚命令:`bash /root/autodl-tmp/_archive-20261020/ROLLBACK.sh`

### 3.3 整理不影响实验的依据

1. 移动前扫描了 4 棵代码树里的全部 sh/yaml/py 文件、数据盘顶层脚本和 `/root/*.sh`,找"引用了待移动文件"的地方,结果为空。
2. 移动后全部环境检查通过:三个版本钉、`kernel-opt` CLI、login shell 下的 opencode/node、ninja、nvcc、KernelBench、权重、provider 权限、L3:43 冠军;另外,保留的 C3 启动脚本和 kb-4090 配置里出现的每一个绝对路径都还存在。
3. `kb-4090/opop` 的 CPU 测试有 126 行失败。为了确认这些失败不是整理造成的,做了一次对照:回滚、重跑、得到**逐行相同**的 126 行,再重新执行整理。所以这些失败在整理之前就存在。原因有两个:
   - `kb-4090/opop/results/` 里缺 `c3-*` 两个符号链接;
   - `test_c2_agent_compatibility` 的 fixture 里写死了 Windows 路径分隔符 `\\`。
   这两项都是旧树本身的问题,而旧树不再修改(§4.3)。

### 3.4 磁盘清理

见 §6。

---

## 4. 代码:现在用哪棵树,怎么用

### 4.1 结论

**新开发统一使用 `/root/autodl-tmp/dev/opop` 加 `/root/autodl-tmp/dev-venv`。** 旧树和旧 venv 保持原样,用于复现已有结果。

### 4.2 新树:`/root/autodl-tmp/dev/opop`

- 来源:`git clone -b v5-C3 https://github.com/Fudan-SMI-lab/opop.git`,HEAD `8de73b4`(2026-10-12)。实测 dirty=0,全部 LF 换行。
- `v5-C3` 是 `v5`(`0e39140`)加 28 个提交,包含 5 个 C3 后期模块(`fast_sites` / `fast_acceptance` / `fast_formal` / `fast_matrix` / `fixture_budget`)、C3 测试、`run_ttft_probe.sh`、`synthesize_c3_matrix.py`、C3 驱动 `results/c3-fast-kernel-iteration/F2-live/{run_f3.py, run_formal_campaign.py, snapshot_code.sh}`,以及论文文档。
- 超时:`configs/default.yaml` 与 `config.py` 的 `request_timeout_s` 都是 **3600**。
- **与现有 C3 证据是同一份实现**:新树、`c2-local-experiment/code/v5`、9/22 的 `probe-norm` 快照、9/22 的 `synth-matrix` 快照,四者的 `implementation_sha256` 都是 **`d9422ad8973c…`**,`schema_sha256` 和 `prompt_sha256` 也相同。
  所以在新树上起的 C3 车道,`framework_id` 中实现相关的部分与 9 月的车道一致。不过 `framework_id` 还哈希了 `agent_config_sha256`(其中包含 `request_timeout_s`)和 contract,而新树默认超时是 3600,9 月的车道是 1500。两者能否合并,仍要按 memory `cross-box-lane-pooling-requires-six-identical-hashes` 逐字段核对。
- 新树与 `c2-local-experiment/code/v5` 的实质差异只有三类,都不在 framework 哈希范围内:
  - `0e39140` 那 3 个文案文件(contract 去掉 fp32 句;`bottleneck.py` / `task_cost.py` 改为 "REFERENCE dtype");
  - 测试文件:新树多出 2 个测试,另有 3 个测试是更新后的版本;
  - 换行符(旧树是 CRLF)。

### 4.3 旧树(只读,不要改)

| 树 | HEAD | 被谁使用 | 为什么不能动 |
|---|---|---|---|
| `kb-4090/opop` | `8baa008`(v5,落后 1) | **`kernel-opt-venv` 的 editable 安装目标**;KernelBench 扩测配置 | 改了它,所有使用 kernel-opt-venv 的旧配置跑的就不再是原来的代码 |
| `c2-local-experiment/code/v5` | `38e6e10`(detached)+59 dirty | 数据盘顶层 `snapshot_code.sh` 的 `SRC` | 未提交的改动在这台机器上只有这一份;对 detached HEAD 做 pull 会冲突 |
| `opop-workspace/opop` | `14fab15`(v2) | `orch-venv`;9 月初 L3 冠军的生产树 | L3 冠军的出处 |
| `work/opop` | `bd65af0`(v4) | 历史 | — |

### 4.4 C3 车道:`/root/autodl-tmp/dev/launch/`

数据盘顶层的 C3 启动脚本把 venv 和代码源写死成了旧路径。为此做了一组**副本**放在 `dev/launch/`,每个文件只改两处:

1. `PY=/root/autodl-tmp/dev-venv/bin/python`(原来是 `kernel-opt-venv`)
2. 快照来源改成 `/root/autodl-tmp/dev/opop`,由 `dev/launch/snapshot_code.sh` 负责。它会在每个快照里写一个 `CODE_COMMIT` 文件,记录 commit 和 dirty 文件数。

| 脚本 | 用途 |
|---|---|
| `run_dev_only.sh <rev> [dev-hours] [gpu] [port]` | 只跑开发阶段,直到 `DEV_GO` |
| `run_goal_site_probe.sh <site> <goal> <card> <tag> [dev-h] [search-h]` | 在一个站点、一个目标上跑完整的 dev→freeze→lane 链。**ttft 也用它**(goal 填 `ttft`) |
| `run_lanes.sh <rev> <dev-epoch> <lanes-csv> [h] [gpu] [port]` | 用已经 DEV_GO 的 epoch 跑车道,不出矩阵 |
| `run_chain.sh` | 完整链路加 36 格矩阵(写死 CVD=0) |
| `merge_and_matrix.sh` | 合并多个 lane 后测一次矩阵 |

- 原脚本都没有改,改动前后 sha256 一致。副本仍然调用数据盘顶层的 `run_f3.py`、`run_formal_campaign.py`、`run_split_matrix.py`、`render_matrix_table.py`。其中 `run_f3.py` 和 `run_formal_campaign.py` 与 git 里 `F2-live/` 下的副本逐字节相同。产物仍写到 `/root/autodl-tmp/c3-fast-kernel-iteration/`,快照仍写到 `/root/autodl-tmp/c3-snapshots/`。
- **故意没有复制 `run_ttft_probe.sh`**:它只设了 `C3_CUDA_INDEX`,没有设 `CUDA_VISIBLE_DEVICES`,所以主评估进程(负责计时的那个)不管传哪张卡,都会落在物理 0 号卡上。见 memory `probe-scripts-omitted-cuda-visible-devices-main-proc-collided`。
- 快照机制本身不能省。`framework_id` 哈希了约 24 个源文件,而且车道在运行过程中会反复校验。如果在车道运行时修改 `dev/opop`,正在跑的车道会被判成 `framework_invalid`。快照是只读的,所以 `dev/opop` 可以继续开发。
- 驱动里有两处硬编码:`ROOT=/root/autodl-tmp/c3-fast-kernel-iteration`,以及 agent 配置模板 `development-996318d/agent-config.json`。后者的 `request_timeout_s` 是 480,`run_f3.py` 会把它覆盖成代码默认值 3600。它的 `server_env.PYTHONPATH` 仍然指向 `c2-local-experiment`,这只影响 agent 子进程,而 agent 用的 helper 命令是按快照路径生成的。**如果要彻底切到新树,应该另做一份 agent-config,不要修改原文件。**

### 4.5 KernelBench 侧(C1/C2 的 L1–L3 任务)

模板在 `/root/autodl-tmp/dev/work/configs/dev_gpu0_template.yaml`(mode 600)。它从 `kb-4090/opop/configs/c4090_a_box4_gpu0_level2_61_glm53.yaml` 复制而来,预算和评估参数逐字不变,只改了以下几项:

| 项 | 值 |
|---|---|
| venv | `dev-venv` |
| `runs_dir` | `/root/autodl-tmp/dev/work/runs-dev-gpu0` |
| `launch_cwd` | `/root/autodl-tmp/dev/work` |
| `sandbox_config_path` | `/root/autodl-tmp/dev/work/.opencode/opencode.jsonc`(zhipuai,是 opop-glm 那份的逐字节副本) |
| `triton_cache_dir` | `/root/autodl-tmp/dev/work/.triton-cache-dev-gpu0` |
| port | 15200 |
| 模型 | `zhipuai/glm-5.3` |

用法:

```bash
cd /root/autodl-tmp/dev/work
cp configs/dev_gpu0_template.yaml configs/<任务>.yaml   # 改 runs_dir / triton_cache_dir / port
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
  bash -lc '/root/autodl-tmp/dev-venv/bin/kernel-opt --config configs/<任务>.yaml run --task level3:43'
```

注意事项:

- `--config` 必须写在子命令**前面**;
- `agent-smoke` 必须带 `--module generator|parameterizer`;
- 要在 login shell(`bash -lc`)里运行,否则 PATH 里可能找不到 opencode;
- `launch_cwd` 和 `runs_dir` 必须都在含 `.opencode/` 的目录下面;
- 同一台机器上同时跑多个 run 时,每个 run 必须有**各自的** port、`runs_dir`、`triton_cache_dir` 和 CVD;
- GPU 锁是每个 run 一把(`<run_dir>/jobs/gpu.lock`),两个 run 之间不会互相阻塞,**同机两个 run 的隔离只能靠 CVD**。

### 4.6 两个 venv

| venv | editable 指向 | 用途 |
|---|---|---|
| `/root/autodl-tmp/dev-venv`(新) | `/root/autodl-tmp/dev/opop/src` | **新开发** |
| `/root/autodl-tmp/kernel-opt-venv`(旧) | `/root/autodl-tmp/kb-4090/opop/src` | 复现 kb-4090 扩测和 9 月的 C3 车道 |
| `/root/autodl-tmp/orch-venv`(旧) | `opop-workspace/opop` | 9 月初的 L3 生产树 |

`dev-venv` 的制作方式:用 `cp -a` 复制 `kernel-opt-venv`,把 `bin/` 下 43 个脚本 shebang 和 `pyvenv.cfg` 里的路径改成新路径,再执行 `pip install --no-deps -e /root/autodl-tmp/dev/opop`。
- 两个 venv 的包集合**完全相同**(各 112 个包,`pip freeze` 差异为 0),所以三个版本钉自然一致。
- 制作前后,旧 venv 的 `.pth` 文件 sha 不变,shebang 也不变。
- 这台机器上**没有 torch/triton 的 wheel 文件**,所以没有从 wheel 重装,而是复制 venv。如果以后要从零重建,torch 需要从 `download.pytorch.org` 下载 cu129 版本:要开 turbo,并加 `--timeout 60 --retries 3`;判断下载是否还在进行,看 `du -sh /tmp/pip-unpack-*`。

---

## 5. 自检结果(2026-10-20)

| 检查 | 结果 |
|---|---|
| `dev-venv` import | `kernel_optimizer` 指向 `dev/opop/src`;torch 2.13.0+cu129 / triton 3.7.1 / transformers 5.16.1;`cuda True 2` |
| `kernel-opt --config dev_gpu0_template.yaml doctor` | **全部 OK**:nvidia-smi、native 拓扑、opencode 1.18.29、KernelBench、torch/triton、kernelbench 可导入、CUDA;cuda backend 可构建(nvcc 和 ninja 都在) |
| 真实 agent 调用 `agent-smoke --module generator --task level1:19`(卡 1) | **成功**:zhipuai/glm-5.3 一次就返回 2 个候选,PARAMS 都合法,费用 $0.0087;结束后没有残留 opencode 进程,端口已释放。smoke 的 run 目录已删除(其中的 sandbox 里有 provider 副本),日志保留在 `dev/work/logs/` |
| 全套 pytest(新树副本,含 `results/c3-*` 链接) | **2743 passed, 7 skipped, 11 failed**,耗时 9 分 25 秒。9/22 的基线是 2741 passed。**11 个失败都与代码无关**,见下表 |

| 失败 | 个数 | 原因 | 处理 |
|---|---:|---|---|
| `test_c3_quality_arithmetic`(参数化) | 7 | 测试写死的 oracle 路径是 `c3-qwen3-4b-operator-goals/readiness-foundation/host-A/native-raw/oracle-a1cbcc11…/oracle.json`,box4-2 上同一个文件在 `readiness-foundation/native-raw/…`(**没有 `host-A` 这一层**)。本地 Windows 的那份是有 `host-A` 的。属于数据布局差异 | 需要这组测试时,在测试副本里补一层链接即可,**不要改数据盘上的证据树** |
| `test_c2_agent_compatibility` ×2、`test_task_agent_profiles::test_cda_contract…` ×1 | 3 | `tests/fixtures/c2_contracts/*.json` 是在 Windows 上录制的,里面写死了 `"<CASE>\\parent.py"`,Linux 生成的是 `<CASE>/parent.py` | fixture 的平台问题,与 C2/C3 逻辑无关 |
| `test_c3_manual_task::test_manual_entry_exists_and_loads_without_model` | 1 | 断言"进程里没有导入 torch",全套连跑时前面的测试已经导入了 torch | **单独运行时 passed** |

pytest 的运行方式:把 `dev/opop` 复制一份到 `c3-snapshots/devtest-full-*`,把副本里 `results/c3-fast-kernel-iteration` 和 `results/c3-qwen3-4b-operator-goals` 换成指向数据盘的符号链接,然后设 `PYTHONPATH=<副本>/src:<副本>`,在副本根目录下运行 `pytest tests`。完整日志在 `dev/work/logs/pytest-full-dev-opop-8de73b4-1020.log`,测试副本用完已删除。
不能直接用 `snapshot_code.sh` 生成的快照来跑全套测试:快照只复制 `examples/src/tests`,没有 `scripts/`,有 30 个 C2/v4 测试会因为 `No module named 'scripts'` 在收集阶段就失败。这些测试本来就不属于 C3 车道。
同样,不要直接在 `dev/opop` 里跑:git 里的 `results/c3-fast-kernel-iteration/` 只带了 3 个驱动文件,测试要找的 `setup/` 等目录不在里面,需要先做上述链接。另外 `dev/opop` 是开发树,不应在里面生成产物。

---

## 6. 磁盘(2026-10-20 已清理)

| | 清理前 | 清理后 |
|---|---:|---:|
| 数据盘 `/root/autodl-tmp`(122G) | 已用 93G / 空闲 30G | **已用 50G / 空闲 73G** |
| 系统盘 `/`(30G) | 已用 5.7G / 空闲 25G | 已用 2.3G / 空闲 28G |

**做了什么**(清理前确认没有任何实验进程在运行):

| 项 | 回收 | 怎么做的 | 为什么安全 |
|---|---:|---|---|
| 29 个 triton 编译缓存目录(15 个 `.triton-cache-*`,14 个 `.triton-b*` / `.triton-matrixfix`) | 36.5G | 删除 | 里面只有按内容哈希命名的编译产物(抽查 `kopt` 和 `v3` 两个,其中 9226 和 4464 个条目全部是哈希名),没有进程打开,也没有符号链接指向它们。triton 发现缓存目录不存在时会自己创建,已实测:删除后编译并运行一个 kernel,结果正确,目录自动重建。`framework_id` 和已有结果都不受影响,只是旧配置第一次重跑时需要重新编译 |
| C3 的 `.f32` logits 张量 | 7.06G | **硬链接去重,一个文件都没删** | 共 434 个文件,内容去重后只有 44 份(10 份 matrix 或 synth-matrix 重复记录了同一组 held-out logits)。每组相同内容只保留一份实体,其余 390 个路径改成指向它的硬链接,所以所有路径和内容都不变。去重前后 434 个路径的 sha256 逐行相同,而且与各 `oracle.json` 里记录的 `logits_sha256` 全部吻合(434/434)。代码只按路径读取、按 sha256 校验,不看 inode |
| `opencode-box4.db.gz`(**含明文 key**)、`box4-full-runs-20260927.tgz`、`box4-xdg-dbs-20260927.tgz`、`_backup/{c3c2-evidence,evidence}-box4-20260924.tgz` | 约 150M | 删除 | 5 个文件的 sha256 与本地 `external_files\box4-backup-20260927\` 下的副本逐一核对一致后才删除 |
| `opencode-bundle.tgz` | 168M | 删除 | 用 `tar --diff` 比对,内容与现有的 `nodejs/` 目录完全一致 |
| `/root/.triton/cache`(系统盘)、`/tmp/__pycache__` | 3.4G | 删除 | 都是缓存,会自动重建 |

**没有动的**:所有代码树、venv、`node/`、`nodejs/`、C3 证据树(`c3-*`)、`_backup/stage-*`、10 月的通信实验目录、`xdg-*`(旧分臂的 opencode 库,本地已有备份),以及 `b26-artifacts.tgz` 和 `c3sync.tgz`(各几十 KB,本地没有找到副本,所以保留)。

清理记录在 `/root/autodl-tmp/_archive-20261020/disk-cleanup-20261020/`,包括被删除的目录清单及各自大小、f32 去重前后的 sha 和 inode、硬链接对照表、包的 sha 校验清单、清理前后的 `df`。

**注意硬链接**:`c3-fast-kernel-iteration` 下的 `.f32` 现在是多路径共享同一个 inode。这些文件是只读证据,不应原地修改;如果原地修改其中一个,所有共享这个 inode 的路径会一起变。新的车道会写新的 oracle 目录,不会碰到旧文件。

**后续空间预算**:一次 C3 矩阵阶段约产生 780M 的 `.f32`,一次 KernelBench run 的 triton 缓存约 1–6G。73G 足够跑若干轮,但长实验开始前仍要看一眼 `df -h /root/autodl-tmp`。

---

## 7. 这台机器上另一个 agent 的实验(10-13 至 10-14)

数据盘上有 6 个目录是另一个 agent 在原 box4 上做的通信/TP 实验,随克隆一起带了过来:

`comm-profiling-check-20261013T234000`、`comm-profiling-nsys-check-20261013T235800`、`ddp-comm-case-20261014T004800`、`qwen3-tp-operator-case-20261014T013300`、`qwen3-fused-operator-check-20261014T020500`、`qwen3-greedy-comm-operator-20261014T030000`,另有 `tools/nsys-2025.3.1`(481M)。

每个目录都有 `README.md`,大多数还有 `SHA256SUMS`。本地备份在 `D:\Pyhon_projects\opop\external_files\` 下的 `box4-comm-profiling-20261013\`、`box4-comm-operator-handover-20261014\`、`ddp-comm-case-20261014\`、`qwen3-tp-operator-case-20261014\{fused-operator,greedy-comm-operator}`,SHA256SUMS 已逐一核对通过。整理时**没有动这些目录**。

---

## 8. 红线(每一条都出过事)

1. **不要修改旧树和旧 venv**,包括 `kb-4090/opop`、`c2-local-experiment/code/v5`、`opop-workspace/opop`、`work/opop`、`kernel-opt-venv`、`orch-venv`,以及数据盘顶层的 C3 启动脚本。它们是已有结果的出处。
2. **不要在车道运行期间修改代码树**。车道跑的是快照;对 `dev/opop` 的修改只对下一次快照生效。
3. **同机两个任务靠 CVD 隔离**:设 `CUDA_DEVICE_ORDER=PCI_BUS_ID` 和 `CUDA_VISIBLE_DEVICES=<卡>`,每个任务用各自的端口、缓存目录和 runs 目录。发车后用 `nvidia-smi --query-compute-apps=pid,gpu_uuid` 对照 pid 和卡的 uuid,确认没有两个任务落在同一张卡上。
4. **判断任务是否还活着**,只看 `/proc/<pid>` 或 `nvidia-smi --query-compute-apps`。`pgrep -c` 在 ssh 里会给出假阴性。
5. **agent 阶段 GPU 0% 不代表卡死**。GLM 流式输出的中位时长是 519–736 秒。判断 agent 是否还在工作,看 opencode.log 里的 `stream` 和 `loop step=N`;判断是否断供,看 ERROR 行里是否有原文"已达到 5 小时使用上限"。
6. **密钥**:每个 run 的 `sandboxes/`,以及 `xdg-*` 和各 runtime 目录下的 opencode 库,都有明文 API key,**不要外发,不要提交**。数据盘顶层的 `opencode-box4.db.gz` 已删除。`.opencode/`、`*.jsonc` 已被 `.gitignore` 排除。交付前需要轮换所有机器上的 API key 和 GitHub PAT。
7. **不要原地修改 `c3-fast-kernel-iteration` 下的 `.f32`**。它们现在是硬链接,改一个路径,共享这个 inode 的所有路径都会跟着变(§6)。
8. **不要在 `/tmp` 下启动 Python**(原因见 §3.1)。
9. **要推送到 GitHub 的代码在本地 worktree 修改**(`D:\Pyhon_projects\opop\v5-C3`,分支 `v5-C3`),推送后在这台机器上 `git pull`。如果要在 `dev/opop` 上直接改,要提交到分支并推送,不要让改动只留在服务器上。

---

## 附:本次实测涉及的文件

- 本地临时脚本:`tidy_box42.sh`(整理)、`verify_box42.sh`(验证)、`make_devvenv.sh`、`make_launch.sh`、`fid_compare.sh`
- 服务器:`/root/autodl-tmp/_archive-20261020/`、`/root/autodl-tmp/dev/{opop,launch,work}`、`/root/autodl-tmp/dev-venv`
- 相关文档:`docs/handover-local-and-box1-2026-10-12.md`(本地结果位置与 box1)、`docs/l3-21-43-champion-locations.md`
