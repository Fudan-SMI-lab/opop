# box1 / box4 服务器交接(2026-09-22,精简版)

> 交给接管这两台服务器做 KernelBench 实验的外部 agent。只讲访问方式和工作区约定,环境配置由你自行完成。

## 1. 访问

| | box1 | box4 |
|---|---|---|
| ssh | `ssh -p 49408 root@connect.bjb3.seetacloud.com` | `ssh -p 37752 root@connect.bjb1.seetacloud.com` |
| GPU | 2×RTX 4090(24GB),当前空闲 | 2×RTX 4090(24GB),当前空闲 |

autodl 平台三个已知坑:
- 单次 ssh 失败可能是瞬时的,重试一次再下结论;
- 访问 GitHub 慢/超时时先 `source /etc/network_turbo`(实测 git clone 从失败变 6s;注意它对 pip 源反而可能更慢);
- python3 只在 login shell 的 PATH 里,非交互命令用 `bash -lc "..."` 或用解释器绝对路径。

## 2. 工作区约定

数据盘是 `/root/autodl-tmp/`(系统盘很小,一切大文件都放数据盘)。

**请在数据盘新建你自己的实验目录**,例如 `/root/autodl-tmp/kb-expansion/`,在其中 clone 框架、放配置和实验产物,例如:

```
/root/autodl-tmp/kb-expansion/
  opop/        # git clone -b v5 https://github.com/Fudan-SMI-lab/opop.git
  runs/        # 实验产物
  configs/     # 你的实验配置
```

**框架代码从 GitHub 自行 clone,用 `v5` 分支**(远端 `8baa008`)。⚠ 盘上已有的几棵旧代码树(`/root/autodl-tmp/work/opop`、`opop-workspace/opop`、`c2-local-experiment/code/v5`)版本各不相同且部分有未提交改动,**不要用它们跑新实验、也不要改动**(`c2-local-experiment` 下还有别的实验的证据数据)。v5 分支包含 agent 自检 helper(self_test,agent 可自愿调用快速正确性评测来分析自己的 kernel)等 v4 里没有的功能。

## 3. 现成可复用的资源(只读引用即可)

- **KernelBench**:两台都在 `/root/KernelBench`,revision `423217d9`(level2/level3 数据集 + 官方 eval)。
- **现成 venv**(可直接用,也可自建):`/root/autodl-tmp/kernel-opt-venv/bin/python`(Python 3.12,torch 2.13.0+cu129,triton 3.7.1,ninja 已装)。nvcc 在 `/usr/local/cuda/bin`。⚠ nohup/非 login shell 下 venv/bin 与 cuda/bin 不在 PATH,编译 CUDA 扩展前需自行注入,否则 `load_inline` 会静默失败回退。
- **agent API**:当前 zhipuai GLM 配置(含 key)在 `/root/autodl-tmp/opop-workspace/opop-glm/.opencode/opencode.jsonc`(mode 600,勿外传)。⚠ zhipuai 有 5 小时用量限额且同机共享额度池。切换到无限额的胜算云的完整手册在仓库 `docs/shengsuanyun-api-config.md`(clone 后即有),目前尚缺胜算云 key,由委托方提供。

## 4. 红线

1. 同机双卡隔离只靠 `CUDA_VISIBLE_DEVICES`,发车后用 `nvidia-smi --query-compute-apps=pid,gpu_uuid` 确认没撞卡。
2. runs 产物里会带 API key 明文副本:runs 目录不外发,交付结果前轮换 key。
3. 长跑判活看 `/proc/<pid>` 或 compute-apps,别信 ssh 里 pgrep 的阴性结果。
