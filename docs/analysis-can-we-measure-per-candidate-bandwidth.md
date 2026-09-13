# 能不能测到每候选的带宽/算力用量:在本机实测过的五条路

**日期** 2026-09-13 · **实测机器** box1(AutoDL,RTX 4090,sm_89,128 SM,384-bit,Triton 3.7.1)
**探针** `scripts/probes/what_profiling_is_available.py`、`can_we_get_per_candidate_bytes.py`、
`ir_derived_bytes.py`(都可复跑)

**结论先说**:硬件计数器这条路在本机**结构性关闭**;但**有一条路成功了** ——
从编译产物推算每候选逐参数组的**逻辑字节数**,在两个已知真值的 kernel 上都精确到 **ratio 1.0000**,
且正对照能区分。它不能替代计数器(看不到 L2 命中、tiling 重读),但它**恰好补上了当前缺的那个性质**:
随候选与参数变化,而不是任务级常数。

---

## 1. 为什么硬件计数器在本机不可用 —— 不是"没装",是被内核模块关掉

| 实测项 | 结果 | 含义 |
|---|---|---|
| `/proc/driver/nvidia/params` | **`RmProfilingAdminOnly: 1`** | **宿主机内核模块**把性能计数器设为仅管理员 —— 这是 `ERR_NVGPUCTRPERM` 的直接原因 |
| `EUID` | `0` | 我们**是** root |
| 容器 capabilities | **`!cap_sys_admin`**、**`!cap_perfmon`** | 但解除该限制所需的两个能力**都被 drop 了** |
| `ncu` 是否存在 | `/usr/local/cuda/bin/ncu` | **存在** |
| **`ncu --metrics dram__bytes.sum` 实跑** | `rc=1`,`ERR_NVGPUCTRPERM`,**拿不到任何 dram 数字** | **存在 ≠ 可用** |
| `nsys` | **ABSENT** | 未安装 |
| `dcgmi` | **ABSENT** | 未安装 |
| `libcupti*` | **none found** | 连 CUPTI 库都不在镜像里 |

**关键点**:`RmProfilingAdminOnly=1` 是**宿主机**设置,租户改不了;而即使它是 0,
容器缺 `cap_sys_admin`/`cap_perfmon` 也照样过不去。
⇒ **这不是"配置没调对",是两道独立的门同时关着。**

**在第二台机器上复验过,不是单机现象。** box4(另一个 AutoDL 实例,
`connect.bjb1.seetacloud.com:37752`)读出**完全相同**的三项:

```
RmProfilingAdminOnly: 1
!cap_sys_admin !cap_perfmon
/usr/local/cuda/bin/ncu        ← ncu 同样存在
```

⇒ **2/2 台 AutoDL 实例相同** ⇒ 这是 AutoDL 的容器策略,不是某台机器的偶然配置。
换一台**同类**租用容器不会解决问题。
(A800/box3 已关机,`Connection refused`,未能验证 —— 若将来开机,值得顺手补这一条:
一条 `grep -i profil /proc/driver/nvidia/params` 即可。)

**顺带确认了一件重要的事:Activity 通道是开的,只有 Profiling 通道关着。**
`torch.profiler` 成功拿到 kernel 级数据(`CUDA_EVENTS 2`,含 `cutlass_80_simt_sgemm_...` 的名字与
318.28 µs 耗时)。⇒ **"kernel 名 + 耗时"随时可得,"硬件计数器"不可得**。
这个区分决定了后面哪些路可行。

---

## 2. Triton Proton:**只报时间,对本问题无用**

Triton 3.7.1 自带 `triton.profiler`(Proton),不需要计数器权限,实跑成功。但它输出的是:

```json
{"frame": {"name": "add_kernel"},
 "metrics": {"count": 1, "device_id": "0", "device_type": "CUDA", "time (ns)": 193092}}
```

`PROTON_MENTIONS_BYTES=False` —— **整份 trace 里没有任何字节量**,只有 `time (ns)`。
⇒ 它给的又是延迟,正是我们已经有的东西。**这条路排除。**

(Proton 的 trace 里附带了有用的设备常量:`bus_width: 384`、`memory_clock_rate: 10501000`、
`num_sms: 128`、`arch: 89` —— 可以用来核对标定的 DRAM 屋顶,但那是任务级常数,不解决本问题。)

---

## 3. 成功的那条路:从编译产物推算逐候选逻辑字节

### 3.1 做法

Triton 编译后的 `kernel.asm["ttir"]` 里,每个 `tt.load` / `tt.store` 都带类型标注:

```
%x_5 = tt.load %x_4, %m_3 : tensor<1024x!tt.ptr<f32>> loc(#loc25)
```

⇒ 一个 program instance 的字节数 = Σ(tensor 元素数 × 元素宽度);
乘上 grid 就是整个 kernel 的逻辑字节数。**不需要任何运行时工具,不需要计数器。**

### 3.2 实测:两个已知真值的 kernel 都精确命中

**kernel 1** — 4096×4096 fp32 逐元素加(读两个、写一个,真值 `3×4096×4096×4 = 201326592`):

```
PER_INSTANCE_BYTES=12288      OP ('load', 1024, 'f32', 4096)
N_INSTANCES=16384             OP ('load', 1024, 'f32', 4096)
IR_DERIVED_TOTAL=201326592    OP ('store', 1024, 'f32', 4096)
KNOWN_TRUE_BYTES=201326592
RATIO=1.0000
```

**随参数变化,且总量守恒**(这是正确性性质:同样的数据,不论怎么切总量不变):

| BLOCK | per_instance | instances | total | ratio |
|---|---|---|---|---|
| 256 | 3072 | 65536 | 201326592 | **1.0000** |
| 512 | 6144 | 32768 | 201326592 | **1.0000** |
| 1024 | 12288 | 16384 | 201326592 | **1.0000** |
| 2048 | 24576 | 8192 | 201326592 | **1.0000** |

**kernel 2(正对照)** — 只读一个、写一个,真值应为上者的 2/3:

```
CONTROL_PER_INSTANCE=8192  (add kernel was 12288)
CONTROL_TOTAL=134217728  CONTROL_TRUE=134217728  ratio=1.0000
CONTROL_DISTINGUISHES=YES
```

⇒ **不是把常数凑对了** —— 换一个流量不同的 kernel,读数按真值改变。
(`probe-needs-a-positive-control`:确认性结论必须有一个会大声失败的正对照。)

### 3.3 这条路的真实边界:tiled matmul 那一格

同一个探针在 2-D tiled matmul 上**当场暴露了两个局限**:

```
MM_2D_POINTER_TENSORS=[('64','1','f32'), ('64','64','f32'), ...] (n=18)
MM_LOADS=2  MM_STORES=1  MM_HAS_LOOP=True
MM_1D_PARSER_READS=0      ← 1-D 解析器读不了 2-D
MM_COMPULSORY_BYTES=12582912
```

**局限一(可修)**:2-D tensor 拼写成 `tensor<64x64x!tt.ptr<f32>>`,需要多一维解析;
另外 `tt.load` 在 `scf.for` 循环体内,必须乘上**循环次数**(K/BK = 16)和 grid(256)。
这是工程量,不是障碍。

**局限二(不可修,必须写在结论里)**:tiled matmul 会**反复重读** A/B 的 tile,
所以**逻辑字节数会远超必需字节数**。这个 case 里必需字节是 12582912,
而逻辑字节 = per_instance × 16 × 256,量级上会高一个数量级。
⇒ **它测的是"这个候选让内存系统看到了多少请求",不是"DRAM 实际搬了多少"。**
L2 把大部分重读吸收掉了,而**逻辑字节看不到 L2** —— 这正是
`pct-of-dram-peak-counts-l2-hits-as-dram-traffic` 记录的同一个缺陷(实测可虚高到屋顶 287%)。

### 3.4 那它到底有什么用 —— 一个诚实的定位

**⚠ 本节早先建议"把它做成 `logical_bytes` 维度"。实测否证,已撤回。见 §3.5。**

| 问题 | 逻辑字节能否回答 |
|---|---|
| 这个候选实际占了多少 DRAM 带宽? | ❌ 不能(看不到 L2) |
| 这个候选比另一个候选**多发起**了多少内存请求? | ✅ 能,而且**逐候选逐参数** |
| 它离物理屋顶多远? | ❌ 不能(分母仍需真实流量) |

### 3.5 决定性实测:它是 tile knob 的代数改写,**不能做成维度**

`scripts/probes/logical_bytes_vs_tile_knobs.py`,box1,真实 tiled matmul(1024³ fp16,18 个 tile 组合):

| 相关性 | ρ | 读法 |
|---|---|---|
| `per_instance` vs tile 乘积 `BM*BN*BK` | **+0.978** | **几乎就是同一个量** |
| 总量 vs `shared_bytes`(编译器给的) | **−0.897** | 强相关,新信息很少 |
| 总量 vs tile 乘积 | **−0.871** | 同上 |
| 总量 vs 延迟 | −0.307 | 中度 |

**根因是代数的,不是测量精度问题。** tiled matmul 的逻辑总量约化为
`M*N*K*width*(1/BM + 1/BN)` —— **只是 tile knob 与问题形状的函数**。
而 tile knob 采样器本来就直接在调,`shared_bytes` 本来就在跟。
⇒ **它是 knob 的第三个名字,不是一个独立的资源维度。**

**第二个否证点:逐元素算子上它恒定。** BLOCK 取 256/512/1024/2048,总量恒为 201326592
—— 这既是正确性性质,也正是致命处:**没有 knob 能移动它,就没有 knob 能被它截断**
⇒ 结构上不可能成为墙。

**第三:L2 盲区被量化了。** 逻辑总量 / 必需字节 = **12.0x 到 32.0x**(18 个组合全部)。
重读绝大多数命中 L2,而逻辑计数**分不出哪些命中** ⇒ 也当不了饱和度的分子。

**这是同一个陷阱第三次出现**:`pct_of_dram_peak` 对 1/latency ρ=+1.000、
SOL 头寸对 1/latency ρ=+1.000,这次是对 tile 乘积 ρ=+0.978。
**共同形态:一个新"维度"其实是已有量的代数改写。** 添加任何新维度前必须先问
"它与哪个已有量相关",这个项目里三次都是这条被跳过。

**`ir_derived_bytes.py` 本身仍有价值** —— 它在两个已知真值的 kernel 上 ratio 1.0000、
正对照通过,是一个可靠的**核对工具**(例如验证某次改写是否真的减少了访存请求),
只是**不该升级成一个参与绑定计数的维度**。

---

## 4. 五条路的汇总

| 路 | 能给什么 | 本机可用性 | 判断 |
|---|---|---|---|
| **Nsight Compute (ncu)** `dram__bytes.sum` | **真实** DRAM 字节、FLOP、L2 命中率 —— 唯一的完整答案 | ❌ **实跑被拒**:`RmProfilingAdminOnly=1` + 容器缺 `cap_sys_admin`/`cap_perfmon`,**2/2 台实例相同** | **需要裸机或有 `CAP_SYS_ADMIN` 的实例**;租用容器无解 |
| **Nsight Systems (nsys)** | 时间线、kernel 序列;**不给** 计数器 | ❌ 未安装 | 即使装上也不给字节 |
| **DCGM (dcgmi)** | 设备级利用率采样 | ❌ 未安装 | 设备级 ≠ 每候选;且是采样百分比不是字节 |
| **Triton Proton** | **实跑成功**,但只有 `time (ns)` | ✅ 可用但无用 | **排除** —— 又是延迟 |
| **编译期 IR 推算** | 逐候选逐参数的逻辑字节,ratio **1.0000** ×2 kernel | ✅ 可用,零成本 | ⚠️ **不能做成维度**(对 tile 乘积 ρ=+0.978,见 §3.5);**可作核对工具** |
| ~~从 tile 几何手写公式~~ | 估计值 | — | **不要做**:手写共享内存约束中位只有真值 **32%**;`BLOCK_M*BLOCK_N*stages` 在 **15/15** 个候选上可行集与不可行集重叠 |

**注意最后一行与 §3 的区别**:§3 **不是**手写公式 —— 它从**编译器自己产出的 IR**
读类型标注,和 `shared_bytes` 直接读 `metadata.shared` 是同一性质
(`triton-flash-attn-shared-memory-formula` 已证伪手写公式,而直接读 profile 是对的)。
**读编译器的输出可靠,自己推导公式不可靠** —— 这条规律在本项目第三次出现。

---

## 5. 建议

1. **`nsys` / `dcgmi` 不必安装** —— 都不给每候选字节。
2. **不要为此换机器** —— **已在 2 台 AutoDL 实例上复验为相同策略**
   (`RmProfilingAdminOnly: 1` + 缺 `cap_sys_admin`/`cap_perfmon`),所以换一台同类容器无效。
   要真计数器需要**裸机或有 `CAP_SYS_ADMIN` 的实例**,那是采购/权限问题,不是配置问题。
   (A800 已关机未能验证;开机时补一条 grep 即可。)
3. **~~值得做的是 §3 的工程化~~ —— 已撤回。** 我曾建议把 IR 解析扩到 2-D + 循环次数、
   做成新维度 `logical_bytes`。**§3.5 的实测否证了它**:对 tile 乘积 ρ=+0.978、
   对 `shared_bytes` ρ=−0.897,逐元素算子上还是常数 ⇒ 它是 knob 的代数改写,不是维度。
   **不要实现。** 若将来仍想做,必须先在真实候选上复测,且与 tile 乘积 |ρ| < 0.6 才有资格。
   `ir_derived_bytes.py` 保留为**核对工具**(验证某次改写是否真减少了访存请求)。
4. **报告里必须保留现状的负结果**:`pct_of_dram_peak` 是任务级常数、就是 1/latency 换刻度
   (ρ=+1.000, 4/4 run)。这一点不因任何新增量而改变。
5. **本轮真正的收获是一条方法论**:这是"新维度其实是已有量的代数改写"第**三**次出现
   (前两次 ρ 都是 +1.000)。⇒ **添加任何新维度前,先测它与已有量的相关性** ——
   三次都是这一步被跳过才走了弯路。
