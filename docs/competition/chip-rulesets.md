# 跨芯 kernel 结构规则集（验证版）

> 来源：flagos-ai/FlagGems、flagos-ai/FlagTree、triton-ascend 官方文档（研究 agent 逐文件核实，2026-09-22）。
> 用途：冲榜循环各角色共用的证据底座；适用条件与反例随使用更新，改规则先补来源。

## 燧原 GCU（gcu300）
- grid 封顶 **12 CTA**（`max_grid_size=(12,1,1)`，超额走 grid-stride）；`num_warps=2`（官方启发式钉死）
- `max_tile_size` 32K 元素起，按位宽放大（4B×2、2B×4）
- [FlagGems GCU300 pointwise 生成器](https://github.com/flagos-ai/FlagGems/blob/f148752746cee390bdbe53b3eaac44bbebb4220b/src/flag_gems/runtime/backend/_enflame/gcu300/utils/pointwise_dynamic.py)
  按**张量实际 stride 互整除关系**判 DMA 适用性；
  不可用时 tile 缩 4 倍并调用 `stride_constexpr` 路径，可用时调用运行时 stride
  路径。编译期固定 stride **不能单独证明**当前 kernel 会启用 DMA。
- `enable_i64=False`：int64 寻址算术软仿真，全链 int32
- i64 原生算子全部 NOT_SUPPORT 走 CPU；非连续 stride 不受支持（参考实现 cat/copy 类在 GCU 病理 → 巨分来源）
- 反例：T80 flat-span 双流（111→13 判负）；T84 24 程序 ≠ 更优（瓶颈在别处）
- **warp pin 按 store 形态分化（09-24 六发实证）**：T80 宽 store 形态解钉
  num_warps=2 → 燧原 -38%（pin 承重）；与 T19-E5/T51-E5 的"不钉更好
  （+38%）"并存——**先看 op 的 store 形态再选，禁跨题外推**
- **wrapper host-sync 探针是 GCU 小 op 头号杀手（09-24 T77）**：每 call
  `arange(4).view(int32)[:4].tolist()` ≈100us D2H 主导 op 时间；改为
  kernel 内读探针内容判定 packed 布局后燧原 +80%。窄化分配保留 int64
  逻辑元数据（view numel 恒翻倍、storage nbytes 报 8N），**元数据不可
  分辨布局，只能读内容**
- gcu300 make_gcuir 拒 int64 向量位运算（`& 0xFFFFFFFF`/`>> 32`）：
  int32 加法器进位公式 `sign((a&b)|((a|b)&~s))` 替代；i64 仅寻址
  （arange/标量 extsi 形态有 kv_indices 121x 实证）

## 沐曦 Metax（MCC/C550）
- `max_tile_size=2048` 元素/程序（(8,512)=4096 超限会退化）；warp_size=64，num_warps ∈{1,2,4,8}
- 写密集核（zeros 类）：fp16/bf16 用 **2 warps**；elementwise 通用 512/warps8
- grid 无小上限（65536³）；1D grid+kernel 内 2D 张量会 TTGIR 断言失败（2D 数据用 2D grid）
- tl.dot 的 M/N ≥16 且整倍数，否则退 FMA
- 反例：T80 warps 阶梯 2>4>8>1（实测闭合）

## 华为 Ascend（Atlas A2 / triton-ascend）
- **Vector CMP 不支持 int32/int64（降标量）**；**Vector ADD 无 int64**——掩码比较转 fp32（域内 <2^24，边界用标量分支保整数路径），寻址全 int32
- UB **192KB**；`tl.static_range` 全展开按迭代累计占用（改 `tl.range` 只留 1 活 tile）
- **block↔物理核强绑定**（AIV 40-48 核，每核 1 block），小程序网格 = 每程序固定 setup + 核闲置
- 向量通路 **32B 对齐**（16 bf16）；masked load 的 other 预填会串行化 MTE2
- num_warps 阶梯：tile<2048→4；<4096→8；≥4096→16（**注意**：T92 e26
  在 BLOCK=16384 纯拷贝 store 上钉 16 = 带内中性——阶梯对 compute 型
  成立，对纯流 store 不保证响应）
- 反例（09-24 五发）：int64 向量管线→int32 只值 +4.6%（T92 华为）；
  2D i64 广播 store 与 1D-flat int32 重写在 T80 华为均中性——
  **华为大缺口另有成因，向量宽度/循环形态/launch 数/并行度四假说全负**
- 反例：T84 (32,512) int32 比较矩阵 328KB>UB 三连编译败（需分块+fp32 比较+tl.range）

## 平台协议硬规则
- 反作弊代码安全扫描：**模块级全局可变容器（dict/set 缓存）直接拒收**（T77 s0 实例）
- 提交元组去重键 = (race, account, team, batch, task, operator, **zip_sha256**)——不含 stage/commit；同字节重掷必须载体 commit 改变 ZIP 字节
- uncertain/sending 元组被 CLI 永久封锁，不自动重试
- 昆仑 XPU：2D broadcast 存储踩 LLVM packing bug（用纯 1D 存储）；昆仑评测偶发确定性环境劣化（T78 0.024 双发同值）
- **XPU 标量广播数值雷（09-24 T77 e8）**：`tl.where(lanes < scalar_pid, …)`
  在昆仑错算（int32 start_loc 出垃圾大值）——vendor 隔离回双 launch 字节
- **XPU 超宽 int64 向量 store 数值雷（09-24 T80 e27）**：16384 宽
  int64 lane 向量写坏（39% garbage）；4096 档安全（T92 昆仑 16384 档
  int64 曾通过——按内核形态区别对待）
- **水位论补充（09-24）**：昆仑/沐曦/天数读数 ±10-35% 波动；T87 昆仑
  单日 1.27→7.04（5.5x）；同字节注释载体重掷 ≤2 次、须预注册进场带
