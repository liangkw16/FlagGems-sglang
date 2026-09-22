# 跨芯 kernel 结构规则集（验证版）

> 来源：flagos-ai/FlagGems、flagos-ai/FlagTree、triton-ascend 官方文档（研究 agent 逐文件核实，2026-09-22）。
> 用途：冲榜循环各角色共用的证据底座；适用条件与反例随使用更新，改规则先补来源。

## 燧原 GCU（gcu300）
- grid 封顶 **12 CTA**（`max_grid_size=(12,1,1)`，超额走 grid-stride）；`num_warps=2`（官方启发式钉死）
- `max_tile_size` 32K 元素起，按位宽放大（4B×2、2B×4）
- **stride 必须编译期互整除才走 DMA**，否则 tile 缩 4 倍走非 DMA——运行时 stride 传参 = DMA 判定失败
- `enable_i64=False`：int64 寻址算术软仿真，全链 int32
- i64 原生算子全部 NOT_SUPPORT 走 CPU；非连续 stride 不受支持（参考实现 cat/copy 类在 GCU 病理 → 巨分来源）
- 反例：T80 flat-span 双流（111→13 判负）；T84 24 程序 ≠ 更优（瓶颈在别处）

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
- num_warps 阶梯：tile<2048→4；<4096→8；≥4096→16
- 反例：T84 (32,512) int32 比较矩阵 328KB>UB 三连编译败（需分块+fp32 比较+tl.range）

## 平台协议硬规则
- 反作弊代码安全扫描：**模块级全局可变容器（dict/set 缓存）直接拒收**（T77 s0 实例）
- 提交元组去重键 = (race, account, team, batch, task, operator, **zip_sha256**)——不含 stage/commit；同字节重掷必须载体 commit 改变 ZIP 字节
- uncertain/sending 元组被 CLI 永久封锁，不自动重试
- 昆仑 XPU：2D broadcast 存储踩 LLVM packing bug（用纯 1D 存储）；昆仑评测偶发确定性环境劣化（T78 0.024 双发同值）
