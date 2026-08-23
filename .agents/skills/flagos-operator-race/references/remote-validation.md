# 远端 GPU 代理验证

只在用户要求开发、修改或验证时连接远端；只读调研、审计或报告请求不产生远端
目录、传输或任务。当前项目默认 SSH alias 是 `gpu`，Python 是
`/home/kevin/notebook/.venv/bin/python`；先用 `ssh -G gpu` 和远端版本命令核对，
不要把解析出的 IP 写进仓库。环境变化时以 SSH config 和实际解释器为准。

## 每次运行

1. 用远端 `mktemp -d /tmp/flagos-<operator>.XXXXXX` 建独立目录并设为 `0700`。
2. 先运行打包器的 dry-run 或 existing 验签，取得 `source commit` 的成员哈希。
   只传本次 generic、vendor、`tests/test_<operator>.py` 及测试实际导入的文件，
   保留仓库相对路径。远端 generic/vendor 哈希必须等于打包器输出的 ZIP 成员哈希；
   测试文件哈希绑定 `verification commit`。传输前后任一不一致就停止，不能拿 HEAD
   源码的测试结果替旧 commit 的 ZIP 背书。
3. 用 `nohup` 在远端后台依次执行：
   - 目标源码和测试的 `py_compile`；
   - Black 79、isort、flake8；
   - `python -m unittest -v tests/test_<operator>.py`。
4. 记录远端目录、PID、日志路径和启动时间；前台继续本地源码审查、上游检索和
   账本准备。轮询日志，不用长时间阻塞 shell。
5. 单测通过后再跑题面主要 shape 的正确性、wrapper-inclusive benchmark 和编译
   资源检查；每组性能数据至少交替运行五组，保留中位数或稳定统计。
6. 记录 GPU、driver、Python、PyTorch、Triton 和 CUDA 版本，以及最大实际验证
   shape。远端 NVIDIA 结果只标记为代理证据。

使用远端临时目录中的源码和测试做验证，不在远端仓库分支提交或 push。日志和
临时目录至少保留到对应实验账本 commit 完成；若要删除，先确认账本已包含复现实验
所需的命令、版本、哈希和结果。
