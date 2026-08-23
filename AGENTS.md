# Project agent instructions

- 需要 GPU/runtime 的开发验证优先在远端后台运行，前台继续后续工作；只读查询、
  静态审计和现有产物验签不连接远端。
- 代码或竞赛账本改完后自动 commit、push；是否触发 CI 以目标分支和仓库规则为准。
- Lark/飞书相关操作使用对应 Lark CLI skill。
- 涉及 FlagOS 算子赛题调研、跨芯开发、ZIP 打包、平台提交、八芯评测或
  vendor 迭代时，先读取并遵循
  [`.agents/skills/flagos-operator-race/SKILL.md`](.agents/skills/flagos-operator-race/SKILL.md)。
- 每次可能消耗团队额度的提交点击都只接受用户当次的一次性授权；确认必须绑定
  race ID/赛季、登录团队、batch、Task/operator、ZIP 绝对路径、完整 SHA-256 和
  实时剩余额度。实际提交产生的平台结果写入对应实验账本；只读状态查询不改文件。
