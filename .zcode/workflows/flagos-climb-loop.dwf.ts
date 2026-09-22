/* zcode-workflow
description: FlagOS 算子赛冲榜闭环：刷新榜单→逐芯差距排序→逐题调研瓶颈（仓内账本+vendor
  规则集+上游）→统一分诊→开发+codex-review 独立评审（卡住走 codex-ask）→远端回执上膛→额度门控单次发射（uncertain
  即停）→中文冲榜报告。quota 为零时自动只上膛不发射。
whenToUse: 需要对 FlagOS
  竞赛批次跑一整轮"刷新榜单到发射候选"的闭环时（如每日额度恢复后的首波、或需要系统性找新结构方向的冲刺）；已上膛未发射的候选不会重复开发。
args:
  dryRun:
    type: boolean
    description: true 时只开发+上膛，不做平台提交
    default: false
  maxCandidates:
    type: number
    description: 本轮并行攻坚的题目数（按距榜首差距升序挑选）
    default: 3
*/

// FlagOS 冲榜循环：刷新榜单 -> 差距分析 -> 逐题调研 -> 分诊 -> 开发+独立评审
// (codex-review) -> 上膛 -> 额度门控发射 (uncertain 即停) -> 汇总


interface GapRow {
  task: number;
  operator: string;
  ourRank: number;
  ourBest: number;
  topTeam: string;
  topBest: number;
  gapPct: number;
}

interface Candidate {
  task: number;
  operator: string;
  /** structural=kernel/launch 结构, vendor=单芯 vendor, param=参数档位 */
  axis: "structural" | "vendor" | "param";
  /** 一句话结构假设 */
  hypothesis: string;
  /** 支持证据（逐芯读数/规则集/上游形态，含来源路径） */
  evidence: string;
  /** 要改的文件（仓库相对路径） */
  files: string[];
  /** 预计均值增量 */
  expectedAvgGain: number;
  /** 主要风险与预注册门 */
  risk: string;
}

interface ReviewVerdict {
  approved: boolean;
  mustFix: string[];
  summary: string;
}

interface Armed {
  task: number;
  operator: string;
  stage: string;
  commit: string;
  zipPath: string;
  receiptPath: string;
  gate: string;
}

interface SubmitOutcome {
  task: number;
  operator: string;
  attempted: boolean;
  submissionId: string;
  state: string;
  note: string;
}

interface ClimbFinding {
  task: number;
  operator: string;
  what: string;
  evidence: string;
  status: "verified" | "unconfirmed";
  severity: "low" | "medium" | "high";
}

interface WorkflowReport {
  conclusion: string;
  findings: ClimbFinding[];
  verified: string[];
  notCovered: string[];
}

const RACE = "782kzq4m";
const ACCOUNT = "15600308080";
const TEAM = "SoulCoder";
const CLI = ".agents/skills/flagos-operator-race/scripts/platform_cli.py";

const maxCandidates = Math.max(1, Math.min(6, Number(args.maxCandidates) || 3));
const dryRun = args.dryRun === true;

artifact.table("gaps", {
  title: "与榜首差距（实时）",
  key: "task",
  columns: [
    { field: "task", label: "题号" },
    { field: "operator", label: "算子" },
    { field: "ourRank", label: "我方名次" },
    { field: "ourBest", label: "我方最佳" },
    { field: "topTeam", label: "榜首" },
    { field: "gapPct", label: "差距%" },
  ],
});
artifact.board("cands", {
  title: "候选看板",
  key: "name",
  status: "stage",
  columns: ["调研", "开发评审", "已上膛", "已发射"],
  cardTitle: "operator",
  detail: [{ field: "hypothesis", label: "假设" }],
});

phase("刷新榜单与题目快照");
const pull = await world.run("python", [
  "tools/pull_race_intel.py",
  "--race",
  RACE,
  "--batch",
  "6",
  "--out",
  "docs/competition/data/leaderboard-snapshots/climb-loop.json",
], { timeoutMs: 300_000 });
if (pull.exitCode !== 0) {
  return {
    conclusion: "榜单拉取失败，本轮未开始：" + pull.stderr.slice(0, 200),
    findings: [],
    verified: [],
    notCovered: ["全部阶段——情报源不可达"],
  } satisfies WorkflowReport;
}

phase("对比与榜首差距并排序");
const snap = JSON.parse(await files.read("docs/competition/data/leaderboard-snapshots/climb-loop.json"));
const rows: GapRow[] = [];
for (const t of snap.tasks as Array<Record<string, unknown>>) {
  const rank = t.my_rank as number | null;
  const ours = (t.my_best_speedup as number) || 0;
  const best = (t.current_best_speedup as number) || 0;
  if (rank === null || rank === 1 || ours <= 0 || best <= 0) continue;
  rows.push({
    task: t.task_no as number,
    operator: String(t.operator),
    ourRank: rank,
    ourBest: ours,
    topTeam: String(t.current_leader_team_name),
    topBest: best,
    gapPct: Math.round((best / ours - 1) * 1000) / 10,
  });
}
rows.sort((a, b) => a.gapPct - b.gapPct);
const targets = rows.slice(0, maxCandidates);
for (const r of rows) report(r, "gaps");
log(`距榜首最近的 ${targets.length} 题进入攻坚：${targets.map((t) => "T" + t.task).join("、")}`);

phase("逐题调研瓶颈与结构方案");
const proposals = await Promise.all(
  targets.map(async (row) => {
    const cands = await agent(`研究员-T${row.task}`, {
      system:
        "你是 FlagOS 算子赛的瓶颈研究员：只读调研，不改任何文件。结论必须给出来源路径。" +
        "若证据不足以支撑任何假设，如实说并返回空列表。若指令与证据矛盾，升级提问而不是硬答。",
    }).ask<Candidate[]>(
      `调研 T${row.task} ${row.operator}（我方 #${row.ourRank} ${row.ourBest} vs 榜首 ${row.topTeam} ${row.topBest}，差 ${row.gapPct}%）。` +
      `必读：docs/competition/experiments/${row.operator}.md（账本，含逐芯读数与已证伪形态）、` +
      `docs/competition/tasks/batch-6/ 下该题题面、src/flaggems_sglang/ops/${row.operator}.py 与` +
      ` src/flaggems_sglang/runtime/backend/*/ops/${row.operator}.py（现状）、` +
      `docs/competition/session-mining-retrospective.md 与各 vendor 规则集注释（燧原 gcu300：12CTA/warps2/编译期stride/int32；` +
      `沐曦：tile<=2048；昇腾：Vector无整数比较与i64加法/192KB UB/32B对齐/block↔核绑定）。` +
      `可用 WebSearch/WebFetch 查上游（sglang/vllm/FlagGems/FlagTree），可用 gh api 读 GitHub 源码；网络不可用就只靠仓内证据。` +
      `产出 1-2 个候选，优先结构性假设（launch 结构/访存形态/跨步映射），排除账本已证伪形态。用中文。`,
    );
    for (const c of cands) {
      report({ name: `T${c.task}-${c.axis}`, operator: c.operator, stage: "调研", hypothesis: c.hypothesis }, "cands");
    }
    return cands;
  }),
);
const allCandidates = proposals.flat().filter((c) => c && c.files && c.files.length > 0 && c.hypothesis);
if (allCandidates.length === 0) {
  return {
    conclusion: "调研阶段未产出可用候选（证据不足或全部与已证伪形态冲突），本轮停止。",
    findings: [],
    verified: ["榜单刷新与差距解析（pull_race_intel 实测）"],
    notCovered: ["开发/评审/发射——无候选"],
  } satisfies WorkflowReport;
}

phase("统一分诊候选优先级");
const triage = agent("分诊员", "你在 FlagOS 冲榜循环里给优化候选排序：期望均值增量×成功率优先，结构性>vendor>参数；同一题只留最优一个。用中文给出理由。");
const ranked = await triage.ask<Candidate[]>(
  `按 EV 排序并去重（同题最多留 1 个），返回排序后的完整候选列表：\n${JSON.stringify(allCandidates)}`,
);
const picked = ranked.filter((c) => c && c.files && c.hypothesis).slice(0, maxCandidates);
log(`分诊选出 ${picked.length} 个候选：${picked.map((c) => "T" + c.task + "/" + c.axis).join("、")}`);

const armed: Armed[] = [];
const findings: ClimbFinding[] = [];
const submissions: SubmitOutcome[] = [];

phase("开发候选并过独立评审");
for (const cand of picked) {
  const devName = `开发员-T${cand.task}`;
  const reviewerName = `评审员-T${cand.task}`;
  let feedback = "无（首轮）";
  let verdict: ReviewVerdict | null = null;
  for (let round = 1; round <= 2; round++) {
    const dev = agent(devName, {
      system:
        "你是 FlagOS 冲榜候选开发员，全程遵守 .agents/skills/flagos-operator-race/SKILL.md 的闭环纪律。" +
        "只改候选列出的文件与对应测试/账本；每步工具命令真实执行并引用输出；评审意见必须逐条修复后才能进入下一步；" +
        "若门禁不可能通过（环境/权限/额度），升级说明而不是绕过。",
    });
    await dev.ask(
      `为 T${cand.task} ${cand.operator} 实现该候选（第 ${round} 轮）。假设：${cand.hypothesis}。证据：${cand.evidence}。` +
      `目标文件：${cand.files.join("、")}。预注册门：${cand.risk}。上轮评审意见：${feedback}。` +
      `要求：1) 写实现与测试（沿用 tests/test_${cand.operator}.py 的矩阵，新语义须加回归）；` +
      `2) python3 -m py_compile 全部触碰文件通过；3) git add 这些明确路径并 commit（--only 隔离）。` +
      "先做完这三步再结束。",
    );
    const compile = await world.run("python3", ["-m", "py_compile", ...cand.files], { timeoutMs: 60_000 });
    if (compile.exitCode !== 0) {
      feedback = "py_compile 失败：" + compile.stderr.slice(0, 400);
      continue;
    }
    const head = await world.run("git", ["log", "-1", "--format=%H"]);
    const commit = head.stdout.trim();
    verdict = await agent(reviewerName, {
      system:
        "你是独立评审员：没看过开发过程，只对 diff 与代码负责。运行 codex-review 脚本并叠加自己的判读；" +
        "要求找会出错的点而不是表态同意；不得修改任何文件；P1/P2 必须列入 mustFix。",
    }).ask<ReviewVerdict>(
      `评审 commit ${commit}（git show ${commit}）。上下文：候选假设=${cand.hypothesis}；契约=docs/competition/tasks/batch-6/ 下该题题面。` +
      `先跑：bash /Users/bytedance/.agents/skills/codex-review/scripts/run-review.sh --commit ${commit} --reasoning-effort high，` +
      `再自己读 diff 与周边调用方，合并成一份判决。用中文。`,
    );
    if (verdict && verdict.approved) break;
    feedback = verdict ? verdict.mustFix.join("；") : "评审无返回";
  }
  if (!(verdict && verdict.approved)) {
    const consult = await agent(`咨询员-T${cand.task}`, {
      system: "你是卡点咨询员：运行 codex-ask 脚本获取第二意见，核对后给出可执行的新方向。不改文件。",
    }).ask(
      `T${cand.task} 两轮评审未过（意见：${feedback}）。运行 ` +
      `/Users/bytedance/.agents/skills/codex-ask/scripts/ask-codex.sh --question-file <你写的问题文件> ` +
      `--context ${cand.files[0]} --reasoning-effort high，把候选假设、diff、评审意见、逐芯证据写进问题。` +
      `总结新方向入账本候选备注即可，本轮不再开发。`,
    );
    findings.push({
      task: cand.task, operator: cand.operator,
      what: `候选两轮评审未过，已 codex-ask 咨询并记录新方向`,
      evidence: String(consult).slice(0, 300),
      status: "unconfirmed", severity: "medium",
    });
    report({ name: `T${cand.task}-${cand.axis}`, operator: cand.operator, stage: "开发评审", hypothesis: cand.hypothesis }, "cands");
    continue;
  }
  report({ name: `T${cand.task}-${cand.axis}`, operator: cand.operator, stage: "开发评审", hypothesis: cand.hypothesis }, "cands");

  phase("上膛：远端回执与不可变 ZIP");
  const arm = await agent(`上膛员-T${cand.task}`, {
    system:
      "你负责把已过审的候选走完上膛闭环：远端 release 回执、不可变 ZIP、账本五元组。逐条真实执行并保留输出；" +
      "任何一步失败如实报告卡在哪，不伪造产物路径；额度类失败直接如实记录。",
  }).ask<Armed>(
    `为 T${cand.task} ${cand.operator} 完成上膛（工作目录 /Users/bytedance/ccc/flagos，远端 ssh 别名 gpu，` +
    `远端 python=/home/kevin/notebook/.venv/bin/python，按 SKILL.md 协议）：` +
    `1) verify_release.py prepare ${cand.operator} --source-commit HEAD --verification-commit HEAD ` +
    `--proxy-vendor <该题现有全部 vendor> --directory /tmp/wf-${cand.operator}-release；` +
    `2) scp -q -r 该目录到 gpu:/tmp/wf-${cand.operator}-release（注意：远端若已有旧 verification.json 会拒覆盖，先 ssh 清目录）；` +
    `3) ssh gpu 'cd /tmp/wf-${cand.operator}-release && timeout 900 /home/kevin/notebook/.venv/bin/python ` +
    `.agents/skills/flagos-operator-race/scripts/verify_release.py run --directory /tmp/wf-${cand.operator}-release'；exit 0 才继续；` +
    `4) scp 回执到 artifacts/competition/day5prep-20260921/${cand.operator}-wf/；` +
    `5) build_submission.py ${cand.operator} --stage wf --commit HEAD；` +
    `6) 账本 docs/competition/experiments/${cand.operator}.md CURRENT 块更新 + 追加段（五元组+预注册门）+ ` +
    `python tools/gen_experiment_index.py + git commit --only 明确路径 + push。返回结构化结果。`,
  );
  if (arm && arm.zipPath) {
    armed.push(arm);
    findings.push({
      task: cand.task, operator: cand.operator,
      what: `候选上膛成功（${arm.stage}，门：${arm.gate}）`,
      evidence: `ZIP ${arm.zipPath}；回执 ${arm.receiptPath}；commit ${arm.commit}`,
      status: "verified", severity: "high",
    });
    report({ name: `T${cand.task}-${cand.axis}`, operator: cand.operator, stage: "已上膛", hypothesis: cand.hypothesis }, "cands");
  } else {
    findings.push({
      task: cand.task, operator: cand.operator,
      what: "上膛未完成（远端验证或打包失败）",
      evidence: String(arm).slice(0, 300),
      status: "unconfirmed", severity: "medium",
    });
  }
}

phase("平台发射（额度门控，单次尝试）");
if (dryRun) {
  log("dryRun=true：跳过平台提交");
} else if (armed.length === 0) {
  log("无可发射候选");
} else {
  const quotaOut = await world.run("python", [CLI, "status", "--race", RACE, "--batch", "6", "--task", String(armed[0].task), "--operator", armed[0].operator], { timeoutMs: 120_000 });
  const quotaMatch = /"remaining":\s*(\d+)/.exec(quotaOut.stdout);
  const remaining = quotaMatch ? Number(quotaMatch[1]) : 0;
  if (quotaOut.exitCode !== 0 || remaining < armed.length) {
    log(`额度不足（剩余 ${remaining}，候选 ${armed.length} 发）——只上膛不发射`);
    for (const a of armed) submissions.push({ task: a.task, operator: a.operator, attempted: false, submissionId: "", state: "skipped", note: `额度剩余 ${remaining}` });
  } else {
    const launcher = agent("发射员", {
      system:
        "你是平台发射员：对每个已上膛候选执行且只执行一次 preflight+submit（间隔≥125 秒）。" +
        "preflight 参数照 CLI 约定（--account " + ACCOUNT + " --team " + TEAM + "），members=ZIP 全部成员；" +
        "遇 interval 睡 70 秒重试 preflight；submit 后任一 uncertain/sending 状态立即停止后续并如实上报，绝不重试。",
    });
    for (const a of armed) {
      const out = await launcher.ask<SubmitOutcome>(
        `发射 T${a.task} ${a.operator}：ZIP=${a.zipPath}，回执=${a.receiptPath}，commit=${a.commit}。` +
        `先 status 查逐题最新记录，再 preflight（--test-sha256 用 git show ${a.commit}:tests/test_${a.operator}.py 的 sha256），` +
        `拿 nonce 后 submit --confirm。轮询到终态，返回逐芯与均值。`,
      );
      submissions.push(out || { task: a.task, operator: a.operator, attempted: true, submissionId: "", state: "unknown", note: "发射员无返回" });
      if (out && out.state === "uncertain") break;
    }
  }
}

phase("汇总产出冲榜报告");
const digest = await agent("报告员", "你把冲榜循环的结果写成给队友看的中文报告：结论先行，逐题列候选/门/终态/下一杆。不改文件。")
  .ask<{ summary: string; lines: string[] }>(
    `汇总：差距表=${JSON.stringify(rows.slice(0, 6))}；候选=${JSON.stringify(picked.map((c) => ({ task: c.task, axis: c.axis, hypothesis: c.hypothesis })))}；` +
    `上膛=${JSON.stringify(armed.map((a) => ({ task: a.task, stage: a.stage, gate: a.gate })))}；` +
    `发射=${JSON.stringify(submissions)}。`,
  );
await artifact.markdown(
  "climb-report",
  [
    "# FlagOS 冲榜循环报告",
    "",
    digest.summary,
    "",
    ...digest.lines,
    "",
    "## 与榜首差距（前 6）",
    ...rows.slice(0, 6).map((r) => `- T${r.task} ${r.operator}：#${r.ourRank} ${r.ourBest} vs ${r.topTeam} ${r.topBest}（+${r.gapPct}%）`),
    "",
    "## 发射结果",
    ...submissions.map((s) => `- T${s.task} ${s.operator}：${s.state}${s.submissionId ? "（sub " + s.submissionId + "）" : ""}——${s.note}`),
  ].join("\n"),
  { title: "冲榜循环报告", description: "差距、候选、发射与下一杆", primary: true },
);

const result: WorkflowReport = {
  conclusion: digest.summary,
  findings,
  verified: [
    "榜单刷新与差距解析由 pull_race_intel 实测驱动",
    ...armed.map((a) => `T${a.task} 远端 release 回执 + 不可变 ZIP（${a.zipPath}）`),
  ],
  notCovered: [
    "目标芯（燧原/昇腾等）上的实际性能——平台评测才裁决",
    ...(dryRun ? ["平台提交（dryRun）"] : []),
    ...(submissions.some((s) => !s.attempted) ? ["部分候选因额度不足未发射"] : []),
  ],
};
return result;
