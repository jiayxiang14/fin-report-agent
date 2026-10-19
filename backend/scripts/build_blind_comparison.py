"""深度分析（Best-of-N）到底比普通分析好吗？——第一步：生成盲评材料。

对每只股票各跑一次普通分析和一次深度分析，把"普通分析的简报"和"深度分析
选出来的简报"随机打乱成A/B两份、隐去来源，生成一个本地HTML页面供人工逐对
比较。评判必须由人来做：被检验的正是"裁判选得对不对"，用LLM裁判来评等于
用被检验的东西证明它自己。

对比的是用户真实拿到的东西：普通分析用产品里普通分析的默认配置，深度分析
用产品里深度分析的完整流程（3个候选+规则分+裁判分+Reflexion）。

输出（都在 .cache/blind_comparison/ 下，不进仓库）：
    raw.jsonl        每只股票两种分析的完整结果，跑完一只追加一行（中断后重跑会跳过已完成的）
    mapping.json     每一对里A/B分别是哪种分析——评完之前不要打开
    labeling.html    浏览器打开，逐对选更好的一份，点"复制结果"把评判结果粘贴保存成 labels.json

评完后运行 scripts/eval_blind_comparison.py 算胜率。

会真实调用LLM和全部数据源（每只股票≈1次普通分析+3次Agent Loop+裁判），
不接入CI。Alpha Vantage每天25次额度：同一只股票的4次Agent Loop共用7天缓存，
每只股票最多消耗1次。

用法（在 backend/ 目录下）：
    python scripts/build_blind_comparison.py                    # 默认10只股票
    python scripts/build_blind_comparison.py --tickers AAPL,KO
    python scripts/build_blind_comparison.py --render-only      # 只用已有raw.jsonl重新生成页面
"""

from __future__ import annotations

import argparse
import asyncio
import html
import json
import random
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.agent.best_of_n import run_best_of_n  # noqa: E402
from app.services.agent.loop import run_agent_loop  # noqa: E402
from app.services.polygon_client import CACHE_DIR  # noqa: E402

OUT_DIR: Path = CACHE_DIR / "blind_comparison"

# 覆盖科技/AI产业链/消费/金融/能源/医药/境外发行人，不只挑项目开发时反复
# 用过的那几只
DEFAULT_TICKERS = ["AAPL", "MSFT", "NVDA", "MU", "KO", "WMT", "JPM", "XOM", "PFE", "NBIS"]


def _load_raw(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


async def _run_one(ticker: str) -> dict[str, Any]:
    record: dict[str, Any] = {"ticker": ticker, "timestamp": datetime.now(UTC).isoformat()}
    try:
        normal = await run_agent_loop(ticker)
        record["normal"] = {
            "final_report": normal.final_report,
            "completed": normal.completed,
            "turns_used": normal.turns_used,
            "estimated_cost_usd": normal.estimated_cost_usd,
        }
    except Exception as exc:  # noqa: BLE001 - 单只股票失败不影响其它股票
        record["normal"] = {"error": str(exc)}
    try:
        deep = await run_best_of_n(ticker)
        record["deep"] = {
            "final_report": deep.selected.final_report,
            "completed": deep.selected.completed,
            "selected_index": deep.selected_index,
            "candidate_scores": [c.total_score for c in deep.candidates],
        }
    except Exception as exc:  # noqa: BLE001
        record["deep"] = {"error": str(exc)}
    return record


def build_pairs(raw: list[dict[str, Any]], seed: int) -> tuple[list[dict[str, Any]], dict[str, dict[str, str]]]:
    """把两种分析都产出了简报的股票做成盲评对，A/B随机分配。返回
    (给评审看的条目, 揭晓用的映射)。"""
    rng = random.Random(seed)
    items: list[dict[str, Any]] = []
    mapping: dict[str, dict[str, str]] = {}
    for record in raw:
        normal_report = record.get("normal", {}).get("final_report")
        deep_report = record.get("deep", {}).get("final_report")
        if not normal_report or not deep_report:
            continue
        pair_id = f"pair-{len(items) + 1:02d}"
        if rng.random() < 0.5:
            a, b, mapping[pair_id] = normal_report, deep_report, {"A": "normal", "B": "deep"}
        else:
            a, b, mapping[pair_id] = deep_report, normal_report, {"A": "deep", "B": "normal"}
        items.append({"id": pair_id, "ticker": record["ticker"], "A": a, "B": b})
    return items, mapping


_PAGE_TEMPLATE = """<!doctype html>
<html lang="zh"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>简报盲评</title>
<style>
:root {{ --bg:#f6f7f8; --card:#fff; --fg:#1d2327; --muted:#5f6b72; --line:#dde2e5; --acc:#1f6f5c; }}
body {{ margin:0; background:var(--bg); color:var(--fg); font:15px/1.75 -apple-system,"PingFang SC","Microsoft YaHei",sans-serif; padding:24px 16px 80px; }}
.wrap {{ max-width:1280px; margin:0 auto; display:grid; gap:28px; }}
h1 {{ font-size:22px; margin:0; }} .lede {{ color:var(--muted); margin:6px 0 0; }}
.pair {{ background:var(--card); border:1px solid var(--line); border-radius:10px; padding:16px; display:grid; gap:12px; }}
.pair h2 {{ font-size:17px; margin:0; }}
.cols {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; }}
.report {{ min-width:0; border:1px solid var(--line); border-radius:8px; padding:12px 14px; white-space:pre-wrap; overflow-wrap:anywhere; max-height:70vh; overflow:auto; font-size:14px; }}
.report h3 {{ margin:0 0 8px; font-size:14px; color:var(--acc); }}
.vote {{ display:flex; flex-wrap:wrap; gap:16px; align-items:center; }}
.vote textarea {{ flex:1 1 320px; min-height:2.5em; font:inherit; padding:6px 8px; border:1px solid var(--line); border-radius:6px; }}
.bar {{ position:fixed; left:0; right:0; bottom:0; background:var(--card); border-top:1px solid var(--line); padding:10px 16px; display:flex; gap:12px; align-items:center; justify-content:center; }}
button {{ font:inherit; padding:6px 14px; border-radius:6px; border:1px solid var(--acc); background:var(--acc); color:#fff; cursor:pointer; }}
#out {{ width:100%; max-width:1280px; margin:0 auto; display:block; min-height:6em; font-family:ui-monospace,Menlo,monospace; font-size:12px; }}
@media (max-width:800px) {{ .cols {{ grid-template-columns:1fr; }} }}
</style></head><body><div class="wrap">
<div><h1>简报盲评（{n} 组）</h1>
<p class="lede">每组是同一只股票的两份简报，来源已隐去、顺序已随机打乱。按你自己的判断选更好的一份：结论是否有依据、有没有自相矛盾、有没有真正有用的判断而不只是复述数字。实在分不出高下再选"差不多"。</p></div>
{pairs}
<textarea id="out" readonly placeholder="点下方按钮后，结果会出现在这里"></textarea>
</div>
<div class="bar"><span id="progress"></span><button type="button" id="copy">复制结果</button></div>
<script>
const ids = {ids};
function collect() {{
  return ids.map(id => {{
    const picked = document.querySelector(`input[name="${{id}}"]:checked`);
    return {{ id, preference: picked ? picked.value : "", reason: document.getElementById(`${{id}}-reason`).value }};
  }});
}}
function refresh() {{
  const done = collect().filter(x => x.preference).length;
  document.getElementById("progress").textContent = `已评 ${{done}} / ${{ids.length}}`;
}}
document.addEventListener("change", refresh); refresh();
document.getElementById("copy").addEventListener("click", async () => {{
  const text = JSON.stringify(collect(), null, 2);
  const out = document.getElementById("out"); out.value = text;
  try {{ await navigator.clipboard.writeText(text); document.getElementById("copy").textContent = "已复制"; }}
  catch {{ out.select(); }}
}});
</script></body></html>
"""

_PAIR_TEMPLATE = """<section class="pair"><h2>第 {num} 组 · {ticker}</h2>
<div class="cols"><div class="report"><h3>简报 A</h3>{a}</div><div class="report"><h3>简报 B</h3>{b}</div></div>
<div class="vote">
<label><input type="radio" name="{id}" value="A"> A 更好</label>
<label><input type="radio" name="{id}" value="B"> B 更好</label>
<label><input type="radio" name="{id}" value="tie"> 差不多</label>
<textarea id="{id}-reason" placeholder="一句话理由（可选）"></textarea>
</div></section>"""


def render_page(items: list[dict[str, Any]]) -> str:
    pairs = "\n".join(
        _PAIR_TEMPLATE.format(
            num=i + 1, ticker=html.escape(item["ticker"]), id=item["id"], a=html.escape(item["A"]), b=html.escape(item["B"])
        )
        for i, item in enumerate(items)
    )
    return _PAGE_TEMPLATE.format(n=len(items), pairs=pairs, ids=json.dumps([item["id"] for item in items]))


async def main_async(args: argparse.Namespace) -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = OUT_DIR / "raw.jsonl"
    if not args.render_only:
        done = {record["ticker"] for record in _load_raw(raw_path)}
        for ticker in args.tickers:
            if ticker in done:
                print(f"{ticker}：已有结果，跳过")
                continue
            print(f"{ticker}：运行普通分析 + 深度分析……", flush=True)
            record = await _run_one(ticker)
            with raw_path.open("a") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            status = {k: ("失败：" + record[k]["error"][:60]) if "error" in record[k] else "完成" for k in ("normal", "deep")}
            print(f"  普通分析 {status['normal']}；深度分析 {status['deep']}", flush=True)

    items, mapping = build_pairs(_load_raw(raw_path), args.seed)
    (OUT_DIR / "mapping.json").write_text(json.dumps(mapping, ensure_ascii=False, indent=2))
    (OUT_DIR / "labeling.html").write_text(render_page(items))
    print(f"\n生成了 {len(items)} 组盲评：{OUT_DIR / 'labeling.html'}")
    print("评完后把结果保存成 labels.json 放在同一目录，再运行 scripts/eval_blind_comparison.py")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tickers", type=lambda s: [t.strip().upper() for t in s.split(",")], default=DEFAULT_TICKERS)
    parser.add_argument("--seed", type=int, default=20261019)
    parser.add_argument("--render-only", action="store_true")
    return asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    sys.exit(main())
