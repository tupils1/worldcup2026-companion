# ⚽ World Cup 2026 Watching Companion / 世界杯 2026 看球伴侣

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](pyproject.toml)
[![WC2026](https://img.shields.io/badge/World%20Cup-2026%20live%20now-brightgreen.svg)](#)

**The betting system that talks me out of betting.** A daily-digest engine for the 2026 World Cup that lands in your Telegram every evening — LLM tactical briefs cross-examined by a Dixon-Coles model and the de-vigged betting market, score "sniffs" with honest probabilities, injury radar, and last-round qualification incentives. Built **enjoyment-first**: its own backtests refuted the edges it hoped for, and it says so in every digest.

**一个劝我别下注的下注系统。** 每晚推送到 Telegram 的世界杯每日简报:LLM 战术推演 × Dixon-Coles 量化模型 × 去水市场盘口三方对照、带概率的比分嗅觉、伤病雷达、末轮出线利益探测。定位**娱乐优先**——它自己的回测否定了主盘 edge,而且每天的简报里都直说。

## What lands in your Telegram / 每晚推送长这样

```text
🌍 世界杯2026 每日雷达  2026-06-11 周四
数据: 盘口快照 06-11 23:00 · 推演生成 06-11 23:15 (北京时间)

▌06-11 墨西哥(传控渗透) × 南非(均衡) [把握一般]
  相撞: 墨西哥控球主导,南非收缩防守,墨西哥边路进攻vs南非密集防线。
  剧本: 墨西哥控球围攻,南非反击机会有限,墨西哥小胜或平局。
  对位: 墨西哥边锋 vs 南非边后卫:速度突破关键
  领先后: 墨西哥先进→控节奏小胜;南非先进→全线退守,围攻难破铁桶。
  总进球: 市场小球(P大球43%) · 战术小球 → ✓战术和市场都看小球——
         没有下注角度,但剧本一致,放心看
  比分嗅觉(DC模型): 0-0 21% · 1-0 20% · 1-1 13%
  角球偏少(南非密集防守,角球机会有限) · 牌偏少(两队风格非侵略性)
```

Every claim is anchored: market lean comes from de-vigged O/U lines across real books, the model lean from a Dixon-Coles fit with Elo-anchored priors, and the LLM is whitelist-constrained so it can't invent players. When all three agree — it tells you to just enjoy the game.

每个判断都有锚:市场倾向来自多书商去水盘口、模型倾向来自 Elo 先验的 Dixon-Coles、LLM 的对位人名被白名单约束不能编。三方一致时,它会告诉你:安心看球。

![architecture](architecture.svg)

## What it does / 它做什么

Every evening (23:00 by default) a launchd job runs the full pipeline and pushes a Chinese-language digest:

- **Tactical 推演 (scout layer)** — LLM-generated team profiles + matchup briefs (formation clash, game script, key battles, swing factors), anchored to real data: Elo, recent form, *verified* current absences, venue weather. Player names are whitelist-constrained so the LLM can't invent people.
- **Three-way total-goals check** — de-vigged market O/U line (the sharpest baseline) vs a Dixon-Coles model vs the tactical lean. Divergence (⚑) is a talking point, not an auto-edge — the digest says so.
- **Score sniff / 比分嗅觉** — top-3 most likely scorelines from the DC score matrix, *with probabilities*, so you can see even the best guess is a ~1-in-8 shot.
- **MD3 incentive detector / 末轮出线利益** — the 2026 48-team format makes last-round incentives weird (mutual draws, dead rubbers, must-win shootouts). Exact 9-outcome enumeration per group, mapped to total-goals leans.
- **News/injury radar** — RSS + structured API injuries, LLM-scored for severity, with a freshness window and recovery-news cancellation (a November red card is not a June absence).
- **Honest data watermark** — every digest stamps when odds and 推演 were last refreshed; stale data gets a loud ⚠.

核心设计原则:**每个判断都要有真凭据**(引用得出锚定数据才许表态)、**市场通常比你聪明**(⚑分歧默认是看点而非 edge)、**数据几岁要标清楚**。

## Stack

Python · SQLite · Dixon-Coles (1997) with Elo-anchored ridge prior & friendly down-weighting · Monte Carlo tournament sim · DeepSeek (LLM briefs & news scoring) · The Odds API / API-Football / Open-Meteo / Polymarket · Telegram Bot API.

## Quickstart

```bash
git clone <this repo> && cd worldcup
python -m venv .venv && .venv/bin/pip install -e .        # or: uv sync

cp configs/secrets.example.env configs/secrets.env         # fill in your keys
# needs: ODDS_API_KEY, API_FOOTBALL_KEY, DEEPSEEK_API_KEY,
#        TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID (optional but the whole point)

.venv/bin/python scripts/init_db.py                        # create + seed SQLite
bash scripts/daily_refresh.sh                              # full pipeline + digest + push
```

One-off tools / 单独使用:

```bash
PYTHONPATH=src .venv/bin/python -m worldcup.strategy.scout --brief GER CUW   # one matchup brief
PYTHONPATH=src .venv/bin/python -m worldcup.strategy.group_incentives        # MD3 incentive board
PYTHONPATH=src .venv/bin/python -m worldcup.models.dixon_coles               # model sanity report
```

### Daily automation (macOS launchd)

```bash
sed "s|/path/to/worldcup|$(pwd)|g" scripts/com.worldcup.dailyrefresh.plist \
  > ~/Library/LaunchAgents/com.worldcup.dailyrefresh.plist
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.worldcup.dailyrefresh.plist
```

Telegram setup: create a bot via `@BotFather`, put the token in `configs/secrets.env`, message the bot once, then `python -m worldcup.notify.telegram --get-chat-id`.

## Design notes / 设计笔记

[TUIYAN_AUDIT_20260610.md](TUIYAN_AUDIT_20260610.md) is a frank pre-tournament audit of this system (26 verified findings) and the improvement roadmap — useful if you want to understand the failure modes of LLM-generated sports analysis (template collapse, stale-cache staleness, fact-supply gaps) and how this repo counters them.

这份开赛前夜的完整审计记录了 LLM 体育推演的典型失效模式(套话坍缩/缓存陈旧/事实供给断档)和对应的工程对策,是本仓库最值得读的文档。

## Disclaimer / 免责声明

This is a **watching companion for entertainment and research**, not betting advice. Its own backtests refuted the main-line edges it once hoped for — the honest conclusion baked into the design is: *the closing line beats you; bet small for fun or not at all.* If you gamble, know your local laws, set a hard cap, and never chase losses.

本项目是**娱乐与研究用途**的看球伴侣,不构成任何投注建议。它自己的回测否定了主盘 edge 的存在——设计里写死的诚实结论是:打不过收盘线,小注图个乐,或者干脆不下。请遵守当地法律,设硬上限,绝不追损。

## License

MIT © 2026 tupils1

---

*If this made your World Cup nights better (or saved you a bad parlay), a ⭐ helps other fans find it. / 如果它让你的看球夜更有意思(或者拦住了一张糊涂串子),点个 ⭐ 让更多球迷看到。*
