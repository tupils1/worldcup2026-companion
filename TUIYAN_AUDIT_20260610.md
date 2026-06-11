# 推演审计报告 — 2026-06-10（开赛前夜）

> 目标:让战术推演**更精准、更有道理**,让推演的**表达更好**。
> 方法:38 个并行审计 agent 深读全仓代码 + 只读查库 + 外部能力调研,67 条原始发现 → 去重 54 条 → 逐条对抗核实,**26 条成立、0 条被反驳**(2 条核实中断按未核实保留,26 条 P2 未核实)。
> 所有 file:line 已逐条核对过;改进路线按「该改动首次被消费的时间」倒排。

---

## A. 先救命:数据流已死(比一切优化都优先)

| # | 问题 | 证据 | 后果 |
|---|---|---|---|
| A1 | **launchd 作业从未安装**,管线 6/1 后一次未跑 | `~/Library/LaunchAgents` 下无 worldcup 作业;`MAX(odds.captured_at)=06-01`,`MAX(daily_tactics.generated_at)=06-08` | 简报的「市场基准」是 9 天前快照;⚑分歧实为「6/1 的市场 vs 6/8 的 LLM」时间差假分歧;6/6 Karl 重伤未进任何基准 |
| A2 | **injuries 表 FK 串号** — `api_football_squad.py:73` 把 API-Football 的球员 id 直插指向 `players.id` 的外键列 | `PRAGMA foreign_keys=ON` 下,开赛后 /injuries 一有数据 → 整批 IntegrityError,且被 daily_refresh 的 `\|\| true` 静默吞掉 | **明天开赛当天必触发**;结构化伤停永远进不来 |
| A3 | API-Football **~6/27 自动降 Free** 卡在小组赛收官/淘汰赛交界 | OPERATIONS.md:9 标成「不用管」 | 淘汰赛 injuries/lineups/赛果回填全部断供 |
| A4 | 管线各步 `\|\| true` 吞错,简报无任何数据时点水印 | daily_refresh.sh 全文;digest 无 captured_at/generated_at 展示 | 断档 9 天用户无从察觉(本次就是实例) |
| A5 | Telegram 429 限流被当 4xx 致命错误,中断推送 | telegram.py:109-111 | 简报变长后推一半静默断尾 |

## B. 推演为什么不够精准(量化口径带病,⚑大半是误差不是洞见)

| # | 问题 | 证据 |
|---|---|---|
| B1 | **scout 把东道主主场算中立场**(全仓唯一这么算的预测路径) | scout.py:374/450 `neutral=True` 写死;calibration.py:89-91、monte_carlo.py:131 都做了 USA/CAN/MEX 修正。揭幕战 MEX-RSA 的 P(o2.5) 0.163→0.211 |
| B2 | 模型 P(over2.5) 对市场**系统性偏低**:27 场有盘比赛 median −0.07,极端(MEX-RSA/IRQ-NOR)−0.31 | 复现:fit(prior=0.5) vs 去水市场。q_lean 被推成 under 25 / over 10 |
| B3 | lean 阈值 0.57/0.47 无据,且与市场侧 0.55/0.45 口径不一 | scout.py:379/452 vs :292;实测改对称后 47 场分类零变化(纯一致性清理,零风险) |
| B4 | 市场 neutral 时 agree **弃判**,22/47 场三方对照空转(其中 9 场量化与战术同向的信息被丢) | scout.py:488-490 显式 `base != "neutral"` |
| B5 | 市场基准无新鲜度校验:9 天前快照照样当「市场」 | scout.py:273 取 MAX(captured_at) 不看几岁 |
| B6 | λ 校准(scale 1.03/floor 0.55)只在 predict_lambda 生效,**MC 模拟器是另一套口径** | monte_carlo.py:78-103 直接用未校准 λ |
| B7 | 训练数据只含 48 强内战,λ 根基薄(Elo ridge 有缓解但 prior=0.5 与 docstring「5-20 合理」矛盾,未核实完) | historical_matches.py:66;dixon_coles.py:252 |
| B8 | R32 对阵图用过时的蛇形假设,冠军/路径概率失真(官方槽位图 2024-02 已公布) | monte_carlo.py:167-177 自承「过时」 |
| B9 | 末轮探测器 `dead` 分支不可达、`THIRD_SAFE` 定义后从未使用 → **已积 4 分大概率以最好第三出线的队会被标 must_win,「低压走过场」误判成「生死对攻↑大球」,方向完全反** | group_incentives.py:19-20/:76-77 |
| B10 | 末轮东道主三场(SUI-CAN/CZE-MEX/TUR-USA)martj42 反向孪生行重复入库 | matches id 6565/6567/6571,主客翻转+日期±1 躲过 upsert 冲突键 |

## C. 推演为什么「没道理」(事实供给脏 + 生成层无约束)

| # | 问题 | 证据 |
|---|---|---|
| C1 | **缺阵判定永不过期、无 recovery 抵消**:C罗 2025-11 红牌旧闻被当「当前缺阵」,驱动 POR 四场推演 under 并固化进缓存 | scout.py:210-215 新闻路径无时间窗;39 条 recovery 行已被 LLM 识别却从不抵消 |
| C2 | **8 队零球员数据(含 MEX!)**,28/48 队 ≤2 人;QAT 锚定全是 2023 旧赛季 → 已退队的 Al Haydos 还在打对位 | players/player_season_stats 按队统计 |
| C3 | key_battles **83% 是位置占位符**(141 条中 117 条至少一侧是「X边锋 vs Y边后卫」),且出现编名(R. Freuler→「Ricardo Freuler」)、**把刚果(金)写成科特迪瓦**(POR-COD 场) | match_tactics 全量扫描 |
| C4 | **推演模板坍缩**:48 场里「反击」46 场、「边路」46 场、「收缩」40 场;archetype 96 槽「均衡」28 个;clash_zh 与 script_zh 互相复读;total_goals 48 场**零 neutral**(永选边,稀释⑚⚑含金量) | match_tactics 全量文本统计 |
| C5 | matchup 置信度恒 0.7/0.8/0.85 三档(47 场),无信息量;digest 还用 `:.1f` 把 0.85 显示成 0.8 | daily_tactics GROUP BY conf;daily_digest.py:207 |
| C6 | **缓存签名只含缺阵**:开赛后近 5 场战绩/天气天天变,但 form/wx 在 early-return 之后根本不会被计算 → 赛中推演必陈旧;profile 也无限期复用 06-01 版本 | scout.py:299-304 |
| C7 | 首发 XI 端点 `fetch_lineup` 写好但**全仓零调用**;赛中实时比分/事件/单场球员数据零摄取;**无任何赛后复盘闭环**(matches 连角球/牌列都没有,已摄取的 home_xg 全库零消费) | api_football_squad.py:94;ingest 全层 |
| C8 | LLM 评新闻只看标题(summary 传空) | llm_news_scorer.py:223-224 |
| C9 | MATCHUP prompt 未给 style 分数语义(defensive_line=4 是「压高」不是「防得好」)、无 Elo 差锚点、无引用义务 → 套话没有代价 | scout.py:122-150 vs SYS:57 |

## D. 表达问题(简报像风控播报,不像懂球老友)

| # | 问题 | 证据 |
|---|---|---|
| D1 | **没有开球时间**(中国用户第一刚需「熬夜还是早起」):摄取层把 kickoff 截成 date-only | api_football.py:231;matches 表无时间列 |
| D2 | 单日限 4 场:06-14/17/21/23 各 5 场、06-24/25 各 7 场、06-27 共 8 场,**必静默丢场** | daily_digest.py:193 LIMIT 4 |
| D3 | 整篇塞单一 `<pre>` 等宽墙:中文 15-17 字硬折行、零层级;Telegram 原生支持 HTML 卡片/折叠引用 expandable blockquote/spoiler 全没用 | telegram.py:88 |
| D4 | verdict 全是下注腔:「✓一致(无独立edge,跳过)」把卡塔尔×瑞士这种双方对攻好局一个「跳过」打发 — 与娱乐优先定位错位 | daily_digest.py:208-213 |
| D5 | over/under/high/low 中英夹杂像调试日志 | daily_digest.py:215-233 |
| D6 | 市场缺线回落模型时仍误称「市场」 | daily_digest.py:209-211 硬编码 |
| D7 | 8 点简报物理上拿不到首发(开球前 ~40 分钟才发布),却把「Zwane 缺阵」写成既定事实 — 措辞无分级 | 时间窗结构性错配 |

---

## 改进路线

### 第一档:今晚(开赛前)— 全部是小改+止血 ✅ 已全部完成(2026-06-10 深夜,见 git diff)
> 落地记录:launchd 已装载验证;管线已补跑(盘口/新闻/天气全部回到 06-10 快照);窗口内 6 场推演已按新规则重生成并推送 Telegram;
> kickoff_utc 已回填 72/75 场;6/26 续费云端提醒已设(`apifootball-renewal-decision`)。三场未回填 kickoff 的是 B10 的孪生行,小组赛期间随去重一并处理。

1. **管线复活**(S,最高优先):安装 launchd(`launchctl bootstrap`)并验证;`\|\| true` 改为记录失败;odds 新鲜度断言;**digest 头部加数据时点水印**(盘口快照时间+推演生成时间,>24h 标 ⚠);telegram 429 重试+片间 sleep。→ A1/A4/A5
2. **事实层止血**(S):scout 主场口径 `neutral = home not in {"USA","CAN","MEX"}`(两处一行);阈值统一 0.55/0.45;缺阵新闻加 21 天时间窗 + recovery 抵消;injuries 串号修复(AF id → 内部 id 映射,单行 try/except)。→ B1/B3/C1/A2
3. **「一眼假」修复**(S):generate_profile/matchup 注入中文队名词典(「POR=葡萄牙,勿与 CIV=科特迪瓦混淆」);key_battles 白名单铁律(只许用清单原文名,禁展开缩写);MATCHUP_SYS 补 style 分数语义行;生成后校验(名单外人名降级、串队名重试)。→ C3/C9
4. **表达速赢**(S):lean 全中文化(大球/小球/偏多/偏少);verdict 改四档朋友话术(含 D6 来源修正、B4 的 neutral 不再弃判改为三方并列);置信改档位文字(把握较高/一般/纯参考);`kickoff_utc` 增列开始积累数据。→ D4/D5/D6/B4
5. 改完跑 `scout --daily --days 3 --force` 重刷被假缺阵污染的场次;OPERATIONS.md 的 6/27 降级改为「6/26 前续费决策」+ 设提醒。→ A3

### 第二档:小组赛期间

6. **缓存时效化**(S,6/12 上):form/wx 上移到缓存检查前、进签名(md5);profile 按「该队有新完赛」失效。→ C6
7. **简报 HTML 卡片化**(M,6/12-14,留 --plain 回退):一场一卡(可见两行=对阵+北京时间+★看点+一句结论,折叠引用收剧本/对位/变量/角球牌);未来 36h 全场次不丢场、按「今晚/明晨」分组;比分嗅觉用 spoiler。→ D1/D2/D3
8. **赛后对答案闭环**(L,最大单点):赛后统计摄取加角球/牌列;新增 eval/tactics_review.py 逐场记「市场/模型/战术」三方命中;简报加「昨日对答案」板块+累计计分板;实际 xG/比分回流 _team_form(「本届已赛: 2-1胜南非(xG 1.8-0.6)」)。**判断开始负责,套话开始有代价** → C7
9. **MATCHUP prompt 证据化**(配合 8 的数据再调,勿盲调):Elo 差锚点输入、why_zh 引用义务(引用不出就必须 neutral)、禁用词表、clash/script 合并为 how_zh、置信程序化锚定 min(matchup, profile_h, profile_a)。→ C4/C5/C9
10. **首发速报**(M):接线 fetch_lineup,开球前 ~1h 静默推「首发速报」,逐条核对早间推演前提(✓在列/✗替补+一句修正)。两段式仪式:早上读剧本,开球前看前提。→ C7/D7
11. **末轮探测器修复**(6/20 前,末轮 6/21 开判):third_safe 状态补全、dead 复活、THIRD_SAFE 接线;孪生行去重(6/21 前)。→ B9/B10

### 第三档:淘汰赛前

12. **R32 官方对阵图查表**(6/26 前):configs/r32_bracket.json 落官方槽位+最佳第三分配表,消费方零改动受益。→ B8
13. MC 模拟器 λ 校准统一(B6)、模型偏差平移校正(B2,以小组赛实赛复盘为据)。

### 今晚别动清单(风险>收益)

- ❌ 扩 teams 全量重 ingest + 重 fit + elo_prior 重校准(全链路回归,白天再做)
- ❌ MC 模拟器口径统一 / hybrid 接入(改变简报数字口径,需对照回测)
- ❌ Telegram HTML 大改版(唯一输出通道,开赛夜只加 429 重试)
- ❌ match_date 合并键语义改造(kickoff_utc 走纯增列即可,动键会再造孪生行)
- ❌ 反套话禁用词表/模型偏差平移(无实赛复盘样本时盲调比不调危险)

---

## 新版推演样例(目标格式,墨西哥×南非)

```
<b>▌揭幕战 墨西哥 × 南非</b> · 明早09:00(北京) ★★★★
看的不是进球是仪式感;市场五五开,无下注角度,安心看球
▼ 点开细节(折叠)
怎么踢: 球权基本归墨西哥(近5场3胜2平只丢1球),边路起球+肋部短传怼南非30米区;
  南非近5场0胜(进5失7)、3月底后无正式比赛,只能收缩等反击第一传。
对位: Zwane(南非进攻核心,难替代)vs 墨西哥双后腰——他被掐死,比赛变半场演练。
变量: 截至6/10两队无已知硬缺阵(首发开球前40分钟才出,以速报为准)。
角球偏少(南非解围果断不送角)· 牌偏少 · 把握:一般(南非档案数据薄)
🫣 比分嗅觉: ‖2-0‖(spoiler,纯娱乐非模型输出)
```

对照旧版同一场:「墨西哥控球主导,南非收缩防守,墨西哥边路进攻遇密集防线」— 同样字数,信息密度与「有真凭据」的差距一目了然。每个判断都点名引用了一条输入事实(近5场数字/Elo差/缺阵/档案标签),引用不出来就必须弃权 — 这是「更有道理」的 prompt 合同。

---

*两条未核实完的发现(elo_prior=0.5 依据、AET/PEN 完赛识别)按存疑保留,动手前先自查。*
