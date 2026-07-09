# Pokémon TCG AI Battle Challenge — 戦略レポート / Strategy Report

> Simulation Track 提出（`submission/main.py` + `submission/deck.csv`）の設計思想・検証結果をまとめたレポート。
> Strategy Track での提出を想定し、日本語版・英語版の両方を1ファイルに収録している。
>
> A report on the design rationale and validation results behind our Simulation Track
> submission. Written for the Strategy Track; both a Japanese and an English version are
> included in this single file.

---

## 目次 / Table of Contents

**日本語**
1. [概要](#概要)
2. [コンペ環境のリバースエンジニアリング](#1-コンペ環境のリバースエンジニアリング)
3. [デッキ設計](#2-デッキ設計)
4. [エージェント戦略](#3-エージェント戦略ルールベース意思決定)
5. [検証方法と結果](#4-検証方法と結果)
6. [実戦データからの改善](#5-実戦データからの改善)
7. [現状と今後の課題](#6-現状と今後の課題)

**English**
1. [Overview](#overview)
2. [Reverse-Engineering the Competition Engine](#1-reverse-engineering-the-competition-engine)
3. [Deck Design](#2-deck-design)
4. [Agent Strategy](#3-agent-strategy-rule-based-decision-making)
5. [Validation Methodology and Results](#4-validation-methodology-and-results)
6. [Improvements Driven by Real Match Data](#5-improvements-driven-by-real-match-data)
7. [Current Standing and Future Work](#6-current-standing-and-future-work)

---

# 日本語版

## 概要

[Pokémon TCG AI Battle](https://www.kaggle.com/competitions/pokemon-tcg-ai-battle) は、
Kaggleのシミュレーション環境（内部名 `cabt`）上でポケモンカードゲームを自動対戦させる形式の
コンペティションである。参加者は `agent(obs: dict) -> list[int]` という関数と60枚のデッキを
提出し、他の参加者のエージェントと自動対戦してEloランキングが決まる。

本チームは以下の方針で開発を進めた。

- **公式ドキュメントに依存せず、実エンジンを直接調査する**。`kaggle-environments` パッケージに
  同梱されている本物のネイティブエンジンを使い、Kaggleへのログインやデータダウンロードなしに
  オフラインで開発・検証を行った。
- **すべての判断を実測データに基づいて行う**。デッキ選択、方策のチューニング、バグ修正のすべてを、
  実エンジンでの自己対戦、および実際にKaggle上で行われた対戦のリプレイ解析に基づいて決定した。
  推測や「たぶんこうだろう」による決定は避けた。

## 1. コンペ環境のリバースエンジニアリング

このコンペは他のKaggleシミュレーション系コンペ（Connect X, Lux AI等）と比べても公式ドキュメントが
薄く、`obs`（観測データ）の構造や合法手の表現方法が文書化されていなかった。そこで、まず
`pip install kaggle-environments==1.30.1` を実行し、パッケージ内を調査した。

**重要な発見**: このパッケージには、Kaggleの本番サーバーで実際に動いているのと同じ
**コンパイル済みネイティブエンジン**（`kaggle_environments/envs/cabt/cg/libcg.so`）がまるごと
同梱されていた。さらに、このライブラリは全カード・全技のデータベースをまるごと返す
`AllCard()` / `AllAttack()` という関数を公開していた（1267枚のカード、1556個の技）。

これにより、以下がすべてオフラインで可能になった。

- 実際のカードデータ（HP、攻撃力、必要エネルギー、進化段階等）を使ったエージェント実装
- `kaggle_environments.make("cabt")` を使った本物のエンジンでの自己対戦テスト
- ランダムエージェント同士を数百戦させることによる `obs`/`action` スキーマの実地調査

調査結果は [`docs/ENGINE_NOTES.md`](docs/ENGINE_NOTES.md) に詳細をまとめている。

## 2. デッキ設計

### 2.1 検討した3案

データドリブンな比較のため、3つのデッキ案を実際に自己対戦させて比較した
（[`tools/build_deck.py`](tools/build_deck.py)）。

| 案 | コンセプト | 結果 |
|---|---|---|
| Kangaskhan swarm（自作） | 無色コストのみの基本ポケモンで統一し、進化・色マッチングの複雑さを排除 | 敗北 |
| Grimmsnarl ex（自作） | 実際の対人メタで最多プレイ率と報告されている進化ライン | 敗北 |
| **カイオーガ/フォレトス→メガユキノオー ex（採用）** | `kaggle-environments` 同梱のサンプルデッキ | **採用** |

「シンプルな構成の方がルールベースBotには扱いやすいはず」という当初の仮説は支持されず、
基本ポケモン（フォレトス、90HP）による繋ぎと、進化後の高火力アタッカー（メガユキノオー ex、
350HP・3エネルギーで200ダメージ＋被ダメージ軽減）を組み合わせたデッキの方が地力が高いという
結果になった。

### 2.2 デッキ構成（60枚）

- ポケモン10枚: カイオーガ×2、フォレトス×4、メガユキノオー ex×4
- トレーナー17枚: サーチ・ドロー・スタジアム等
- 基本{W}エネルギー33枚

詳細な採用理由は [`submission/deck.csv`](submission/deck.csv) のコメントに記載している。

## 3. エージェント戦略（ルールベース意思決定）

### 3.1 設計方針

1. **絶対にクラッシュしない・不正な選択をしない**。タイムアウトや不正選択は即敗北のルールのため、
   すべての判断ロジックを例外処理で保護し、失敗時は安全なフォールバック（最小限の合法手）を返す。
2. **実カードデータベースに基づいて判断する**。`AllCard()`/`AllAttack()` から取得した実データ
   （HP、ダメージ、必要エネルギー数、ex/メガフラグ等）を使い、盲目的なヒューリスティックを避ける。
3. **`option`の種類（進化・エネルギー付け・プレイ・攻撃・退避・ターン終了等）に基づいて判断する**。
   ゲーム内の文脈（`context`）は一部しか解読できなかったが、`option[i].type` は常に判断材料になる。

### 3.2 優先順位付けスコアリング

1ターン内の行動（`MAIN`コンテキスト）は、以下の優先順位で評価する。

```
進化(EVOLVE) > エネルギー付け(ATTACH) > カードプレイ(PLAY) > 攻撃(ATTACK) > 退避(RETREAT) > ターン終了(END)
```

同じ優先度内では、カード価値関数 `card_value()`（HP・ex/メガボーナス・進化段階・
サーチ/ドロー効果等から算出）や、攻撃の期待ダメージ（相手を倒せる場合はボーナス）で
細かく比較する。

## 4. 検証方法と結果

### 4.1 自己対戦による検証

`kaggle_environments.make("cabt")` を使い、エンジン標準のベースラインエージェント2種と
実際に対戦させて検証した（[`tools/evaluate.py`](tools/evaluate.py)）。

| 対戦相手 | 勝率 |
|---|---|
| `random_agent`（ランダム） | 87.5〜97.5% |
| `first_agent`（常に先頭の選択肢を選ぶ決定的Bot） | 35〜65%（試行間の分散大） |
| クラッシュ・不正選択 | 0件 / 100戦超 |

`first_agent`への勝率の分散が大きい理由は、この決定的Botがエンジンの選択肢配列の並び順に
依存しており、状況によって偶然「まともな手」を選んでしまうことがあるためと考えている。

### 4.2 実戦（Kaggleリーダーボード）での検証

実際のリーダーボードでの対戦は、他の参加者との実力差・メタゲームの影響を含むため、
自己対戦だけでは見えない問題を発見する上で不可欠だった。詳細は次章。

## 5. 実戦データからの改善

実際のKaggle対戦のリプレイ（JSON形式でダウンロード可能）を解析し、2つの具体的な
バグを発見・修正した。

### バグ1: 先発ポケモンの技が実質ダメージ0

先発ポケモンには常にカイオーガ（HP150、exボーナスなし）を選んでいたが、その低コスト技
「うずしお」は「トラッシュの基本水エネルギー1枚につき20ダメージ」という条件付き技で、
序盤はトラッシュが空のため**実質ダメージ0**だった。複数の敗戦リプレイで、この技を
何度も選択し続けて実質何もしていないことが確認された。

一方、ベンチで待機していたフォレトス（HP90）は同コストで確定10〜30ダメージを出せた。

**修正**: 先発ポケモン選択のスコアリングに「低コスト技の確定ダメージ」を加味し、
「見た目のステータス」だけでなく「実際に殴れるか」を評価するようにした。

### バグ2: 進化パーツより余剰エネルギーを残す誤判断

「Ultra Ball」使用時の「手札2枚を捨てる」コストで、進化ラインの要である**フォレトスを
捨ててしまい**、余っている基本エネルギーを手札に残す、という敗戦リプレイが見つかった。
エネルギーの価値評価が甘く、貴重な進化パーツとほぼ同等に見えてしまっていたことが原因。

**修正**: エネルギーカードの評価値を明確に最低にし、捨て札選択では常にエネルギーが
最優先で捨てられるようにした（デッキに33枚あるエネルギーは1枚捨てても実質無傷だが、
進化パーツは唯一無二）。

両修正とも、対戦相手固定のベースライン検証だけでは発見できず、**実際の対人対戦データが
不可欠だった**ことを強調したい。

## 6. 現状と今後の課題

修正後もまだ改善の余地は大きい。リーダーボードのEloは569〜600程度で推移しており、
確認できているリーダーボード上位（1060以上）とはまだ差がある。

今後の改善候補:

1. **デッキ再検討**: 実際の対人メタで強いと報告されているアーキタイプ（Grimmsnarl ex、
   Kangaskhan ex、Cynthia's Garchomp ex等）との比較を継続する。
2. **弱点・抵抗の反映**: 現在は攻撃ダメージを額面値のみで評価しており、タイプ相性を
   考慮していない。
3. **簡易先読み**: 「この攻撃をした場合、次の相手の番で倒され返すか」等、1手先を
   考慮した評価を導入する。
4. **リプレイ解析の継続**: 今回のように実戦データから具体的な問題を見つけて修正する
   サイクルを継続する。

---

# English

## Overview

[Pokémon TCG AI Battle](https://www.kaggle.com/competitions/pokemon-tcg-ai-battle) is a Kaggle
simulation competition in which the Pokémon Trading Card Game is played automatically inside
Kaggle's simulation environment (internal name `cabt`). Participants submit an
`agent(obs: dict) -> list[int]` function together with a 60-card deck; submissions are matched
against each other automatically and ranked on an Elo ladder.

Our approach was guided by two principles:

- **Investigate the real engine directly rather than relying on documentation.** We used the
  actual native engine bundled inside the `kaggle-environments` package to develop and validate
  entirely offline, with no Kaggle login or data download required.
- **Ground every decision in measured evidence.** Deck selection, policy tuning, and bug fixes
  were all driven by real self-play against the actual engine and by analyzing replays of real
  matches played on Kaggle — not by guesswork.

## 1. Reverse-Engineering the Competition Engine

Official documentation for this competition is thinner than for other Kaggle simulation
competitions (e.g. Connect X, Lux AI); the structure of `obs` and the legal-action representation
were undocumented. We started by running `pip install kaggle-environments==1.30.1` and inspecting
the package contents.

**Key discovery**: the package bundles the exact same **compiled native engine**
(`kaggle_environments/envs/cabt/cg/libcg.so`) that runs on Kaggle's production servers, including
`AllCard()`/`AllAttack()` functions that return the entire card and attack database (1267 cards,
1556 attacks).

This made the following possible entirely offline:

- Implementing an agent driven by real card data (HP, attack damage, energy cost, evolution
  stage, etc.)
- Running real self-play against the actual engine via `kaggle_environments.make("cabt")`
- Empirically mapping the `obs`/`action` schema by running hundreds of games between random
  agents

Full findings are documented in [`docs/ENGINE_NOTES.md`](docs/ENGINE_NOTES.md).

## 2. Deck Design

### 2.1 Three candidates compared

To make deck selection data-driven rather than a guess, we A/B-tested three candidate decks
against each other using the real engine (see
[`tools/build_deck.py`](tools/build_deck.py)).

| Candidate | Concept | Result |
|---|---|---|
| Kangaskhan swarm (custom) | All-Colorless-cost Basics, avoiding evolution/color-matching complexity | Lost |
| Grimmsnarl ex (custom) | Reportedly the most-played archetype in the real ladder meta | Lost |
| **Kyogre/Snover → Mega Abomasnow ex (adopted)** | The sample deck bundled with `kaggle-environments` | **Adopted** |

Our initial hypothesis — that a simpler deck would be easier for a rule-based bot to pilot well —
was not supported by the data. The deck combining a cheap Basic attacker (Snover, 90 HP) with a
powerful evolved finisher (Mega Abomasnow ex: 350 HP, 200 damage for 3 energy plus a same-turn
damage-reduction effect) outperformed both simpler alternatives.

### 2.2 Final deck (60 cards)

- 10 Pokemon: Kyogre x2, Snover x4, Mega Abomasnow ex x4
- 17 Trainers: search, draw, and stadium support
- 33 Basic {W} Energy

Card-by-card rationale is documented as comments in
[`submission/deck.csv`](submission/deck.csv).

## 3. Agent Strategy (Rule-Based Decision Making)

### 3.1 Design principles

1. **Never crash, never return an illegal action.** A timeout or illegal action is an instant
   loss under the competition's rules, so every decision path is wrapped in exception handling
   with a safe fallback (a minimal legal action) on failure.
2. **Decide using the real card database.** We use actual data from `AllCard()`/`AllAttack()`
   (HP, damage, energy cost, ex/mega flags, etc.) rather than blind heuristics.
3. **Reason on `option[i].type`** (evolve, attach energy, play, attack, retreat, end turn, etc.)
   rather than `select.context`, since only a subset of context codes could be identified
   empirically, while the option type is always informative.

### 3.2 Tiered scoring

Within a single turn (the `MAIN` context), actions are prioritized as:

```
EVOLVE > ATTACH ENERGY > PLAY A CARD > ATTACK > RETREAT > END TURN
```

Within the same tier, decisions are refined using `card_value()` (a heuristic combining HP,
ex/mega bonuses, evolution stage, and search/draw text) and expected attack damage (with a
bonus when it would knock out the opponent's active Pokemon).

## 4. Validation Methodology and Results

### 4.1 Self-play against the engine's baselines

We ran real matches via `kaggle_environments.make("cabt")` against the engine's two built-in
baseline agents (see [`tools/evaluate.py`](tools/evaluate.py)).

| Opponent | Win rate |
|---|---|
| `random_agent` | 87.5-97.5% |
| `first_agent` (deterministic, always picks the first listed option) | 35-65% (high variance across runs) |
| Crashes / illegal actions | 0 across 100+ games |

We believe the variance against `first_agent` comes from that baseline's behavior depending on
incidental option-array ordering in the engine, which occasionally happens to look like
reasonable play.

### 4.2 Validation against real ladder opponents

Self-play alone cannot surface issues that only appear against skilled human-designed opponents
and the real metagame. The next section covers what we found there.

## 5. Improvements Driven by Real Match Data

By downloading and analyzing JSON replays of actual Kaggle ladder matches, we found and fixed two
concrete bugs.

### Bug 1: the starting Pokemon's attack dealt effectively zero damage

We always started **Kyogre** (150 HP, no ex bonus) as active, but its cheap attack **Riptide**
does "20 damage for each Basic Water Energy card in your discard pile" — with an empty discard
pile early game, that's **zero damage**. Multiple loss replays showed the agent repeatedly
choosing this attack turn after turn, accomplishing nothing.

**Snover**, sitting unused on the bench, has a flat 10-30 damage attack at the same energy cost.

**Fix**: active-Pokemon selection now factors in the guaranteed damage of a Pokemon's cheap
attacks, not just its raw stats — so it stops picking a "bigger on paper" Pokemon whose cheap
attack is conditional or worthless.

### Bug 2: keeping spare Energy over an evolution piece

A replay showed Ultra Ball's "discard 2 other cards" cost choosing to discard a **Snover** — our
only path to Mega Abomasnow ex — while keeping a spare Basic Energy card. The value heuristic
scored plain Energy too close to a small Basic Pokemon, even though the deck runs 33 copies of
that Energy card (so any one of them is nearly free to give up) while Snover is scarce and
irreplaceable.

**Fix**: Basic/Special Energy now scores explicitly lower than every Pokemon and every other
trainer, so it's always the first thing discarded when it's among the options.

Neither of these bugs was discoverable from self-play against fixed baselines alone — **real
opponent data was essential** to find them.

## 6. Current Standing and Future Work

There is still significant room for improvement after these fixes. Our leaderboard Elo has been
running around 569-600, still below the visible top of the leaderboard (1060+).

Candidate next steps:

1. **Revisit deck choice** against archetypes reported to be strong in the real ladder meta
   (Grimmsnarl ex, Kangaskhan ex, Cynthia's Garchomp ex, etc.).
2. **Model weakness/resistance** — attack damage is currently evaluated at face value only, with
   no type-matchup adjustment.
3. **Add shallow lookahead** — e.g. "would this attack leave us open to a KO next turn?" — rather
   than purely static per-turn scoring.
4. **Keep mining match replays** — continue the cycle of finding concrete, evidence-backed issues
   from real match data and fixing them, as demonstrated in Section 5.
