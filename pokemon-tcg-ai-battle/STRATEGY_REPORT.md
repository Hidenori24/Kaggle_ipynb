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
3. [エージェント戦略：スコアリングの構造](#2-エージェント戦略スコアリングの構造)
4. [デッキ設計：なぜ水デッキから闘デッキに変えたか](#3-デッキ設計なぜ水デッキから闘デッキに変えたか)
5. [検証方法と結果](#4-検証方法と結果)
6. [実戦データからの改善サイクル](#5-実戦データからの改善サイクル)
7. [試して失敗したアイデア（正直な記録）](#6-試して失敗したアイデア正直な記録)
8. [現状と今後の課題](#7-現状と今後の課題)

**English**
1. [Overview](#overview)
2. [Reverse-Engineering the Competition Engine](#1-reverse-engineering-the-competition-engine)
3. [Agent Strategy: The Shape of the Scoring Logic](#2-agent-strategy-the-shape-of-the-scoring-logic)
4. [Deck Design: Why We Moved From Water to Fighting](#3-deck-design-why-we-moved-from-water-to-fighting)
5. [Validation Methodology and Results](#4-validation-methodology-and-results)
6. [The Real-Match-Data Improvement Cycle](#5-the-real-match-data-improvement-cycle)
7. [Ideas We Tried and Rejected (an Honest Record)](#6-ideas-we-tried-and-rejected-an-honest-record)
8. [Current Standing and Future Work](#7-current-standing-and-future-work)

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
  推測や「たぶんこうだろう」による決定は避け、A/Bテストで裏付けが取れないアイデアは**採用しても
  すぐに撤回した**（6章で詳述）。
- **うまくいかなかったことも正直に記録する**。直感的に「これは良いはずだ」と思えたロジック変更
  やデッキ変更が、実測すると勝率を下げるケースが複数あった。それらを隠さず記録することが、
  同じ失敗を繰り返さないための最良の方法だと考えている。

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
- Kaggle上の実際の対戦リプレイ（JSON形式）を同じエンジンに再入力して、行動の再現・検証を行う

調査結果は [`docs/ENGINE_NOTES.md`](docs/ENGINE_NOTES.md) に詳細をまとめている（リプレイ解析
特有のハマりどころ——`action`は1つ前の`observation`への応答であること、`INACTIVE`側の
`observation`は古いスナップショットのまま更新されないことがある点等——も含む）。

## 2. エージェント戦略：スコアリングの構造

### 2.1 設計方針

1. **絶対にクラッシュしない・不正な選択をしない**。タイムアウトや不正選択は即敗北のルールのため、
   すべての判断ロジックを例外処理で保護し、失敗時は安全なフォールバック（最小限の合法手）を返す。
2. **実カードデータベースに基づいて判断する**。`AllCard()`/`AllAttack()` から取得した実データ
   （HP、ダメージ、必要エネルギー数、ex/メガフラグ等）を使い、盲目的なヒューリスティックを避ける。
3. **`option`の種類（進化・エネルギー付け・プレイ・攻撃・退避・ターン終了等）に基づいて判断する**。
   ゲーム内の文脈（`context`）は一部しか解読できなかったが、`option[i].type` は常に判断材料になる。

### 2.2 基本の優先順位（固定tier）

1ターン内の行動（`MAIN`コンテキスト）は、まず以下の固定優先順位で評価する。

```
進化(EVOLVE) > エネルギー付け(ATTACH) > カードプレイ(PLAY) > 攻撃(ATTACK) > 退避(RETREAT) > ターン終了(END)
```

同じ優先度内では、カード価値関数 `card_value()`（HP・ex/メガボーナス・進化段階・
サーチ/ドロー効果等から算出）や、攻撃の期待ダメージ（相手を倒せる場合はボーナス）で
細かく比較する。

### 2.3 状況に応じた優先順位の上書き（動的オーバーライド）

固定順位だけでは対応できない場面が実戦データから複数見つかったため、**単純な条件判定で
tierそのものを動的に持ち上げる**という、複雑にしすぎない形の拡張を積み重ねた。いずれも
「盤面のごく単純な特徴を1つ見て、tierを1段階上げ下げする」だけの実装で、深い探索や評価関数
の全面刷新は行っていない。

| オーバーライド | 条件 | 効果 |
|---|---|---|
| `attack_is_lethal` | この攻撃で相手アクティブを確殺できる | 攻撃のtierを最上位(11)に |
| `bench_is_thin` / `active_in_danger` | 自分のベンチが0〜1体、または自分のアクティブが低HP | ベンチにポケモンを出す/退避のtierを最上位付近(10 / 8.7)に |
| `opponent_underprepared` | 相手アクティブにエネルギーがなく、相手ベンチも0体 | 攻撃のtierをエネルギー付けより上(8.5)に |
| ベンチが薄く手札に出せるポケモンがない | 上記2条件の複合 | 大量ドロー系サポーターのtierをエネルギー付けより上(8.6)に |
| `_expected_discard_damage` / `_discard_pile_damage` | 攻撃の`damage`が0だが効果テキストに条件付きダメージの記述がある | ダメージを0のままにせず、デッキのエネルギー比率や実際の捨て札枚数から推定/実測して評価 |

最後の項目は少し補足が必要。`AllAttack()`が返す`damage`フィールドは、攻撃によっては
「額面0＝実質無効」ではなく「条件付きで、条件を満たさなければ0」という意味になっている
ことがある（5章で詳述）。これを素朴に0として扱うと、実際には強力な技を一度も選ばなくなる
という見落としが生まれる——これが実戦データから見つかった最大の改善点だった。

## 3. デッキ設計：なぜ水デッキから闘デッキに変えたか

### 3.1 最初の3案（水デッキを採用した経緯）

開発初期、データドリブンな比較のため3つのデッキ案を実際に自己対戦させて比較した
（[`tools/build_deck.py`](tools/build_deck.py)）。

| 案 | コンセプト | 結果 |
|---|---|---|
| Kangaskhan swarm（自作） | 無色コストのみの基本ポケモンで統一し、進化・色マッチングの複雑さを排除 | 敗北 |
| Grimmsnarl ex（自作） | 実際の対人メタで最多プレイ率と報告されている進化ライン | 敗北 |
| **カイオーガ/フォレトス→メガユキノオー ex（初期採用）** | `kaggle-environments` 同梱のサンプルデッキ | **採用** |

「シンプルな構成の方がルールベースBotには扱いやすいはず」という当初の仮説は支持されず、
基本ポケモン（フォレトス、90HP）による繋ぎと、進化後の高火力アタッカー（メガユキノオー ex、
350HP・3エネルギーで200ダメージ＋被ダメージ軽減）を組み合わせたデッキの方が地力が高いという
結果になった。この構成は基本エネルギー33枚（60枚中55%）と、実際のTCG構築の目安（通常
10〜15枚程度）から大きく外れた比率だったが、当時のロジックとの組み合わせではこれが最も
勝てるデッキだった。

### 3.2 本番リプレイ解析で見つかった2つの構造的弱点

その後、実際のランクマッチのリプレイ13戦（すべて敗戦）を解析したところ、水デッキには
ロジックの微調整では解決できない2つの構造的な弱点があることが分かった。

1. **ポケモンが10枚しかなく、実質的な基本ポケモンはカイオーガ2枚とフォレトス4枚の6枚のみ**。
   複数の敗戦で「ベンチが1試合を通して0体のまま、アクティブが倒されて即敗北」というパターンが
   繰り返し観測された。エネルギー比率を下げて枚数を空けようとする調整も試したが、6章で述べる
   通りこれは逆に勝率を下げた。
2. **対戦相手の方がダメージ／エネルギー比で明確に優れたアタッカーを使っていた**。例えば
   `Dragapult ex` の「Phantom Dive」は2エネルギーで200ダメージ＋相手ベンチへの追加ダメージまで
   持つのに対し、自分の「Frost Barrier」は3エネルギーで同じ200ダメージ。同じ土俵で戦うと
   コスト面で不利だった。

### 3.3 カードプール全体を調査してのデッキ乗り換え

上記2つの弱点を踏まえ、`AllCard()`/`AllAttack()` の全カードプール（ポケモンだけで約1000種）
から、ダメージ／エネルギー比の高い技を持つポケモンを機械的に抽出し、以下の基準で絞り込んだ。

- **進化が浅いこと**（基本→1回進化のみ。旧水デッキと同じ深さ）。`Dragapult ex`のような
  3段階進化は展開の遅さ・不安定さのリスクが上がるため候補から外した。
- **単色エネルギーで完結すること**。`Dragapult ex`は炎+超の2色を要求し、エネルギー付けの
  判断ロジックが複雑になる。単色なら既存のロジックがそのまま活きる。
- **`damage`フィールドが額面で高いこと**（2章で述べた「`damage:0`だが実は強い」パターンに
  頼らない、素直に強いカードを選ぶ）。

最終的に選んだのが **Riolu → Mega Lucario ex**（闘タイプ）である。

- Riolu（基本, 70HP）: `Buddy-Buddy Poffin`（HP70以下の基本ポケモンを2枚まで直接ベンチに
  出すグッズ）の対象になる——旧デッキの「ベンチが育たない」問題への直接的な回答。
- Mega Lucario ex（340HP, 進化元Riolu）: 「Aura Jab」は1エネルギーで130ダメージ、かつ
  捨て札の闘エネルギーを最大3枚までベンチのポケモンに再アタッチする（Powerglass的な効果を
  内蔵）。「Mega Brave」は2エネルギーで270ダメージ（次のターンは使用不可）。

デッキの残りは`Buddy-Buddy Poffin`・`Ultra Ball`・サーチ／ドロー系サポーター2種
（`Team Rocket's Petrel`, `Lillie's Determination`）、妨害用の`Boss's Orders`・`Judge`、
`Powerglass`で構成し、基本エネルギーは旧デッキの55%から40%へ引き下げた（この攻撃セットは
Hammer-lancheのようなエネルギー枚数依存の技を持たないため、大量のエネルギーを積む理由がない）。

新デッキは同じエージェントロジックのまま旧デッキとの直接対戦を350戦行い、**約54〜56%の
勝率**を確認した上で採用した（4章）。カード一枚一枚の採用理由は
[`submission/deck.csv`](submission/deck.csv) のコメントに記載している。

## 4. 検証方法と結果

### 4.1 自己対戦による検証

`kaggle_environments.make("cabt")` を使い、エンジン標準のベースラインエージェント2種と
実際に対戦させて検証した（[`tools/evaluate.py`](tools/evaluate.py)）。

| 対戦相手 | 勝率 |
|---|---|
| `random_agent`（ランダム） | 92〜98% |
| `first_agent`（常に先頭の選択肢を選ぶ決定的Bot） | 54〜67%（試行間の分散大） |
| クラッシュ・不正選択 | 0件 / 100戦超 |

`first_agent`への勝率の分散が大きい理由は、この決定的Botがエンジンの選択肢配列の並び順に
依存しており、状況によって偶然「まともな手」を選んでしまうことがあるためと考えている。

### 4.2 デッキ単体のA/Bテスト（乱数シード固定）

デッキやロジックの変更を評価する際、`first_agent`だけを相手にした比較は分散が大きすぎて
判断を誤りやすい。そこで、`random.seed()`を各試合前に固定した上で、変更前・変更後の
エージェントを同じシード列で対戦させる方法を多用した（`tools/build_deck.py`の
`make_deck_agent`を使い、同じ方策で異なるデッキを直接対戦させる比較も行った）。
100〜390戦単位のサンプルで、数%ポイントの差でも一貫した傾向として現れるかを確認してから
採用・不採用を決めている。

### 4.3 実戦（Kaggleリーダーボード）での検証

実際のリーダーボードでの対戦は、他の参加者との実力差・メタゲームの影響を含むため、
自己対戦だけでは見えない問題を発見する上で不可欠だった。詳細は次章。

## 5. 実戦データからの改善サイクル

実際のKaggle対戦のリプレイ（JSON形式でダウンロード可能）を解析し、方策・デッキ双方に
複数の具体的な改善を加えた。

### 5.1 主力技が一度も選ばれていなかった（Hammer-lanche問題）

ミラーマッチ（相手も同じ水デッキ）のリプレイで、相手がメガユキノオー exの「Hammer-lanche」
（デッキの上から6枚を捨て、捨てた基本水エネルギーの枚数×100ダメージ）を使って
300〜500ダメージを連発しているのに対し、自分は一度もこの技を選んでいなかった。原因は、
この技が`AllAttack()`上で`damage:0`と記録されており、`attack_score()`がそれをそのまま
「ダメージ0の技」として評価していたこと。自分のデッキのエネルギー比率（55%）から実際の
期待ダメージ（6枚×55%×100 ≈ 330）を計算するよう修正したところ、240戦のA/Bテストで
**勝率55.4% → 63.75%** という、今回の調査で最も大きな単発の改善が得られた。

同じ理屈で、カイオーガの「リップタイド」（捨て札の基本水エネルギー枚数×20ダメージ）も
`damage:0`のまま見落とされていた。こちらは自分の捨て札を実際に数えれば正確な値が分かる
（デッキ比率からの推定ではない）ため、同様に修正した。ただしこのポケモンはデッキ60枚中2枚
しかなく出番自体が少ないため、390戦のA/Bテストでは勝率への影響はほぼ中立（59.5% → 59.0%）
だった。悪化はしていないため、正確性の改善として維持している。

### 5.2 ベンチが育たず、アクティブ被KOで即敗北するパターン

13戦の敗戦リプレイを通じて、「試合を通じてベンチが一度も1体以上にならず、アクティブが
きぜつした時点でそのまま敗北」というパターンが繰り返し観測された。まずロジック側で
「ベンチが薄い・アクティブが危険なら、ベンチ展開や退避を優先する」オーバーライドを追加し
（`bench_is_thin`/`active_in_danger`、40戦で`first_agent`への勝率30%→47.5%）、さらに
「ベンチが薄く手札に基本ポケモンがない場合は大量ドロー系サポーターを優先する」を追加した
（390戦でベンチ0体のまま終わる試合の割合が46.2%→31.5%に減少、勝率への影響はほぼ中立）。

しかし、これらのロジック側の対策だけでは根本原因（デッキの基本ポケモンがそもそも6枚しか
ない）を解消できないと判断し、最終的に3章で述べたデッキの乗り換えに至った。

### 5.3 対戦相手のメタ分析

13戦の敗戦リプレイの相手デッキを一覧化したところ、`Dragapult ex`（ドラゴン）、
`Mega Gardevoir ex`（超・進化でエネルギー加速）、`Mega Lucario ex`（闘）、
`Archaludon ex`（鋼）等、ダメージ効率の高い`ex`/メガ進化アタッカーを中心にした
デッキが多数を占めていた。この観察が、3章のデッキ乗り換えでカードプール全体を
ダメージ／エネルギー比で機械的にスクリーニングする方針につながっている。

### 5.4 サーチ効果が「進化先の重複コピー」を選んでしまう問題（ようやく見つかった当たり）

6章で記録している通り、v2デッキ採用後も「負けた試合の約半数がベンチ0体で終わる」問題は
ロジック側・デッキ側それぞれ複数のアイデアを試しても解消しなかった。ここで方針を変え、
実際のローカル対戦（自己対戦ミラーマッチ）で早期のベンチ0敗北を再現し、その試合で
エージェントが打った手を1手ずつ追跡するデバッグを行った（実は本番のKaggleラダーと
同程度の頻度——ミラーマッチ150戦中55.7%——でローカルでも再現することが判明。それまでの
すべてのA/Bテストが「起きてもいない問題」を測っていたわけではなかったと確認できた）。

追跡の結果、ターン6・ベンチ0体で敗北した試合で、決定的な悪手を発見した:
ベンチが0体の状態でUltra Ballを使い、**すでに場に出ているMega Lucario exの
もう1枚のコピーをサーチしてしまっていた**——場にRioluがいなければ絶対に使えない
（進化させる元がない）ため完全に無駄な選択で、代わりにRiolu自体（Basicなので
ベンチに直接出せる）を探すべきだった。原因は`score_option`の`OPT_CARD`（サーチ・
選択系カードの評価）が、単純に`card_value()`（生のHPベース）で最良のカードを選ぶ
ようになっており、「ベンチが空でBasicが1体もいない」という盤面状況を一切考慮していなかった
ため。

`bench_is_empty(obs)`（既存の`bench_is_thin`=ベンチ0-1体、よりさらに厳しいベンチ完全0体
判定）を追加し、ベンチが完全に空の時だけ、サーチ選択でBasicポケモンに大きなボーナスを
与えて進化カードより優先させるよう修正。**`bench_is_thin`（0-1体）ではなく`bench_is_empty`
（0体のみ）で発動条件を絞ったのが重要**——0-1体で発動する版を最初に試したところ、
ベンチにまだ1体残っている状態でも発動してしまい、300戦のA/Bテストで明確に悪化した
（このデッキの勝ち筋は進化を素早く進めることなので、ベンチにまだ保険が1体残っている
状況でまで進化カードのサーチを後回しにするのは損だった）。範囲を完全ベンチ0体に絞った版は、
複数回に分けて計900戦以上のA/Bテストで一貫して改善を示した（単発のシード単位では45〜60%と
分散があるものの、600戦を1回の実行でまとめて計測した際は6バッチ全てで候補側が上回り、
合計54%）。**採用**。

この過程で、`current["result"]`フィールドが試合終了後の最後の`observation`でも
`-1`（継続中）のままになりうるという、これまで見落としていたエンジンの挙動も発見した
（`docs/ENGINE_NOTES.md`参照。勝敗判定は`current.result`ではなく、そのステップ自身の
`status`/`reward`を見る必要がある）。この修正なしでは、今回の追跡調査自体が
「負けた試合が1件も見つからない」という誤った結果を出し続けていた。

### 5.5 弱点・抵抗ダメージ補正を未実装のまま放置していた問題

`docs/ENGINE_NOTES.md`の7章に「弱点2倍・抵抗-30が実際に適用されるかは未検証」と長らく
記録されたままの項目があった。ユーザーから「持ち時間があるならロジック改善をもっと深く
考えてよい」と指示を受けたのを機に、これを実際に検証した。

固定ダメージの技（Riolu の Quick Attack、Mega Lucario ex の Aura Jab）を、`AllCard()`で
`weakness`/`resistance`が確認できる相手カード（弱点=闘のTeam Rocket's Kangaskhan ex、
抵抗=闘のIron Crown ex）を含む対戦相手デッキにぶつけ、実際のHP変化を1手ずつ追跡する
プローブスクリプトを書いて検証したところ、**弱点一致で額面ダメージのx2、抵抗一致で
-30（0未満にはならない）**という、実際のポケモンカードの現行ルールと完全に一致する
補正が確認できた。

`attack_score()`と`attack_is_lethal()`は、この検証まで額面の`damage`フィールドだけで
「攻撃で何点入るか」「これは確殺か」を判定していたため、相手が弱点で本来なら確殺できる
一撃を「まだ倒せない」と誤判定し、代わりに進化やエネルギー付けなど別の行動を優先してしまう
（`OPT_ATTACK`の基本tierは6で、`OPT_ATTACH`(8)・`OPT_EVOLVE`(9)より低いため、確殺と誤判定
されない限り攻撃より他の行動が優先される構造になっている）ケースが構造的に発生していた。
`apply_weakness_resistance()`を追加し、両関数の適用箇所に組み込んだ。

自己対戦での検証には構造的な限界があった。ミラーマッチはそもそも自分のデッキ（Riolu/
Mega Lucario exはどちらも弱点=超で闘には無関係）を相手にするため弱点・抵抗が一切発動せず、
`first_agent`の参照デッキ（Kyogre/Snover/Mega Abomasnow ex、弱点=雷/鋼）も同様に無関係
だった。そこで意図的に「闘弱点のポケモンだけで組まれた対戦相手デッキ」を用意して検証した
——過去に検討したが不採用にした候補デッキ`KANGASKHAN_SWARM`（弱点=闘の3種のカンガルーン
系ポケモンのみ）と、こちらのデッキと同じ構造（1進化・同じトレーナー構成）でElectrike→
Mega Manectric ex（弱点=闘）を使う新規デッキ`MANECTRIC_TEST`の2種。ところが両方とも
既存ロジックの時点でこちらのデッキに95〜99%勝っており、勝率という指標そのものが
天井（ceiling effect）に達してしまっていて、A/Bで有意差を測るだけの「伸びる余地」が
ほぼ残っていなかった（500戦規模でも旧98.8%・新98.8%、旧95.2%・新95.6%という具合）。

そこで勝率の代わりに、「`attack_is_lethal`の判定が旧ロジックと新ロジックで実際に食い違う
回数」を直接計測する手法に切り替えた。弱点デッキ2種（合計120戦）で**141回**の食い違いを
観測し、そのすべてが「旧ロジックでは確殺と判定できなかったが、弱点込みなら実際には確殺
だった」ケース（1試合あたり平均約1.2回）。一方、弱点・抵抗が無関係な中立デッキ2種
（`GRIMMSNARL_EX`・参照Abomasnowデッキ、合計120戦）では食い違いは**0回**——修正が
意図した通りの場面だけで発動し、無関係な場面には一切影響しないことも確認できた
（中立デッキに対する勝率A/Bも95.0%→93.3%と、120戦規模のノイズの範囲内で変化なし）。

**採用**。判断の根拠は「勝率A/Bの有意な改善」ではなく（自己対戦の練習相手が強すぎて
天井効果でそもそも測れない）、(1)補正ルール自体をエンジンに対して直接検証済み、
(2)中立マッチアップでの後退がゼロと確認済み、(3)発動する場面・回数を直接計測して
矢面から漏れていた確殺を正しく検出できることを確認済み、という3点。実際のランクマッチの
対戦相手はこちらの手持ちの練習デッキよりずっと多様なので、闘に弱い/強いポケモンを含む
デッキと当たる場面はローカルの天井効果よりも本番で意味を持つ可能性が高いと判断した。

### 5.6 「進化元Basicへの評価ボーナス」で、5連敗した2種目Basic案がようやく成立した

6章に記録している通り、「保険用の2種目Basicポケモンを足す」というアイデアは、Sawk
（3variant）・Farfetch'd（2variant）と合計5回試して全て不採用に終わっていた。ユーザーから
「もったいない負け方をしてもいいので一度試してみてもいい」と許可を得て、根本原因の
仮説——`card_value()`が全てのBasicポケモンを生のHPだけで評価するため、Rioluと同HP
（70）の2種目Basicが完全に同値になり、限られたサーチ・捨て札判断の一部がRiolu以外に
流れてしまう——を実際に修正して検証した。

`OWN_DECK_EVOLUTION_BASES`（自分のデッキの中で、他のカードの`evolvesFrom`に名前が
挙がっているBasicの集合。CARD_DBとload_deck()から動的に計算し、"Riolu"をハードコード
しない、このプロジェクトの一貫した方針に忠実な実装）を追加し、該当するBasicの評価に
+10のボーナスを与えるよう`card_value()`を修正。これによりRioluの評価は7.0→17.0となり、
同HPのFarfetch'd（7.0のまま、変化なし）との同値関係が解消された。

この修正の効果を2段階で検証した。まず修正単体の効果を分離するテスト（同じFarfetch'd
入りデッキで、修正あり/なしのロジックを比較）では、600戦で**57.8% vs 42.2%**
（6バッチ中5つが有利）という明確な改善が見られ、仮説が正しかったことを直接確認できた。
続いて、この修正込みのFarfetch'd入りデッキ全体を、現行の本番デッキ（Farfetch'dなし）と
比較した。最初はPowerglassを削る構成（5.4節で既にPowerglassの重要性が判明していたため
不利）で試したところ48.5%と僅かに届かず、代わりに**Petrel**（「トレーナーカードを
1枚サーチ」——Buddy-Buddy Poffin・Ultra Ball・Brock's Scouting・Cyranoが既に揃っている
今、最も冗長度が高いカードと判断）を削る構成に変更したところ、600戦で52.5%、追加で
1200戦を投じて52.2%、さらに1800戦で51.2%——**3回の独立した検証（合計3,600戦）が
いずれも51〜53%という狭い範囲に収まり**、Maximum Belt検証時（42%〜67%まで大きく
振れ続けた）とは対照的に安定した傾向を示した（36バッチ中22バッチが有利）。

**採用**。単発の勝率としては控えめ（51.75%）だが、(1) 修正単体の効果は57.8%という
大きく明確な数字で直接確認済み、(2) デッキ全体としての効果も3回の独立検証全てで
同じ狭い範囲に収まる一貫性を示した、という2点から、ノイズと判断せず採用に踏み切った。
`submission/deck.csv`・`DEFAULT_DECK`をPetrel抜き・Farfetch'd入りの構成
（v3）に更新し、`card_value()`の修正も本体に組み込んだ。

v3採用直後、さらなる改善余地を2つ検証したが、いずれも不採用となった。
- **Farfetch'd → Hawlucha（退却コスト0、こちらの闘エネルギー・超弱点と完全一致）への
  差し替え**: 理屈上はFarfetch'dより優れているはずだったが、600戦でv3（現行）51.5% vs
  Hawlucha版48.5%（6バッチ中3勝3敗）と、僅かに劣る結果だった。ミラー戦では両者とも
  相手が特殊エネルギーを使わないため、Farfetch'dの「相手の特殊エネルギーをディスカード」
  効果は無関係——純粋なステータス比較でもHawluchaが上回らなかったのは意外だったが、
  差自体もノイズに近い範囲。**不採用**。
- **Farfetch'dを4枚→2枚に減らし、Petrelを2枚戻す「軽量版」**: 最初の600戦では
  v3 45.0% vs 軽量版55.0%（6バッチ中5つが有利）という強い改善が見えたため、1200戦の
  追加検証を実施したところv3 50.8% vs 軽量版49.2%（12バッチ中3つのみ有利）と明確に
  反転した。合計1,800戦では51.1%とわずかに軽量版が上回るものの、バッチ有利率は
  44%（18バッチ中8つ）に低下し、最初の好結果は単なる分散だったと判断せざるを得ない。
  **不採用**——この一件は、単発の600戦だけで判断せず必ず追加検証を行うという
  このプロジェクトの方針の重要性を改めて裏付けた。

### 5.7 一般戦略のリサーチと「メガEx被撃破＝3プライズ」という重大な未知仕様の発見

ユーザーから「一般的なポケモンカードの戦略（特に上級者向け）をルールベースに落とし込み、
デッキ構築セオリー・ベストプラクティスも採用する」という方針を受け、実装前にWeb検索で
実際のポケモンカードの競技シーンの戦略を調査した。

調査結果の要点:
- **プライズトレード数学**：相手の高価値ポケモン（ex/VMAX等）を狩ってプライズを多く稼ぎ、
  自分の高価値ポケモンは守る、という損得計算が現代競技シーンの中核戦略。
- **テンポ**：エネルギー・手数を先行して盤面優位に転換する概念。
- **ガスト効果**（`Boss's Orders`等、相手のベンチを強制的にアクティブへ出す）でベンチの
  弱ったポケモンを狙い撃つのが上級テクニックとして定着している。
- **モダンな構築比率**：概ねポケモン12-16／トレーナー30-36／エネルギー8-12（60枚換算）。
  現行デッキ（ポケモン12／トレーナー24／エネルギー24）はエネルギー比率が著しく高い。
- **特殊状態**（毒・やけど・こんらん・まひ・ねむり）：現在の技構成（Aura Jab / Mega Brave /
  Quick Attack / Mach Cut）はいずれも状態異常を付与しないため、直接の適用先は見当たらない。

（出典は英語版5.7節末に一覧を記載）

このリサーチを実装に落とし込む前に、「プライズトレード数学」がそもそも本エンジンに
実在する仕組みかどうかをCARD_DB・実際の対戦ログで検証した。`obs.current.players[i].prize`
が6要素のリスト（`null`=未獲得）であることを発見し、対戦を1ステップ単位で追跡して
「どのポケモンが撃破された直後に、撃破した側のプライズ残数が何枚減るか」を100戦・
293回のプライズ獲得イベントで計測したところ、**`megaEx`（Mega Lucario ex）の撃破は
例外なく3枚減少、それ以外の無印Basic（Riolu / Farfetch'd）の撃破は例外なく1枚減少**——
293件全てが完全にこの2値のどちらかであり、ブレは一切なかった。実物のポケモンカードの
「`ex`=2プライズ」ルールとは異なり、本エンジンでは`megaEx`のみが3プライズという独自仕様
であることが確認できた。これはこれまでのプロジェクトの調査でも一度も把握していなかった、
エンジンの重大な未知仕様である。

### 5.8 プライズ価値を考慮した退却判定（`prize_value`／`active_in_danger`の閾値変更）

5.7節の発見（megaEx撃破＝3プライズ）を踏まえると、自分のMega Lucario exが撃破される
たびに相手へ一気に3プライズ（6枚中の半分）を献上していることになる。従来の
`active_in_danger()`はHP35%未満で退却を検討する固定閾値だったが、これは「撃破されても
1プライズしか献上しない」使い捨てのBasicポケモンには適切でも、3プライズを背負う
Mega Lucario exには危険すぎる（手遅れになるまで退却を検討しない）閾値だった。

`prize_value(card)`（`megaEx`なら3、それ以外は1を返す。5.7節の実測に基づく）を追加し、
`active_in_danger()`の閾値を、アクティブが3プライズ級（`megaEx`）の場合は55%、それ以外は
既存の35%のままとする条件分岐に変更した。これにより、Mega Lucario exは体力がまだ半分
以上残っている段階で使い捨てのBasic（Riolu / Farfetch'd、1プライズ）に交代し、高価値
アタッカーをより長く盤面に残せるようになる。

600戦を1回とする独立した検証を3回実施（毎回フレッシュな自己対戦、シード固定なし）:

| 実行回 | 新ロジック（プライズ価値考慮）勝率 |
|---|---|
| 1回目（600戦） | 53.8% |
| 2回目（600戦） | 55.0% |
| 3回目（600戦） | 52.5% |
| 合計（1,800戦） | 53.8% |

5.6節のv3採用時（51〜53%の狭い帯に3回収束）と同様の基準で、3回とも独立に50%を
明確に上回り、安定した改善を示した。**採用**。

今後の検討候補（今回は未実装）:
- `Boss's Orders`（cardId 1182、「相手のベンチポケモン1体を強制的にアクティブへ」——
  本カードプールに存在することを確認済み）を採用し、相手の`ex`/`megaEx`をベンチ狙撃する
  「ガスト」戦術を実装する。現行デッキは60枚中24枚（40%）がエネルギーで、リサーチで
  判明したモダンな比率（8-12枚、13-20%）よりかなり高いため、これを削ってBoss's Orders等の
  妨害トレーナーを増やす余地がある。ただしデッキ構成そのものを変える大きな変更になるため、
  別途独立したA/Bテストが必要。
- 特殊状態は現行の技構成に付与手段が一切なく、現時点では適用対象なしと判断（技構成が
  変わった場合は再検討）。

## 6. 試して失敗したアイデア（正直な記録）

「理屈の上では良さそうに見えたが、A/Bテストすると勝率を下げた」アイデアが複数あった。
今後同じ失敗を繰り返さないために記録しておく。

- **劣勢なら攻撃を優先する（`we_are_behind`）**: サイド枚数差または場のポケモン数差で
  「劣勢」と判定したら攻撃のtierを上げる、という一般的なゲーム戦略の発想。しかし
  このデッキの勝ち筋は「重いフィニッシャーへじっくりエネルギーを注ぎ込む」ことなので、
  劣勢時に無理に攻撃してエネルギー投資を後回しにするのは逆効果だった。300戦のA/Bテストで
  勝率55.3%→48.7%と明確に悪化し、削除した。
- **アクティブが最大エネルギーに達したらベンチへ分散する**: 「1体に過剰にエネルギーを
  積みすぎている」という観察は事実だったが、実装すると別の問題を生んだ。ATTACH（エネルギー
  付け）はもともとATTACK（攻撃）より常に高い優先度だったため、「ベンチに回せる」選択肢が
  常にある状態になると、アクティブが一度もATTACKせずに毎ターンベンチへエネルギーを配り
  続けてしまった。300戦で55.3%→48.7%の悪化を確認して削除。
- **デッキのエネルギーを減らしてポケモン／トレーナーを増やす（水デッキのまま）**: 「エネルギー
  55%は明らかに多すぎる」という直感から、非エネルギーカードを4枚上限まで増やしエネルギーを
  33→23枚に削った変種を試した。しかし増やしたトレーナー（サーチ系）はポケモンや進化先を
  探すだけでエネルギー自体は探せないため、「ポケモンは引けるがエネルギー不足で攻撃できない」
  という別の問題を生み、120戦で47%→43%、120戦で51%→37%と悪化。この経験（と5.1節の
  Hammer-lanche発見）から、「このデッキにとってエネルギー55%は見た目以上に合理的だった」
  と結論し、水デッキ自体の微調整ではなく3章のデッキ乗り換えに方針を切り替えた。
- **ベンチが薄い時、ポケモンを直接ベンチに出す手だけでなく「ポケモンをサーチする」
  トレーナー（Buddy-Buddy Poffin・Ultra Ball・Brock's Scouting・Cyrano）も同じ緊急tierに
  上げる**: `tools/kaggle_watch.py`が本番の実戦データから収集した負け試合の詳細（後述）を
  見ると、v2デッキ採用後も**負けた試合の約半数がベンチ0体の状態で終わっていた**——
  デッキにサーチ手段を増やしたのに、この失敗パターン自体は解消されていなかった。そこで
  `searches_for_pokemon(card)`を追加し、`bench_is_thin`時にこれらのトレーナーを
  プレイする優先度を、ポケモンを直接ベンチに出す時と同じtier(10、EVOLVEの9より上)まで
  上げてみた。しかし300戦のA/Bテスト（旧ロジック52.7% vs 新ロジック47.3%、5シード中4つが
  旧ロジック優位）で明確に悪化。おそらく原因は、ベンチがまだ1体残っている「薄いが空でない」
  状態でもこの上書きが発動し、進化(EVOLVE)より優先されてしまうため——このデッキの勝ち筋は
  Mega Lucario exへの進化を素早く進めることなので、ベンチにまだ1体いる状況で進化を後回しに
  してサーチカードを打つのは損だったと考えられる。「ベンチが完全に0の時だけ発動」に絞った
  改良版も試したが（300戦で52%程度、5シード中3勝1敗1分）、A/Bテストの一貫性という基準
  （デッキ乗り換え時の5シード全勝・55〜67%のような明確な差）には届かず、ノイズと区別できない
  弱い効果だったため、いずれも不採用として撤回した。**「負けの半数がベンチ0体」という
  実戦シグナル自体は依然解消されていない課題として残る**——次に試すべきは意思決定ロジックの
  微調整ではなく、デッキ構成側（サーチカードの密度、Riolu以外のBasic採用等）の見直しかもしれない。
- **保険用の2枚目のBasicポケモン（Sawk）をデッキに追加する**: 上記の直後、実際にその
  「デッキ構成側の見直し」を実ポケモンカードの定石（メインの進化ラインが引けない/落ちた時の
  保険として、無関係な別のBasicを少数差す、という実際の競技デッキで一般的な手法）に基づいて
  試した。カードプール全体をHP≧100（Rioluの70より頑丈）・にげるコスト≦1・1エネルギーで
  条件無しの確定ダメージを持つ・闘/無色タイプ、という条件でスクリーニングし、**Sawk**
  （HP110、1エネルギーで確定30ダメージ——コイントス依存のRioluのQuick Attackより明確に上位、
  さらに相手が{ex}ならもう1エネルギーの技で90ダメージ）を最有力候補として特定。
  Powerglass（このデッキの現在の課題に対しては比較的重要度が低いエネルギー再利用ツール）を
  削ってSawkを4枚採用する案（300戦、v2が53%・5シード中4つがv2優位）、Sawkを2枚に抑え
  Powerglassも2枚だけ削る控えめな案（300戦、v2が53.7%・5シード中4つがv2優位）の両方を
  A/Bテストしたが、いずれも改善どころかむしろ悪化。
  調査の過程で`card_value()`の実際の欠陥にも気づいた——ヒューリスティックは素のHPで
  カードを評価しているため、サーチ・ディスカード時の比較で**HP110のSawkがHP70のRioluより
  「価値が高い」と判定されてしまい**、進化ラインの本体であるRiolu自体を差し置いてSawkを
  優先してしまう可能性があった。「他のカードの`evolvesFrom`に名前が挙がっているBasicには
  ボーナスを与える」という汎用的な修正（Rioluをカード名でハードコードせず、CARD_DB全体から
  動的に「進化先を持つBasicの集合」を計算する、カード名依存を避けるこのプロジェクトの方針に
  忠実な実装）を試作しRiolu>Sawkの評価順を正しく直したが、この修正込みでも300戦のA/Bテストで
  v2が53.3%・5シード中4つが依然v2優位という結果は変わらず、**すべて撤回した**（`main.py`・
  `tools/build_deck.py`・テストとも変更なしの状態に戻し、この投稿だけをコミットしている）。
  合計で約1,200戦のA/Bテストを重ねたが、デッキ乗り換え時のような「5シード全部が明確に有利」
  という一貫した signal は一度も得られなかった。**「進化しない保険用Basicを足す」という
  実TCGの定石自体は理論的には正しいはずだが、少なくともこのエージェントの意思決定ロジックの
  現状の巧妙さでは、うまく活用できていない**可能性がある——たとえばサーチカードで
  「Rioluを優先的に探す」ような、単純なHP/ステータス評価を超えた「デッキプラン全体の理解」が
  伴わないと、保険用カード自体が本来のエンジンの引きを薄める副作用の方が大きく出てしまうのでは
  ないか、というのが現時点での仮説。

  **追記（5.6節参照）**: この時点で試した価値評価修正はSawk自体には効かなかったが、後日
  Farfetch'd——HP70でBuddy-Buddy Poffinの上限とも噛み合う——に対して同じ修正を適用した
  ところ明確に機能した。SawkはHP110のためBuddy-Buddy Poffinから最初から候補として
  提示されず、価値評価をいくら直しても「そもそも選ばれる機会」自体が他の候補より少なかった
  ことが、修正が効かなかった一因だったと考えられる。
- **エネルギー付け先に「もう十分なエネルギーがある」ガードを入れる**: 5.4節と同じ実戦データ
  （負けの一定割合がベンチ2体以上でもエネルギーが偏って残っていたパターン）から、
  「攻撃で使い切れる上限を超えてエネルギーを1体に積み続けるのは無駄ではないか」という
  仮説を立てた。`energy_need_gap()`——そのポケモンが持つ最もエネルギーコストの高い技を
  基準に「まだ何枚エネルギーを活かせるか」を計算するヘルパー——を追加し、すでに使い切れる分の
  エネルギーが付いているスロットへの`OPT_ATTACH`/`OPT_ENERGY`スコアを下げ、まだエネルギーが
  必要な他のスロット（ベンチの控えMega Lucario exなど）を優先させるよう変更した。600戦
  （1回の継続実行、6バッチ）のA/Bテストの結果は旧49.7%・新50.3%で、6バッチ中3対3の
  分裂——ノイズと区別できなかった。理由は単純で、このデッキのMega Lucario exは
  Aura Jab（1エネルギー）・Mega Brave（2エネルギー）のどちらも要求エネルギーが非常に
  少なく、そもそも「使い切れないほどエネルギーを積む」機会自体が稀だったため。**不採用**。
- **`active_in_danger`の35%固定閾値を、相手の実際の即死圏判定に置き換える**: Mega Brave
  （270ダメージ、2エネルギー）は自分自身の最大HP340の35%（119）よりずっと高い、HP120〜269
  という広い範囲で「まだ安全」と誤判定してしまう構造的なギャップがあると気づいた。
  `opponent_can_lethal_us()`——相手のアクティブが今すぐ払える技（現在の付着エネルギー数で
  判定）のうち、こちらのHPを超えるダメージを出せる技が1つでもあるかを直接計算するヘルパー
  ——を追加し、`active_in_danger`の判定に「35%未満」とのOR条件で組み込んだ。600戦
  （1回の継続実行、6バッチ）のA/Bテストの結果は旧50.7%・新49.3%で、6バッチ中3対3の
  分裂——理屈は正しいはずだが、実測ではノイズと区別できなかった。考えられる原因は2つ:
  (1) まだ十分健康な状態で早期に逃げること自体に「その分攻撃・進化が遅れる」という機会損失が
  あり、理論上の危険回避のメリットを相殺してしまった。(2) Mega Braveは「使った次の番は
  使えない」という自己制限があるが、`opponent_can_lethal_us()`はこの制限を見ておらず、
  実際には撃てない一時的な見せかけの脅威を検知してしまうケースがあった可能性がある。
  **不採用**。
- **保険用の2枚目のBasicポケモン（Farfetch'd）を、Cyrano／Powerglassと差し替える**:
  実戦データで「負けの約35%がターン3〜9という早期のブリック（事故）で終わる」ことが分かり、
  根本原因はこのデッキの基本ポケモンがRiolu×4・Mega Lucario ex×4の合計8枚しかないことだと
  特定した。前回のSawk案（3.55節参照）が失敗した一因は、SawkのHPが110でBuddy-Buddy Poffin
  の「HP70以下」という上限を超えており、最強のサーチカードと噛み合っていなかったことだと
  分析。カードプール全体をHP70以下・コイントス等の運要素なし・1エネルギー（無色または闘）で
  確定20ダメージ以上、という条件で再スクリーニングし、**Farfetch'd**（HP70、無色1エネルギーで
  確定30ダメージ＋相手の特殊エネルギーを1枚ディスカード）を新候補として特定。Buddy-Buddy
  Poffinの上限にぴたり収まり、Sawkでの失敗要因を理論上解消したはずだった。
  Cyranoを削ってFarfetch'd×4に差し替えた版（DECK_V3）を600戦A/Bテストしたところ、
  **旧デッキ64.7% vs 新デッキ35.3%、6バッチ全てで新デッキが劣勢**という、今回の調査で
  最も大きな悪化を記録した。原因を分析すると、Cyranoの「{ex}を最大3枚まで検索して手札に
  加える」効果は、実は**Mega Lucario exの複数コピーを見つけるための一貫性エンジン**として
  非常に重要だったことが判明——「進化先しか探せないから重要度が低い」という当初の判断は誤り
  だった。そこでCyranoを残し、代わりにPowerglass（エネルギー再利用ツール）を削る版
  （DECK_V4）を試したが、これも600戦で旧デッキ60.8% vs 新デッキ39.2%、6バッチ全てで
  劣勢という明確な悪化に終わった。
  Powerglassも予想以上に重要（アクティブに付けておくと毎ターン終了時に捨て札から基本
  エネルギーを1枚自動で再アタッチする——Aura Jabが自分から捨てたエネルギーを回収する
  役割を果たしていた）だったことが分かる一方、**2種類とも削って試した結果が軒並み大敗**
  という事実は、「何を削るか」よりも**Farfetch'd（2種目のBasicポケモン）自体を含めること
  自体が問題**である可能性を強く示している。Sawk（3変種）・Farfetch'd（2変種）と、今回の
  プロジェクト全体で「進化しない保険用の2種目Basicを足す」という実TCGの定石を**合計5variant
  試して全て失敗**しており、単なる個別カードの相性問題ではなく構造的な失敗パターンと見るべき
  段階に来ている。考えられる仮説: サーチ・ディスカード等の場面で`card_value()`がRioluと
  2種目のBasicを同程度に評価してしまう（HPが同じ70なら値が同じ）ため、限られたサーチ機会の
  一部がRiolu以外に流れてしまい、進化ライン（＝唯一の勝ち筋）にアクセスする実質的な確率を
  下げている可能性がある——Sawk検証時に試した「他カードの`evolvesFrom`に載っているBasicへ
  ボーナスを与える」汎用修正でも改善しなかったことを踏まえると、単純な価値評価の同点問題
  以上に、この意思決定ロジックのアーキテクチャ自体が「複数種のBasicが同居する状況」を
  苦手としている可能性が高い。**不採用**（DECK_V3・DECK_V4両方）。この軸（2種目の
  Basic追加）はこれ以上単発の差し替えを試すのではなく、根本的なロジック改修
  （進化ラインの構成要素に明確なボーナスを与える等、価値評価の同点問題を解消する以上の
  変更）と併せてでない限り、今後も同じ結果になる可能性が高いと考えられる。

  **追記（5.6節参照）**: まさにこの「根本的なロジック改修」を実施したところ、6回目の
  挑戦でこの軸は最終的に成立した。`card_value()`に進化元Basicへの評価ボーナスを追加し、
  Cyrano・Powerglassの代わりにPetrelを削る構成に変えたところ、3,600戦（3回の独立検証）
  で51.75%という安定した改善を確認し、**採用**。上記の「アーキテクチャ自体が複数種の
  Basicが同居する状況を苦手としている」という分析は、価値評価の同点問題という具体的な
  原因まで踏み込めば解消可能だったことになる。
- **ポケモン種を増やさず、ACE SPECの「Maximum Belt」をPowerglass1枚と差し替える（最終判定：
  不採用）**: 2種目Basic案が5連敗したことを踏まえ、ポケモン種を増やさない別軸を探索。
  Powerglass×3+Maximum Belt×1（ACE SPECのため元々1枚上限）——「装着ポケモンの技が相手の
  アクティブ{ex}に+50ダメージ（弱点・抵抗適用前）」という効果で、このカードプールが実際の
  対戦環境を模しているため{ex}/megaEx採用デッキが多いと想定し、有望と考えた。
  ユーザーから「さらに試合数を投じて判断を固めてよい」と指示を受け、合計4回に分けて
  A/Bを実施: 600戦で53.3%（6バッチ中5つが有利）、600戦で51.7%（3対3で分裂）、
  1200戦の継続実行で50.7%（12バッチ中7つが有利）、さらに3000戦の継続実行で50.9%
  （20バッチ中12勝8敗）。**4回の合計5,400戦で新デッキ51.2%・旧デッキ48.8%**——
  方向性は一貫してプラス寄りだが、単発バッチの結果は42%〜62%まで振れ続け、試合数を
  重ねても「ほぼ全バッチで明確に有利」という水準（デッキ乗り換え時の5シード全勝、
  `bench_is_empty`修正の600戦6バッチ全勝など）には最後まで届かなかった。この規模
  （5,400戦、今回のセッションで最大の検証量）まで投じても性質が変わらなかったことから、
  これ以上試合数を積んでも結論が変わる見込みは低いと判断し、**最終的に不採用**とする。
  効果自体が完全にゼロとは言い切れない（総合勝率は一貫して50%をわずかに超えている）ため、
  将来Maximum Beltを他のトレーナー1枚（例えばPetrelの1枚）と差し替えるバリエーションを
  試す価値は残るが、現時点でこのまま採用するには確信が持てない。
- **健康な同種の複製が控えている時、ダメージを受けたアクティブから積極的に退却する
  （`has_healthier_duplicate_on_bench`）**: ユーザーからの指示で「1手先読み」の方向性を
  検討する過程で、実戦データ（bench=1で終わる負けが全体の46.2%——bench=0の
  ブリック系（53.8%）に次ぐ大きな割合）を分析。ローカル自己対戦で該当パターンを4件
  追跡したところ、全件で同じ構図——**アクティブが力尽きて0になった瞬間、ベンチには
  満タンHP（またはそれに近い）の2匹目のMega Lucario exが手つかずのまま残っていた**。
  アクティブが50%未満まで削られ、同じ種類のベンチポケモンが80%以上のHPを保っている時に
  退却の優先度を上げる修正を実装（`active_in_danger`にOR条件で追加）。
  600戦のA/Bテストの結果は旧49.0% vs 新51.0%、6バッチ中2勝2敗2分——きれいに
  ノイズレベルで一貫した改善は見られなかった。Mega Lucario exの`retreatCost`は2
  エネルギーと決して軽くなく、退却すること自体に確かな資源コストがかかる
  （退却する側のエネルギーを2枚失う）。これは今回のセッションで検証した
  「エネルギー配分の最適化」「相手の即死圏の厳密判定」と同じ系統の「防御的な立ち回りの
  ためにエネルギー・テンポを差し出す」変更であり、そのいずれもノイズレベルの結果に
  終わっていることから、**このデッキ・このロジックのアーキテクチャでは、防御目的で
  エネルギーを消費する変更は一貫して効果が出にくい**という傾向が、今回で3例目、
  さらに強く裏付けられた形になる。**不採用**。

## 7. 現状と今後の課題

デッキ乗り換え後も改善の余地は大きい。今後の改善候補:

1. ~~弱点・抵抗の反映~~ → 5.5節で検証・実装済み。
2. **簡易先読み**: 「この攻撃をした場合、次の相手の番で倒され返すか」等、1手先を
   考慮した評価を導入する。
3. **新デッキの追加チューニング**: `Boss's Orders`（相手の弱いベンチを狙い撃ち）や
   `Judge`（手札リセット）は現在の`score_option()`では他の汎用トレーナーと同じ扱いしか
   受けておらず、これらの妨害効果に特化した評価はまだ入れていない。
4. **他アーキタイプとの継続比較**: 3章の選定はダメージ／エネルギー比という単一の軸での
   スクリーニングであり、`Mega Gardevoir ex`系のエネルギー加速アーキタイプ等、他の強い
   候補との直接対決A/Bテストはまだ行っていない。
5. **リプレイ解析の継続**: 今回のように実戦データから具体的な問題を見つけて修正する
   サイクルを継続する。

---

# English

## Overview

[Pokémon TCG AI Battle](https://www.kaggle.com/competitions/pokemon-tcg-ai-battle) is a Kaggle
simulation competition in which the Pokémon Trading Card Game is played automatically inside
Kaggle's simulation environment (internal name `cabt`). Participants submit an
`agent(obs: dict) -> list[int]` function together with a 60-card deck; submissions are matched
against each other automatically and ranked on an Elo ladder.

Our approach was guided by three principles:

- **Investigate the real engine directly rather than relying on documentation.** We used the
  actual native engine bundled inside the `kaggle-environments` package to develop and validate
  entirely offline, with no Kaggle login or data download required.
- **Ground every decision in measured evidence.** Deck selection, policy tuning, and bug fixes
  were all driven by real self-play against the actual engine and by analyzing replays of real
  matches played on Kaggle — not by guesswork. Ideas that couldn't survive an A/B test were
  reverted, even when they looked reasonable on paper (Section 6).
- **Record what didn't work, honestly.** Several logic and deck changes that seemed intuitively
  correct turned out to lower the win rate once measured. We keep those on record rather than
  quietly dropping them, since that's the best defense against repeating the same mistake.

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
- Feeding real Kaggle match replay JSON back through the same engine to reproduce and verify
  what our agent actually chose in each situation

Full findings are documented in [`docs/ENGINE_NOTES.md`](docs/ENGINE_NOTES.md), including
replay-analysis-specific pitfalls (an `action` is a response to the *previous* step's
`observation`, not the same-indexed one; an `INACTIVE` player's `observation` can go stale for
several steps at a time).

## 2. Agent Strategy: The Shape of the Scoring Logic

### 2.1 Design principles

1. **Never crash, never return an illegal action.** A timeout or illegal action is an instant
   loss under the competition's rules, so every decision path is wrapped in exception handling
   with a safe fallback (a minimal legal action) on failure.
2. **Decide using the real card database.** We use actual data from `AllCard()`/`AllAttack()`
   (HP, damage, energy cost, ex/mega flags, etc.) rather than blind heuristics.
3. **Reason on `option[i].type`** (evolve, attach energy, play, attack, retreat, end turn, etc.)
   rather than `select.context`, since only a subset of context codes could be identified
   empirically, while the option type is always informative.

### 2.2 The base priority order (fixed tiers)

Within a single turn (the `MAIN` context), actions are first evaluated against a fixed priority
order:

```
EVOLVE > ATTACH ENERGY > PLAY A CARD > ATTACK > RETREAT > END TURN
```

Within the same tier, decisions are refined using `card_value()` (a heuristic combining HP,
ex/mega bonuses, evolution stage, and search/draw text) and expected attack damage (with a
bonus when it would knock out the opponent's active Pokemon).

### 2.3 Situational overrides on top of the fixed order

Real match data repeatedly surfaced situations the fixed order alone couldn't handle. Rather
than rewriting the evaluation function wholesale, we layered on a series of deliberately simple
overrides — each one checks a single, cheap-to-compute board-state condition and bumps a tier up
or down accordingly; none of them involve search or a rewritten value function.

| Override | Condition | Effect |
|---|---|---|
| `attack_is_lethal` | This attack would KO the opponent's active right now | Attack's tier jumps to the top (11) |
| `bench_is_thin` / `active_in_danger` | Our bench has 0-1 Pokemon, or our active is low on HP | Playing a Pokemon to the bench / retreating jumps near the top (10 / 8.7) |
| `opponent_underprepared` | Opponent's active has no energy and their bench is empty | Attack's tier rises above attaching energy (8.5) |
| Thin bench + no Pokemon in hand | Combination of the above two | A big-draw Supporter's tier rises above attaching energy (8.6) |
| `_expected_discard_damage` / `_discard_pile_damage` | An attack's raw `damage` is 0 but its text describes a conditional payoff | Estimate/measure the real damage instead of treating it as zero |

The last row deserves elaboration. `AllAttack()`'s `damage` field sometimes means "zero unless a
condition is met" rather than "genuinely does nothing" (Section 5). Taking it at face value
means never selecting an attack that's actually strong — this was the single biggest blind spot
we found in real match data.

## 3. Deck Design: Why We Moved From Water to Fighting

### 3.1 The first three candidates (how we picked the water deck)

Early in development, we A/B-tested three candidate decks against each other using the real
engine to make deck selection data-driven (see [`tools/build_deck.py`](tools/build_deck.py)).

| Candidate | Concept | Result |
|---|---|---|
| Kangaskhan swarm (custom) | All-Colorless-cost Basics, avoiding evolution/color-matching complexity | Lost |
| Grimmsnarl ex (custom) | Reportedly the most-played archetype in the real ladder meta | Lost |
| **Kyogre/Snover → Mega Abomasnow ex (initially adopted)** | The sample deck bundled with `kaggle-environments` | **Adopted** |

Our initial hypothesis — that a simpler deck would be easier for a rule-based bot to pilot well —
was not supported by the data. The deck combining a cheap Basic attacker (Snover, 90 HP) with a
powerful evolved finisher (Mega Abomasnow ex: 350 HP, 200 damage for 3 energy plus a same-turn
damage-reduction effect) outperformed both simpler alternatives. This deck ran 33 Basic Energy
(55% of 60 cards), well outside typical TCG deckbuilding guidance (usually 10-15), but it was
still the strongest option against that era's agent logic.

### 3.2 Two structural weaknesses found in real ranked replays

Analyzing 13 real ranked-ladder losses later revealed two structural problems with the water deck
that no amount of logic tuning could fix:

1. **Only 10 Pokemon total, and only 6 of them Basics** (Kyogre x2, Snover x4). Multiple losses
   showed the exact same pattern: the bench stayed empty for the entire game, and the moment the
   active was KO'd, the game was over. We tried lowering the energy count to make room for more
   Pokemon, but as covered in Section 6, that measurably made things worse.
2. **Several opponents ran clearly more cost-efficient attackers.** For example, `Dragapult ex`'s
   Phantom Dive deals 200 damage for 2 Energy and also damages the opponent's bench, while our
   own Frost Barrier deals the same 200 damage for 3 Energy. We were simply less efficient at the
   same power level.

### 3.3 Surveying the full card pool and switching decks

Given those two weaknesses, we mechanically scanned the entire card pool returned by
`AllCard()`/`AllAttack()` (roughly 1000 Pokemon) for attacks with a high damage-per-Energy
ratio, filtering on:

- **A shallow evolution line** (Basic -> one evolution only, the same depth as the old Snover
  line). A 3-stage line like `Dragapult ex` adds more setup risk and instability.
- **A single Energy type**, so the existing energy-attach logic doesn't need color-matching
  complexity. `Dragapult ex` needs two colors (Fire + Psychic); a single-type attacker keeps our
  logic as-is.
- **A genuinely high face-value `damage`** (rather than leaning on the "`damage:0` but actually
  strong" pattern from Section 2.3 — pick a card that's honestly strong instead).

We landed on **Riolu -> Mega Lucario ex** (Fighting type).

- Riolu (Basic, 70 HP): qualifies for `Buddy-Buddy Poffin` (an item that searches up to 2 Basic
  Pokemon with <=70 HP straight onto the bench) — a direct answer to the old deck's "empty
  bench" problem.
- Mega Lucario ex (340 HP, evolves from Riolu): Aura Jab deals 130 damage for 1 Energy and also
  re-attaches up to 3 discarded Fighting Energy onto the bench (a built-in Powerglass-like
  effect). Mega Brave deals 270 damage for 2 Energy (can't be used again the following turn).

The rest of the deck is `Buddy-Buddy Poffin`, `Ultra Ball`, two search/draw Supporters (`Team
Rocket's Petrel`, `Lillie's Determination`), disruption Supporters (`Boss's Orders`, `Judge`),
and `Powerglass`, with Basic Energy trimmed from the old deck's 55% down to 40% (this attack set
has no Hammer-lanche-style payoff for running a huge energy count, so there's no reason to).

We validated the new deck by running 350 games of direct self-play against the old deck (same
agent logic on both sides) before adopting it, confirming a **~54-56% win rate** (Section 4).
Card-by-card rationale is in [`submission/deck.csv`](submission/deck.csv)'s comments.

## 4. Validation Methodology and Results

### 4.1 Self-play against the engine's baselines

We ran real matches via `kaggle_environments.make("cabt")` against the engine's two built-in
baseline agents (see [`tools/evaluate.py`](tools/evaluate.py)).

| Opponent | Win rate |
|---|---|
| `random_agent` | 92-98% |
| `first_agent` (deterministic, always picks the first listed option) | 54-67% (high variance across runs) |
| Crashes / illegal actions | 0 across 100+ games |

We believe the variance against `first_agent` comes from that baseline's behavior depending on
incidental option-array ordering in the engine, which occasionally happens to look like
reasonable play.

### 4.2 Seeded deck-vs-deck A/B testing

A comparison against `first_agent` alone is too noisy to reliably judge a small logic or deck
change. We fixed `random.seed()` before each game and ran the before/after agent through the
same seed sequence (and, for deck comparisons, pitted two decks directly against each other under
identical agent logic via `tools/build_deck.py`'s `make_deck_agent`). We required a consistent
effect across 100-390 games before accepting or rejecting a change.

### 4.3 Validation against real ladder opponents

Self-play alone cannot surface issues that only appear against skilled human-designed opponents
and the real metagame. The next section covers what we found there.

## 5. The Real-Match-Data Improvement Cycle

By downloading and analyzing JSON replays of actual Kaggle ladder matches, we made several
concrete improvements to both the policy and the deck.

### 5.1 Our best attack had never once been selected (the Hammer-lanche bug)

In a mirror-match replay (opponent running the same water deck), the opponent repeatedly used
Mega Abomasnow ex's Hammer-lanche ("discard the top 6 cards of your deck, 100 damage for each
Basic Water Energy discarded") for 300-500 damage per hit, while our own agent had never once
selected that attack. The cause: `AllAttack()` reports `damage:0` for it, and `attack_score()`
took that at face value. Estimating the real expected damage from our own deck's Basic Energy
ratio (55%; 6 cards x 55% x 100 ≈ 330) gave the attack the priority it deserved, producing the
single largest improvement found in this investigation: **55.4% -> 63.75%** win rate over 240
seeded self-play games.

The same class of bug applied to Kyogre's Riptide ("20 damage for each Basic Water Energy card
in your discard pile"), also reported as `damage:0`. Unlike Hammer-lanche, this one scales with
a number we can count exactly (the actual discard pile) rather than estimate. Since Kyogre was
only 2 of 60 cards, the fix was roughly neutral on win rate (59.5% -> 59.0% over 390 games) but
is kept as a genuine accuracy improvement with no measured downside.

### 5.2 The empty-bench, instant-loss pattern

Across the 13 loss replays, a recurring pattern was "the bench never has a single Pokemon on it
all game, and the moment the active is KO'd, the game ends." We first added logic-side
mitigations — prioritizing bench-building/retreating when the bench is thin or the active is low
on HP (`bench_is_thin`/`active_in_danger`, 30% -> 47.5% vs `first_agent` over 40 games), then
prioritizing a big-draw Supporter over attaching energy when the bench is thin and no Basic
Pokemon is in hand (cut the rate of games ending with a permanently empty bench from 46.2% to
31.5% over 390 games, roughly neutral on win rate).

These logic-side mitigations couldn't fix the root cause — the deck simply didn't have enough
Basic Pokemon — which is what ultimately drove the deck switch in Section 3.

### 5.3 Metagame observations

Cataloging the opponent decks across the 13 loss replays showed a clear pattern: `Dragapult ex`
(Dragon), `Mega Gardevoir ex` (Psychic, energy acceleration via evolution), `Mega Lucario ex`
(Fighting), and `Archaludon ex` (Metal) — cost-efficient `ex`/Mega attackers — appeared
repeatedly. This observation directly motivated the mechanical damage-per-Energy screening of
the full card pool in Section 3.

### 5.4 A search effect kept fetching a redundant copy of the evolution (the fix that finally worked)

As recorded in Section 6, the "half of all losses end at 0 bench" pattern survived several
rejected logic and deck attempts even after the v2 deck rework. At that point the approach
changed: rather than guessing at another fix, we reproduced an early bench-0 loss locally
(mirror self-play) and traced every decision the agent made turn by turn. It turned out this
pattern reproduces locally at almost the same rate as the real ladder (55.7% of 150 mirror-match
losses) — every earlier A/B test had, in fact, been measuring a real phenomenon, not a
non-issue.

The trace of one turn-6, bench-0 loss showed the actual mistake: with the bench at 0, the agent
used Ultra Ball and **fetched a second copy of Mega Lucario ex that was already on the field** —
completely useless, since there was no Riolu in play to evolve it from. Riolu itself (a Basic,
playable straight onto the bench) was available as a search target and should have been
preferred instead. The cause: `score_option`'s handling of `OPT_CARD` (search/reveal-style card
choices) ranks candidates purely by `card_value()` (raw HP-based), with no awareness of "the
bench is empty and there's no Basic in play to build on."

Added `bench_is_empty(obs)` (stricter than the existing `bench_is_thin`, which is 0-1) and gave a
Basic Pokemon a large bonus over an evolution card in search-target scoring specifically when the
bench is completely empty. **Scoping this to `bench_is_empty` rather than `bench_is_thin` turned
out to matter**: a first attempt firing at bench<=1 measured as a clear regression over 300 seeded
games — with one backup Pokemon still down, fetching the evolution is still correct, and this
deck's win condition depends on evolving quickly, so delaying that even slightly cost more than
it helped. Narrowing the trigger to a literally empty bench turned this into a consistent
improvement across multiple test batches totaling 900+ games (a single continuous 600-game run
showed every one of 6 sub-batches favoring the fix, 54% overall). **Adopted.**

Along the way, this investigation also surfaced a previously-unknown engine quirk: the
`current["result"]` field in the very last recorded observation of a finished game can still read
`-1` ("in progress") even though the game has actually ended — the step's own `status`/`reward`
fields are the authoritative outcome (see `docs/ENGINE_NOTES.md`). Before this was found, the
local loss-tracing tooling itself was silently reporting "no losses found," which would have
blocked this whole investigation.

### 5.5 Weakness/Resistance damage adjustment had been left unverified and unimplemented

`docs/ENGINE_NOTES.md` had long carried an open item: whether the engine actually applies the
real "Weakness x2 / Resistance -30" damage rule was never checked, and `attack_score`/
`attack_is_lethal` only ever compared the raw `damage` field to the opponent's HP. Prompted by
the user's explicit invitation to dig deeper into logic-level improvements, this was verified
directly: a probe script fired known-damage attacks (Riolu's Quick Attack, Mega Lucario ex's
Aura Jab) at fixed opponent decks built from cards with a known `weakness`/`resistance` value
(Team Rocket's Kangaskhan ex, weak to Fighting; Iron Crown ex, resistant to Fighting) and tracked
the actual HP change turn by turn. The result matched the real Pokemon TCG rule exactly: **x2
damage on a Weakness match, -30 (floored at 0) on a Resistance match**.

Both `attack_score()` and `attack_is_lethal()` previously judged "how much damage" and "is this
lethal right now" using only the flat `damage` field, so a hit that was actually lethal thanks to
Weakness could be scored as merely a setup move — a real missed opportunity, since `OPT_ATTACK`'s
base tier (6) already sits below `OPT_ATTACH` (8) and `OPT_EVOLVE` (9), so unless an attack is
recognized as lethal, the agent will happily attach energy or evolve instead of finishing the
opponent off. Added `apply_weakness_resistance()` and wired it into both functions.

Self-play A/B testing hit a structural ceiling here. Mirror matches can't exercise the fix at all
(our own Pokemon are weak to Psychic, not Fighting), and neither can `first_agent`'s reference
deck (Water/Grass types, also unrelated to Fighting). Two purpose-built opponent decks that *are*
weak to Fighting — `KANGASKHAN_SWARM` (a previously-rejected candidate deck) and a new
`MANECTRIC_TEST` deck built with the exact same structure as our own — were tried, but both
already lost to our deck ~95-99% of the time even with the *old* logic, leaving essentially no
headroom for a win-rate delta to show up (500-game runs: old 98.8% vs fix 98.8%; old 95.2% vs fix
95.6%).

Rather than rely on a saturated win-rate metric, the fix's actual effect was measured directly: how
often does `attack_is_lethal`'s verdict change between the old and new logic? Across 120 games
against the two Fighting-weak decks, this happened **141 times** (~1.2 per game), and every single
one was a newly-detected lethal hit that the flat-damage check had missed. Across 120 games against
two neutral decks (`GRIMMSNARL_EX`, the Water/Grass reference deck), the count was **exactly 0** —
confirming the fix is a true no-op outside the situations it targets (win rate against the neutral
deck was also unchanged within noise: 95.0% -> 93.3% over 120 games). **Adopted**, on the strength
of (1) the mechanic itself being independently verified against the engine, (2) zero measured
regression on neutral matchups, and (3) a concrete, well-scoped positive effect where it matters —
not a win-rate lift, which the available local practice decks structurally cannot show. Real ladder
opponents are far more diverse than our own practice-deck roster, so this has real room to matter
in production even though it's invisible to local self-play.

### 5.6 An "evolution-base" value bonus finally made the 5-times-rejected second-Basic idea work

As recorded in Section 6, "add an insurance second Basic Pokemon" had been tried and rejected 5
times (3x Sawk, 2x Farfetch'd). With the user's explicit permission to accept some wasted-loss
risk and try it once more, we finally fixed the root cause instead of another card swap:
`card_value()` scored every Basic Pokemon by raw HP alone, so a same-HP rival Basic (Farfetch'd,
70 HP, same as Riolu) tied with Riolu in search/discard comparisons, diverting some search hits
away from the one card that actually opens the evolution line.

Added `OWN_DECK_EVOLUTION_BASES` — the set of names referenced by some other deck card's
`evolvesFrom`, computed dynamically from `CARD_DB`/`load_deck()` rather than hardcoded to "Riolu"
— and gave any Basic in that set a +10 value bonus. Riolu's value went from 7.0 to 17.0;
Farfetch'd, referenced by nothing, stayed at 7.0, breaking the tie cleanly.

Verified in two steps. First, isolating the fix's own effect (same Farfetch'd deck, fix vs
no-fix): **57.8% vs 42.2% over 600 games, 5/6 batches favorable** — a large, clean swing that
directly confirmed the hypothesis. Second, testing the fixed deck as a whole against the current
shipped baseline: an initial attempt cutting Powerglass (already known to be load-bearing from
Section 6) came up just short at 48.5%; cutting **Petrel** instead (the most redundant search
Trainer once Buddy-Buddy Poffin/Ultra Ball/Brock's Scouting/Cyrano are already in the deck) landed
at 52.5% over 600 games, then 52.2% over 1200, then 51.2% over 1800 — **three independent runs
(3,600 games total) all landing in a narrow 51-53% band**, a sharp contrast with the Maximum Belt
experiment's wild 42-67% batch-level swings that never settled down even after 5,400 games.

**Adopted.** The aggregate win-rate lift (51.75% pooled) is modest by this project's usual
standard, but two things separate it from a coin flip: (1) the underlying mechanism was
independently confirmed with a large, clean swing (57.8% vs 42.2%) when isolated from the deck
change itself, and (2) three separate large-scale runs landed in the same narrow band instead of
swinging unpredictably. Updated `submission/deck.csv`/`DEFAULT_DECK` to this v3 composition
(Farfetch'd in, Petrel out) and shipped the `card_value()` fix alongside it.

Right after shipping v3, tried two further tweaks; both rejected:
- **Swap Farfetch'd for Hawlucha** (retreatCost 0, and its weakness/energyType exactly match our
  own Pokemon vs Farfetch'd's colorless attack cost) — looked strictly better on paper, but
  measured slightly worse over 600 games: v3 (current) 51.5% vs Hawlucha-swap 48.5% (split 3-3
  across 6 batches). In mirror self-play neither side ever uses Special Energy, so Farfetch'd's
  "discard the opponent's Special Energy" clause never fires either way — this was meant to be a
  clean stat comparison, and Hawlucha still came up short. **Rejected.**
- **Cut Farfetch'd from 4 copies to 2, backfilling with 2 Petrel** ("light touch," diluting
  Riolu's search pool less while keeping some bench insurance): the first 600-game batch looked
  like a strong win (v3 45.0% vs light-touch 55.0%, 5/6 batches favorable), but a 1200-game
  confirmation run reversed cleanly (v3 50.8% vs light-touch 49.2%, only 3/12 batches favorable).
  Pooled across both (1,800 games): light-touch at 51.1%, but batch favorability dropped to 44%
  (8/18) — the initial result was simply favorable variance. **Rejected** — and a good reminder of
  exactly why this project never ships on a single 600-game batch.

### 5.7 Researching general strategy, and discovering a major unknown mechanic: KOing a megaEx awards 3 prizes

Per the user's direction to "encode general Pokémon TCG strategy — especially advanced,
tournament-level strategy — into the rule-based logic, and adopt general deck-building theory and
best practices," we researched real competitive Pokémon TCG strategy via web search before
touching any code.

Key findings:
- **Prize-trade math**: hunting the opponent's high-value Pokemon (ex/VMAX etc.) for extra prizes
  while protecting your own is a core piece of modern competitive strategy.
- **Tempo**: converting an energy/turn lead into board advantage.
- **Gusting** (e.g. `Boss's Orders`, which forces a specific benched Pokemon into the active spot)
  to snipe a weakened bench Pokemon is an established advanced technique.
- **Modern deck-building ratios**: roughly 12-16 Pokemon / 30-36 Trainers / 8-12 Energy (out of
  60). Our current deck (12 Pokemon / 24 Trainers / 24 Energy) runs a far higher Energy count than
  that norm.
- **Special conditions** (Poison/Burn/Confusion/Paralysis/Asleep): none of our current attacks
  (Aura Jab / Mega Brave / Quick Attack / Mach Cut) inflict any condition, so there's no direct
  hook for this in the current card pool.

(Sources listed at the end of this subsection.)

Before building on this research, we checked whether "prize-trade math" is even a real mechanic in
this engine. We found `obs.current.players[i].prize` is a length-6 list (`null` = not yet claimed),
and by tracking every step of real games we measured, for each prize-count drop, which opposing
Pokemon had just been KO'd. Across 100 games and 293 sampled prize-taking events, the result was
completely deterministic with zero exceptions: **KOing a `megaEx` Pokemon (Mega Lucario ex) always
dropped the KO-er's remaining-prize count by exactly 3; KOing any plain non-ex Basic (Riolu or
Farfetch'd) always dropped it by exactly 1.** Unlike the real paper TCG's "ex = 2 prizes" rule, this
engine gives only `megaEx` a multi-prize penalty, and it's 3, not 2. This is a significant engine
mechanic that no prior investigation in this project had uncovered.

### 5.8 Prize-value-aware retreat threshold (`prize_value` / `active_in_danger`)

Given Section 5.7's finding, every time our Mega Lucario ex gets KO'd we hand the opponent 3 prizes
at once — half the game's prizes in one hit. The existing `active_in_danger()` used a flat "retreat
below 35% HP" threshold, which is fine for a disposable Basic that only costs 1 prize if lost, but
dangerously late for a Pokemon that costs 3.

Added `prize_value(card)` (returns 3 for `megaEx`, else 1 — grounded directly in the Section 5.7
measurement) and changed `active_in_danger()`'s threshold to 55% when the active Pokemon is
3-prize-value (`megaEx`), keeping the existing 35% otherwise. This lets Mega Lucario ex swap out to
a disposable Basic (Riolu/Farfetch'd, 1 prize) while it still has roughly half its HP left, instead
of only once it's nearly dead — trading down to a cheap Pokemon before the expensive one is lost.

Ran three independent 600-game batches (each a fresh self-play run, no fixed seed):

| Run | New-logic (prize-value-aware) win rate |
|---|---|
| 1 (600 games) | 53.8% |
| 2 (600 games) | 55.0% |
| 3 (600 games) | 52.5% |
| Pooled (1,800 games) | 53.8% |

Using the same bar as Section 5.6's v3 shipment (three runs converging on a narrow 51-53% band),
all three runs here independently landed clearly above 50%, showing the same kind of consistency.
**Adopted.**

Candidates considered but not yet implemented:
- Adding `Boss's Orders` (cardId 1182, "switch in 1 of your opponent's benched pokemon to the
  active spot" — confirmed present in this card pool) to enable gust-style bench-sniping of the
  opponent's `ex`/`megaEx` Pokemon. Our deck currently runs 24/60 (40%) Energy, well above the
  8-12/60 (13-20%) modern norm found in research, so there's room to trim Energy and add
  disruption Trainers like this. This is a larger, structural deck change that needs its own
  separate A/B test.
- Special conditions: no current attack in our card pool inflicts any, so there's no hook to act
  on yet (worth revisiting if the attack lineup changes).

**Sources** (per this project's practice of citing web research used to inform implementation):
[Pokémon TCG Prize Mapping Guide 2026](https://tcgprotectors.com/blogs/pokemon-blog/pokemon-tcg-prize-mapping-guide-2026),
[Pokémon TCG Prize Trade Guide](https://tcgprotectors.com/blogs/pokemon-blog/pokemon-tcg-prize-trade-guide-advanced-prize-mapping),
[Don't Have a Tempo Tantrum (SixPrizes)](https://sixprizes.com/2013/03/29/dont-have-a-tempo/),
[Intermediate Pokémon TCG Strategy Guide](https://tcgprotectors.com/blogs/pokemon-deck-guides/pokemon-tcg-intermediate-strategy-guide),
[Deep Dive into Energy (2026)](https://tcgprotectors.com/blogs/pokemon-blog/pokemon-tcg-energy-guide-2026-special-acceleration),
[Deck Strategy — JustInBasil](https://www.justinbasil.com/guide/deck-strategy),
[Competition-Ready Basic Strategies — Pokemon.com](https://www.pokemon.com/us/strategy/competition-ready-basic-strategies),
[Pokémon TCG Deck Building Guide (2025)](https://tcgprotectors.com/blogs/pokemon-deck-guides/how-to-build-pokemon-tcg-deck-guide),
[Deck Ratio? — PokéBeach](https://www.pokebeach.com/forums/threads/deck-ratio.96041/),
[Introduction to Deckbuilding — Pokemon.com](https://www.pokemon.com/us/strategy/designing-a-deck-from-scratch),
[Gusting and Repulsion — JustInBasil](https://www.justinbasil.com/guide/gusting),
[Mastering Disruption 2026](https://tcgprotectors.com/blogs/pokemon-blog/mastering-disruption-pokemon-tcg-2026-hand-board-ability-lock-strategies),
[Boss's Orders Targeting — Pokémon Forums](https://community.pokemon.com/en-us/discussion/6961/bosss-orders-targeting),
[Boss's Orders — Pokémon Rulings Compendium](https://compendium.pokegym.net/category/5-trainers/bosss-orders/),
[Special Condition (TCG) — Bulbapedia](https://bulbapedia.bulbagarden.net/wiki/Special_Condition_(TCG)),
[Special Conditions Guide — TCG Protectors](https://tcgprotectors.com/blogs/pokemon-beginners-guide/pokemon-tcg-special-conditions-guide),
[Pokémon TCG Rules PDF — Pokemon.com](https://www.pokemon.com/static-assets/content-assets/cms2/pdf/trading-card-game/rulebook/par_rulebook_en.pdf).

## 6. Ideas We Tried and Rejected (an Honest Record)

Several ideas that looked reasonable in theory measurably lowered the win rate once A/B-tested.
We record them here so they aren't retried blind.

- **Attack more when behind on prizes (`we_are_behind`)**: a common general-strategy intuition —
  press harder when losing. But this deck's win condition is patiently loading energy onto a
  heavy finisher; attacking prematurely when behind meant skipping that energy investment at
  exactly the wrong time. Measured 55.3% -> 48.7% over 300 seeded games and was removed.
- **Spread energy to the bench once the active is maxed out**: the observation ("we're piling
  too much energy onto one attacker") was accurate, but the fix backfired. ATTACH already always
  outranked ATTACK regardless of target, so once attaching could usefully go to the bench
  forever, the active simply stopped attacking and spent every turn feeding the bench instead.
  Measured 55.3% -> 48.7% over 300 games and was removed.
- **Trim deck energy to make room for more Pokemon/trainers (while keeping the water deck)**: the
  intuition that "55% energy is obviously too much" led us to max every non-energy card to 4
  copies and cut energy from 33 to 23. The added trainers (search cards) could only find Pokemon
  or evolution targets, not energy itself, so the deck ended up drawing Pokemon it couldn't
  actually power up — a different failure mode. Measured 47% -> 43% over 120 games and 51% -> 37%
  over another 120. Combined with the Hammer-lanche discovery in Section 5.1, this taught us that
  55% energy was more load-bearing for this specific deck than it looked, which is part of why we
  ultimately switched decks (Section 3) rather than keep tuning the water deck's energy count.
- **Give Pokemon-searching Trainers (Buddy-Buddy Poffin / Ultra Ball / Brock's Scouting / Cyrano)
  the same urgent tier as playing a Pokemon directly, whenever the bench is thin**: `tools/
  kaggle_watch.py`'s automated per-loss diagnostics (Section 5) surfaced a persistent pattern in
  real ladder play even after the v2 deck rework added these exact search cards — **roughly half
  of all losses still ended with 0 Pokemon on the bench**. `searches_for_pokemon(card)` was added
  to detect these Trainers by effect text, and playing one was bumped to the same tier (10, above
  EVOLVE's 9) as playing a Pokemon straight from hand whenever `bench_is_thin`. A/B testing (300
  seeded games, direct mirror match) measured this as a clear regression: 52.7% for the old logic
  vs. 47.3% for the new one, unfavorable in 4 of 5 seeds. The likely cause: this fired even with 1
  Pokemon still on the bench (not just 0), outranking EVOLVE — and this deck's win condition is
  evolving into Mega Lucario ex quickly, so deferring that to search for more bodies when the
  bench wasn't actually empty cost more than it helped. A narrower variant (only override at a
  literally empty bench, otherwise unchanged) came back roughly neutral (300 games, 3 seeds
  favorable / 1 tied / 1 unfavorable, ~52% overall) — not the kind of consistent, seed-independent
  signal this project requires before shipping (contrast the deck switch's 60% across all 5 seeds
  individually). Both variants were reverted. **The underlying signal — half of all real losses
  ending at 0 bench — is still real and still unresolved**; the next thing worth trying is
  probably deck composition (search-card density, a second cheap Basic alongside Riolu) rather
  than further tuning the decision logic, which already seems to be making a reasonable choice
  given what's actually in hand.
- **Add a second, non-evolving "insurance" Basic Pokemon (Sawk) to the deck**: acted on the
  previous idea's own conclusion, using a real-TCG deckbuilding pattern (running a small,
  unrelated Basic purely as insurance against the main evolution engine bricking or dying early
  — a genuinely common technique in real competitive decks). Screened the card pool for Basic
  Pokemon with HP >= 100 (tougher than Riolu's 70), retreat cost <= 1, a real unconditional
  1-Energy attack, and Fighting/Colorless typing so it wouldn't need a second energy color. Sawk
  (110 HP; Elbow Strike: 1 {F} Energy for a flat 30 damage — strictly better than Riolu's own
  coin-flip-dependent Quick Attack; Rising Chop: 1 Energy for 90 damage against a Pokemon {ex}
  active) was the clear standout. A/B-tested two versions: 4 copies replacing Powerglass (300
  games, existing deck won 53%, unfavorable in 4 of 5 seeds) and a lighter 2-copy version cutting
  only 2 Powerglass (300 games, existing deck won 53.7%, again unfavorable in 4 of 5 seeds) — both
  measured as regressions, not improvements.
  Along the way, this surfaced a real flaw in `card_value()`: it scores Pokemon by raw HP, so in
  any search/discard comparison **Sawk's 110 HP outranked Riolu's 70**, even though Riolu is the
  actual engine piece the whole deck is built around. Implemented a generic fix (a bonus for any
  Basic whose name appears as some other real card's `evolvesFrom` — computed dynamically from
  the full card database, not hardcoded to "Riolu", staying consistent with this project's
  data-driven-not-card-name-specific approach) that correctly restored Riolu > Sawk. Re-tested the
  4-copy Sawk variant with this fix in place: still 53.3% for the existing deck, unfavorable in 4
  of 5 seeds — no better. **All of it was reverted** (`submission/main.py`, `tools/build_deck.py`,
  and the tests are all back to their pre-experiment state; only this write-up ships).
  Across roughly 1,200 self-play games total, none of it produced the kind of consistent,
  seed-independent signal this project requires before shipping (contrast the deck switch's 60%
  across all 5 individual seeds). The real-TCG insurance-Basic pattern is theoretically sound, but
  apparently isn't translating into a measured win here — a plausible reason is that this agent's
  decision logic doesn't yet distinguish "a card that happens to have good stats" from "the
  specific card this deck's plan actually depends on" when picking among multiple Basics (e.g.
  during a search effect), so adding a second Basic option, even a statistically better one,
  seems to dilute the odds of assembling the real plan more than it helps survive early aggro.

  **Update (see Section 5.6):** the value-fix tried here didn't rescue Sawk itself, but applying
  the same fix to Farfetch'd later (70 HP, which fits Buddy-Buddy Poffin's cap) worked cleanly.
  Sawk's 110 HP meant Buddy-Buddy Poffin never offered it as a candidate in the first place — no
  amount of fixing the value comparison helps a card that isn't even in the running.
- **Cap energy-attach priority once a Pokemon already has enough Energy for its best attack**:
  motivated by the same real-match data as Section 5.4 (some losses left energy stacked unevenly
  across a bench with 2+ Pokemon). Added `energy_need_gap()` — computes how many more Energy a
  Pokemon could still use based on the highest cost among its known attacks — and penalized
  `OPT_ATTACH`/`OPT_ENERGY` targets that had already met that cap, so a slot that still needed
  Energy (e.g. a second Mega Lucario ex sitting on the bench) would be preferred instead. A
  600-game single continuous A/B run (6 batches) came back at old 49.7% vs fix 50.3%, split 3-3
  across batches — indistinguishable from noise. The likely reason: both of Mega Lucario ex's
  attacks are cheap (1-2 Energy), so the "energy already maxed out on one attacker" scenario this
  fix targets is simply rare for this specific deck. **Rejected.**
- **Replace `active_in_danger`'s flat 35%-HP threshold with an exact "opponent has a lethal
  attack ready right now" check**: Mega Brave (270 damage, 2 Energy) exposes a real structural
  gap — 35% of its own 340 max HP is only 119, so anything sitting at 120-269 HP reads as "safe"
  under the old threshold while a 2-Energy-loaded opponent Mega Lucario ex would still one-shot
  it. Added `opponent_can_lethal_us()` — checks every attack the opponent's active already has
  enough Energy to pay for right now against our current HP — and OR'd it into `active_in_danger`.
  A 600-game single continuous A/B run (6 batches) came back at old 50.7% vs fix 49.3%, split 3-3
  across batches — the reasoning was sound but didn't translate into a measured improvement.
  Two plausible reasons: (1) retreating pre-emptively while still fairly healthy has its own
  opportunity cost (that turn doesn't attack or evolve), which may have offset the theoretical
  safety gain; (2) Mega Brave carries a self-imposed "can't use it again next turn" restriction
  that `opponent_can_lethal_us()` doesn't check for, so it can flag a threat that isn't actually
  available yet. **Rejected.**
- **A second insurance Basic (Farfetch'd), swapped in for Cyrano or Powerglass**: real ladder data
  showed ~35% of recent losses ending in an early brick (turn 3-9), traced to the deck running only
  8 Pokemon total (4 Riolu, 4 Mega Lucario ex). The earlier Sawk experiment (Section 3.5-equivalent)
  likely failed partly because Sawk's 110 HP exceeded Buddy-Buddy Poffin's "70 HP or less" cap, so
  our best search card could never fetch it. Re-screened the full card pool for a Basic at HP<=70
  with a non-conditional 1-Energy (colorless or Fighting) attack dealing >=20 damage, and found
  **Farfetch'd** (HP70, 1 colorless Energy for a guaranteed 30 damage plus discarding a Special
  Energy from the opponent's active) — fits Buddy-Buddy Poffin's cap exactly, fixing the specific
  gap that sank Sawk.
  Swapping out Cyrano for 4x Farfetch'd (DECK_V3) and A/B testing over 600 games came back as the
  single largest regression measured in this whole investigation: **64.7% old vs 35.3% new, every
  one of 6 batches unfavorable**. Digging into why revealed that Cyrano's "search for up to 3
  Pokemon {ex}" is actually a major consistency engine for finding extra copies of Mega Lucario ex
  — the original assessment that it was "the least useful card" was wrong. Keeping Cyrano and
  swapping out Powerglass instead (DECK_V4) was tried next; still a clear loss, 600 games at 60.8%
  old vs 39.2% new, again every batch unfavorable.
  Powerglass turned out to matter more than expected too (auto-reattaching a discarded Basic Energy
  to the active Pokemon each end of turn — recovering exactly what Aura Jab discards from itself),
  but the fact that cutting *either* card for Farfetch'd produced a decisive loss suggests the
  problem isn't which card to cut — it's that **including a second Basic Pokemon species at all**
  hurts this deck under the current decision logic. Counting this project's full history, that's
  now **5 total variants across 2 different insurance-Basic candidates (3x Sawk, 2x Farfetch'd),
  all rejected**. A plausible mechanism: `card_value()` scores Riolu and any other 70-HP Basic
  identically, so a fraction of limited search hits get diverted away from Riolu — the only card
  that actually opens the evolution line — even though a generic "bonus for any Basic that appears
  in some other card's `evolvesFrom`" fix was already tried during the Sawk investigation and still
  didn't rescue the idea, suggesting the issue runs deeper than a simple value tie. **Rejected**
  (both DECK_V3 and DECK_V4). This axis (adding a second Basic species) likely needs a genuine
  decision-logic change — not just another card swap — before it's worth retrying.

  **Update (see Section 5.6):** that genuine decision-logic change was exactly what made this axis
  finally work on the 6th attempt. Adding the `card_value()` evolution-base bonus and cutting
  Petrel instead of Cyrano/Powerglass landed at a stable 51.75% over 3,600 games across three
  independent runs. **Adopted.**
- **Swap 1 copy of Powerglass for the ACE SPEC "Maximum Belt," without adding a new Pokemon
  species (final verdict: rejected)**: after 5 straight losses on the "add a second Basic" axis,
  explored a different lever instead. Maximum Belt ("attacks used by the Pokemon this card is
  attached to do 50 more damage to your opponent's Active Pokemon {ex}, before Weakness and
  Resistance") looked promising since this card pool's real-meta-inspired design means {ex}/megaEx
  opponents are common, and it's ACE SPEC (capped at 1 copy anyway, so a low-risk single-card
  swap). At the user's explicit request to pour in more games and settle the question, ran four
  A/B batches total: 600 games at 53.3% (5/6 favorable), 600 at 51.7% (split 3-3), a 1200-game
  continuous run at 50.7% (7/12 favorable), and a 3000-game continuous run at 50.9% (12/20
  favorable). **Pooled across all four (5,400 games total): 51.2% new vs 48.8% old** — directionally
  positive throughout, but individual 100-game batches kept swinging from 42% to 62% no matter how
  much data was added, never converging toward the "nearly every batch favorable" consistency this
  project requires before shipping. At this scale (5,400 games, the largest test volume in this
  session), further data collection is unlikely to change the picture. **Rejected**, though not
  because the effect is necessarily zero (the pooled rate stayed persistently just above 50%) — a
  future variant swapping a different card to make room for Maximum Belt (e.g. 1 copy of Petrel
  instead of Powerglass) might still be worth trying, but this exact swap isn't confident enough
  to ship.
- **Retreat proactively out of a damaged active when a healthy duplicate of the same species is
  idling on the bench**: while scoping the user-requested "1-ply lookahead" direction, real ladder
  data showed 46.2% of recent losses end with bench=1 (second-largest category after the 53.8%
  bench=0 brick pattern). Tracing 4 matching local self-play losses turned up the same shape every
  time: **our active grinds down to 0 while a second, undamaged (or nearly so) Mega Lucario ex sits
  idle on the bench the whole game.** Added `has_healthier_duplicate_on_bench()` — fires when our
  active is below 50% HP and a bench Pokemon of the same species is above 80% — OR'd into
  `active_in_danger` to bump retreat priority in that situation.
  600-game A/B: 49.0% old vs 51.0% fix, split 2 favorable / 2 unfavorable / 2 tied across 6 batches
  — cleanly noise-level. Mega Lucario ex's `retreatCost` is 2 Energy, a real resource cost for
  switching. This is the same underlying shape as two other rejected ideas this session (the
  energy-allocation cap and the exact lethal-threat retreat check) — trading Energy/tempo for
  defensive positioning. With all three now tested and rejected, there's a fairly strong pattern
  emerging: **this deck's decision logic doesn't have room to spend Energy on defense and still
  come out ahead** — its win condition is too tightly coupled to keeping the current attacker
  fed. **Rejected.**

## 7. Current Standing and Future Work

There is still significant room for improvement after the deck switch. Candidate next steps:

1. ~~Model weakness/resistance~~ → verified and implemented, see Section 5.5.
2. **Add shallow lookahead** — e.g. "would this attack leave us open to a KO next turn?" — rather
   than purely static per-turn scoring.
3. **Tune the new deck's disruption cards further** — `Boss's Orders` (pulling a specific
   Benched Pokemon into the Active Spot) and `Judge` (resetting both hands) are currently scored
   like any other generic trainer; we haven't yet added situational logic specific to their
   disruption value.
4. **Compare against more archetypes** — Section 3's deck selection screened on a single axis
   (damage-per-Energy); we haven't yet run a direct A/B test against other strong candidates like
   a `Mega Gardevoir ex`-style energy-acceleration archetype.
5. **Keep mining match replays** — continue the cycle of finding concrete, evidence-backed issues
   from real match data and fixing them, as demonstrated in Section 5.
