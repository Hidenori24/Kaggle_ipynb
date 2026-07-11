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

## 7. 現状と今後の課題

デッキ乗り換え後も改善の余地は大きい。今後の改善候補:

1. **弱点・抵抗の反映**: 現在は攻撃ダメージを額面値のみで評価しており、タイプ相性を
   考慮していない。`CARD_DB`に`weakness`/`resistance`の値自体はあるため、実装コストは低い。
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

## 7. Current Standing and Future Work

There is still significant room for improvement after the deck switch. Candidate next steps:

1. **Model weakness/resistance** — attack damage is currently evaluated at face value only, with
   no type-matchup adjustment. `CARD_DB` already exposes `weakness`/`resistance` values, so this
   is a low-cost addition, just not yet implemented.
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
