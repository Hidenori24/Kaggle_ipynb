# CABT エンジン仕様メモ（リバースエンジニアリング結果）

このコンペ（[Pokémon TCG AI Battle](https://www.kaggle.com/competitions/pokemon-tcg-ai-battle)、
Kaggle 環境名 `cabt` = **C**ard **A**I **B**attle **T**ournament）は公式ドキュメントが薄いため、
`kaggle-environments==1.30.1` に同梱されているネイティブエンジン（`kaggle_environments/envs/cabt/cg/libcg.so`）
を実際にロードし、ランダムエージェント同士で数百戦させて観測データ（`obs`）を実地調査した。
この文書はその結果をまとめたものであり、Kaggle公式仕様書の代替ではない（不明点は都度明記する）。

## 0. エンジンの入手方法

`cg` ディレクトリ（`libcg.so` / `cg.dll` というコンパイル済みネイティブライブラリを含む）は
`pip install kaggle-environments==1.30.1` で自動的にダウンロードされる。Kaggle側の実行環境にも
同じライブラリが積まれているため、ローカルで `pip install kaggle-environments==1.30.1` するだけで
**本物のゲームエンジンを使ったオフライン検証**が可能。追加のデータダウンロードやKaggle認証は不要。

```python
from kaggle_environments import make
env = make("cabt")
env.run([agent_a, agent_b])
```

## 1. `agent(obs: dict) -> list[int]` の全体フロー

- `obs["select"] is None` の場合 → デッキ選択フェーズ。**60枚のカードID配列**を返す。
  60枚ちょうどでないと `INVALID` 判定で即敗北する（`interpreter()` 参照）。
- それ以外 → `obs["select"]["option"]` から `minCount`〜`maxCount` 個のインデックスを選んで返す。
- 返り値は必ず「選択肢のインデックス」であり、カードIDそのものではない（デッキ選択フェーズのみ例外）。
- タイムアウトやクラッシュ（例外送出）は即座に `TIMEOUT`/`ERROR` → 敗北。**エージェントは絶対にクラッシュしてはいけない。**

## 2. `obs` のトップレベルキー

| key | 内容 |
|---|---|
| `select` | 現在の意思決定要求。`None` ならデッキ提出フェーズ。 |
| `current` | 盤面全体のスナップショット（自分・相手の場、手札枚数など）。まれに `None` になる（後述の注意点）。 |
| `logs` | 直近のイベントログ（型番号ベース、詳細は未解読）。 |
| `search_begin_input` | 内部エンジン用のraw dataで、通常のエージェント実装では未使用。 |

⚠️ **注意**: `current` が `None` になるフレームが低確率で観測された（数百戦に1回程度）。
`select` が非 `None` でも `current` が `None` の可能性があるため、必ず `None` チェックを入れること。

## 3. `select` オブジェクト

```json
{
  "type": 0,
  "context": 0,
  "minCount": 1,
  "maxCount": 1,
  "option": [ ... ],
  "deck": null,
  "contextCard": null,
  "effect": null
}
```

- `context`: 意思決定の種類（下記表）。
- `option[i].type`: 選択肢の種類（OptionType、下記表）。**エージェントの分岐は基本的にこちらを使う方が頑健**
  （`context` の全網羅は未完了のため）。
- `deck`: サーチ系の効果（山札から手札に加える等）の時だけ、**山札全体**が
  `[{"id":cardId,"serial":..,"playerIndex":..}, ...]` の配列で渡される。
  `option[i]` が `{"type":3,"area":1,"index":k}` の場合、対象カードは `deck[k]`（山札配列内のインデックス）。
- `effect`: そのSelectがトリガーカードの効果による場合、`{"id":cardId,"serial":..,"playerIndex":..}` が入る
  （例: 「Powerglass」の自動エネルギー付け替え効果）。

### 3.1 SelectContext（判明分。番号は実測 or 参考実装からの類推）

| 値 | 意味（推定） |
|---|---|
| 0 | MAIN（1ターン中の行動選択。プレイ/進化/エネルギー付け/攻撃/ターン終了がすべてここに混在） |
| 1 | SETUP_ACTIVE（初期バトル場ポケモン選択） |
| 2 | SETUP_BENCH（初期ベンチ配置） |
| 3 | SWITCH（にげる等による強制/任意交代） |
| 4 | TO_ACTIVE（きぜつ後などにベンチ→バトル場） |
| 5 | TO_BENCH | 
| 7 | TO_HAND（サーチ効果で山札から手札へ） |
| 8 | DISCARD（手札やその他のトラッシュ） |
| 21 | ATTACH_FROM |
| 22 | ATTACH_TO（例: Powerglassのトラッシュからのエネルギー付け） |
| 38 | 不明（実測: `NUMBER` 選択肢が複数出るコンテキスト。用途未特定） |
| 41 | IS_FIRST（先攻/後攻の選択、YES/NO） |
| 42 | MULLIGAN（未実測、参考実装からの類推） |

**未知の `context` 番号に遭遇した場合、本実装は `option[i].type`（OptionType）ベースの
汎用ロジックにフォールバックする**ため、コンテキスト網羅率の不完全さは致命的にならない設計にしてある。

### 3.2 OptionType（実測で確認済み）

| 値 | 意味 | 追加フィールド |
|---|---|---|
| 0 | NUMBER | `number` |
| 1 | YES | - |
| 2 | NO | - |
| 3 | CARD | `area`, `index`, `playerIndex` |
| 7 | PLAY（手札のカードをプレイ） | `index`（手札インデックス） |
| 8 | ATTACH（エネルギーを付ける） | `area`,`index`(手札側), `inPlayArea`,`inPlayIndex`(対象ポケモン) |
| 9 | EVOLVE（進化） | `area`,`index`(手札側), `inPlayArea`,`inPlayIndex`(対象ポケモン) |
| 13 | ATTACK | `attackId` |
| 14 | END（このステップでの選択肢なし/ターン終了） | - |

`6`(ENERGY), `10`(ABILITY), `12`(RETREAT) は参考実装のドキュメントに記載があるが今回の実測では未出現。
本実装ではこれらも安全に処理できるよう `.get()` ベースの汎用フォールバックを用意している。

### 3.3 `area` コード（推定）

| 値 | 推定意味 |
|---|---|
| 1 | 山札（deck） |
| 2 | 手札（hand） |
| 3 | トラッシュ（discard） |
| 4 | 場（in-play。`inPlayIndex` と併用し、0=バトル場, 1-5=ベンチ） |
| 5 | ベンチ（`area`単体でCARD型に使われる場合） |

## 4. `current` の構造（プレイヤー視点）

```json
{
  "turn": 2,
  "yourIndex": 0,
  "firstPlayer": 1,
  "supporterPlayed": false,
  "stadiumPlayed": false,
  "energyAttached": false,
  "retreated": false,
  "result": -1,
  "players": [
    {
      "active": [{"id":722,"serial":5,"hp":90,"maxHp":90,"energies":[3],"energyCards":[...],"tools":[],"preEvolution":[]}],
      "bench": [...],
      "hand": [{"id":723,"serial":9,"playerIndex":0}, ...],
      "handCount": 7,
      "deckCount": 46,
      "discard": [...],
      "prize": [null, null, ...]
    },
    { "...": "相手視点。hand は常に null（非公開情報）" }
  ]
}
```

- `result`: `-1`=継続中, `0`=プレイヤー0勝利, `1`=プレイヤー1勝利, `2`=引き分け。
- 自分の手札は常にフルオブジェクト、相手の手札は `null`（`handCount` のみ判明）。
- ポケモンオブジェクトの `hp` は**現在HP**（`maxHp` が最大HP）。

## 5. カードデータベース（`AllCard` / `AllAttack`）

ネイティブライブラリは `GetBattleData` 等に加え、**カード全件・技全件を返すAPI**を公開している
（これは第三者実装 `cg.api.all_card_data()` 等が内部で使っているものと同一のシンボル）。

```python
import ctypes, json
lib = ctypes.cdll.LoadLibrary(".../libcg.so")
lib.GameInitialize()
lib.AllCard.restype = ctypes.c_char_p
lib.AllAttack.restype = ctypes.c_char_p
cards = json.loads(lib.AllCard())      # 1267枚（実測時点のバージョン）
attacks = json.loads(lib.AllAttack())  # 1556技
```

- `cardType`: 0=ポケモン, 1=グッズ, 2=ポケモンのどうぐ, 3=サポート, 4=スタジアム, 5=基本エネルギー, 6=特殊エネルギー
- `pokemonType`/`weakness`/`resistance`/`energyType`: 0=無色, 1=草, 2=炎, 3=水, 4=雷, 5=超, 6=闘, 7=悪, 8=鋼, 9=ドラゴン
  （基本エネルギーのID1-8の並びから逆算。ドラゴン/無色は他カードとの整合性からの推定）
- カード名を見る限り、このコンペのカードプールは実際の最新ポケモンカードの環境
  （Marnie's Grimmsnarl ex、Iono's Bellibolt ex、Cynthia's Garchomp ex 等）を模した独自デジタル実装であり、
  Mega Evolution ex のような独自メカニクスも含む（現実のスタンダードレギュレーションと完全一致はしない）。

## 6. ハマりどころ：エージェント関数のシグネチャ

`kaggle_environments` はエージェントの呼び出し方をシグネチャから動的に判定する。
`def agent(obs, deck=some_list):` のように**2引数目（デフォルト値つき）を足すと**、
`kaggle_environments` はそれを「`agent(observation, configuration)` 形式」だと解釈し、
2つ目の引数に（意図した既定値ではなく）環境の `configuration` オブジェクトを渡してくる。
結果として `deck` 変数が silently に別物へ差し替わり、デッキ提出フェーズで
不正な値を返して即負け（`INVALID`）になる——にもかかわらず例外は出ないため非常に気づきにくい。

対策: 複数のデッキ／方策を使い分けたい場合は、デフォルト引数ではなく
**クロージャ（`def make_agent(deck): def _agent(obs): ...; return _agent`）**で
1引数関数を生成すること（`tools/build_deck.py` の `make_deck_agent` を参照）。
`submission/main.py` の `agent(obs)` 自体は最初から1引数なのでこの問題の影響を受けない。

## 7. 既知の制約・未解決事項

- `NUMBER` 型選択肢の正確な意味（何を選ぶと何が起きるか）は文脈ごとに未解読。本実装は
  「基本的に選べる最大値に寄せる」というヒューリスティックで対応している。
- 攻撃のダメージ計算に「弱点2倍・抵抗-30」等の実際の補正ルールが適用されるかは未検証
  （`attacks[].damage` の額面値のみで評価している）。
- 特性（Ability）・とくしゅエネルギーの詳細効果テキストの自動解釈は行っていない
  （カード名・cardType・数値ステータスのみに基づくヒューリスティック）。
