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
| 6 | **未解読**。実測: `context=7`（TO_HAND）の`CARD`型選択肢で出現、`deck`/`effect`/`contextCard`
     が全て`null`のため発生源のカードを特定できなかった。サイド（`prize`）関連の可能性があるが未確認。
     本実装は`resolve_card_by_area()`が未知の`area`では`None`を返し、`card_value(None)==0.0`で
     全選択肢が同スコアになる（先頭が選ばれる）ため、致命的な誤動作にはならない設計。

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
- `prize`: 各プレイヤー自身の残りサイド枚数を表す配列。**要素の値は常に`null`で使えない
  ——`len(prize)` が「残りサイド枚数」そのもの**（初期値は観測範囲では6）。自分がKOする度に
  自分の`prize`が1減る（＝取ったサイドの枚数ではなく残数）。したがって
  `(相手のlen(prize)) - (自分のlen(prize))` が「自分が取ったサイド数 − 相手が取ったサイド数」
  になり、勝敗の進み具合を測る簡単な指標として使える（`submission/main.py`では試したが
  A/Bテストで悪化したため不採用——7章参照）。
- ⚠️ `INACTIVE`（自分の手番でない）側の`observation`は、直後のリプレイJSONを見ると**次に
  自分の手番が回ってくるまで更新されない**ことがある（同じ`current`スナップショットが複数
  ステップにわたって繰り返し記録される）。リプレイを解析する際、「自分がINACTIVEの間に
  記録された`current`」は相手の1ターン分のサブアクション消化中の古い状態である可能性が高く、
  実際の最新盤面ではない点に注意（例: 自分視点で「サイドを2枚多く取っていた」ように見えた
  試合が、実際にはその後のINACTIVE区間で逆転されて負けていた、という観測あり）。

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
- `weakness`/`resistance`/`energyType`: 0=無色, 1=草, 2=炎, 3=水, 4=雷, 5=超, 6=闘, 7=悪, 8=鋼, 9=ドラゴン
  （基本エネルギーのID1-8の並びから逆算。ドラゴン/無色は他カードとの整合性からの推定）。
  ⚠️ `pokemonType`は**このマッピングと一致しないことがある**（実測: Mega Lucario ex は
  `energyType=6`(闘)で技コストも闘エネルギーのみだが`pokemonType=4`(雷相当の値)——おそらく
  `pokemonType`は別の内部分類で、技コスト判定には`energyType`を使うべき）。
- ポケモンカードの主なフィールド: `hp`, `basic`/`stage1`/`stage2`（進化段階、いずれもFalseなら
  他の段階と推定）, `evolvesFrom`（進化元の名前文字列、`None`なら基本ポケモン）, `ex`/`megaEx`
  （通常exとMega進化exは別フラグ——実測でMega Lucario exは`ex=False, megaEx=True`）,
  `retreatCost`（にげるコスト）, `attacks`（`attackId`のリスト）。
- 技（`AllAttack()`)の主なフィールド: `damage`（額面ダメージ、**0は「ダメージなし」を意味しない
  場合がある**、下記参照）, `energies`（コストのエネルギータイプIDのリスト、長さ=必要エネルギー数）,
  `text`（効果テキスト、英語）。
- **`damage:0`だが実際はダメージが出る技に要注意**。効果テキストで条件付きダメージを記述して
  いる技は `damage` フィールドが `0` になっている（そのカード自身の"確定ダメージ"ではないという
  意味だと思われる）。実測で2パターン確認:
  - 「デッキ上からN枚を今すぐ捨て、捨てた基本Xエネルギーの枚数×Yダメージ」
    （例: Mega Abomasnow exのHammer-lanche）→ 自分のデッキのエネルギー比率から期待値を推定できる。
  - 「捨て札にある基本Xエネルギーの枚数×Yダメージ」（例: Kyogreのリップタイド）→
    今の捨て札を数えれば正確な値が分かる（推定ではなく実測できる）。
  `attack_score()`はこの2パターンを正規表現でテキストから検出し、`damage:0`のまま評価する
  のを避けている（`submission/main.py`の`_expected_discard_damage`/`_discard_pile_damage`）。
  **他にも同種のテキストパターン（例えば相手の捨て札依存、手札枚数依存等）が存在する可能性が
  高く、今回発見した2つで全てではないと考えるべき。**
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
  （`attacks[].damage` の額面値のみで評価している）。弱点・抵抗の数値自体は`CARD_DB`から
  読み取れるので、実装コストは低い（未着手なだけ）。
- 特性（Ability）・とくしゅエネルギーの詳細効果テキストの自動解釈は行っていない
  （カード名・cardType・数値ステータスのみに基づくヒューリスティック）。
- `area=6`（3.3節）や`evolutionType`の意味は未解読のまま。
- `pokemonType`と`energyType`が食い違うケースがある（5章）ため、タイプ関連の判定は
  必ず`energyType`（技コストの実際のタイプ）を使うこと。`pokemonType`をタイプ相性判定に
  使う場合は要検証。

## 8. リプレイJSON解析で分かったこと（本番ランクマッチの後付け調査）

自己対戦（`tools/evaluate.py`）とは別に、Kaggle上で実際に行われた対戦のリプレイJSON
（`env.run()`が返す形式と同じ、Kaggleの対戦詳細ページからダウンロード可能）を解析して
方策改善のヒントを得るサイクルを回した。この節はそのために必要な知識をまとめる。

- **`steps[i][player]["action"]` は `steps[i-1][player]["observation"]` に対する応答**——
  同じインデックス`i`の`observation`とは対応しない（1つ前を見る必要がある）。これを間違えると
  「エージェントが選ぶはずのない選択肢を選んでいる」という誤った結論に至る（実際に一度
  誤認した）。
- 自分がどちらの`playerIndex`か分からない場合、`steps[0][p]["visualize"][0]["action"]`
  （両プレイヤーの提出デッキがカードIDの配列で入っている）を自分の`load_deck()`の結果と
  比較すれば判定できる。
- `status`が`INACTIVE`の側は、その区間のアクションが`[]`（空）で記録される。相手が1ターン内で
  複数のサブアクション（進化・エネルギー付け・カードプレイ等）を行う間、こちら側は複数ステップ
  連続で`INACTIVE`のままになるのが正常（異常ではない）。
- `logs`（2章）の各要素の`type`は依然完全には解読していないが、実測で `type:15`=攻撃実行
  （`attackId`, `cardId`, `playerIndex`付き）, `type:16`=ダメージ付与（`value`が負数、対象は
  `cardId`+`serial`）, `type:6`/`type:7`=カード移動（`fromArea`/`toArea`、`toArea:3`＝トラッシュ
  への移動＝多くはKO）と推定できる。これだけで「誰が何にどれだけダメージを与え、何がKOされたか」
  の再構築が可能。
