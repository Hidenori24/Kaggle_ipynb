# Pokémon TCG AI Battle Challenge

[コンペティションページ](https://www.kaggle.com/competitions/pokemon-tcg-ai-battle)
（The Pokémon Company × Kaggle）向けの、ルールベース対戦エージェントの実装一式。

このコンペは通常のCSV予測コンペではなく、**Kaggleのシミュレーション環境
（`kaggle_environments`、内部名 `cabt`）上でポケモンカードゲームを自動対戦させる**形式。
参加者は `agent(obs: dict) -> list[int]` という関数と60枚のデッキを提出し、
他の参加者のエージェントと自動で対戦してEloランキングが決まる。

## 何をやったか（要約）

1. **公式ドキュメントが薄いため、まず対戦エンジンをリバースエンジニアリングした。**
   `pip install kaggle-environments==1.30.1` すると、実際にKaggleサーバー上で
   動いているのと同じコンパイル済みネイティブエンジン（`libcg.so`、全1267枚の
   カードデータベースと1556個の技データベースを返す `AllCard`/`AllAttack` API込み）が
   丸ごと手元に来ることを発見。これにより **Kaggleへのログインなし・データダウンロードなしで、
   本物のエンジンを使ったオフライン開発・検証**が可能になった
   （詳細: [`docs/ENGINE_NOTES.md`](docs/ENGINE_NOTES.md)）。
2. 実際にランダムエージェント同士を数百戦させて `obs` の構造を実地調査し、
   カードデータベースを引きながら行動を採点するルールベースエージェントを実装
   （[`submission/main.py`](submission/main.py)）。
3. デッキ案を実際のエンジンで自己対戦させて比較し（
   [`tools/build_deck.py`](tools/build_deck.py)、
   [`notebooks/02_agent_evaluation.ipynb`](notebooks/02_agent_evaluation.ipynb)）、
   最も勝率の高かったデッキを [`submission/deck.csv`](submission/deck.csv) として採用。
   当初はサンプル付属の「Mega Abomasnow ex」水デッキを使っていたが、実際のランクマッチの
   負けリプレイ13戦を分析した結果、Basicポケモンが6枚しかなくベンチが育たない構造的弱点と、
   対戦相手（Dragapult ex等）の方がダメージ/エネルギー比で明確に優れていたことが判明。
   カードプール全体を調査して「Mega Lucario ex」闘デッキに切り替え、旧デッキとの直接対決で
   350戦中 約54〜56% の勝率を確認して採用。その後さらに新デッキでの実戦5連敗を分析した結果、
   相手の技効率よりも「進化先（Mega Lucario ex）への到達率の低さ」自体が主因と判明したため、
   軸（Riolu→Mega Lucario ex）は変えずに、進化ラインを直接サーチできるサポーター
  （Brock's Scouting・Cyrano）でBoss's Orders/Judgeを置き換える v2 改訂を実施。
   同じエージェントロジックでの直接対決（5シード・260戦）で v2 が約60%の勝率を確認して採用
  （詳細は deck.csv のコメント参照）。その後、実戦データで判明した「負けの一定割合が
   基本ポケモンの手詰まり（ブリック）で終わる」問題に対し、2種目の基本ポケモン
  （Farfetch'd）を追加する案を検証。単純にカードを差し替えるだけでは5回連続で不採用と
   なったが、`card_value()`にRioluのような「進化元」への評価ボーナスを追加する
   ロジック修正と組み合わせたところ、Petrelとの差し替えで3,600戦を通じて安定した
   改善（約51.75%）を確認し、v3として採用（詳細は
   [`STRATEGY_REPORT.md`](STRATEGY_REPORT.md) 5.6節参照）。
   さらに、一般的なポケモンカードの上級者戦略（プライズトレード等）をルールベースに
   落とし込む方向性で調査を行い、実測により「megaExポケモンの被撃破は3プライズ
  （通常のポケモンは1プライズ）」という本エンジン独自の未知仕様を発見。これを踏まえ、
   Mega Lucario exが低HPになった際の退却判断をより早めるロジック（プライズ価値を考慮した
  `active_in_danger()`の閾値変更）を追加し、独立した3回の検証（合計1,800戦）で
   一貫した改善（約53.8%）を確認して採用（詳細は
   [`STRATEGY_REPORT.md`](STRATEGY_REPORT.md) 5.7〜5.8節参照）。さらに`prize_value`を
   無印`ex`（2プライズ）にも一般化した上で、Boss's Ordersを使った相手ベンチ狙撃ロジックを
   実装・検証したが、既存デッキの非エネルギーカードは全て4枚上限まで最適化済みで
   エネルギーを削る以外に空きがなく、2回の独立した600戦検証（45.7%・42.7%）で明確な
   悪化を確認したため不採用とした（削減幅を半分にした軽量版も51.0%とノイズと区別できず
   採用基準に届かず）。実装自体の正しさ（複数候補からの選択が常に最良のものになること）は
   150戦で直接確認済みだが、デッキ全体としては見送り（詳細は
   [`STRATEGY_REPORT.md`](STRATEGY_REPORT.md) 5.9節参照）。その後、実際のKaggle
   ラダーの負けリプレイ3件をJSONレベルで直接解析し、「HP満タンのまま負ける」という
   長年の謎の一部が`kaggle_watch.py`の観測スナップショット遅延によるロギング上の
   アーティファクトだったことを特定するとともに、相手の手札枚数に比例して際限なく
   伸びる一撃必殺技（例: Alakazamの「Powerful Hand」）に対する完全な盲点を発見。
   `opponent_lethal_threat_damage()`を追加し、HP割合に関わらずこの種の致死級の
   脅威を検知して退却優先度を上げるよう`active_in_danger()`を拡張した（詳細は
   [`STRATEGY_REPORT.md`](STRATEGY_REPORT.md) 5.10節参照）。さらに、コインフリップ技
  （自分のRioluの「Quick Attack」自体がこのパターン）の期待値がダメージ見積もりに
   一切反映されていなかった問題を修正。`attack_score()`側の期待値反映は自己対戦A/B
  （5回・3,000戦・プールして51.4%）で改善を確認して採用したが、`attack_is_lethal()`
   側で「表が出れば倒せるなら賭ける」という再分類まで行うと逆にA/Bで悪化したため
  （3回・1,800戦・プールして48.8%）、そちらは確定ダメージのみに差し戻した——1つの
   変更を丸ごと採用/却下ではなく、A/Bで裏目に出た部分だけを切り分けて差し戻した
   初めてのケース（詳細は[`STRATEGY_REPORT.md`](STRATEGY_REPORT.md) 5.11節参照）。
   その後さらに、同一シグネチャ（turn11・HP満タン被撃破）の4件目の負けリプレイを
  `obs.logs`のイベントログレベルで精査したところ、相手がKadabra→Alakazamへの
   進化とPowerful Handの発動を同一ターン内で完結させる、より複雑な形の一撃必殺
   だったことが判明。既存の検知ロジック（5.10節）は「今このターンに使える技」しか
   見ない設計であるため原理的に検知範囲外であり、コード側の不具合ではないと結論。
   対処には1ply相手先読み（未着手のバックログ項目）が必要だが、追加確認1件のみでは
   実装に踏み切る根拠として不十分と判断し、今回はコード変更を行わず調査結果のみを
   記録した（詳細は[`STRATEGY_REPORT.md`](STRATEGY_REPORT.md) 5.12節参照）。その後も
   実戦成績を追跡した結果（61.8%まで改善）、同系統の不自然な負けが依然散見されたため
   判断を見直し、`opponent_lethal_threat_damage()`を相手アクティブの現在の技だけでなく
   1段先の進化先の技までチェックするよう拡張（`CARD_DB`全体から動的に構築する進化
   逆引きテーブルを追加）。実リプレイへの再適用で正しく発火することを確認し、自分の
   進化ラインには手札スケーリング技がないため自己対戦でもノーオペ（300戦・47.7%、
   ノイズ帯内）であることを確認して採用した（詳細は
   [`STRATEGY_REPORT.md`](STRATEGY_REPORT.md) 5.13節参照）。
4. 実測の勝率・カードプールの分析結果を Jupyter Notebook にまとめた
   （捏造データなし、すべて実エンジンでの実行結果）。

## 実測結果（自己対戦、実エンジン使用）

| 対戦相手 | 勝率 | 備考 |
|---|---|---|
| `random_agent`（ランダム） | 92〜98% | 複数回の試行で安定して高勝率 |
| `first_agent`（常に先頭の選択肢を選ぶ決定的Bot） | 54〜70%（試行間で分散大） | 「Mega Lucario ex」v2デッキ採用後の実測値 |
| クラッシュ・不正選択 | 0件 / 100戦超 | タイムアウト即敗北ルール下での最優先事項 |

`first_agent` に対する勝率が試行によってばらつく理由や、その他の限界・今後の改善案は
[`notebooks/02_agent_evaluation.ipynb`](notebooks/02_agent_evaluation.ipynb) の
「まとめと今後の課題」に正直にまとめてある。

## リポジトリ構成

```
pokemon-tcg-ai-battle/
├── submission/
│   ├── main.py        # 提出用エージェント本体（自己完結）
│   └── deck.csv        # 提出用デッキ（60枚）
├── tools/
│   ├── evaluate.py             # 実エンジンでの自己対戦・勝率計測CLI
│   ├── build_deck.py           # デッキ案のA/Bテスト（deck.csv の選定根拠を再現）
│   ├── kaggle_submit.sh        # submission/ をzip化してKaggleに提出
│   ├── kaggle_status.sh        # 提出履歴・リーダーボードを確認
│   ├── build_kaggle_kernel.py  # main.pyを自己完結ノートブックに変換（Kaggle Notebook用）
│   ├── kaggle_push_kernel.sh   # ↑をKaggleにpush（Kaggleのランタイムで実行）
│   └── kaggle_kernel_status.sh # Kaggle Notebookの実行結果を確認・ダウンロード
├── tests/
│   ├── test_policy.py      # 実エンジンでのクラッシュ・不正選択防止テスト（pytest）
│   └── test_heuristics.py  # score_option()内の個別ロジック（bench_is_thin等）の単体テスト
├── notebooks/
│   ├── 01_card_pool_eda.ipynb      # 実カードデータベースのEDA
│   └── 02_agent_evaluation.ipynb   # デッキ比較・勝率検証（実行済み）
├── docs/
│   └── ENGINE_NOTES.md  # obs/action スキーマのリバースエンジニアリング結果
│                         # ＋実戦リプレイJSON解析のハマりどころ
├── SUBMISSION.md         # Kaggle提出手順（CLI・Kaggle Notebook・GitHub Actions）
├── STRATEGY_REPORT.md    # 戦略トラック向けレポート（日本語・英語両方収録）
├── requirements.txt      # 提出物の実行に必要な最小限の依存
└── requirements-dev.txt  # ＋ notebooks/ 再実行用（jupyter, matplotlib, pandas等）

../.github/workflows/
├── pokemon-tcg-ci.yml                # push/PRで自動: 実エンジンでのテスト・勝率計測
├── pokemon-tcg-kaggle-submit.yml     # 手動実行: テスト→Kaggle提出→結果をジョブサマリーに表示
├── pokemon-tcg-kaggle-kernel.yml     # 手動実行: Kaggle Notebookのpush→ステータス確認
└── pokemon-tcg-kaggle-watch.yml      # 3時間おきに自動実行: 提出結果をポーリングしIssueに報告
```

## 使い方

```bash
pip install -r requirements.txt

# クラッシュ・不正選択がないことを確認
python -m pytest tests/

# ベースラインへの勝率を実測
python tools/evaluate.py --games 40 --opponent random
python tools/evaluate.py --games 40 --opponent first

# デッキ案のA/Bテストを再現
python tools/build_deck.py --games 20
```

Jupyter Notebook を再実行する場合（`notebooks/` から相対パスで `submission/main.py` を
読み込む想定。EDA・グラフ描画用の追加パッケージが必要）:

```bash
pip install -r requirements-dev.txt
cd notebooks && jupyter nbconvert --to notebook --execute --inplace 01_card_pool_eda.ipynb 02_agent_evaluation.ipynb
```

## Kaggleへの提出・結果確認・Kaggle Notebookでの実行

Kaggle CLI (`kaggle competitions submit`) での提出、`kaggle kernels push` によるKaggle
Notebook（Kaggleのランタイム上）での実行確認、GitHub Actionsでの「pushして自動テスト、
ボタン一つで提出→結果確認」の自動化については [`SUBMISSION.md`](SUBMISSION.md) を参照。

## 戦略トラック向けレポート

戦略トラック（Strategy Category）向けの、デッキ設計・エージェント戦略・検証結果・
実戦データから見つけた具体的な改善点をまとめたレポートは
[`STRATEGY_REPORT.md`](STRATEGY_REPORT.md)（日本語・英語両方収録）を参照。

## 参考

- コンペ公式: [Pokémon TCG AI Battle](https://www.kaggle.com/competitions/pokemon-tcg-ai-battle)
- 戦略トラック: [PTCG AI Battle Challenge Strategy](https://www.kaggle.com/competitions/pokemon-tcg-ai-battle-challenge-strategy)
- 参考ノートブック: [pok-mon-tcg-ai-strategy-analysis](https://www.kaggle.com/code/nmatsumoto24/pok-mon-tcg-ai-strategy-analysis)（`nmatsumoto24` 氏）
