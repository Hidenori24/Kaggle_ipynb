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
  （詳細は deck.csv のコメント参照）。
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
