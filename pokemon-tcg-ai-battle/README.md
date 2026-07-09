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
3. 3つのデッキ案を実際のエンジンで自己対戦させて比較し（
   [`tools/build_deck.py`](tools/build_deck.py)、
   [`notebooks/02_agent_evaluation.ipynb`](notebooks/02_agent_evaluation.ipynb)）、
   最も勝率の高かったデッキを [`submission/deck.csv`](submission/deck.csv) として採用。
4. 実測の勝率・カードプールの分析結果を Jupyter Notebook にまとめた
   （捏造データなし、すべて実エンジンでの実行結果）。

## 実測結果（自己対戦、実エンジン使用）

| 対戦相手 | 勝率 | 備考 |
|---|---|---|
| `random_agent`（ランダム） | 87.5〜95% | 複数回の試行で安定して高勝率 |
| `first_agent`（常に先頭の選択肢を選ぶ決定的Bot） | 37〜63%（試行間で分散大） | 詳細と考察は notebook 参照 |
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
│   ├── evaluate.py      # 実エンジンでの自己対戦・勝率計測CLI
│   └── build_deck.py    # デッキ案のA/Bテスト（deck.csv の選定根拠を再現）
├── tests/
│   └── test_policy.py   # 実エンジンでのクラッシュ・不正選択防止テスト（pytest）
├── notebooks/
│   ├── 01_card_pool_eda.ipynb      # 実カードデータベースのEDA
│   └── 02_agent_evaluation.ipynb   # デッキ比較・勝率検証（実行済み）
├── docs/
│   └── ENGINE_NOTES.md  # obs/action スキーマのリバースエンジニアリング結果
├── SUBMISSION.md         # Kaggle提出手順
└── requirements.txt
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
読み込む想定）:

```bash
cd notebooks && jupyter nbconvert --to notebook --execute --inplace 01_card_pool_eda.ipynb 02_agent_evaluation.ipynb
```

## 参考

- コンペ公式: [Pokémon TCG AI Battle](https://www.kaggle.com/competitions/pokemon-tcg-ai-battle)
- 戦略トラック: [PTCG AI Battle Challenge Strategy](https://www.kaggle.com/competitions/pokemon-tcg-ai-battle-challenge-strategy)
- 参考ノートブック: [pok-mon-tcg-ai-strategy-analysis](https://www.kaggle.com/code/nmatsumoto24/pok-mon-tcg-ai-strategy-analysis)（`nmatsumoto24` 氏）
