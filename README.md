# Kaggle_ipynb

Kaggle/SIGNATEコンペティションへの取り組みをまとめるリポジトリ。

## コンペ一覧

- [`pokemon-tcg-ai-battle/`](pokemon-tcg-ai-battle/) — [Pokémon TCG AI Battle Challenge](https://www.kaggle.com/competitions/pokemon-tcg-ai-battle)
  向けのルールベース対戦エージェント。実際のKaggle対戦エンジンをリバースエンジニアリングし、
  オフラインで自己対戦検証を行った上でエージェント・デッキを構築している。詳細は
  [`pokemon-tcg-ai-battle/README.md`](pokemon-tcg-ai-battle/README.md) を参照。
- [`baggage-loading-robot/`](baggage-loading-robot/) — SIGNATE「グランドハンドリング
  （手荷物積付ロボット）」チャレンジ向けのハイブリッド（オフライン最適化＋オンライン
  逐次配置）ヒューリスティックエージェント。ハイトマップベースの3D bin-packing探索で
  積付位置・向き・コンテナを決定する。詳細は
  [`baggage-loading-robot/README.md`](baggage-loading-robot/README.md) を参照。
- [`rsna-knee-abnormality-detection/`](rsna-knee-abnormality-detection/) — [RSNA Knee Abnormality Detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection)
  膝MRIの12所見予測（マクロAUC）。Python学習＋C++は前処理/推論に限定する方針。詳細は
  [`rsna-knee-abnormality-detection/README.md`](rsna-knee-abnormality-detection/README.md) を参照。
