# 提出方法

## 提出物

- `submission/main.py` — `agent(obs: dict) -> list[int]` を定義したエージェント本体。
- `submission/deck.csv` — 60枚のデッキ（カードID、1行1枚）。

この2ファイルだけで完結する（カードデータベースは `kaggle-environments` に同梱されている
ネイティブライブラリから実行時に読み込むため、別途データファイルを同梱する必要はない）。

## 提出前のローカル検証

```bash
pip install -r requirements.txt
python -m pytest tests/            # クラッシュ・不正選択が無いことを確認
python tools/evaluate.py --games 40 --opponent random
python tools/evaluate.py --games 40 --opponent first
```

`tests/test_policy.py` は実際に `kaggle-environments` 同梱のネイティブエンジンで
複数戦フルプレイし、エージェントのステータスが常に `DONE`（`TIMEOUT`/`ERROR`/`INVALID`
にならない）ことを検証する。タイムアウト・クラッシュ・不正選択は即敗北というルールのため、
これが提出前に確認すべき最優先事項になる。

## 注意（本リポジトリの開発環境について）

このリポジトリはKaggleアカウントの認証情報を持たないサンドボックス環境で開発した。
`kaggle-environments==1.30.1` の pip パッケージにゲームエンジン本体
（`kaggle_environments/envs/cabt/cg/libcg.so`、カード・技データベースAPIを含む）が
まるごと同梱されていたため、**Kaggleへのログインや対戦データのダウンロードなしに、
本物のエンジンでオフライン検証ができた**（`docs/ENGINE_NOTES.md` 参照）。

一方で、Kaggle サイト上の実際の「エージェント提出」UI（アップロード形式・ファイルサイズ制限・
tarballが必要かどうか等）は未確認である。提出時は
[コンペティションページ](https://www.kaggle.com/competitions/pokemon-tcg-ai-battle)の
公式の提出手順を必ず確認してほしい。
