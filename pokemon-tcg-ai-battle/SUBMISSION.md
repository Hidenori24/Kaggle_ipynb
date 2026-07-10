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

## Kaggle CLIでの提出・結果確認（ローカル）

```bash
pip install kaggle
# https://www.kaggle.com/settings -> API -> Create New Token -> ~/.kaggle/kaggle.json に保存
chmod 600 ~/.kaggle/kaggle.json

bash tools/kaggle_submit.sh "任意のメッセージ"   # submission/{main.py,deck.csv} をzip化して提出
bash tools/kaggle_status.sh                      # 提出履歴とリーダーボードを確認
```

## Kaggleのランタイムでの実行確認（Kaggle Notebook）

`submission/main.py` はGitHub上では相対パスでファイルを読むが、Kaggle Notebook（Kernel）は
独立した実行環境でリポジトリの他ファイルを読めないため、`tools/build_kaggle_kernel.py` が
`main.py`の内容をそのままセルに埋め込んだ**自己完結型のノートブック**を生成する。

```bash
python tools/build_kaggle_kernel.py       # kaggle_kernel/ に notebook.ipynb + kernel-metadata.json を生成
#   -> kaggle_kernel/kernel-metadata.json の "id" が自動検出できなければ手動で
#      "<あなたのKaggleユーザー名>/pokemon-tcg-ai-battle-agent" に書き換える
bash tools/kaggle_push_kernel.sh          # Kaggleにpush（`kaggle kernels push`）
bash tools/kaggle_kernel_status.sh        # 実行ステータス確認 + 実行結果(ログ・実行済みnotebook)をダウンロード
```

このノートブックはKaggle上で `random_agent`/`first_agent` との自己対戦を実際に走らせるので、
提出前にKaggle側のランタイムでもクラッシュしないことを確認できる（`kaggle_kernel/`は
`.gitignore`対象——`main.py`から再生成する使い捨てのビルド成果物のため）。

## GitHub Actionsでの自動化

`.github/workflows/` に4つのワークフローがある。

| ワークフロー | トリガー | 内容 |
|---|---|---|
| `pokemon-tcg-ci.yml` | push / PR（自動） | 実エンジンでのテスト・勝率計測（`test`ジョブ）＋ `notebooks/` が実際に最後まで実行できるかの検証（`notebooks`ジョブ、独立実行なので一方の失敗が他方を隠さない） |
| `pokemon-tcg-kaggle-submit.yml` | 手動（Actionsタブから実行） | テスト→Kaggleへ提出→結果をジョブサマリーに表示 |
| `pokemon-tcg-kaggle-kernel.yml` | 手動 | Kaggle Notebookのpush→ステータス確認 |
| `pokemon-tcg-kaggle-watch.yml` | 3時間おき（自動）＋手動 | 提出結果をポーリングし、新しい結果だけをリポジトリの「Kaggle submission watch」Issueにコメントで報告 |

`pip`の依存関係キャッシュ（`actions/setup-python`の`cache: pip`）を全ワークフローで有効化しているほか、
`pokemon-tcg-ci.yml`は同じブランチ／PRへの新しいpushが来たら実行中の古いジョブを自動キャンセルする
（`concurrency`設定）。

提出系の2つはリポジトリの **Settings → Secrets and variables → Actions** に
`KAGGLE_USERNAME` と `KAGGLE_KEY`（Kaggle APIトークンの中身）を登録すれば動く。
1日あたりの提出回数制限があるため、`push`のたびに自動提出はされない設計にしている
（意図的に手動トリガーのみ）。これで「pushしてActionsタブのボタンを押すだけで
提出→結果確認まで完結する」環境になる。

### 提出結果の自動監視（`pokemon-tcg-kaggle-watch.yml`）

Kaggle APIには「提出が採点されたら通知する」という push型（Webhook）の仕組みは無く、
`kaggle competitions submissions`を定期的にポーリングして状況を見に行くしかない。
このワークフローは3時間おきに自動実行され、前回までに報告済みの結果（ステータス・スコアの
組み合わせをフィンガープリント化してIssueコメントに埋め込んで判定、`tools/kaggle_watch.py`）
と比較して、**新しい結果（新規提出、または pending→complete などの状態変化）だけ**を
リポジトリの Issue「Kaggle submission watch」にコメントとして追加する。Issueが無ければ
自動作成される。追加のSecret登録は不要（既存の`KAGGLE_USERNAME`/`KAGGLE_KEY`と、
ワークフロー自身の`GITHUB_TOKEN`のみで動く）。

このコンペはシミュレーション（対戦ラダー）形式のため、`publicScore`単体では
「何と何が起きているか」が分かりにくい。そこでこのワークフローは提出ステータスの
コメントに加えて、直近の提出について**個々の対戦（Episode）の結果**も
`kagglesdk`（`kaggle`パッケージ内部のpythonクライアント、素の`kaggle` CLIの表には
出ないネストされた`agents`情報を含む）経由でポーリングし、前回チェック以降に
終了した対戦について:

- 勝敗数（W-L-D）
- 対戦相手（チーム名ごとの対戦回数）
- クラッシュ・タイムアウト・不正選択などエラー終了した対戦の内訳

をまとめてIssueにコメントする。さらに、**負けた対戦のリプレイJSON**は
自動でダウンロードされ、そのワークフロー実行のGitHub Actions Artifact
（`kaggle-loss-replays-<run id>`、30日間保持）としてアップロードされる。
実戦の負けリプレイをアップロードして分析する、というこれまでこのリポジトリで
何度も手動でやってきた作業（`docs/ENGINE_NOTES.md`参照）を、自動収集の部分だけ
先取りしておく位置づけ。

ダウンロードしたリプレイJSONは、ワークフロー実行環境（GitHub Actionsのランナー）上で
その場で軽く解析もされ、Issueコメントに1行ずつ追記される: どのターンで負けたか、
負けた瞬間のベンチ枚数・アクティブHP、そして最も重要な点として**「実際にクラッシュ/
タイムアウトして負けたのか、それとも普通にKOされて負けたのか」**（`status`が`DONE`
以外なら前者）。開発環境（サンドボックス）はネットワーク制限でKaggleのリプレイ格納先
（Azure Blob Storage）に直接アクセスできないため、この解析ロジック自体は実際にダウンロード
された本番リプレイJSONではまだ検証できていない——`tools/kaggle_watch.py`のコード内コメントに
その旨を明記している。ローカルの実エンジンで生成した負けリプレイでは動作確認済み。

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
