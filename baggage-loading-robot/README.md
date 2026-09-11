# baggage-loading-robot

SIGNATE「グランドハンドリング（手荷物積付ロボット）」チャレンジ向けの提出エージェント。
公式シミュレータの `Agent`（`__init__` / `get_init_states` / `optimize` / `policy`）
仕様に準拠したハイブリッド（オフライン最適化 + オンライン逐次配置）アルゴリズムを実装している。

アルゴリズムの設計意図・前提・既知の制約は [`docs/DESIGN.md`](docs/DESIGN.md) を参照。

## ディレクトリ構成

```
baggage-loading-robot/
├── submit/                    : 提出物（このディレクトリをそのままzip化する）
│   ├── agent.py                 : エージェントのエントリポイント
│   ├── requirements.txt
│   └── gh_baggage_core/         : 実装本体（ハイトマップ再構築・配置探索・並び替え）
│       ├── geometry.py
│       ├── container_state.py
│       ├── packing.py
│       ├── ordering.py
│       └── policy.py
├── tests/                      : pybullet不要のロジック単体テスト・契約テスト
├── dev/                        : 実機シミュレータ向けのローカル回帰ベンチマーク（下記参照）
└── docs/
    └── DESIGN.md                : アルゴリズム設計メモ
```

公式のシミュレータ本体（PyBulletベースの物理演算環境、`agents/`, `configs/`,
`scripts/run_test.py` など）はSIGNATE配布のスターターキットに含まれるものであり、
本リポジトリには含めていない。ローカルで配布キットを展開し、
`agents/submit/` 以下に `submit/` の中身を配置した上で

```bash
cd /path/to/simulator
python -m scripts.run_test --module-path agents/submit/
```

を実行すると、実際の物理シミュレーション込みで動作確認できる。

## テストの実行

```bash
cd baggage-loading-robot
pip install pytest numpy
python -m pytest tests/ -q
```

`tests/test_agent_contract.py` は、PyBulletを使わない軽量なモック環境で
`Agent` の3メソッドを一通り呼び出し、全手荷物が例外なく配置されること・
アクションのフォーマットが仕様通りであること・優先手荷物が優先コンテナへ
ルーティングされることなどを検証する。ただし物理的な妥当性（搬入経路の
干渉、配置後の安定性など）までは検証できないため、提出前には必ず公式
シミュレータ (`scripts/run_test.py`) で実行確認すること。

## 実機シミュレータでのローカル回帰ベンチマーク

配布キットの `configs/sample_config.json` にはタスクが2つ（`000`/`001`）しか
入っておらず、これだけでは実際のSIGNATE評価基盤にある「コンテナ数・
サイズ・荷物構成の異なる多数の隠しテストケース」を代表できない
（実際、`docs/DESIGN.md`に記録の通り、ローカルの2ケースでは無害に見えた
変更が実提出でスコアを大きく下げたことがある）。この2ケースだけで
「変更が安全そうだ」と判断するのは危険なので、`dev/`に、より多様な
条件（複数コンテナ、優先コンテナ、棚あり、優先/ソフト貨物混在、
荷物サイズの偏りなど）を機械的に生成するスクリプトを用意している。

```bash
# 1. 配布キットの configs/ 以下にベンチマーク用configを生成
python baggage-loading-robot/dev/gen_benchmark_configs.py --out-dir /path/to/simulator/configs/bench

# 2. 配布キットのルートから、各configを実機シミュレータで実行
cd /path/to/simulator
for f in configs/bench/*.json; do
  name=$(basename "$f" .json)
  python -m scripts.run_test --module-path agents/submit/ --config-path "$f" \
      --result-dir results/ --result-fname "bench_${name}.json"
done

# 3. 結果を一覧表示（fill_score・配置率・打ち切り理由を要約）
python /path/to/baggage-loading-robot/dev/summarize_bench_results.py --result-dir results/ --prefix bench_
```

コードのスコアリングロジックを変更したときは、`sample_config.json`の2
タスクだけでなく、このベンチマーク一式でも退行がないことを確認してから
提出することを推奨する（それでも本番の隠しテストケースを完全には
代表できない点に注意 -- あくまで「明らかな退行」を検出するための
追加の安全網）。

## 提出用zipの作成

```bash
cd baggage-loading-robot/submit
zip -r ../submit.zip .
```
