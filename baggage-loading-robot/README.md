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

## 提出用zipの作成

```bash
cd baggage-loading-robot/submit
zip -r ../submit.zip .
```
