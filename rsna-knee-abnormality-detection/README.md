# RSNA Knee Abnormality Detection

[コンペ](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection) — 膝MRI(DICOM)から12所見の
study単位確率を予測。評価は12ラベルのマクロ平均AUC。

## C++ で出るか？（判断）

**結論: 全部C++は現実的でない。Pythonで学習し、C++は「DICOM前処理」と「推論」の一部に限定する。**

| 工程 | 言語 | 理由 |
|---|---|---|
| レポートからの疑似ラベル生成 | Python | 多言語テキスト処理。ラベル付きstudyはごく一部 |
| CNN学習 | Python(PyTorch/GPU) | C++(LibTorch)で学習するメリットが薄い |
| DICOM読込(JPEG Lossless/JPEG2000/Implicit VR混在) | Python(pydicom+gdcm/pylibjpeg) | C++ではGDCM/DCMTKが必要。Kaggleのコードコンペ環境(オフライン)で
ビルド・依存導入が最大のリスク |
| 前処理の高速化・推論 | C++(任意) | ONNX Runtime + GDCM が使えるなら可能。まず`g++`と各ライブラリがKaggle Notebookで使えるか要検証 |

まずPythonでスコアを出し、ボトルネックが見えたらC++化する。

## 方針（Python ベースライン）

1. `train.csv`のレポートからキーワードで全studyの疑似ラベルを作る（`src/pseudo_labels.py`）。正解ラベルがある行はそちらを優先。
2. `train_series.csv`のメタデータで、所見ごとに有効な series（例: Sagittal+Fluid_Sensitive）を選ぶ。
3. 各seriesから中央付近のスライスを固定枚数サンプルし、2D CNN特徴→study単位で集約→12出力。
4. テストにはReportが無いので、画像のみで推論する。

## 注意

- 有病率は train / public / private で異なるため、AUC(順位)重視で校正は不要。
- コードコンペの場合、提出Notebookはインターネット不可。事前学習重みはKaggle Datasetsとして追加すること。

## 現状

- [x] 方針決定・雛形
- [x] レポート疑似ラベル（英語キーワードのみ。他言語は未対応）
- [x] DICOM読込・series選択（`src/dicom_io.py`, `src/features.py`。合成DICOMで動作確認のみ）
- [x] ベースラインNotebook（`notebooks/01_baseline.ipynb`: 手作り特徴+LightGBM。Kaggle上で未実行）
- [ ] モデル学習
- [ ] 提出Notebook / C++ 推論の検証

## Kaggle Notebook での使い方

1. コンペページの Code → New Notebook（データは `/kaggle/input/rsna-knee-abnormality-detection/` に置かれる）。
2. `notebooks/01_baseline.ipynb` をアップロードして実行（Internet ONで `src/` を git clone する。OFFにする提出時は `src/` をDatasetとして追加し `SRC` を変更）。
3. 最初は `N=600` で動作確認 → 全件へ。LightGBMのCV AUCは疑似ラベルに対する値で、LBとは一致しない。
4. 次段階: 2D/3D CNN（事前学習重みをDataset化）、ラベル付きstudyのみでの検証分割。

## 自動実行（GitHub → Kaggle）

`.github/workflows/rsna-knee-kaggle-run.yml` が `kaggle kernels push` でNotebookをKaggleに送って実行し、出力(`submission.csv`, ログ)をArtifactに保存する。
一度だけ: GitHubリポジトリの Settings → Secrets → Actions に `KAGGLE_USERNAME` / `KAGGLE_KEY` を登録（pokemonと共通）。
その後は `main` へマージ、または Actions から手動実行で走る。Kaggle側でコンペの Join Competition は済ませておくこと。

## ノートブック

| ファイル | 内容 | 結果(N=600) |
|---|---|---|
| `01_baseline.ipynb` | 手作り特徴+LightGBM | 疑似ラベルAUC 0.72 / 正解ラベルAUC 0.55(n=58) |
| `02_cnn_embed.ipynb` | ResNet18(ImageNet)のスライス埋め込み+LightGBM。GPU | 疑似0.725 / 正解0.594。N=4000でLB 0.641 |
| `04_finetune.ipynb` | ResNet18を3断面×16スライス(224px)でエンドツーエンド学習。スライス間attention pooling。GPU | 8epoch: 疑似0.886 / 正解0.644。**LB 0.659** |

Actionsの実行対象は手動実行時の `notebook` 入力で選ぶ（既定は `02_cnn_embed.ipynb`）。

`02` は次版で、英語以外のレポートを疑似ラベル学習から除外し、N=600→2000に増やした。

## 提出フロー（コードコンペ・Internet OFF）

1. 学習: `rsna-knee Kaggle run`（`02_cnn_embed.ipynb`）。終了後、重み+LightGBMを private Dataset `rsna-knee-model` に自動公開する。
2. 提出: Actions の **rsna-knee Kaggle SUBMIT** を手動実行（`confirm` に `SUBMIT` と入力、`message` 必須）。
   `src/` を Dataset `rsna-knee-src` に更新 → `03_submit.ipynb`（推論のみ・Internet OFF・GPU）をKaggleで実行 → `kaggle competitions submit -k` で提出。
   **提出枠を消費する唯一のworkflow**。自動では走らない。
3. 提出の順序: 先に学習workflowを最低1回成功させる（`rsna-knee-model` が無いと提出workflowは失敗する）。

公開リポジトリのため、実行出力（submission.csv・ログ）はArtifactにしない。結果はKaggle側で確認する。

## 04_finetune の構成（`src/finetune.py`）

- 入力: 各studyの Sagittal/Coronal/Axial から選んだseries × 16スライス × 224px を uint8 でディスクにキャッシュ（`/kaggle/temp`。`/kaggle/working` に置くと出力に含まれる）。
- モデル: 1ch化したImageNet ResNet18を全スライスで共有 → スライス間attention pooling → 断面ごとのベクトルを連結(欠けた断面は0) → 12出力。左右反転の拡張はしない(medial/lateral が入れ替わるため)。
- 学習: 英語レポート由来の疑似ラベル。正解ラベル58件は評価専用、疑似ラベルの10%を検証に使い、検証AUCが最良のepochを保存。
- 学習workflowの既定は `04_finetune.ipynb`。`03_submit.ipynb` は `ft_model.pt` があればそれを使い、なければ従来のLightGBM版を使う。
- 学習workflowは `src/**` の変更でも走る(GPU時間を使う)。README等の変更では走らない。

## 提出履歴

| 版 | LB |
|---|---|
| 02 埋め込み+LightGBM (N=4000) | 0.641 |
| 04 ファインチューニング 8epoch | 0.659 |

## 多言語の疑似ラベル（`src/multilingual_labels.py`）

英語以外のレポート(1240/4407件)は、トルコ語・ギリシャ語・ブルガリア語が大半（Kaggleログの文字種と頻出語から判断）。
`detect_lang` で言語を判定し、言語別のキーワード規則でラベルを作る。否定の位置が言語で違う
（トルコ語は文末の動詞、ギリシャ語・ブルガリア語は前）ので、スコープを言語ごとに切り替えている。
規則は手書きで、**実データでは未検証**（作例の文でのみ確認）。`04_finetune` は言語別の疑似陽性率を表示するので、
英語と桁が違う言語があれば規則を疑う。`USE_LANGS` で使う言語を切り替えられる（`['en']` なら従来どおり英語のみ）。
上記3言語以外(スペイン語など)は `other` として学習から除外する。

## 学習結果の記録

| 実行 | 使用データ | 正解ラベルAUC(n=58) | 備考 |
|---|---|---|---|
| 04 14epoch・英語のみ(ASCII判定) | 3167件 | epoch8で0.67、その後0.64 | epoch8以降は過学習(lossだけ下がる) |
| 04 14epoch・多言語(en/tr/el/bg) | 3259件(en 2157, tr 548, el 321, bg 220) | 0.65〜0.73、後半0.72前後で安定 | `other`(1161件)は未使用。ASCIIのみの非英語レポートをenから外した効果も含む |

言語別の疑似陽性率: ギリシャ語は英語に近い。ブルガリア語(MCL/OA/Synovitis/Baker)とトルコ語(OA/MCL/Synovitis)は極端に低く、語彙の取りこぼしが疑われる。

### スペイン語・ドイツ語の追加と語彙の補強

`other`(1161件)の頻出語から、大半がスペイン語(impresión, rotura, derrame, menisco…)、一部がドイツ語(Gelenkerguss, Innenmeniskus, Rissbildung…)と判明したため規則を追加。
ポルトガル語はスペイン語と単語が重なるが否定語が違う(sem/não)ので `other` に残す。ブルガリア語(скъсване, увреда, бекерова киста, костномозъчен едем)とトルコ語(medyal, sprain, kıkırdak kaybı, daralma)の取りこぼしも補強。
既知の限界: 1文に内側・外側の半月板が並び片方だけ裂傷、という書き方だと両方が陽性になる(文単位の判定)。

## 実行結果(続き)と次の版

| 実行 | 使用データ | 正解ラベルAUC(n=58) | LB |
|---|---|---|---|
| 04 多言語(en/tr/el/bg)・14epoch | 3259件 | 0.65〜0.73 | **0.743** |
| 04 多言語+es/de・10epoch | 4300件前後 | 0.66〜0.75(ブレる) | **0.768** |

58件の評価は実行間で±0.03ほどブレる。ラベル改善の効果はLBで見る。

次の版(`04_finetune`): フランス語規則を追加し、スペイン語のOA規則を補強(cóndilo/platillo/rótula/tróclea)。
乱数と検証分割を変えた3モデルを学習し(約3時間)、`03_submit` は `ft_model*.pt` をすべて読んで予測を平均する。
学習workflowの待ち時間上限は約5.8時間。Kaggleの週あたりGPU時間(約30時間)に注意。

### 3モデルのアンサンブル(10epoch×3)の結果と、epochを6に減らす理由

正解ラベルAUC(n=58): seed0/1/2 = 0.715/0.698/0.684(平均0.699)、アンサンブル0.715。
3シードとも、正解ラベルAUCは epoch 4〜6 でピーク(0.73〜0.77)を打ち、その後0.03〜0.07下がる。一方、疑似ラベルAUCは epoch 8〜9 まで上がる。
→ epoch 5以降は疑似ラベルのノイズを覚え始めている可能性が高い。保存epochは疑似ラベルAUCで選ぶため、後半(ノイズを覚えた側)が選ばれていた。
対策として EPOCHS を 10→6(学習率スケジュールも6epochで完結)。学習時間は約3時間→約1.4時間。

## 次の版: 英語ルールの強化・EMA・診断

- 英語ルール(学習データの約半分)を強化: `torn`/`meniscal`/`oedema`/`synovial thickening`/`chondromalacia`/`osteophyte`/`joint space narrowing` などを追加、文末の否定(`not seen`)に対応、半月板の変性だけの文はOAにしない。
- `finetune.fit` に重みのEMA(0.998)。epoch後半で疑似ラベルのノイズに寄っていく挙動を平滑化し、EMA重みで評価・保存する。
- `04_finetune` が、規則ラベルと正解ラベル58件の一致(所見ごとのTP/FP/FN)と、アンサンブルの所見ごとのAUCを表示する。

## 準備済みで既定オフの2機能(`04_finetune.ipynb` のフラグ)

- `CENTER = 0.7`: 各seriesの中央70%のスライスからK枚を取る(既定1.0=全体から均等)。膝の端のスライスを避ける。
- `USE_META = True`: DICOMヘッダー(メーカー・スライス厚・ピクセル間隔・行列サイズ・TR/TE・フリップ角・磁場強度・スライス枚数)を
  モデルの最終層に連結する(`src/dicom_meta.py`)。画像は正規化されるので施設・スキャナーの手がかりが消えているため。学習中は30%の確率でメタ全体をドロップして画像側を鍛える。
どちらも既定はオフで、#53(英語ルール+EMA)の結果と混ざらないようにしてある。`03_submit` は保存されたチェックポイントの `center` / `meta` を読んで同じ設定で推論する(古い形式のチェックポイントも可)。
試す順序: #53の結果 → `CENTER=0.7` → `USE_META=True`。LBは提出枠の都合で、小さな変更は同時に入れてもよい。

## #53 の結果(LB未提出)と次の版

規則ラベル vs 正解58件: 適合率0.65・再現率0.60。再現率が低い所見ほどAUCが低い
(Synovitis 再現率0.44→AUC0.605、PF OA 0.33→0.651、Medial OA 0.33→0.645、ACL 0.79→0.804)。
正解ラベルAUCは3シード×3実験すべてで epoch 3〜4 がピーク(0.74〜0.76)、epoch 6 では0.71〜0.72(EMAでも防げない)。
3モデルのアンサンブル0.725は単体(0.713〜0.724)とほぼ同じ(モデルが似すぎている)。

- `EPOCHS` 6→4(学習時間も約1時間に)。
- 英語: tricompartmental / multicompartmental / diffuse osteoarthritis の総称表現を、3つのOAラベルすべてに付ける。
- `05_label_diag.ipynb`(GPU不要・約1分): 規則ラベルと正解58件の一致表、所見ごとに「正解陽性で多い語」「規則が見逃した報告で多い語」「誤検出で多い語」(語の文書頻度のみ、本文は出さない)。
  Actions の `rsna-knee Kaggle run` を手動実行し、`notebook` に `05_label_diag.ipynb` を指定する。

## LLMでレポートを読んでラベルを作る(`06_llm_labels.ipynb`, `src/llm_labels.py`)

規則ラベルは正解58件との比較で適合率0.65・再現率0.60。規則を手で直す代わりに、KaggleのGPU上でオープンな重みの
LLM(Qwen2.5-7B-Instruct 4bit。入らなければ3B fp16)にレポートを読ませ、12所見を判定させる。レポートはKaggle上でローカルに処理するだけで外に出さない。
- 出力は `{"ACL":0,...}` のJSON。値のトークンを生成する瞬間の「1」対「0」の確率から、所見ごとのソフトラベルが1パスで得られる。
- ステージ3(この版): 正解58件だけで、規則とLLMを同じ表(適合率・再現率・AUC・両者の平均)で比較する。学習は走らせない。
  `rsna-knee Kaggle run` を手動実行し、`notebook` に `06_llm_labels.ipynb` を指定する(GPUを使う。1分あたりの処理速度から全件の所要時間も出す)。
- ステージ4: 良ければ `RUN_ALL=True` で全レポートを処理して `labels_out/llm_labels.csv` を作り、`04_finetune` のラベルに使う(規則との平均なども可)。
