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

## 05_label_diag の結果と、それを受けた修正

- 正解58件の言語内訳は en 31 / es 13 / tr 6 / el 3 / bg 3 / de 2 で `other` が無い。一方、規則が見逃した報告の頻出語には
  オランダ語(scheur, botoedeem, besluit, knie)とクロアチア語(zgloba, križni, stražnji, hrskavice)が並ぶ。
  ASCII中心の文字なので `en` と判定され、英語規則で全ラベル0(=読めなかっただけの偽陰性)になっていた。訓練データ全体にも同じ混入があるはず。
- `detect_lang`: ASCII中心のテキストは英語の機能語(the/and/with/there…)が2種類以上ないと `en` にしない(無ければ `other`)。
  オランダ語(`nl`)・クロアチア語/セルビア語(`hr`)の判定と規則を追加。
- 誤検出の修正(精度が低かった所見): Contusion は変性・嚢胞・骨棘と同じ文の骨髄浮腫を除外(`subchondral edema` / `bone marrow lesion` も外した)。
  MCL は `grade 2`(半月板のシグナル等級)を損傷語から外した。半月板の裂傷語は、その言及の前後100文字・同じ節(`;`で区切る)の中にあるものだけを採用する。
- `kaggle_run.sh`: Kaggle側でカーネルがERRORになってもActionsが緑になっていたため、失敗時は終了コード1にし、Kaggleログの末尾を表示する。
- 05 に言語別(en/es/tr/…)の規則 vs 正解の適合率・再現率と、訓練データ全体の言語内訳を追加。
- 注意: 正解58件は訓練データ全体より陽性が多い(例 Synovitis 0.47 vs 規則0.11、Fracture 0.31 vs 0.07)。ラベル付きの部分集合は異常を多く含むよう選ばれているらしく、
  Synovitis など画像から付けられたと思われるラベルはレポートに書かれていない場合があり、規則の再現率には上限がある。

## 06_llm_labels の1回目の失敗と修正

- 1回目: GPUメモリ不足(OOM)。長いレポート(ギリシャ語・トルコ語はトークンが多い)をバッチ8でまとめたため、注意機構の行列が3.5GBを超えた。
  `label_reports` を、バッチサイズ×最長プロンプト ≤ 5000トークンになるようバッチ分けし(レポートは1200トークンで切る)、
  それでもOOMになったバッチは半分に割って再試行、1件でもOOMなら0のまま続行、とした。`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` も設定。
- 2回目: `Maximum batch GPU session count of 2 reached`。Kaggleの同時GPUセッションは2つまで。#54・#55のマージで `04_finetune` が自動で2つ走っていた。
  → `rsna-knee Kaggle run` は手動実行のみにした(pushでは走らない)。`kaggle_run.sh` は送信に失敗したら終了コード1で止まる(古い実行のステータスを読まない)。
- ノートブックのリポジトリのclone先を `/kaggle/working/repo` → `/kaggle/temp/repo` に変更(実行の出力に.gitや全ファイルが入って、Actionsのログが埋もれていた)。

## 06 の結果: LLMラベルは規則より大幅に良い → 本番の学習に使う

正解58件に対する macro AUC: 規則 0.729 / **LLM 0.864** / 平均 0.853(LLM単体のほうが良い)。
所見別(規則→LLM): Medial OA 0.67→0.96、Medial Meniscus 0.71→0.92、Lateral Meniscus 0.67→0.87、PF OA 0.64→0.84、Synovitis 0.66→0.78。
英語以外の報告で再現率 0.56→0.79(オランダ語・クロアチア語・その他も規則なしで読める)。プロンプトはコンペの説明の定義から書き、58件では調整していない。
画像モデルの正解ラベルAUC(約0.72)は規則ラベルのAUC(0.729)とほぼ同じ=教わったラベルの質が天井になっていた。

使い方(手動実行、順番に):
1. `rsna-knee Kaggle run` に `06_llm_labels.ipynb`: 58件の評価表のあと全レポートを処理(`RUN_ALL=True`)し `labels_out/llm_labels.csv`(確率つき)を作る。
   正常終了すると workflow が private Dataset `rsna-knee-labels` に公開する(失敗した実行では公開しない)。
2. `rsna-knee Kaggle run` に `04_finetune.ipynb`: Dataset `rsna-knee-labels` があれば自動でアタッチされ(`kaggle_run.sh`)、
   `LABEL_SOURCE='llm'` でソフトラベル(確率)を学習ターゲットにする。言語の制限がなくなり全レポートが使える。無ければ規則ラベルに戻る。
   学習中の検証AUCはソフトラベルを0/1に直して計算する。正解ラベル付き58件は従来どおり評価専用。

## LLMラベルで学習した04の結果(規則ラベル版との比較)

| | 正解58件 macro AUC |
|---|---|
| 規則ラベル(3シードのアンサンブル) | 0.735 |
| **LLMラベル(3シード)** | seed0/1/2 = 0.837/0.832/0.846、アンサンブル **0.849** |

epoch 1 の時点で0.79〜0.80(規則ラベルでは0.68〜0.70)。所見別(アンサンブル): Baker's 0.984、Medial OA 0.953、Contusion 0.918、Fracture 0.913、Effusion 0.909、Medial Meniscus 0.843、ACL 0.825、Lateral OA 0.820、
Lateral Meniscus 0.783、PF OA 0.766、Synovitis 0.747、MCL 0.726。
モデル vs LLMラベル自身のAUC(06): Effusion/Baker's/Contusion/Fracture はモデルがラベルを上回る(画像からラベル以上を読めている)。
ACL(0.83 vs 0.91)・MCL(0.73 vs 0.93)・Medial Meniscus(0.84 vs 0.92)・Lateral Meniscus・PF OA は届いていない=靱帯・半月板などの細かい構造は画像側が限界。

## 次の設定B: 画像側の改善と、設定違いモデルの平均

- `04_finetune`: `CENTER = 0.7`(各seriesの中央70%から16枚)、`USE_META = True`(DICOMヘッダーを最終層へ)、`EPOCHS = 5`、`RUN_TAG = 'b'`。
  モデルは `ft_model_<RUN_TAG>_s<seed>.pt` で保存する。
- 学習workflow: 新しいDatasetのversionは全ファイルを置き換えるため、公開の前に既存の `rsna-knee-model` をダウンロードして新しいモデルを足す
  (前の設定のモデル=タグなしの `ft_model_s0..2.pt` は残る)。
- `03_submit`: すべての `ft_model*.pt` を読み、(size, k, center) が同じモデルごとにテスト画像のキャッシュを作って、全モデルの予測を平均する。
- 古い設定のモデルを消すには、Kaggleの Datasets で `rsna-knee-model` の該当ファイルを削除する(または新しいDatasetを作る)。

## LLMラベルの質を上げる(`07_llm_prompts.ipynb`, `src/llm_labels.py` のプロンプト版)

LLMラベルのAUC(正解58件)は0.864。弱い所見は Effusion(0.78、適合率0.71・再現率0.97=陽性と答えすぎ)、Synovitis(0.78)、Contusion(0.82)、PF OA(0.84)。
同じモデル・同じ58件で、プロンプトの版を比べる:
- `v0`: 今のプロンプト(基準)。
- `v1`: 3段階の回答(0=なし/未記載、1=疑い・軽度、2=明らかにあり)。トークンの確率から P(2)+0.5×P(1) を使う。「疑い」を区別して順位づけを細かくする。
- `v2`: 所見ごとの詳しい基準(同義語、半月板は変性だけなら0、OAは軟骨の菲薄化/骨棘/関節裂隙狭小化、三区画のOAは3つとも、Synovitis は滑膜肥厚とHoffa脂肪体の炎症、など)。
07 は3版を同じ58件で評価し、版の平均(v0+v2、v0+v1、全部)も表にする(約15〜30分)。`RUN_VARIANT = 'v2'` のように設定すると、その版で全レポートにラベルを付けて
`labels_out/llm_labels_v2.csv` に書く(約4時間)。workflow は、既存の `rsna-knee-labels` を取得して新しいファイルを足して公開する。
`04_finetune` は Dataset 内の `llm_labels*.csv` をすべて読み、研究ごとに平均して学習のターゲットにする。
流れ: 07を実行 → 良い版を選んで `RUN_VARIANT` を設定するPR → 07を再実行(全件) → 04を実行。

## 04 設定B と 07 プロンプト比較の結果

04(LLMラベル、3シード×5epoch): 設定A(中央スライスなし・メタなし)のアンサンブル0.849 → **設定B(中央70%・DICOMヘッダー)のアンサンブル0.859**。
3シードとも B ≥ A。所見別: ACL 0.825→0.887、Medial Meniscus 0.843→0.867、Lateral OA 0.820→0.855、PF OA 0.766→0.790、Synovitis 0.747→0.765 は上がり、
Lateral Meniscus 0.783→0.752、MCL 0.726→0.714 は少し下がった。`rsna-knee-model` にはAとBの6モデルが入る(提出時は6モデルの平均)。

07(同じ58件・同じ7Bモデル、macro AUC): v0 0.864 / v2 0.868 / **v1 0.879** / v0+v1 0.878 / v0+v1+v2 0.882 / 規則 0.729。
所要時間(1レポートあたり→全4407件): v0 3.04秒→3.7時間、v1 3.15秒→3.9時間、v2 4.44秒→5.4時間。
v2 を足しても +0.004 で5.4時間に見合わないので、v1 だけを全件に流す。v0のラベルは既にあり、04は `llm_labels*.csv` を平均する(v0+v1)。
`07_llm_prompts.ipynb` は `RUN_VARIANT = 'v1'`(58件で確認 → 全件に付けて `llm_labels_v1.csv`)。`None` にすると3版を58件で比べるだけ。

## 上位は0.96台 → ラベルを作るLLMの質が鍵

公開LBの上位5人は 0.961〜0.964。画像モデルは教わったラベルの質にほぼ1対1でついていく(規則ラベルの正解58件AUC 0.729 → モデル 0.735、LLM v0 0.864 → モデル 0.849〜0.859)ので、
上位との差は「レポートから正解に近いラベルを作る」精度の差である可能性が高い。7B・4bitのLLMは弱い。
`08_llm_models.ipynb`: 58件だけで、(1) Qwen2.5-14B-Instruct と Mistral-Nemo-Instruct(12B)を7Bと比べ、(2) 正解58件で全スコアを補正(所見ごとのロジスティック回帰、繰り返し5分割交差検証)した効果を見る(約30〜45分)。
大きなモデルの58件AUCが十分高ければ、それで全件にラベルを付け直す(14Bは約2倍遅いので、2セッションに半分ずつ並列)。

## 08 の結果と、次の設定C(画像側の強化)

08(大きいLLM・58件での補正): 大きいLLMでもラベル精度は上がらず、58件での stacking は逆効果だった。
ラベルの質は 0.886 付近で頭打ちなので、画像側が主なレバーになる。

設定C(`04_finetune.ipynb`、`RUN_TAG='c'`): 細かい構造を見えやすくする。
- 隣のスライス(前・今・次)を3チャンネルに積む(`ADJ=True`)
- 視野の中央85%を切り出して 256px(`CROP=0.85`, `SIZE=256`)
- スライス 24 枚(`K=24`)、中央70%・ヘッダー・5 epoch は設定Bのまま
- キャッシュ前に最大形状で forward/backward を試し、OOMならバッチを下げる
- 所要は約3.5時間(キャッシュ約75分 + 学習)

`03_submit` は `(size,k,center,crop)` ごとにテストキャッシュを作り、A+B+C 全モデルを平均する。
