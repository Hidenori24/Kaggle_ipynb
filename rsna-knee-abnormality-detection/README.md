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
| `02_cnn_embed.ipynb` | ResNet18(ImageNet)のスライス埋め込み+LightGBM。GPU | 疑似0.725 / 正解0.594 |

Actionsの実行対象は手動実行時の `notebook` 入力で選ぶ（既定は `02_cnn_embed.ipynb`）。

`02` は次版で、英語以外のレポートを疑似ラベル学習から除外し、N=600→2000に増やした。
