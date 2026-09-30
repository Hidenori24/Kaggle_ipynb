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
- [ ] DICOM読込・series選択
- [ ] モデル学習
- [ ] 提出Notebook / C++ 推論の検証
