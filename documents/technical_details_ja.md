# TinyVCの技術的な詳細
このドキュメントでは、モデルアーキテクチャ等のTinyVCの技術的な詳細について述べる。

# モデル設計
![](../images/tinyvc_architecture.png)
Fig. 1は、全体的な構造を示し、Fig.2 では加算シンセサイザの構造を示す。

## SSL Distillated Model
WavLM Baseの4層目の特徴量を推定するようにConvNeXt v2スタックを学習する。この層を選んだ理由はkNN-VCとWavLMの論文を参照。
(話者の特徴量(SID)が多く含まれる層がここ)
また、同時にピッチ推定のタスクも学習させる。ピッチのアノテーションには~~WORLDのHarvestアルゴリズム~~FCPEを用いた。

## kNN Matching
kNN-VCと同様に類似した特徴ベクトルを検索し、上位数個の平均をとることで、話者特徴の変換を行う。

## Additive Synthesizer
F0(ピッチ)の整数倍の周波数を持つ正弦波をいくつか生成し、それに対してニューラルネットが推定した係数をかけ手合計を取ることで、微分可能な加算シンセサイザとして機能する。類似したニューラルボコーダーとしてDDSPなどがあげられる。
また、ノイズなどの非周期成分は、ガウシアンノイズに対してイコライザーをかけることによって生成する。

## U-Net
FastSVC, NeuCoSVCを参考に軽量なU-Netを設計。CPUでもリアルタイムで推論できるほど軽いもの。

## Discriminator
Multi Periodic Discriminator (MPD) と Multi Resolution Discriminator (MRD) を使用する。
MRDはスペクトルのオーバースムージングを回避する目的で導入した。