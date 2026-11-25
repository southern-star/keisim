# KeiReality Phase 3 の長時間走行実験

Phase 3a は 10 Hz の各運転時刻に 1 枚を生成する。学習に使った 8 町、重みの選択に使った 2 町、最終評価のために追加収録した 4 町を分けた。[判定・比較動画](../runs/reality/p3/report.md) ・ [収録計画](../configs/reality/p3_episodes.json) ・ [評価前に固定した設定](../configs/reality/p3_full.json)

1 万更新を完了し、4 町すべてで 32 秒の生成を完走した。G3 の総合判定は保留。動体の誤描画を測る分類器のテスト教師上の IoU は 0.305 で、事前基準 0.40 を下回った。分類器から独立した静止面のちらつきは、Phase 2 比で 1 回生成が 2.11 倍、4 回生成が 4.61 倍となった。表示の基準モデルには Phase 2 の履歴版を継続して使う。[全指標](../runs/reality/p3/metrics.json)

## 学習と推論

運転用 mc14 と Phase 2 の履歴付き描画器を凍結した。MeanFlow の U-Net は 12,381,043 パラメータで、128×64 ピクセルの RGB を生成する。入力は、現在の Phase 2 の RGB・観測支持度・最終特徴、現在と前時点の視点、自車の相対移動、経過時間、有効フラグ、直前に自分が生成した画像。初回の履歴はゼロにする。ノイズは時点・視点ごとに独立に引き、同じ入力には両 NFE で同じ乱数を使う。町・時点の識別子は乱数の再現にだけ使う。

標準正規ノイズを時刻 1、教師画像を時刻 0 とした。固定条件の下で瞬間速度を接ベクトルとする JVP を取り、平均速度の回帰目標への勾配を止める。Phase 2 の RGB を中心とする標準偏差 0.18 のガウス分布の解析的な平均速度場に、U-Net の残差を加えた。独立した端点ノイズからの RGB L1 も重み 0.5 で加える。車両・歩行者の画素では、この L1 の重みを 2 倍にした。これらの事前分布と補助損失は本実験の選択である。[原論文の式 9–11](https://arxiv.org/html/2505.13447v1#S4.SS1) ・ [数値検証](../runs/reality/p3_preflight/numerics.json)

学習データは 8 町の 1,024 時点、中央と左右 36° の 3 視点。長さ 10 時点の連続区間を 4 本ずつ扱い、各時点で独立のノイズから生成した 1 回推論の出力を次に渡す。最初の一巡で全 3,072 組を通し、その後は区間内の視点切替を許す。前画像を伏せる確率は 10%。FP32 で AdamW を 10,000 更新し、2,000 更新ごとに検証 2 町の 96 時点をそれぞれ通しで生成する。1 回・4 回推論の検証 L1 の平均で重みを選ぶ。[学習ログ](../runs/reality/p3/meanflow/progress.json) ・ [選択した重み](../runs/reality/p3/meanflow/best.pt)

対照実験では同じ U-Net を通常の flow matching で 2,000 更新し、瞬間速度を Euler 法で積分する。同じ 2,000 更新の MeanFlow と検証 2 町の 1・4・8 NFE を比較する。NFE は 1 枚の生成で速度場を評価する回数。[通常の flow matching](../runs/reality/p3/flow_matching_control/training.json)

## G3 の測り方

町 80–83 は KeiReality の過去の収録とも重ならない。各町の 320 時点を同時刻の 5 視点で収録し、各視点の出力を個別に次へ渡す。各動画は 32.0 秒で、先頭と末尾の観測時刻差は 31.9 秒。別に、視点を 4 秒ごとに切り替える系列と、中央で前回画像を毎回伏せる系列を生成する。テスト教師は生成を終えたファイルと時刻・視点で照合する。履歴支持と再投影評価には収録した深度・意味ラベルを使う。[入力と教師の照合](../runs/reality/p3/integrity.json) ・ [自己回帰出力の記録](../runs/reality/p3/rollouts/manifest.json)

主比較は 4 町の固定 5 視点を合算する。128×64 の共通解像度で RGB L1・SSIM・PSNR を測る。256×128 の表示解像度では Phase 2 の元の画像と、バイリニア拡大した MeanFlow の両方を元の教師と照合する。静止面のちらつきは、同じ静止クラスで深度が 3% 以内の画素を前時点へ再投影したときの RGB 残差の変化。残像は動体が去った新しい静止面で測る。

誤って描かれた車両・歩行者は独立した RGB 分類器で測る。分類器を学習用の教師 RGB と意味ラベルで学習し、検証用の教師で選ぶ。静止面を動体とした割合、実際の動体の見落とし率、車両・歩行者など個別クラスの IoU を記録する。この測定は分類器の認識精度に依存する。検証とテストの教師 RGB の両方で動体 IoU 0.40 以上を判定の前提とした。[分類器の学習](../runs/reality/p3/rgb_probe/training.json) ・ [較正とクラス別集計](../runs/reality/p3/metrics.json)

品質または処理時間に事前規定の改善があり、画質、ちらつき、残像、誤描画、終盤の劣化、100 ms の時間予算をすべて満たした方式を G3 通過とする。処理時間はバッチ 1 で入力 2 カメラの CPU→GPU 転送から、運転特徴の抽出、履歴更新、生成、RGB と支持度の CPU 転送完了までを同期計測する。各町の先頭 8 時点はウォームアップに使う。シミュレータの撮影・物理計算、ブラウザ合成、モニターへの提示は計測外。[判定条件](../configs/reality/p3_full.json) ・ [区間別レイテンシと VRAM](../runs/reality/p3/runtime.json)

## 再現手順

リポジトリ直下から、Phase 2 のデータ・重みと同じローカル CUDA 環境で実行する。収録は KeiView のハードウェア描画を使う。学習の `last.pt` は 500 更新ごと、長時間生成の中間結果は各町が完了するごとに確定する。

```bash
py=.venv/bin/python
cfg=configs/reality/p3_full.json
train=data/reality/p2
data=data/reality/p3_long
feature=data/reality/p3_long_spatial
labels=data/reality/p3_long_support
cache=data/reality/p3_long_conditions
tc=data/reality/p3_conditions
pilot=runs/keipilot_mc.pt
p2=runs/reality/p2/temporal/best.pt
out=runs/reality/p3
flow=$out/meanflow/best.pt
$py scripts/collect_reality.py --gl hw --orbit --audit --plan configs/reality/p3_episodes.json --out "$data"
$py scripts/cache_reality_features.py --pyramid --data "$data" --out "$feature" --ckpt "$pilot"
OPENBLAS_NUM_THREADS=1 $py scripts/label_reality_history.py --data "$data" --out "$labels"
$py scripts/cache_reality_flow.py --data "$train" --features data/reality/p2_spatial --renderer "$p2" --splits train val --out "$tc"
$py scripts/cache_reality_flow.py --data "$data" --features "$feature" --renderer "$p2" --spatial-reference runs/reality/p1_spatial/best.pt --splits test --out "$cache"
$py scripts/train_reality_flow.py --config "$cfg" --data "$train" --cache "$tc" --mode meanflow --out "$out/meanflow"
$py scripts/train_reality_flow.py --config "$cfg" --data "$train" --cache "$tc" --mode flow_matching --out "$out/flow_matching_control"
$py scripts/train_reality_rgb_probe.py --config "$cfg" --data "$train" --out "$out/rgb_probe"
$py scripts/rollout_reality_flow.py --config "$cfg" --data "$data" --cache "$cache" --training-data "$train" --model "$flow" --out "$out/rollouts"
$py scripts/profile_reality_flow.py --data "$data" --cache "$cache" --config "$cfg" --pilot "$pilot" --model "$flow" --renderer "$p2" --rollouts "$out/rollouts" --out "$out/runtime.json"
OPENBLAS_NUM_THREADS=1 $py scripts/eval_reality_flow.py --data "$data" --cache "$cache" --config "$cfg" --probe "$out/rgb_probe/best.pt" --profile "$out/runtime.json" --labels "$labels" --rollouts "$out/rollouts" --out "$out"
$py scripts/present_reality_flow.py video --data "$data" --cache "$cache" --rollouts "$out/rollouts" --out "$out"
MPLCONFIGDIR=/tmp/keireality-mpl python3 scripts/present_reality_flow.py plot --out "$out"
```

グラフは Matplotlib、動画は FFmpeg、比較映像の日本語ラベルは Noto Sans CJK を使う。運転回帰は収録した RGB と同じ走行状態を元の運転モデルに渡し、特徴読み出し付きの運転モデルの全出力と照合する。

```bash
$py scripts/audit_reality_flow.py --config "$cfg" --data "$data" --features "$feature" --labels "$labels" --cache "$cache" --pilot "$pilot" --renderer "$p2" --prior-data data/reality/p1 "$train" --previous-manifest runs/reality/p2/manifest.json --out "$out/integrity.json"
for town in 80 81 82 83; do
  $py scripts/check_reality_driver.py --data "$data" --ckpt "$pilot" --pyramid --frames 320 --town "$town" --out "$out/driver-$town.json"
done
```
