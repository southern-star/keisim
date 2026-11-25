# KeiReality 実験記録

## Phase 0

**G0 は PASS。** 町 5、エピソード 1 を別々の Chrome プロセスで 32 時点と 128 時点収集した。共通する 32 時点すべてで、2 入力・教師 RGB、深度、ラベル、可視性、自車姿勢、時刻、アクター状態のハッシュが一致した。教師を加えた場合と既存 2 カメラだけの場合の比較は、両入力とも RGB 最大差 0、ラベル不一致 0 画素。自車の屋根を投影した画素は 32 回とも道路だった。

| 128 時点での測定 | 結果 |
|---|---:|
| 取得中の 1 時点あたり中央値 / p95 | 267.5 / 351.4 ms |
| 同期取得時間 / 起動等を含む所要時間 | 34.52 / 67.34 秒 |
| 保存した配列 | 108,044,416 byte |
| カード全体の使用 VRAM（開始 / サンプリング最大） | 732 / 1,462 MiB |
| 共通 32 時点の観測済み / 未観測 / 判定不能画素 | 446,228 / 435,222 / 167,126 |

作業ブランチは `experiment/keireality`。公開 main は `a332fe554d56b5c3043c5c4855ab96c1aa3dcf47`。設計時点以降に追加された ReconView の追跡カメラ定義を再利用した。Ubuntu 22.04.5、Python 3.12.11、PyTorch 2.14.0+cu130、CUDA 13.0、RTX 3060 12 GB、ドライバー 580.196.07、OpenCV 5.0、Node 20.20.2。公開 mc14 の SHA-256 は `fcee0654c22448f86f47a00da5e4f4b0232ab74b466d13336ea544b62aee58db`。

32 時点の H.264 動画は `runs/reality/g0/synchronized.mp4`、詳細は `runs/reality/g0/audit.json`。追加テストの 11 ケースと既存の rig・depth・recon の 24 ケースが通った。公開 ONNX とのビット単位の回帰テストには、公開 ONNX の元になった FP32 の mc14 学習チェックポイントを `runs/keipilot_mc14/last.pt` に指定し、onnx・onnxruntime・onnxscript を検証環境に追加した。KeiReality の特徴抽出には公開 FP16 配布版を使用する。

## 再現

リポジトリ直下で実行する。Python は `uv sync --frozen --extra cu130`、KeiView は `npm ci --prefix web` で準備する。今回は既存の uv キャッシュと同一 package-lock のローカル Node パッケージを利用した。重みは公開 v0.12.0 を `runs/keipilot_mc.pt`、既存 ReconView の mc14 深度ヘッドを `runs/reference/mc14-depth.pt` に置く。後者の SHA-256 は `94fac9af15a4df6a396e568dee3046d77ebe5886dbfb145fa6ed80cf0b8dfb87`。

```bash
.venv/bin/python scripts/collect_reality.py --gl hw --frames 32 --town 5 --seed 1 --audit --out data/reality/smoke
.venv/bin/python scripts/collect_reality.py --gl hw --frames 128 --town 5 --seed 1 --audit --out data/reality/benchmark
.venv/bin/python scripts/audit_reality.py --data data/reality/smoke --recapture data/reality/benchmark --out runs/reality/g0
.venv/bin/python scripts/check_reality_driver.py --data data/reality/smoke --ckpt runs/keipilot_mc.pt --out runs/reality/g0/driver-before.json
.venv/bin/python -m pytest -q tests/test_reality.py
KEIPILOT_MC_ONNX=runs/reference/release.onnx .venv/bin/python -m pytest -q tests/test_rig.py tests/test_recon.py tests/test_depth.py
.venv/bin/python scripts/cache_reality_features.py --data data/reality/benchmark --out data/reality/benchmark_mem_cpu --ckpt runs/keipilot_mc.pt --device cpu --bs 4
.venv/bin/python scripts/train_reality.py --config configs/reality/p1_smoke.json --data data/reality/benchmark --features data/reality/benchmark_mem_cpu --out runs/reality/tiny --device cpu
```

12 時点の過学習検証は CPU で 300 ステップ。RGB L1 は 0.20792 から 0.10623。特徴を別の時点に入れ替えると 0.14112、ゼロにすると 0.12564 になった。計算時間は 189.9 秒。

## Phase 1

**G1 は事前に定めた条件を通過した。生成 RGB では道路の曲率や道端が見える一方、観測済みの車両が消える。** 学習は 8 町・4,000 時点、検証は別の 2 町・500 時点、テストは別の 4 町・500 時点。各エピソードは 125 時点で、分割は収集前に固定した。ここでいう未見は KeiReality の学習に対するもの。

| テストの観測済み領域 | 生成器 | 学習データから求めた基準 |
|---|---:|---:|
| RGB L1（小さいほど良い） | 0.1387 | 0.1457 |
| RGB SSIM | 0.3333 | 0.3092 |
| RGB PSNR | 14.99 dB | 15.16 dB |
| 道路 IoU（補助出力） | 0.781 | 0.694 |
| 車両 IoU（補助出力） | 0.277 | 0.000 |

RGB の基準は位置ごとの平均画像、意味ラベルの基準は位置ごとの最頻クラス。観測済み RGB L1 は 4 町中 3 町で改善し、町 61 では 0.1481 から 0.1565 に悪化した。直線の赤いトラック、交差点の青い車は生成 RGB 上で識別できず、観測済みの歩行者・赤信号・青信号の補助 IoU は 0。道路の形が出たことと動体の描画品質に差がある。

収集プロセスは 4,750 時点保存後に終了コード 143 で中断した。同じ計画で保存済みシャードを照合して残り 250 時点を収集し、全 160 シャードの配列と特徴のハッシュ・件数を検査した。計画作成から完了まで 21 分 28 秒。配列 4.220 GB、特徴キャッシュ 1.280 GB、特徴抽出 18.83 秒。生成器は RTX 3060 上で 4 エポック・2,000 ステップを 121.4 秒で実行した。学習時の PyTorch 最大割当は 562.8 MiB。生成器単体の FP32・バッチ 1 は p50 3.43 ms、p95 3.87 ms、p99 3.96 ms。

学習前後の CPU FP32 運転リプレイ 32 時点は、ベースコミットとの経路、速度、安全層を通した制御、信号出力の最大差がすべて 0。重み・BatchNorm を含むハッシュと制御ログのハッシュも一致した。

### 本実験の再現

```bash
.venv/bin/python scripts/collect_reality.py --gl hw --plan configs/reality/p1_episodes.json --out data/reality/p1
.venv/bin/python scripts/cache_reality_features.py --data data/reality/p1 --out data/reality/p1_mem --ckpt runs/keipilot_mc.pt --device cuda --bs 16
.venv/bin/python scripts/train_reality.py --config configs/reality/p1_pilot.json --data data/reality/p1 --features data/reality/p1_mem --out runs/reality/p1 --device cuda
.venv/bin/python scripts/recon_reality.py --data data/reality/p1 --depth-ckpt runs/reference/mc14-depth.pt --out runs/reality/recon-test --split test --gl hw
.venv/bin/python scripts/eval_reality.py --data data/reality/p1 --features data/reality/p1_mem --ckpt runs/reality/p1/best.pt --recon runs/reality/recon-test --out runs/reality/p1 --split test --device cuda
.venv/bin/python scripts/demo_reality.py --data data/reality/p1 --recon runs/reality/recon-test --eval runs/reality/p1 --clips configs/reality/p1_clips.json --out runs/reality/p1
.venv/bin/python scripts/check_reality_driver.py --data data/reality/smoke --ckpt runs/keipilot_mc.pt --out runs/reality/p1/driver-after.json
```

詳細な比較値、失敗例、追加実験の予算は [ローカル実験レポート](../runs/reality/p1/report.md) にある。[3 シナリオの比較動画](../runs/reality/p1/compare.mp4) ・ [学習曲線](../runs/reality/p1/learning_curve.png) ・ [テストの集計値](../runs/reality/p1/metrics.json) ・ [学習済み生成器](../runs/reality/p1/best.pt)

## 凍結した空間特徴の比較

同じ 5,000 時点と同じ学習条件で、運転トークンに c2・c3 とカメラ投影を加えた。未見 4 町で観測済み車両の RGB L1 が 0.1735 から 0.1360、補助出力の車両 IoU が 0.277 から 0.744 になった。赤いトラックの色・輪郭は不鮮明。観測済み歩行者の補助 IoU は 0。[比較画像と詳細](../runs/reality/p1_spatial/report.md)

## Phase 2

14 エピソード・1,600 時点の各時点に 5 視点の教師を収録した。中央と左右 36° で学習し、左右 18° は評価専用とした。現在時点だけを使うモデルと、過去 0.7 秒の特徴と自車移動を使うモデルを同じ空間モデルから初期化する。各モデルに同じ順序で 3,072 組の時点・視点を 3 回学習させ、検証損失が最も低い重みで比較する。[評価条件](../configs/reality/p2_pilot.json) ・ [未見 4 町での結果](../runs/reality/p2/report.md)

G2 は通過した。未見 4 町の中央視点で、過去だけに見えた静止面 254.8 万画素の RGB L1 は単一時点版 0.12725、履歴版 0.12373、履歴無効化 0.12936、相対移動をゼロにすると 0.12958。中央専用の空間モデルは 0.11423 だった。発進直後と観測済みの車両の RGB 誤差には課題が残る。[視点・支持度の比較ビューア](../runs/reality/p2/viewer.html) ・ [連続回転の動画](../runs/reality/p2/rotation.mp4)

### 再現コマンド

以下のコマンドはリポジトリのルートで実行する。学習にはローカル CUDA、収集には同じ KeiView のハードウェア描画を使う。

```bash
py=.venv/bin/python
pilot=runs/keipilot_mc.pt
p1=runs/reality/p1/best.pt
spatial=runs/reality/p1_spatial/best.pt
out=runs/reality/p2
features=data/reality/p2_spatial
labels=data/reality/p2_support
$py scripts/cache_reality_features.py --pyramid --data data/reality/p1 --out data/reality/p1_spatial --ckpt "$pilot"
$py scripts/train_reality.py --arch spatial --config configs/reality/p1_pilot.json --data data/reality/p1 --features data/reality/p1_spatial --out runs/reality/p1_spatial
$py scripts/eval_reality_temporal.py --data data/reality/p1 --features data/reality/p1_spatial --p1 "$p1" --spatial "$spatial" --out runs/reality/p1_spatial
$py scripts/collect_reality.py --gl hw --orbit --audit --plan configs/reality/p2_episodes.json --out data/reality/p2
$py scripts/cache_reality_features.py --pyramid --data data/reality/p2 --out "$features" --ckpt "$pilot"
OPENBLAS_NUM_THREADS=1 $py scripts/label_reality_history.py --data data/reality/p2 --out "$labels"
common=(--config configs/reality/p2_pilot.json --data data/reality/p2 --features "$features" --labels "$labels" --init "$spatial")
$py scripts/train_reality.py "${common[@]}" --arch spatial --out "$out/single"
$py scripts/train_reality.py "${common[@]}" --arch temporal --out "$out/temporal"
$py scripts/eval_reality_temporal.py --data data/reality/p2 --features "$features" --labels "$labels" --p1 "$p1" --spatial "$spatial" --single "$out/single/best.pt" --temporal "$out/temporal/best.pt" --out "$out"
$py scripts/replay_reality_temporal.py --data data/reality/p2 --features "$features" --pilot "$pilot" --temporal "$out/temporal/best.pt" --out "$out"
$py scripts/build_reality_viewer.py --out "$out"
```

## Phase 3 の数値検証

条件付き MeanFlow の回帰目標と 1・4・8 回の区間サンプリングを実装した。FP32 の全微分、中央差分、回帰目標への勾配停止、区間長ゼロ、ノイズからデータへ向かう符号を確認した。実データ 4 時点での検査に用いた条件は、凍結履歴モデルの生成 RGB と観測支持度。G3 の長時間自己回帰と画質比較は今後の評価対象。[数値レポート](../runs/reality/p3_preflight/numerics.json) ・ [原論文](https://arxiv.org/html/2505.13447v1#S4.SS1)

実入力 12 時点・3 視点で 1,233 万パラメータの U-Net 候補を 100 更新した。RTX 3060、64×128、FP32、バッチ 4 で平均 0.124 秒/更新。初回フレーム条件での演算時間から、1 万更新分は約 21 分と見積もられる。[予算計測](../runs/reality/p3_preflight/throughput.json)

```bash
.venv/bin/python scripts/check_reality_meanflow.py --data data/reality/p2 --features data/reality/p2_spatial --temporal runs/reality/p2/temporal/best.pt --out runs/reality/p3_preflight
.venv/bin/python scripts/benchmark_reality_meanflow.py --data data/reality/p2 --features data/reality/p2_spatial --temporal runs/reality/p2/temporal/best.pt --out runs/reality/p3_preflight
```
