# KeiSim + KeiPilot — 軽量クローズドループ自動運転環境

**軽 (Kei) = light.** ゲームエンジンを使わず `numpy + OpenCV + PyTorch` だけで動く、
「**シミュレータ × 軽量 E2E モデル**」を最初から一体で設計したクローズドループ環境です。

> **English summary** — KeiSim is a procedurally generated, Japan-style (left-hand traffic) driving simulator
> written in plain numpy + OpenCV (no game engine, no GPU needed; ~30× real time without rendering).
> KeiPilot is a 14.9M-parameter camera-only end-to-end driving model trained inside KeiSim by imitation of a
> privileged expert plus DAgger. On 40 routes in unseen towns it completes 100% of the routes with zero
> collisions (Driving Score 0.96). Quick start: `bash setup.sh`, download `keipilot.pt` from Releases, then
> `uv run scripts/demo.py --agent model --ckpt runs/keipilot.pt --show`.
> The same towns can also be rendered by KeiView (an anime cel-shaded three.js renderer in `web/`). The KeiSim-trained
> model cannot even pull away there (DS 0.008); after fine-tuning plus DAgger on KeiView frames it scores DS 0.955 / 0.974
> on unseen towns and keeps its KeiSim score (section 6).

![gallery](docs/gallery.jpg)

| | CARLA | MetaDrive + 既存モデル | **KeiSim + KeiPilot** |
|---|---|---|---|
| 依存 | UE4, GPU 必須, 数十 GB | Panda3D | numpy / OpenCV / PyTorch のみ |
| 起動 | 数十秒〜 | 数秒 | 街生成 0.14 s + テクスチャ 0.73 s |
| 速度 (1 プロセス, i5-10400F) | ≲ 実時間 | 高速 | 描画なし **24.6×** 実時間 / カメラ込み **10.8×** |
| 認識のドメインギャップ | 小 (ただし重い) | **大 (既存モデルが見えない)** | **ゼロ: 同じ世界で学習する** |
| クローズドループ評価 | ○ | ○ | ○ (DS/RC/IS, 違反ログ, 動画, 決定論的リプレイ) |

---

## 1. 何が問題で、どう解いたか

* CARLA は重い（UE4・GPU・巨大アセット・client/server 同期）。
* MetaDrive は軽いが、実データや CARLA で学習したモデルからすると
  **描画の分布が違う (ドメインギャップ)** ため、車も信号も「認識されない」。

→ **見た目をリアルに寄せる競争をやめ、シミュレータとモデルをセットで設計する。**

1. シミュレータは **特権情報付きエキスパート** と、ピクセル単位で正確な **セマンティックラベル** を標準装備する。
2. モデルはその世界のカメラ画像だけを入力に、エキスパートを模倣して学習する（LBC / Roach と同じ系統）。
3. 閉ループでの誤差蓄積は、**ノイズ注入 (DART)**・**仮想カメラずらし**・**DAgger** で潰す。

手続き生成なので「学習に使っていない街」で汎化を測れます（学習: town 0–399、評価: town 1000–1009）。

## 2. 結果 (すべて `runs/eval/*.json`)

600 m ルート × 20 本のスイート。`(town_seed, episode_seed)` で地図・経路・交通・歩行者・信号位相・天候が完全に再現されます。
DS = Route Completion × 違反ペナルティ（車両 ×0.60、歩行者 ×0.50、静止物 ×0.65、信号無視 ×0.70。衝突時点で打ち切り）。

| エージェント | 未見の街 A | 未見の街 B | 既知の街・新ルート | 高密度交通 (未見 A) |
|---|---|---|---|---|
| 特権エキスパート (正解情報) | 1.000 | 1.000 | 1.000 | 1.000 |
| **KeiPilot · BC** (カメラのみ) | 0.945 | 0.871 | 0.985 | 0.798 |
| **KeiPilot · DAgger** (カメラのみ) | **0.970** | **0.955** | **0.970** | **0.921** |

* KeiPilot (DAgger) は未見の街 40 ルート (24.3 km) で **完走率 100%、衝突 0 件**。残る減点は信号無視 5 件 (0.21/km) のみ。
* 高密度交通 (車両・歩行者が約 2 倍、横断多発) でも DS 0.921、20 ルート中 19 本を完走 (歩行者との接触 1 件)。
* 未見の街 6,000 フレームでのオフライン指標 (BC → DAgger): 経路 ADE 0.100 → 0.090 m、信号 4 クラス認識 95.9 → 96.2 %、セグメンテーション mIoU 0.819 → 0.829。

### 見つけた弱点と対処（開発ログ）

1. **BC の信号無視**: 信号の認識自体は正しい (P(赤) ≈ 1.0) が、停止直前の 1〜2 m で距離を読み違えてオーバーシュートしていた。停止線が画面下端から消えると、学習データ上の「線を越えた＝進んでよい」と区別できず、再加速していた。
   → 停止目標を線の 2.5 m 手前に変更 (線が画面に残る)。停止直前の速度プロファイルを √ 型から「√ + 線形徐行」に変更 (距離誤差への感度を下げる)。「線を少し越えても赤なら止まる」ラベルを追加。旧データは同じ式で再ラベル (`keipilot/data.py: relabel_v1`)。DAgger も実施。
   → 未見の街の信号無視 14 → 5 件。
2. **残る弱点**: 黄信号のジレンマ（エキスパートは黄の残り時間と自車速度で判断するが、速度入力のない 1 フレームモデルでは原理的に再現不能）と、遠方で隣の灯器と取り違えるケース。
   次の一手は、自車速度 (ドロップアウト付き) か 2〜3 フレームの時間文脈の入力、または信号用の望遠クロップ。
3. **高密度時のグリッドロック** (シミュレータ側の問題): NPC が出口の詰まった交差点に進入して互いを塞いでいた → 「出口に空きがなければ停止線で待つ」ルールを NPC とエキスパートに追加。

## 3. 構成

```
pyproject.toml, uv.lock  依存関係 (uv)。setup.sh が GPU/CPU を判定して .venv を作る
keisim/                  シミュレータ (numpy + OpenCV)
  roadnet.py             手続き型の街: ジッタ格子 + カーブ道路 + 信号交差点 + 横断歩道 + 建物/街路樹
  traffic.py             NPC 車両 (レーン追従 IDM, キャッシュ付き前方コリドー検索) / 歩行者 (歩道歩行 + 乱横断)
  vehicle.py, control.py 自車 (運動学的自転車モデル) / Pure Pursuit + 非対称ゲイン PI 速度制御
  expert.py              特権エキスパート (経路中心線 + 距離ベースの目標速度 → 1 フレームから学習可能なラベル)
  world.py, env.py       World / Gym 風 Env (違反判定, DS/RC/IS, env.vector_obs())
  render/                CPU レンダラ: 地面 = 厳密な平面ホモグラフィ (ミップ帯域), 物体 = ペインタ法
                         + ピクセル単位セマンティック + BEV 可視化
                         keiview.py: KeiView (web/) を headless Chrome で動かし、自車カメラとして使う
  viz.py                 ダッシュボード合成, H.264 動画書き出し
keipilot/                軽量 E2E モデル (ResNet-18 + 小型 Transformer デコーダ, 14.9M params, 6.9 ms/frame)
  model.py, data.py, agent.py
scripts/
  collect.py             データ収集 (expert: ノイズ注入 + 仮想カメラ / dagger: モデルが運転しエキスパートがラベル)
  train.py               学習 (bf16, 重み付きサンプリング, 補助セグメンテーション)
  evaluate.py            クローズドループ評価スイート (train / test / test2 / showcase, --dense)
  eval_perception.py     オフライン指標 (ADE, 信号認識, mIoU)
  demo.py                1 エピソード実行 + ダッシュボード動画 (--show でライブ表示)
  bench_speed.py         速度・決定論性の計測
  make_clips.py, build_report.py, summarize.py   レポート作成
tests/test_keisim.py     不変条件 (レーンが道路外に出ない, 信号の排他, 決定論的リプレイ, エキスパート完走, 描画形状)
scripts/export_town.py   街を JSON に書き出す (KeiView 用)
web/                     KeiView: 同じ街をアニメ調 (セル調) で歩ける three.js ビューア。KeiPilot の学習用カメラにもなる (6 章)
docs/                    ギャラリー画像, 上面図, ハイライト動画, 結果ページ (docs/report/index.html)
runs/eval/               評価結果 JSON (README の数値の出所)。学習済みモデルは GitHub Releases の keipilot.pt
data/                    収集データ (リポジトリ外, 約 3 GB。scripts/collect.py で再生成できる)
```

### KeiSim の設計ポイント

* **日本仕様**: 左側通行、交差点奥の横型信号（青・黄・赤）、停止線 + 横断歩道、軽自動車サイズの車も混在。`TownConfig(left_hand_traffic=False)` で右側通行にも切り替え可能。
* **手続き生成の街**: ジッタ格子 (3〜5×3〜5)、S 字 / 円弧カーブ、T 字・十字・斜め交差、ベンド。シード = 街。
* **衝突しない信号制御**: 流入方向ごとのスプリット現示 (青→黄→全赤)。右折待ちのデッドロックが原理的に起きない。
* **交通**: NPC は IDM + カーブ減速 + 信号遵守 + 前方コリドー (歩行者・自車も考慮) + 「出口が詰まっていたら交差点に入らない」。
  歩行者は歩道を歩き、安全を確認してから乱横断する (自車の前方でも横断イベントが起きる)。密度はエピソードごとにランダム。
* **レンダラ**: 地面は 1 枚の上面テクスチャを **厳密なホモグラフィ** で透視投影する（距離帯ごとにミップレベルを選択）。
  物体は低ポリゴンの凸パーツを、背面カリング + 近平面クリップ + ランバート陰影 + 距離フォグで描画する。
  ブレーキランプ・窓・信号灯はデカール。天候は晴れ / 曇り / 夕焼け / 霧 + ランダム化。
* **ラベルが全部タダ**: RGB と同じ幾何でセマンティック画像を描くので、ラベルはピクセル単位で一致する。
  エキスパートの経路・目標速度・関連信号の状態も毎フレーム出る。任意の姿勢から再描画できるので、仮想カメラ拡張もタダ。
* **決定論的**: `(town_seed, episode_seed)` で完全に再現できる (テストで検証済み)。

### KeiPilot（軽量モデル）

```
camera 320x160 ──ResNet-18──┬─ FPN ─→ セマンティックセグメンテーション (補助タスク / 可視化)
                            └─ 1/16,1/32 トークン ─┐
nav command + target point ─→ 条件埋め込み ────────┤
                             12 learned queries ──→ Transformer decoder (3 層)
                                                     ├→ 経路 10 点 (2 m 間隔)
                                                     ├→ 目標速度 (two-hot 分布, 0..11 m/s)
                                                     └→ 関連信号の状態 (赤/黄/青/なし)
経路 + 目標速度 ─→ エキスパートと同一の Pure Pursuit / PI 制御 ─→ steer / throttle / brake
```

* 入力は **カメラ画像 1 枚 + ナビ指示 (次の交差点での左/直/右 + 目標点) のみ**。自車速度は入れない (慣性問題の回避)。
* エキスパートの目標速度は「見えるものまでの距離」だけで決まるように設計しているので、1 フレームから原理的に推定できる。
* 初期重みは torchvision の ImageNet 学習済み ResNet-34 の各ステージ先頭 2 ブロック (`~/.cache/torch/hub/checkpoints` にあれば流用、なければランダム初期化)。公開している keipilot.pt もこの重みから学習したものです。
* 学習: BC 12 エポック (160k フレーム, RTX 3060 で約 55 分) → DAgger 60k フレーム + 歩行者強化 30k フレームで 5 エポック追加学習。

## 4. セットアップ (uv)

```bash
git clone https://github.com/southern-star/keisim.git && cd keisim
bash setup.sh
gh release download v0.1.0 -R southern-star/keisim -p keipilot.pt -D runs   # 学習済みモデル (30 MB)
```

`gh` がない場合は `curl -L --create-dirs -o runs/keipilot.pt https://github.com/southern-star/keisim/releases/download/v0.1.0/keipilot.pt` でも取得できます。

`nvidia-smi` を見て PyTorch のビルドを自動で選び、`.venv` を作ります（このPCでは約 1.5 分）。

| 状況 | 選ばれる extra | 手動で指定する場合 |
|---|---|---|
| NVIDIA ドライバが CUDA 13.0 以上に対応 (R580〜) | `cu130` | `uv sync --extra cu130` |
| CUDA 12.8〜12.9 対応 | `cu128` | `uv sync --extra cu128` |
| CUDA 12.0〜12.7 対応 | `cu126` | `uv sync --extra cu126` |
| GPU なし・CI・macOS | `cpu` | `uv sync --extra cpu` |

* Python 3.12 (uv が自動で用意)、numpy・OpenCV・PyTorch・pytest、H.264 対応の ffmpeg (imageio-ffmpeg 同梱) まで入ります。システムの Python や ffmpeg は不要です。
* バージョンは `uv.lock` で固定しています (現在 torch 2.14 / OpenCV 5.0 / numpy 2.5)。
* 同じシードなら、ライブラリのバージョンが違っても同じ街・同じ走行になります（numpy 1.21 + OpenCV 4.5 と numpy 2.5 + OpenCV 5.0 で、街・ルート・6 秒後の自車位置が一致）。uv 環境でもベンチマーク結果が再現します（未見の街 A で DS 0.970）。

## 5. 使い方

```bash
# 動作確認・速度計測 (作業ディレクトリ直下で実行)
uv run pytest
uv run scripts/bench_speed.py runs/keipilot.pt

# 学習済みモデルのデモ (動画保存, --show でウィンドウ表示)
uv run scripts/demo.py --agent model --ckpt runs/keipilot.pt --town 1001 --episode 3 --out runs/demo_model.mp4 --show
uv run scripts/demo.py --agent expert --town 1000 --episode 3 --out runs/demo_expert.mp4

# ベンチマーク
uv run scripts/evaluate.py --agent model --ckpt runs/keipilot.pt --suite test --videos 3
uv run scripts/evaluate.py --agent model --ckpt runs/keipilot.pt --suite test --dense --max_steps 6000
uv run scripts/summarize.py

# 再現: データ収集 → BC → DAgger → 追加学習
uv run scripts/collect.py --out data/expert --frames 160000 --workers 10
uv run scripts/collect.py --out data/expert_ped --frames 30000 --workers 5 --ego_cross_rate 0.45 --ped_spacing 14 26
uv run scripts/train.py --data data/expert --out runs/keipilot_bc --epochs 12
uv run scripts/collect.py --mode dagger --ckpt runs/keipilot_bc/last.pt --out data/dagger1 --frames 60000 --workers 6 --ego_cross_rate 0.3
uv run scripts/train.py --data data/expert data/expert_ped data/dagger1 --init runs/keipilot_bc/last.pt --out runs/keipilot_dagger --epochs 5
uv run scripts/export_weights.py runs/keipilot_dagger/last.pt runs/keipilot.pt   # 推論用 fp16 (Release の keipilot.pt)
```

uv を使わない場合は、`pip install -e .` のあとに PyTorch を別途入れ、`python3 scripts/...` で同じように動きます。

Gym 風 API:

```python
from keisim.env import KeiEnv
env = KeiEnv()
obs = env.reset(town_seed=1000, episode_seed=0)      # obs: rgb, seg, speed, command, target_point, expert{path,target_speed,tl_state}
while True:
    obs, reward, done, info = env.step(env.expert_action())   # action = [steer, throttle, brake]
    if done:
        break
print(info)            # status, RC, IS, DS, infractions, ...
env.vector_obs()       # 物体レベルの観測 (PlanT 系プランナ向け)
```

自前のモデルを載せる場合は、`obs["rgb"]` (BGR uint8, 320×160) を入力にして `[steer, throttle, brake]` を返せば `env.step()` に入れられます。
経路 + 目標速度を出すモデルなら、`keisim.control.PlanFollower` がエキスパートと同じ制御器として使えます。

## 6. KeiView — 同じ街をアニメ調で歩く

![KeiView](docs/keiview_gallery.jpg)

`web/` は、KeiSim の街を [Sakuragaoka Station](https://github.com/Kenton-GMI/sakuragaoka-station) の
セル調レンダラと家の生成器で描く three.js ビューアです。道路・歩道・白線・信号は KeiSim のジオメトリそのもので、
信号の灯火も KeiSim と同じ現示で切り替わります。KeiSim の建物の箱は区画として扱い、
家（間取り・ベランダ・洗濯物・塀）やマンションになります。街路樹は桜、電柱と電線も生成します。

```bash
uv run scripts/export_town.py --town 1000     # web/towns/town_1000.json (1000〜1003 は同梱)
cd web && node tools/serve.mjs                # http://localhost:5174/?town=1000
```

詳しくは [web/README.md](web/README.md)。

### KeiView の画像で KeiPilot を学習する

![学習前後の比較](docs/keiview_before_after.jpg)

KeiSim の画像だけで学習した KeiPilot を KeiView の画像で走らせると、**20 ルートすべてで発進できません**（DS 0.008）。
アニメ調の画像では青信号を赤と読み違え、いない歩行者や車まで見えてしまうためです（上の図の 1 段目）。
冒頭の「MetaDrive では既存のモデルが認識しない」と同じドメインギャップが、2 つのレンダラの間でも起きています。

対処は KeiSim のときと同じです。KeiView の画像にもエキスパートの操作とピクセル単位の正解ラベルを付けて学習し、最後に DAgger をかけます。

- `--renderer keiview` を付けると、カメラ画像だけを KeiView で描きます（headless Chrome を GPU で動かします）。
  - シミュレーション・エキスパート・ラベルは KeiSim のままなので、同じシードなら同じエピソードになります。
  - 仕組みは [web/README.md](web/README.md) の「エゴモード」の節にあります。
- 必要なもの: Node.js 18 以上、Chrome または Chromium、`cd web && npm install`。
  - GPU があると実用的な速度になります。RTX 3060 で 1 フレーム約 17 ms、6 並列の収集で約 90 fps です。
  - 下の手順は RTX 3060 で合計約 2 時間です（収集 22 分 → 学習 50 分 → DAgger の収集 26 分 → 学習 20 分）。
- 学習は KeiSim の既存データと混ぜ、KeiSim で学習したモデルから追加学習します。
  - 1 バッチのうち 55〜60% が KeiView の画像です。
  - こうすると、KeiSim での性能を保ったまま、KeiView でも走れるようになります。

```bash
cd web && npm install && cd ..
uv run scripts/collect.py --renderer keiview --episodes_per_town 3 --out data/kv_expert --frames 120000 --workers 6 --seed 31 --ego_cross_rate 0.2
uv run scripts/train.py --data data/kv_expert data/expert data/expert_ped data/dagger1 --init runs/keipilot_dagger/last.pt \
    --out runs/keipilot_kv --epochs 8 --samples_per_epoch 200000 --kv_weight 3 --dagger_weight 2
uv run scripts/collect.py --mode dagger --ckpt runs/keipilot_kv/last.pt --renderer keiview --episodes_per_town 3 --out data/kv_dagger1 \
    --frames 60000 --workers 6 --seed 41 --ego_cross_rate 0.3
uv run scripts/train.py --data data/kv_expert data/kv_dagger1 data/expert data/expert_ped data/dagger1 --init runs/keipilot_kv/last.pt \
    --out runs/keipilot_kv_dagger --epochs 4 --samples_per_epoch 200000 --kv_weight 2 --dagger_weight 2
uv run scripts/evaluate.py --agent model --ckpt runs/keipilot_kv_dagger/last.pt --suite test --renderer keiview --videos 3
uv run scripts/demo.py --agent model --ckpt runs/keipilot_kv_dagger/last.pt --renderer keiview --town 1001 --episode 3 --out runs/demo_kv.mp4
```

結果はすべて未見の街でのものです。

- 学習前: KeiSim だけで学習した `keipilot.pt`
- 追加学習: `runs/keipilot_kv/last.pt`
- DAgger: `runs/keipilot_kv_dagger/last.pt`（最終モデル）

| | 学習前 | KeiView で追加学習 | + KeiView で DAgger |
|---|---|---|---|
| KeiView で走行・未見の街 A (DS) | 0.008（20 ルートとも発進できず） | 0.970 | 0.955 |
| KeiView で走行・未見の街 B (DS) | — | 0.922 | **0.974** |
| KeiView の 40 ルート: 完走 / 衝突 / 信号無視 | — | 39 / 0 / 7 | 39 / 0 / **4** |
| KeiSim で走行・未見の街 A (DS) | 0.970 | 0.985 | 0.970 |
| KeiView 画像 6,000 枚: 経路 ADE / 目標速度 MAE | 0.455 m / 2.08 m/s | 0.100 m / 0.16 m/s | 0.093 m / 0.14 m/s |
| KeiView 画像 6,000 枚: 信号 4 クラス / mIoU / 歩行者 IoU | 69.2 % / 0.373 / 0.04 | 91.9 % / 0.737 / 0.54 | 92.7 % / 0.743 / 0.55 |
| KeiSim 画像 6,000 枚: 信号 4 クラス / mIoU | 96.2 % / 0.829 | 96.0 % / 0.809 | 96.0 % / 0.807 |

走行動画（未見の街、最終モデル）:

- [赤信号で止まり、青になってから左折](docs/clips/kv_redlight_turn.mp4)
- [横断する歩行者の前で止まり、渡り終えてから進む](docs/clips/kv_pedestrian.mp4)

**残る弱点**

- **信号無視**: DAgger の前の 7 件を再現して調べました。
  - 4 件は、停止線の手前まで減速したのに、線の直前で「進む」に切り替わるものでした。線が画面から消えると、越えた後と区別できなくなるためです。DAgger で狙った種類の失敗です。
  - 3 件は黄信号のジレンマでした。KeiSim と同じ弱点です（[2. 結果](#2-結果-すべて-runsevaljson) を参照）。
  - DAgger の後は 4 件に減りました。20 ルートでは信号無視 1 件の差は誤差の範囲なので、街 A と街 B の増減より、合計で見てください。
- **変則的な交差点**: 右に曲がれる道が 2 本ある交差点で、違う方に入ってルートを外れました（街 B の 1 ルート）。同じルートは、KeiSim の画像なら完走できます。
- **信号の認識**: KeiView では 92.7 % で、KeiSim の 96 % より低いままです。灯器が小さく、ブルームでにじむためです。

## ライセンス

[MIT License](LICENSE)。学習済みモデル `keipilot.pt` のバックボーンは torchvision の ImageNet 学習済み ResNet-34 で初期化しています。
`web/vendor/sakuragaoka/` は Sakuragaoka Station（MIT, © 2026 Sakuragaoka Station contributors）のコードです（[NOTICE](web/vendor/sakuragaoka/NOTICE.md)）。
