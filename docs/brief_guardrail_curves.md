# 指示書: 白線のない道の急カーブで、ガードレールに引き寄せられる癖を学習で直す

GPU のある手元のマシンで、学習と評価をするための指示書です。人が読んでも、Claude Code に「この指示書を読んで進めて」と渡しても使えるように書いています。
くわしい経緯と数字は、README 2 章の「カーブの試験コース」にあります。

## 1. 問題

- **きっかけ:** 別のシミュレーターで、白線のない左カーブでだんだん内側へ切り込む癖が出ました。白線がなくガードレールのある道でも、同じように切り込みました。
- **KeiSim での再現:** KeiPilot v0.10 (`keipilot_v24`) の癖は、KeiSim の描画で再現しました。原因は 2 つです。
  1. **ガードレールの陰影 (太陽の向き):**
     - 晴れの日に車の左のガードレールの道路側が日陰になると、モデルは左へ寄ります。直線でも、ガードレールに当たるまで寄ることがあります。
     - ガードレールを陰影なしで描くと (`CameraConfig.guardrail_light = "flat"`)、この寄りは消えます。
  2. **急カーブでの引き寄せ (陰影なしでも起きる):** 半径 15 m 前後のカーブでは、モデルの予測する道がガードレールの方へ約 0.8 m ずれます。カーブの形は合っていて、車線の中心を見誤っています。
     - 右カーブで車の左 (外側) にガードレールがあると、外へふくらんで当たります。
     - 左カーブで車の右 (外側) にあると、外へ 0.9 m ふくらみます。
     - 左カーブで車の左 (内側) にあると、内側へ切り込みます。低いガードレール (0.1〜0.3 m) ほど強く引かれます。
- **白線があると起きません。**
- **KeiView ではほとんど出ません。** ブラウザで見ても大きく破綻しないのは、このためです。直す対象は KeiSim で出ている癖ですが、KeiView を悪くしてはいけません。
- **考えられる理由:** 学習データに、白線のないガードレールの道がありません。そのため、白く光る縦の線を、車線の境目の代わりに使っているのだと考えています。

## 2. 目標 (合格の条件)

評価はすべて、同じマシンで v0.10 と新しいモデルを並べて走らせて比べます。下の v0.10 の数字は、クラウドの CPU で回したものです (README)。GPU では計算の細かい違いで、ずれが数 cm 変わることがあります。そのため、手元で v0.10 も回し直してください (手順 3.1)。

新しいモデルの名前は `keipilot_v26` です。`keipilot_v25` は、見える車だけで譲るエキスパートの実験 (README の開発ログ 10.) で使った名前なので、使わないでください。

| 評価 | v0.10 (README、CPU) | 必須 | 目標 |
|---|---|---|---|
| **弱点の試験コース `weak`** (KeiSim、晴れのエピソード 0〜3 × 両方向 + 陰影なし × 両方向、10 回) | 10 回すべてで接触 (1 回あたり 13〜63 フレーム)。x は 1 回あたり 4〜9 場面 | 接触 0 件 | x が 0。o が 1 回あたり 14 場面以上 (16 場面中) |
| `weak_lines` (晴れのエピソード 0〜3 × 両方向、8 回) | 接触 0、o 15〜16 | 接触 0、o 15 以上 | 同じ |
| `weak` の KeiView (エピソード 0、両方向) | 接触 0、o 14 / 13 | 接触 0、o 13 以上 | 同じ |
| カーブの試験コース `nolines_guard` / `nolines_guard_nobld` (KeiSim、晴れのエピソード 0〜3、両方向、16 回) | 16 回中 12 回接触 | 接触 0 件 | 同じ |
| カーブの試験コース `lines_guard` (同じ 8 回) | 接触 0、完走 7 (1 回は blocked) | 接触 0 | 同じ |
| 町のベンチマーク (下の 3.5) | `runs/eval/keipilot_v10_hold_red_*.json` と README 2 章 | v0.10 より悪くならない。完走・違反・衝突が同じかよい | 同じ |

必須を満たせば v0.11 として出せます。目標は、次の学習で目指すところです。

o / ^ / x は `scripts/weak_report.py` の印です。場面の中で、車線の中心からのずれがいちばん大きいところを見ます。

| 印 | ずれ |
|---|---|
| o | 0.4 m 未満 |
| ^ | 0.4〜0.75 m |
| x | 0.75 m 以上 (車体が車線を出る)、または接触 |

## 3. 手順

### 3.0 準備

```bash
git pull                       # main (PR #27〜#29 が入っていること)
uv run pytest -q               # 53 件通ること
# KeiView を使う評価・収集は Chrome が要る (keisim/render/keiview.py。見つからないときは CHROME=/path/to/chrome)
```

- **v0.10 の重み:**
  - 学習の始まり (`--init`) と DAgger (`--ckpt`) には、`runs/keipilot_v24/last.pt` (v0.10 の学習のチェックポイント) を使います。
  - 手元にないときは、Release v0.10.0 の `keipilot.pt` (fp16) で代わりになります。`--init` と `--ckpt` のどちらでも読めます。
  - 評価には、推論用の重みを `runs/keipilot_v10.pt` として置きます。
  ```bash
  uv run scripts/export_weights.py runs/keipilot_v24/last.pt runs/keipilot_v10.pt
  # または: gh release download v0.10.0 -R southern-star/keisim -p keipilot.pt -O runs/keipilot_v10.pt
  ```
- **v0.10 の学習データ:**
  - README 5 章「再現」の v0.10 までの節にある `data/...` が、手元にそろっている必要があります。
  - 新しい組だけで学習すると、町での運転が崩れるおそれがあります。そろっていないときは、学習に進む前に知らせてください。

### 3.1 手元で v0.10 の基準を取り直す

これが比べる相手です。

```bash
CK=runs/keipilot_v10.pt        # v0.10 の推論用の重み (3.0)
for ep in 0 1 2 3; do uv run scripts/curve_test.py --agent model --ckpt $CK --brake_hold --red_hold --tp ahead \
    --weather noon --episode $ep --variants weak weak_lines --tag v10_gpu_weak_ep$ep; done
uv run scripts/curve_test.py --agent model --ckpt $CK --brake_hold --red_hold --tp ahead --weather noon \
    --variants weak weak_lines --guardrail_light flat --tag v10_gpu_weak_flat
uv run scripts/curve_test.py --agent model --ckpt $CK --brake_hold --red_hold --tp ahead --renderer keiview \
    --variants weak --tag v10_gpu_weak_keiview
for ep in 0 1 2 3; do uv run scripts/curve_test.py --agent model --ckpt $CK --brake_hold --red_hold --tp ahead \
    --weather noon --episode $ep --variants nolines_guard nolines_guard_nobld lines_guard --tag v10_gpu_curve_ep$ep; done
uv run scripts/weak_report.py runs/eval/curve/v10_gpu_weak_ep{0,1,2,3}.json runs/eval/curve/v10_gpu_weak_flat.json
```

README の表 (CPU) と、印がほぼ同じになることを確かめます。大きく違うときは、学習に進む前に原因を調べてください。

### 3.2 データを集める (学習用の事例)

`--curve_scenes 1` にすると、町がすべてランダムな曲がりくねった 1 本道になります (`keisim/curve_course.py` の `scene_town`)。
- 道は急カーブが多く、ガードレールは左・右・両側・なしです。高さは、下端が 0.35〜0.55 m、上端が 0.65〜0.95 m です。
- 白線はなしが 6 割で、建物は半分の町にあります。
- 天候と太陽の向きはランダムで、目標点は 50 m 先です。
- 周りの車はいません。
- 試験コース (`curve_town`、`weak_town`) とは別の道です。

```bash
uv run scripts/collect.py --out data/guard_expert --frames 40000 --workers 10 --curve_scenes 1 --seed 3101
uv run scripts/collect.py --out data/guard_kv_expert --frames 10000 --workers 3 --curve_scenes 1 --renderer keiview \
    --episodes_per_town 3 --seed 3102
uv run scripts/collect.py --mode dagger --ckpt runs/keipilot_v24/last.pt --out data/guard_dagger --frames 30000 \
    --workers 8 --curve_scenes 1 --seed 3103
```

- **エキスパート:** ノイズと仮想カメラ (左右 ±1.2 m、±9 度) で、車がガードレールに寄った場面と、そこから戻るラベルが入ります。
- **DAgger:** v0.10 自身が寄っていく場面が入ります。
- 集め終わると、`collect.py` が「frames on the guardrail scenes」を出します。100% になっていることを確かめてください。

### 3.3 学習する (v0.10 から追加学習)

v0.10 の 9 回目 (`osm_train runs/keipilot_v24 v23 v22 --mem_input`) と同じデータに、新しい 3 組を重み 2 で足し、v24 から 2 エポック学習します。

- `$D7 $FL $NP $T $W9`、`osm_sets`、`osm_w` は、README 5 章「再現」の v0.10 までの節で定義されています。先にその節の変数と関数の定義を、シェルに読み込んでください。
- 学習データはすべてメモリに読み込みます。約 40 GB 要ります (README)。

```bash
uv run scripts/train.py --data $D7 $FL $(osm_sets v23) $(osm_sets v22) $NP data/guard_expert data/guard_kv_expert data/guard_dagger \
    --init runs/keipilot_v24/last.pt --out runs/keipilot_v26 --epochs 2 $T --aux_weight 0.5 --mem_input $W9 \
    $(osm_w v23 1.5) $(osm_w v22 1) data/osm_dagger_v11_np=1.5 data/osm_kv_dagger_v11_np=1.5 \
    data/guard_expert=2 data/guard_kv_expert=2 data/guard_dagger=2
uv run scripts/export_weights.py runs/keipilot_v26/last.pt runs/keipilot_v26.pt
```

### 3.4 弱点のコースとカーブのコースで評価する

3.1 と同じコマンドで、`CK=runs/keipilot_v26.pt`、タグを `v26_gpu_...` にして回します。そのあと、v0.10 と並べた表を作ります。

```bash
uv run scripts/weak_report.py runs/eval/curve/v10_gpu_weak_ep{0,1,2,3}.json runs/eval/curve/v10_gpu_weak_flat.json \
    runs/eval/curve/v26_gpu_weak_ep{0,1,2,3}.json runs/eval/curve/v26_gpu_weak_flat.json    # 列名にモデルの名前が付く
uv run scripts/weak_report.py runs/eval/curve/v10_gpu_weak_keiview.json runs/eval/curve/v26_gpu_weak_keiview.json
uv run scripts/weak_report.py --variant weak_lines runs/eval/curve/v10_gpu_weak_ep0.json runs/eval/curve/v26_gpu_weak_ep0.json
```

カーブの試験コースは `curve_test.py` の出力の `end:` の行 (完走かどうか) と、`runs/eval/curve/v26_gpu_curve_ep*.json` の `status` で見ます。

### 3.5 町のベンチマークで、悪くなっていないことを確かめる

v0.10 の数字は、町のスイートが `runs/eval/keipilot_v10_hold_red_*.json`、札幌が `runs/eval/osm_sapporo/keipilot_v24_hold_red*.json` にあります (README 2 章)。
札幌のエピソード 200〜 / 300〜 は、`--out` を付けないと 100〜 の結果に上書きされます。

```bash
E="uv run scripts/evaluate.py --agent model --ckpt runs/keipilot_v26.pt --brake_hold --red_hold"
$E --suite test && $E --suite test2 && $E --suite train && $E --suite test --dense --max_steps 6000   # KeiSim の 4 スイート
$E --suite test --renderer keiview && $E --suite test2 --renderer keiview                          # KeiView
$E --suite long3c && $E --suite long3c --dense && $E --suite long3d --renderer keiview               # 2 現示の確認用
O="uv run scripts/evaluate.py --agent model --ckpt runs/keipilot_v26.pt --brake_hold --red_hold --osm_town osm_sapporo --routes 20"
S=runs/eval/osm_sapporo/keipilot_v26_hold_red
$O && $O --episode0 200 --out ${S}_ep200.json && $O --episode0 300 --out ${S}_ep300.json             # 札幌 60 ルート
uv run scripts/summarize.py
```

ブラウザ版でも見るときは、ONNX を書き出します。
```bash
uv run --with onnx --with onnxruntime scripts/export_onnx.py runs/keipilot_v26/last.pt web/models/keipilot_v26.onnx
cd web && node tools/serve.mjs   # → http://localhost:5174/?town=curve_weak_w1&pilot=1&model=models/keipilot_v26.onnx
node tools/pilot_check.mjs --town curve_weak_w1 --sec 120 --model models/keipilot_v26.onnx   # 画面なしで走らせる
```

## 4. 判断のしかた

| 結果 | 次にやること |
|---|---|
| 合格の条件をすべて満たす | 手順 5 で報告する。README の開発ログに 11. として書く |
| 弱点のコースで、外側のガードレール (右カーブ R16・車の左) にまだ当たる | v26 が運転する DAgger をもう 1 組集める (`--mode dagger --ckpt runs/keipilot_v26/last.pt --out data/guard_dagger2 --seed 3104`)。前の組に足して、v26 から学習し直す |
| 低いガードレール (左カーブ R14・車の左・低い・内側) だけ ^ か x が残る | 学習用の事例の高さ (`scene_town` の `lo = rng.uniform(0.35, 0.55)`) は、わざと 0.35 m からにしてある。0.1〜0.3 m の試験コースの低いガードレールは、見たことのない高さとして残してある。これを下げる (例 0.10〜0.55) と、試験は「見たことのある高さ」になるので、そう書き残す |
| 町のベンチマークが悪くなる | 新しい組の重みを 1 に下げる。どのスイートの、どの失敗かを `runs/failures/` で確かめる |
| 弱点のコースは良くなったが、KeiView の `weak` が悪くなる | KeiView の組 (`guard_kv_expert`) を増やす (例 2 万フレーム) か、重みを上げる |
| `weak_lines` が悪くなる | 白線があるときの見方まで変わっている。新しい組の重みを下げる |

### 変えてはいけないもの

- **試験コース:** `curve_town` (`COURSE_VERSION = "v3"`) と `weak_town` (`WEAK_VERSION = "w1"`)。直すときは版を上げて、v0.10 も回し直す。
- **試験コースで学習しないこと:** `--curve_scenes` の道は試験コースとは別。試験コースの町を `collect.py` に渡さない。
- **合格の条件:** 結果を見てから変えない。変えるなら、理由を書き残す。
- **陰影なしは調べるための設定です。** `guardrail_light = "flat"` で学習しない。直したいのは、陰影があっても寄らないことです。
- **`Raster.fill` の半画素のずれ (5 cm、README):** 直すと町の生成が変わる。この作業では触らない。

## 5. 報告すること

- 3.4 の `weak_report.py` の表 (v0.10 と v26 を並べたもの)。KeiSim の 10 回と KeiView の 2 回です。
- カーブの試験コースの接触の数 (v0.10 と v26)。
- 3.5 の各スイートの数字 (完走・DS・違反・衝突)。v0.10 の数字と並べます。
- **重み:** `runs/*` は git に入りません (`.gitignore` の例外は `runs/eval/` だけ)。クラウドでも評価するなら、`runs/keipilot_v26.pt` (約 30 MB) と ONNX を Release に置いてください (今までの版と同じ `keipilot.pt` / `keipilot.onnx`)。
- **結果のファイル:** `runs/eval/curve/` と `runs/eval/*.json` は commit する。集めたデータ (`data/`) はリポジトリに入れない。

## 6. 関係するファイル

| ファイル | 中身 |
|---|---|
| `keisim/curve_course.py` | 試験コース (`curve_town`、`weak_town`、`course_route`、`course_pieces`) と、学習用の事例 (`scene_town`、`scene_route`) |
| `scripts/curve_test.py` | 試験コースを走らせる (`--variants weak weak_lines`、`--episode`、`--guardrail_light flat`、`--renderer keiview`)。弱点のコースでは、接触しても走り続けて場面ごとに数える |
| `scripts/weak_report.py` | 弱点のコースの、場面ごとの表 |
| `scripts/curve_diag.py` | 1 ステップごとに、モデルの道とエキスパートの道を比べる (原因を調べるとき) |
| `scripts/collect.py` | `--curve_scenes P`: 町を確率 P で学習用の事例にする |
| `keisim/roadnet.py` | `RoadSpec.guardrail` / `guardrail_spans` / `guardrail_beam` |
| `keisim/render/camera.py` | `CameraConfig.guardrail_light` (`sun` / `flat`) |
| `web/towns/town_curve_*.json` | ブラウザ版のコース (`?town=curve_weak_w1&pilot=1`)。`scripts/export_town.py --curve` で書き出し直せる |
