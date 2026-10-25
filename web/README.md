# KeiView — KeiSim の街をアニメ調で歩く

KeiSim の手続き生成の街（道路網・交差点・信号・建物の区画・街路樹）を、
[Sakuragaoka Station](https://github.com/Kenton-GMI/sakuragaoka-station)（MIT）の
**家の生成器**と**セル調（三渲二）レンダラ**で描く three.js ビューアです。ビルド不要、外部アセットなし。

![KeiView](../docs/keiview_gallery.jpg)

## 仕組み

```
KeiSim Town(seed)  ──scripts/export_town.py──▶  web/towns/town_<seed>.json  ──▶  web/ (three.js)
  道路中心線・交差点ポリゴン              道路 / 歩道の輪郭（ラスタ化して結合）       アスファルト・縁石付き歩道・白線・横断歩道
  白線・横断歩道・停止線                   建物ごとの接道・側・セットバック             ↳ 各建物 = 区画 → Sakuragaoka の家の生成器
  信号（位置・現示・周期）                  追加の区画（すき間・二列目・街区の内側）      ↳ 高い建物 = 4〜10 階のマンション
  建物の箱・街路樹                                                                    ↳ 街路樹 = 桜の生成器、信号 = KeiSim と同じ現示
                                                                                      ↳ 電柱と電線（引込線は家の壁に raycast）
```

* **道路は KeiSim と完全に一致**します。歩道・車道の形は KeiSim 自身のポリゴン（`maptex.py` と同じ重ね順）をラスタ化して輪郭を取り直したもので、ブラウザ側でポリゴン演算をせずに縁石付きの歩道を押し出せます。
* **建物は KeiSim の箱をそのまま「区画」として使います。** 箱の向き・接している道路・セットバックを復元し、区画の前面を歩道の外縁に合わせて Sakuragaoka の `buildLot()`（間取り、階数、玄関、ベランダ、洗濯物、塀、庭…）に渡します。16 m 以上の箱は `mansion.js` のマンションになります（42 m までを 4〜10 階に圧縮）。
* **KeiSim が空けている場所**（建物のすき間、二列目、街区の内側）は、エクスポート時に追加の区画で埋めます（家・畑・月極駐車場・空き地・小さな公園）。すべて道路外なので、KeiSim のシミュレーションには影響しません。
* **信号の灯火は KeiSim と同じ式**（`Junction.signal_state`）で時刻 t から決まります。
* 街の外側は Sakuragaoka の遠景の家並み（`houses/far.js`）で囲みます。
* すべての乱数はシード付きです。同じ `town` なら毎回同じ街になります。

## 使い方

```bash
uv run scripts/export_town.py --town 1000 1001 1002 1003     # web/towns/ に出力（サンプル 4 つは同梱）
uv run scripts/export_town.py --town 1015 1018 --style twophase   # 二段階の信号の町 → town_1015-98782ab7.json など（同梱）
uv run scripts/import_osm.py --name osm_sapporo --out web/towns     # 実在の町（札幌、OpenStreetMap）→ town_osm_sapporo.json（同梱、data/osm/ の抜き出しから）
cd web && node tools/serve.mjs                               # → http://localhost:5174/?town=1000
```

任意の静的サーバで動きます（ES modules なので `file://` 不可）。three.js とフォントは jsDelivr / Google Fonts から読み込みます。
`npm install` しておくと、`serve.mjs` と `shot.mjs` は `node_modules` の three.js を使います（オフラインや CDN が塞がれた環境向け）。

| 操作 | |
|---|---|
| マウス / WASD / Shift / Space | 視点 / 歩く / 走る / ジャンプ |
| F | 飛行モード |
| 1〜5 | 視点プリセット（通り・交差点・俯瞰・街区・遠くの交差点） |
| R / H | 最初の視点に戻る / UI を隠す |

URL パラメータ: `?town=1001`、`?view=3`、`?t=20`（シミュレーション時刻 → 信号の現示）、`?q=medium`、`?roads=detail`（路面の作り込み、下）、`?only=roads,houses`、`?stats`、`?fly`、`?tdir=.towns`（KeiSim が書き出した街のフォルダ）

## 路面の作り込み（`?roads=detail`）

`src/world/roaddeco.js` が、`roads.js` の路面の上に次のものを足します。画面右上のメニュー「路面: 作り込み」でも切り替えられます。

- 路面標示: 信号のある交差点の停止線の 30 m と 50 m 手前の◇（横断歩道あり）、道路の始まりの黄色の最高速度（規制標示、道路の制限速度を 10 km/h 単位に丸めたもの）、歩道のある道路の青い矢羽根（自転車の通行位置）
- 舗装: 信号のある交差点の 35 % を赤茶色のカラー舗装に。生活道路（道路の種類）の半分は路側帯を緑に（通学路）
- 路面のふた: 道路の中央付近のマンホール（亀甲模様・桜のデザイン・同心円の 3 種）、消火栓（黄色の縁取り）、仕切弁とガスの小さなふた、側溝のグレーチング
- 補修跡: 打ち換えた舗装・横断する掘削跡・ひび割れのシール
- 歩道: 横断歩道の端の点状ブロックと、そこから歩道に沿う線状ブロック、縁石沿いのガードパイプ
- 道路の付属物: カーブ（2 本の道路だけがつながる角）の外側のカーブミラー、最高速度・横断歩道・駐車禁止の標識

配置はすべて街のデータから決まり、乱数はシード付きです（同じ街なら毎回同じ）。道路の形は変えず、車線の上に立つものも置きません。
既定（`plain`）ではこのモジュールを読み込まないので、今までの見た目と 1 ピクセルも変わりません。
これまでの KeiView の学習データと評価画像はすべて `plain` です。学習済みのモデルを `detail` で走らせると、モデルにとっては初めて見る絵になります。

エゴモードのセマンティックラベルは、路面標示が marking、ふた・補修跡・カラー舗装が road、点字ブロックが sidewalk、ガードパイプ・標識・カーブミラーが pole です。

## 道路の種類

KeiSim の roadtypes の街（`--town_style roadtypes`、[../README.md](../README.md) の 3 章）では、道路ごとに幅・歩道・白線が違います。
エクスポートは道路ごとの断面（`kind`、車線幅、路側帯、歩道の幅、制限速度）と黄色のセンターライン（`markings_yellow`）を足します。
ビューアは、歩道のない道路では電柱を道路のすぐ外に、視点プリセットを路側帯に置きます。

```bash
uv run scripts/export_town.py --town 1000 --style roadtypes     # web/towns/town_1000-<設定のハッシュ>.json（1000 は同梱）
```

## 開発ツール

```bash
cd web && npm install                         # three + puppeteer-core（ツール用のみ）
node tools/check.mjs --town 1000              # GPU なしのビルド検査（モジュールごとのエラー・ポリゴン数・時間）
node tools/check.mjs --town 1000 ground roads houses trees signals poles roaddeco   # 路面の作り込みも
node tools/shot.mjs --town 1000 --views 1,2,3 --out shots/t1000     # 実レンダラでスクリーンショット（--roads detail、--tdir .towns も可）
```

`shot.mjs` は Linux では SwiftShader（CPU 描画）を使うので、GPU のないクラウドコンテナでも動きます（4 コアで準備に約 30 秒、`--q medium` で 1 枚あたり約 20〜30 秒）。
Windows / macOS では GPU を使います。ブラウザは `$CHROME`、Playwright の Chromium、Edge、Chrome の順に探します。

## KeiSim のカメラとして使う（エゴモード）

`?ego=1` を付けると、アニメーションループを回さず、KeiSim の自車カメラ 1 フレームを要求に応じて描画するモードになります。
KeiPilot の学習データ収集とクローズドループ評価で使います（学習の手順と結果は [../README.md](../README.md) の 6 章）。

- `src/ego.js` の `window.__ego(req)` は、カメラ・時刻・周囲の車と歩行者を受け取り、RGB 画像と **ピクセル単位で正確なセマンティックラベル** を返します。ラベルは KeiSim と同じ 13 クラスです。
  - 静的メッシュはバッチング前に `userData.sem` でクラスを付けます（`vendor/.../batch2.js` はクラスごとに別々にまとめます）。
  - ラベル画像は、マテリアルをクラス色に差し替えた 2 回目の描画で作ります。
- `src/world/actors.js` は KeiSim の車（セダン・軽・バン・トラック、ブレーキランプ付き）と歩行者を同じセル調で描きます。
  - 学習と評価のフレームは、この簡易な形の車で描きます。パイロットモードの表示に使う詳しい形の車（`src/world/cars.js`）は、`?cardetail=1` を付けたときだけ描きます。
- `tools/ego_server.mjs` は headless Chrome を GPU で動かし、標準入出力の JSON 行で Python（`keisim/render/keiview.py`）とやり取りします。
  - RTX 3060 では、ラベル込みで 1 フレーム約 17 ms、RGB のみで約 12 ms です。
  - 街の読み込みは約 8 秒です。
  - `--gl auto`（既定）は、GPU で WebGL2 が使えなければ SwiftShader（CPU 描画）に自動で切り替えます。`--gl hw` / `--gl soft` で固定もできます（Python 側は `EnvConfig.keiview_gl`）。
  - `--roads detail` で路面の作り込みを足して描きます（Python 側は `EnvConfig.keiview_roads`、スクリプトは `--keiview_roads detail`）。既定は `plain` です。
  - SwiftShader では 1 フレーム約 2 秒です（4 vCPU のクラウド VM で計測。並列にしてもほぼ伸びません）。動作確認やデモ動画 1 本向けです。
- エゴモードでは信号の灯器を 2.2 倍で描きます（`?sigscale=` で変更可）。
  - 実寸（灯火 30 cm）だと、320×160 の画像では 30 m 先の灯火が 1〜2 ピクセルになり、見えないフレームが多いためです。
  - 1.6 倍でも、広い交差点の向こう側の灯火は 3 ピクセルしかなく、読み取れませんでした。
  - KeiSim 側の描画も同じ理由で灯火を誇張しています。
- 灯火には向きがあります（実物のひさしと LED と同じ）。
  - 正面から 30° までは明るく、60° より横からは消灯と同じに見えます。
  - ラベル画像でも、45° より横からの灯火は「点灯」に数えません。
  - 消灯中の灯火は、昼間の実物と同じくほぼ黒です。
- 太陽の向き・露出・霞はエピソードごとにランダムです。乱数は KeiSim とは別系列なので、同じシードなら KeiSim で描いたときと同じエピソードになります。

```bash
cd web && npm install && cd ..
uv run scripts/demo.py --agent model --ckpt runs/keipilot.pt --renderer keiview --town 1001 --episode 3 --out runs/demo_kv.mp4   # Release v0.10.0 の重み
```

## ブラウザで KeiPilot を走らせる（パイロットモード）

`?pilot=1` を付けると、運転モデル KeiPilot がブラウザの中で車を運転します（Python も GPU サーバも不要）。
公開版: <https://southern-star.github.io/keisim/?pilot=1>

```bash
uv run --with onnx --with onnxruntime scripts/export_onnx.py runs/keipilot_v24/last.pt web/models/keipilot.onnx
#   （または Release v0.10.0 の keipilot.onnx を web/models/ に置く）
cd web && node tools/serve.mjs                    # → http://localhost:5174/?pilot=1&town=1000
```

- KeiSim の閉ループをそのまま JavaScript に移しています。
  - 毎秒 10 回、自車カメラをエゴモードと同じ 320×160 で描きます（表示用の高解像度の描画とは別のパイプライン）。
  - モデル（`scripts/export_onnx.py` で ONNX にしたもの）がその画像から経路と目標速度を出します。実行は [onnxruntime-web](https://onnxruntime.ai/) で、WebGPU があれば GPU、なければ WASM（Worker の中で実行し、描画は止めません）です。
  - 車は KeiSim と同じ制御器（Pure Pursuit と PI 速度制御、`keisim/control.py`）と車両モデル（`keisim/vehicle.py`、20 Hz）で走ります。
  - 安全層 BrakeHold / RedHold も `keipilot/agent.py` と同じものが入っています。
  - ルートとモデルへの 2 つの入力（次の信号交差点で曲がる方向と、その交差点を出て 4 m 先の目標点）は、`src/nav.js` が `keisim/route.py` と同じ規則で作ります。
  - v0.10 のモデルには、最後に止まってから走った距離（10 m で頭打ち）も入れます。止まれの線の手前で「まだ止まっていない」のか「止まったあとで発進した」のかは、1 枚の画像では分からないためです。`src/pilot.js` が `keipilot/data.py` の `stand_update` と同じく制御のたびに数えます（ONNX に `stand` 入力があるときだけ。前の版のモデルもそのまま動きます）。
- 曲がる方向はキー ← ↑ →（または画面のボタン）で、次の交差点について指定します。
  - 指定しなければランダムに曲がります。
  - Space で一時停止、R で最初から、T で周りの車の表示を切り替えます。
- 画面の線の意味:
  - ピンクはモデルが描いた走行経路です。
  - 水色はナビのルートです。
  - 右下に、モデルへの入力画像とモデルの領域分割を表示します。
- 信号は KeiSim と同じ現示で切り替わります。
  - town 1000〜1003 は、流入方向ごとに 1 方向ずつ青になるスプリット現示です。交差点の中で進路が交わらないので、右折待ちはありません。
  - town 1015 と 1018（`?town=1015-98782ab7`、`?town=1018-98782ab7`、右上のメニューでも選べます）は、日本でふつうの二段階の信号です。向かい合う方向が同時に青になり、右折は交差点の中の待ち位置で対向車の切れ目を待ちます（long3 と同じ街のタイプ）。KeiPilot は v0.6 からこの右折待ちを学んでいます。
- 周りの車（`src/traffic.js`）が街を走ります。歩行者はいません。
  - 台数は車線の長さに合わせて決めます（車線 1 km あたり 12 台、最大 40 台。town 1000 では 28 台）。
  - 動きは `keisim/traffic.py` の NPC を軽くしたものです。交差点ごとにランダムに曲がり、IDM で前の車（と自車）との車間を取り、赤と黄色で止まります（黄色は止まれるときだけ）。交差点の中に別の現示の車がいるときと、出口が詰まっているときは、停止線で待ちます。
  - 二段階の信号の町では、KeiSim と同じ右折のルールで動きます。
  - 実在の町「札幌」（`?town=osm_sapporo`、[../README.md](../README.md) 2 章の「実在の町」）では、信号のない交差点で KeiSim と同じ優先道路の規則で動きます（細い道から来た車が優先道路のすべての動きに譲る、右折は対向の左折を先に行かせる、車は 6 m まで）。細い道の入口には止まれの標識と停止線があり、周りの車はそこで一度止まってから譲ります。地図データは © OpenStreetMap contributors (ODbL) で、画面の右下に出典を出します。
    - 進路が交わる・合流する動きは、直進 > 左折 > 右折 の順に優先します。
    - 同じ向きの動きどうしが交わるとき（斜めの交差点の右折どうし）は、片方が譲ります。決め方も KeiSim と同じです（`Town._equal_gives_way`）。
      town 1015 と 1018 にはこの交差はありません。
    - 右折は待ち位置で、優先の車（自車も含む）が 5 秒以内に交わる所に来るあいだは待ちます。
    - 待ち位置と、譲る相手は、KeiSim の `Town._make_yields` と同じ手順で JavaScript で求めます。town 1015 では、KeiSim と同じ 14 本の右折が譲り、待ち位置の差は 0.25 m 以内です。
  - 自車のカメラに見えない所で 40 秒以上止まっている車は、別の場所に移します。
  - 自車と車体が重なると「接触」として数え、HUD に表示します。
  - 描画が重いときは、画面の「周りの車」ボタンか T キーで消せます（選んだ設定はブラウザに残ります）。`?traffic=0` でも消せます。周りの車の描画は、1 台を材質ごとに 1 つのメッシュにまとめて軽くしています（車 24 台が見えているとき、描画命令は 520 → 740、RTX 3060・1280×720 で 1 フレームあたり 2〜3 ms 増）。
- カメラから 70 m 以内の周りの車は、詳しい形のモデル（`src/world/cars.js`）で描きます。
  - 車種は KeiSim が決めた車の大きさで選びます。軽ハイトワゴン、軽ハッチ、軽トラ（白い軽に多い）、コンパクト、セダン、SUV、ミニバン、小型トラックの 8 種類です。
  - 車体は、横から見た輪郭を車幅の方向に押し出して角を丸めたものです。暗い色ガラス、窓枠とピラー、ドアの継ぎ目とノブ、ホイール、ブレーキで光るテールランプ（ハイマウントも）、ナンバー（軽は黄色）を付けています。
  - モデルへの入力（320×160）は、学習のときと同じ簡易な形の車で描きます。表示の車を変えても、モデルの入力は変わりません。
  - 1 台は 1,400〜2,300 ポリゴンで、これも材質ごとに 1 つのメッシュにまとめています。簡易な形と比べて、描画命令は数個増えるだけで、描画時間の差は測れないほどです（RTX 3060・1280×720）。
  - `?cardetail=0` で、表示も簡易な形の車に戻せます。

  ![周りの車の詳しい形](../docs/keiview_cars.jpg)
- ルートから 6 m 外れるか、90 秒動けないと、最初からやり直します（KeiSim の打ち切り条件と同じ）。
- 推論 1 回の時間:
  - WebGPU（RTX 3060）は約 20〜40 ms です。
  - WASM（1 スレッド）は約 0.2〜0.35 秒です。WASM でも閉ループは回りますが、計画の更新が遅れるぶん反応が遅くなります。
- モデルの重みは float16 で保存し、グラフの中で float32 に戻します（30 MB、計算は float32）。値は Release の重みと同じです。

URL パラメータ: `?town=1001`（二段階の信号の町は `?town=1015-98782ab7`、札幌は `?town=osm_sapporo`）、`?q=low`（表示の画質。モデルの入力は常に medium）、`?ep=wasm` / `?ep=webgpu`（推論エンジンを固定）、`?model=<URL>`、`?traffic=0`（周りの車なし）、`?cars=20`（周りの車の台数）、`?cardetail=0`（周りの車を簡易な形で表示）、`?tdir=.towns`（`scripts/import_osm.py` が `web/.towns` に書き出した町、例 `?tdir=.towns&town=osm_nishioi`: 一方通行の道路は中心線の上の車線 1 本で、周りの車もその向きにだけ走ります）

```bash
node tools/pilot_check.mjs --town 1000 --sec 60   # headless Chrome で 60 秒（シミュレーション時間）走らせて記録を表示
node tools/pilot_check.mjs --url https://southern-star.github.io/keisim/ --sec 10   # 公開版を確かめる
node tools/pilot_check.mjs --town osm_sapporo --sec 120 --model models/keipilot_v24.onnx   # 札幌で、別の ONNX（web/ の下）を走らせる
```

`pilot_check.mjs` は、モデルの計算を毎ステップ待つので、KeiSim と同じく推論の遅れがない状態で走ります。
Linux では Vulkan を使い、GPU の WebGPU で推論します。フラグなしの headless Chrome の WebGPU は SwiftShader（CPU）で、1 回 6 秒かかります。
ページ側も、SwiftShader の WebGPU しかないブラウザでは WASM を使います。

## 構成

```
index.html, src/main.js     ブートストラップ（レンダラ・空・モジュール構築・バッチング・ループ・HUD・撮影用フック）
src/town.js                 KeiSim → three.js 座標変換、路面グリッド（歩道の高さ・道路までの距離）、視点プリセット
src/world/ground.js         地面
src/world/roads.js          車道・側溝・縁石付き歩道・白線・横断歩道
src/world/houses.js         KeiSim の建物と追加区画 → 家 / マンション / 畑など（Sakuragaoka の家キットを使用）
src/world/mansion.js        マンション生成器
src/world/plots.js          畑・月極駐車場・空き地・公園
src/world/trees.js          街路樹 → 桜
src/world/signals.js        信号機（灯火は KeiSim の現示どおり）
src/world/poles.js          電柱・電線・引込線・支線
src/world/roaddeco.js       路面の作り込み（?roads=detail）: 路面標示・カラー舗装・ふた・点字ブロック・ガードパイプ・標識
src/world/actors.js         KeiSim の車と歩行者（エゴモード、パイロットモードの周りの車）
src/world/cars.js           表示用の詳しい形の車（8 車種）
src/ego.js                  エゴモード（1 フレーム描画・セマンティックラベル・ライティングのランダム化）
src/pilot.js                パイロットモード（モデルの入力の描画、onnxruntime-web、制御器と車両モデル、安全層、表示）
src/nav.js                  パイロットモードのルート、コマンド、目標点（keisim/route.py と同じ規則）
vendor/sakuragaoka/         Sakuragaoka Station から持ってきたコード（MIT、変更点は NOTICE.md）
towns/                      エクスポート済みの街（1000〜1003、二段階の信号の 1015 と 1018、道路の種類の 1000-dae67d73）。エゴモードの街は .towns/ に自動で出力（git 対象外）
tools/                      serve / check / shot / ego_server / pilot_check
```

## クレジット

レンダラ、マテリアル、家・桜・電柱の生成器は [Sakuragaoka Station](https://github.com/Kenton-GMI/sakuragaoka-station)
（MIT, © 2026 Sakuragaoka Station contributors）によるものです。持ってきたファイルと変更点は
[vendor/sakuragaoka/NOTICE.md](vendor/sakuragaoka/NOTICE.md) を参照してください。
