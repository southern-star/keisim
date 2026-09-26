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

URL パラメータ: `?town=1001`、`?view=3`、`?t=20`（シミュレーション時刻 → 信号の現示）、`?q=medium`、`?only=roads,houses`、`?stats`、`?fly`

## 開発ツール

```bash
cd web && npm install                         # three + puppeteer-core（ツール用のみ）
node tools/check.mjs --town 1000              # GPU なしのビルド検査（モジュールごとのエラー・ポリゴン数・時間）
node tools/shot.mjs --town 1000 --views 1,2,3 --out shots/t1000     # 実レンダラでスクリーンショット
```

`shot.mjs` は Linux では SwiftShader（CPU 描画）を使うので、GPU のないクラウドコンテナでも動きます（4 コアで準備に約 30 秒、`--q medium` で 1 枚あたり約 20〜30 秒）。
Windows / macOS では GPU を使います。ブラウザは `$CHROME`、Playwright の Chromium、Edge、Chrome の順に探します。

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
vendor/sakuragaoka/         Sakuragaoka Station から持ってきたコード（MIT、変更点は NOTICE.md）
towns/                      エクスポート済みの街（1000〜1003）
tools/                      serve / check / shot
```

## クレジット

レンダラ、マテリアル、家・桜・電柱の生成器は [Sakuragaoka Station](https://github.com/Kenton-GMI/sakuragaoka-station)
（MIT, © 2026 Sakuragaoka Station contributors）によるものです。持ってきたファイルと変更点は
[vendor/sakuragaoka/NOTICE.md](vendor/sakuragaoka/NOTICE.md) を参照してください。
