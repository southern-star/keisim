# 読み込み器に一方通行を足す計画 (品川区 西大井・二葉 に向けて)

作成日 2026-10-25。対象: `keisim/realworld.py` (OSM → 道路グラフ)、`keisim/roadnet.py` (`Town.from_graph` / `_build_from_graph`)、`scripts/export_town.py`、`web/src/{nav,traffic,town}.js`、`web/src/world/roaddeco.js`、`tests/test_realworld.py`。

## なぜ必要か (西大井の数字)

OSM の `oneway` を落とす今の読み込み器では、西大井の二車線道路網がばらばらになり、行き止まり刈り込みと最大連結成分の選択でほとんど消えます。一方通行を向き付きの辺として残すと、走れる道路はおよそ 2.5〜3 倍になります。

| 箱 | 現状の町 (二車線のみ) | 一方通行込みの概算 (最大強連結成分、整形前) | 内訳 |
|---|---|---|---|
| A 35.616〜35.625 N, 139.723〜139.734 E | 2.52 km、交差点 20 (信号 4) | 約 6.8 km | 二車線 5.2 km + 一方通行 1.6 km |
| B 35.612〜35.621 N, 139.726〜139.737 E | 3.53 km、交差点 46 (信号 0) | 約 8.5 km | 二車線 6.9 km + 一方通行 1.6 km |
| (参考) 札幌 | 18.1 km、交差点 240 (信号 11) | — | — |

西大井の一方通行は `lanes=1` か `lanes` なし (residential / unclassified / tertiary、幅タグなし) が 4〜5 km、`lanes=2` の一方通行は 0.12 km しかなく、残りは山手通りの上下線 (primary、今まで通り落とす) です。つまり「片側 1 車線だけの道路」を足せば、西大井の一方通行はほぼ全部表現できます。

ついでに分かったこと: 二車線道路のうち `lanes=1` (中央線なしの 4 m 道路) が A で 1.6 km、B で 2.0 km あります。今は 6 m の residential として作られます。本計画の範囲外ですが、alley 相当の断面に落とす小さな追加を末尾に挙げます。

## 設計

### 道路の表現

- `RoadSpec` に `oneway: bool` を足す。一方通行の断面は同じ道路種の **車線 1 本を中心線上に置き**、`half_width = lane_width / 2 + shoulder`、`walk_outer = half_width + sidewalk`。residential の一方通行は 4.5 m 幅 (3.0 + 0.75 × 2)、street は 5.0 m 幅 + 歩道。OSM に幅タグがないので当面はこの固定値、PLATEAU 道路 LOD3 の `uro:RoadStructureAttribute width` を使う拡張は別件。
- `Road` の向きの約束: 一方通行の道路は **a → b が進行方向**。`Road.lane_ab` が唯一の車線、`Road.lane_ba = -1`。
- 速度制限・路面種別 (center "none"、edge "lane"/"classic") は道路種のまま。中央線は引かない。

### グラフ (`keisim/realworld.py`)

1. `road_kind` は一方通行を落とさなくなり、別関数 `oneway_of(tags)` が `yes/1/true → +1`、`-1 → 反転`、それ以外 0 を返す。`lanes >= 2` の一方通行 (複数車線の上下線分離) と roundabout は引き続き落とす (`dropped["multi-lane one-way"]`)。
2. 辺に `oneway` を持たせ、`-1` の way は頂点列を反転して登録 (**辺の pts は常に進行方向**)。`Graph.oriented()` を使う箇所は向きを失うので、各整形パスで扱いを決める:
   - `merge_straight`: 両辺が同じ `oneway` で、直進方向が一致する (p1 が n に入り p2 が n から出る向きで両方とも進行方向) ときだけ結合。
   - `drop_duplicates`: 同じ 2 節点を結ぶ平行な辺が **逆向きの一方通行同士** なら 1 本の二車線辺にまとめる (中央分離帯を 2 本の way で描いた道路)、同じ向きなら今まで通り片方を落とす。
   - `contract_short`: 節点の統合で辺の向きは変わらないのでそのまま。自己ループの削除も同じ。
   - `split_sharp`、`drop_acute_arms`、`clip`: 向きを引き継ぐだけ。
3. `prune_dead_ends` を **向きを見る刈り込み** に置き換える。節点ごとに、入ってくる腕 (二車線 or 一方通行の終点) と出ていく腕 (二車線 or 一方通行の始点) を数え、
   - 入ってくる腕のそれぞれについて別の道路の出ていく腕がない (U ターンしかない)、または
   - 出ていく腕のそれぞれについて別の道路の入ってくる腕がない (到達できない車線)
   なら、その辺を落として繰り返す。二車線の行き止まり (次数 1) は従来どおりここで消える。
4. `largest_component` を **最大強連結成分** (Kosaraju) に置き換える。二車線辺は両向きの弧として扱う。
5. `est_radius` / `Town.graph_min_radius`: 一方通行の腕は車線が中心線上にあるので、曲がりの弧半径は二車線より少し大きくなる方向。式は最大車線幅のまま使い、`tests` の expert 走行で確認する (6 m で起きた PurePursuit の出口ドリフトは半径 8 m で収まっているので、余裕側)。
6. `graph_town`: 道路の `(a, b, pts, spec)` の正規化で `a > b` のとき頂点列を反転しているが、一方通行の辺は **反転しない** (a → b が進行方向のまま)。`RoadSpec.oneway` で `Town.from_graph` に伝える。
7. `osm_town(..., oneway=True)` を引数にする。**同梱の札幌は `oneway=False` のまま** (下記「札幌を変えない」)。`scripts/import_osm.py` に `--no-oneway`。`graph_report` に一方通行の km を出す。

### 町の生成 (`keisim/roadnet.py`)

1. `_build_from_graph` の車線生成: `rs.oneway` なら `offset_polyline(road.center, 0)` の 1 本だけ (`lane_ab`)、`lane_ba = -1`。
2. 腕 (`arm["in"]` / `arm["out"]`): 無い側は `-1`。`J.incoming` / `J.outgoing` は `-1` を除く。
3. 接続レーン: `arm_in["in"] >= 0` かつ `arm_out["out"] >= 0` の組だけ作る (同じ道路への U ターン除外は今のまま)。
4. 信号: `arm["in"] < 0` の腕には `signal` / `stop_s` を付けない。`_two_phase_groups` は腕の角度だけ見るので変更なし (出るだけの腕も相を持つが使われない)。`_make_signal_heads` は `arm["in"] < 0` を飛ばす。
5. `_priority_groups`: 優先道路の対は **全腕の幾何** で選び (出るだけの腕も対の候補)、戻り値は `in` 車線がある腕だけに絞る。`_make_yields` / `_equal_gives_way` は `pred[0]` 経由なので変更なし。bend の `narrow` 規則は接続レーンが 1 本なら自然に何もしない。
6. 標示 `_make_markings`: 一方通行は中央線なし。縁線は `rs.edge` に従い車線端 (`lane_width / 2 + ...`) に、停止線は車線 1 本の幅 (±lane_width/2、classic は路肩まで) に。横断歩道は道路幅いっぱいで今のまま。一方通行の矢印は描かない (日本では標識。モデルも読まない)。
7. `_finalize_static`: `road_lanes` / `truck_lanes` / `walkways` はそのままで通る (walkways は道路両側、`in_road` の 路側帯 も両側)。
8. `Town.from_graph` の `Road` 生成は `RoadSpec` 経由で向きが来るので引数の形は変えない。

### 書き出しとブラウザ

- `scripts/export_town.py`: `roads[].oneway = true` を追加 (`half_width` などは既に道路ごとに出ている)。`format` version は 1 のまま (追加フィールドのみ)。`real_town_fields` には変更なし。
- `web/src/nav.js` `buildLanes`: `r.oneway` なら中心線上の 1 車線 (a → b) だけ作る。接続レーン生成は `o.from === L.to` で動くので変更なし。
- `web/src/traffic.js` `priorityGroups`: 腕を `conns` の `from` 車線からだけ集めているので、出るだけの腕が候補から漏れる。**その交差点から出る車線 (`L.from === J.id`) も腕として集め**、Python の `_priority_groups` と同じ対を選ぶ。`phaseOf` は信号頭の向きで決めるので変更なし。
- `web/src/town.js` `section()`: `oneway` を持ち回す。`web/src/world/roaddeco.js`: `for (const dir of [1, -1])` を一方通行は `dir = 1` のみ、横位置 `side(F, r.laneW / 2)` を中心 (`0`) に。`roads.js` の塗装は書き出し済みポリゴンなので変更なし。
- `tools/pilot_check.mjs --yields` が一方通行の町で通ることを確認 (web と Python の yield が一致)。

### 札幌を変えない

`data/osm/osm_sapporo.json` には 東15丁目屯田通 (tertiary、上下線を別 way で描いた一方通行 2 本、箱内外で 6.2 km) が入っています。既定を一方通行ありにすると同梱の町と学習・評価の基準 (osm-retrain の走行データ、`runs/eval/osm_sapporo`) が変わるので、**札幌は `oneway=False` で固定**し、`test_bundled_sapporo_export_unchanged` もその引数で比較します。屯田通を取り込むのは、v0.9 系の評価が落ち着いてから別 PR で (上下線 2 本を `drop_duplicates` の「逆向き一方通行の統合」で 1 本の二車線にするか、分離帯つきの 2 本にするかを決めてから)。

## 作業の順番 (すべて 1 本の PR、ブランチは `claude/keisim-realworld-expansion-669ae7` から切り直す)

1. **グラフ**: `oneway_of`、辺の向き、整形パス 5 つの向き対応、向きつき刈り込み、強連結成分、`osm_town(oneway=)`、`--no-oneway`。`tests/test_realworld.py` の `mini_osm` にある一方通行リンク (grid(1,0) → (0,-140) → grid(2,0)) が **残る** ようになるので、期待する交差点位置・腕数を更新し (曲がり角 (0,-140) が bend 交差点に、(0,-80) の腕が 1 本増える)、新しい小さな fixture を 2 つ足す: 逆向き一方通行 2 本の統合、入るだけ/出るだけの節点の刈り込み。
2. **町**: 車線 1 本、腕の `-1`、接続レーン、信号・停止線・標示、`_priority_groups`。テスト: 一方通行の道路は `lane_ba == -1` で車線が中心線上、すべての道路車線に `succ` と `pred` がある、expert が一方通行を含む経路を完走する (`DS > 0.99`)。
3. **書き出しとブラウザ**: `oneway` フィールド、`nav.js`、`traffic.js`、`roaddeco.js`。`node tools/check.mjs --tdir .towns --town osm_nishioi`、`pilot_check.mjs --yields`。ブラウザペインは WebGL が使えないので headless Chrome (GPU) で盤面を撮る。
4. **西大井の町**: `uv run scripts/import_osm.py --bbox 35.616 139.723 35.625 139.734 --name osm_nishioi --routes 8` (箱 B とも比較し、信号 4 つのある A を本命に)。`graph_report` で一方通行の km、交差点数、落とした理由を記録し、`docs/report/` に短い報告を足す。KeiPilot v0.9 のゼロショット (`evaluate.py --osm_town`) は参考値として 1 回だけ取る (中心線のない 4.5 m の一方通行は学習に無い見え方なので、下がるのは想定内)。
5. **PR**: README の「現実の町」節に一方通行対応と西大井を追記、`data/osm/osm_nishioi.json` を同梱 (ODbL 表記は既存の README に箱を 1 行追加)。公開版 (Pages) への追加は札幌と同じ手順だが、学習前の町を公開するかは別途判断。

見積り: 1〜3 で Python と JS 合わせて 300 行前後の差分、4 は走行込みで半日分の計算。

## リスクと決めごと

- **PurePursuit と交差点半径**: 中心線上の車線から曲がると、二車線より弧がわずかに大きくなるので運転は楽になる側。逆に一方通行 → 二車線の左折は目標車線が手前に寄るぶん弧がきつい。`graph_min_radius` は変えず、expert の完走テストで見る。問題が出たら一方通行の腕だけ `lane_width / 2` を半径の式に足す。
- **信号のある一方通行**: 西大井 A の信号 4 か所のうち一方通行が絡む交差点では、二相信号の相は腕の角度で決まるので、出るだけの腕が「相を占める」ことはあるが矛盾はしない (その相の間は誰も入らない)。NPC の `box_occupants` は入る車線だけ見る。
- **NPC の生成**: `road_lanes` から等確率に置くので一方通行には半分しか置かれない。密度調整は不要 (現実も同じ)。
- **PLATEAU の幅員**: OSM の幅がないので断面は固定値。LOD3 の `uro:RoadStructureAttribute` (車道部の幅) を道路ごとに当てるのは次の段階 (`tools/site_survey/b3dm_meta.py` で読める)。
- **二車線 `lanes=1` 道路 (A 1.6 km、B 2.0 km)**: 今は 6 m の residential。alley (2.75 m × 2) に落とすと見た目は近づくが、alley は NPC のトラックが入らず速度 20 km/h になる。本 PR には入れず、西大井の町を一度走らせてから決める。

## 確認済みで変更が要らないもの

`keisim/traffic.py`、`keisim/expert.py`、`keisim/route.py`、`keisim/world.py`、`keisim/render/*` は `lane_ab` / `lane_ba` を参照せず、車線の `succ` / `pred` / `road` だけで動く。`web/src/world/roads.js` と `maptex.py` は道路ポリゴン (`half_width`) から描く。`ego_server` と KeiView の学習用レンダラは同じ JSON を読む。
