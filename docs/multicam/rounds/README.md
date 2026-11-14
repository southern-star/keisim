# KeiPilot Mc の学習の各回のスクリプト

KeiPilot Mc (v0.11.0) を作ったときに、実際に走らせたスクリプトです。上から順に走らせました ([../../multicam.md](../../multicam.md) の 3 章)。
作業中は `runs/mc_work/` に置き、systemd のユーザー unit で動かしていました。最初の行の `cd` だけ、リポジトリの直下に移るように書き換えています。

| スクリプト | 中身 | できるモデル |
|---|---|---|
| `round1.sh` | エキスパートのデータ (格子の町と札幌の学習区画) → mc1 → mc1 の DAgger → mc2 | mc1, mc2 |
| `sapporo_more.sh` | 札幌の評価を 300 ルートに (v0.10 と mc2) | — |
| `round2.sh` | mc2 の DAgger (歩行者・追従と信号・右折・札幌、KeiView の組) → mc3 | mc3 |
| `round3.sh` | mc3 の DAgger、固まったらエキスパートに渡す → mc4 | mc4 |
| `round4.sh`, `round4b.sh` | エキスパートのラベル 6 (止まれのあとの徐行) で mc4 の DAgger → mc5、外し方を変えた mc5b | mc5, mc5b |
| `round5.sh`, `round5_fix.sh` | `--ego_patience 3` で mc5b の DAgger → mc6 | mc6 |
| `round6.sh` | `--stand_pert` で mc6 の DAgger → mc7 | mc7 |
| `round7.sh` | mc7 の DAgger と 2 現示の町の高密度の DAgger → mc8 | mc8 |
| `yieldhold.sh`, `yieldhold3.sh` | 安全層 YieldHold の評価 (2 つ目の版と、いまの版) | — |
| `suburban0.sh` | 片側 2 車線の町での mc8 と v0.10 の出発点の評価 | — |
| `round8.sh` | 片側 2 車線の町の DAgger → mc9 | mc9 |
| `round9.sh` | ランプの手前の食い違うフレームを捨てた DAgger → mc10 | mc10 |
| `round10.sh` | エキスパートのラベル 7 で mc10 の DAgger → mc11 | mc11 |
| `round11.sh` | ガードレールの急カーブの事例 → mc12 (= v0.11.0) | mc12 |
| `round12.sh` | 横断歩道を渡る人・大きな交差点の街 (bigjunction)・自転車の事例を 1 回で → mc13。クラウドで書いた案で、まだ走らせていません (収集のコマンドは 100〜200 フレームずつ試してあります)。`MMAP=1` で、古い組を半分にしない形 (`train.py --mmap`) | mc13 |
| `v010_r11.sh` | 比べるための v0.10 の評価 (11 回目と同じコード・同じルート) | — |

* データは `data/mc_*` に集まります。1 回目から 11 回目までで約 120 組です。
* メモリー (32 GB) に収めるため、学習の前に古い組を半分にしています (`_half` の付いた組)。どの組を半分にしたかは各スクリプトにあります。
  `train.py --mmap` (画像をディスクのシャードから読む) があれば、`round12.sh` は `MMAP=1` で古い組を半分にせずに学習します。
* コメントの `HANDOVER.md` は作業中のメモで、公開していません。
