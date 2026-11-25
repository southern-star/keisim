# KeiReality for Android

背面カメラを入力に、KeiRealityが生成した384×192の追従視点だけを横画面に大きく表示します。映像は縦横比を保ち、全体が収まる大きさまで拡大します。モデルはAPKに同梱され、撮影から表示まで端末内のCPUで動きます。アプリが必要とする権限はカメラだけです。

## Pixelで試す

初版のテスト用APKとSHA256SUMSを同じフォルダーにダウンロードします。Linuxでは `sha256sum --ignore-missing -c SHA256SUMS` で `KeiReality-Pixel11.apk: OK` を確認できます。[テスト用APK](https://github.com/southern-star/keisim/releases/download/keireality-384-v1.0.0/KeiReality-Pixel11.apk) ・ [SHA256SUMS](https://github.com/southern-star/keisim/releases/download/keireality-384-v1.0.0/SHA256SUMS)

APKをPixelに転送して開き、インストールします。端末を横向きにして「カメラを開始」を押し、カメラの使用を許可してください。最初の生成映像が出ると操作欄が隠れます。映像をタップすると「停止」ボタン、表示コマ数、推論時間、平均fpsが現れ、もう一度タップすると隠れます。「停止」を押すか、ホームに戻るとカメラが閉じます。

## 入力と出力

カメラを水平に構える前提で、背面レンズ1本の映像を学習時の広角・望遠の2つの画角に変換します。端末の画角外には観測マスクを適用します。手持ち撮影用の初版は1コマずつ処理し、固定した追従視点に描画します。シミュレーターの道路で学習したモデルのため、室内などでは映っていない道路や車に似た形も生成します。

## ビルド

KeiSimの学習環境で、選定済みの重みと町6・7の検証用画像がある状態から実行します。エクスポートにはPythonの `onnx` と `onnxruntime` も必要です。使用したバージョンは順に1.23.2と1.32.0です。Android側はSDK 36とJDK 21でビルドし、ONNX Runtime Android 1.23.2を同梱します。

```sh
.venv/bin/python scripts/export_reality_android.py
cd android
./gradlew :app:assembleDebug
```

生成先は `android/app/build/outputs/apk/debug/app-debug.apk` です。ビルド済みAPKにはARM64とx86-64の両方のライブラリが入っています。機内モードのAndroidエミュレーター（x86-64、API 36）での実表示は2.33 fpsでした。[モデルの数値照合](../docs/keireality/384-v1/android-export.json) ・ [エミュレーターの表示と停止・再開](../docs/keireality/384-v1/android-emulator.json)
