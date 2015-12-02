<!--
#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
#  KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#
-->

# cordova-cli

> ビルド、配置、および[コルドバ](http://cordova.io)を管理するコマンド ライン ツール-ベースのアプリケーション。

[Apache のコルドバ](http://cordova.io)は、HTML、CSS、JavaScript を使用してネイティブ モバイル アプリケーションを構築することができます。 このツールは、コルドバのプラグインの統合と同様、マルチプラット フォーム向け cordova アプリの管理に役立ちます。

コルドバのサブ プロジェクトを操作する方法の詳細については、[入門ガイド](http://cordova.apache.org/docs/en/edge/)をチェックしてください。

# サポートされているコルドバのプラットフォーム

  * アマゾン火 OS
  * アンドロイド
  * ブラックベリー 10
  * Firefox の OS
  * iOS
  * Ubuntu
  * Windows Phone 8
  * Windows 8

# 要件

  * [Node.js](http://nodejs.org/)
  * 各プラットフォーム用の Sdk をサポートします。 
      * **Android**:[アンドロイド SDK](http://developer.android.com) -**注**Android SDK のすべてのコンポーネントの絶対の最新の更新プログラムを持っていない限り、このツールは動作しません。 SDK の`ツール`を必要があります、あなたの**システムのパス**それ以外の場合 Android 対応の`プラットフォーム固有のツール`ディレクトリが失敗します。
      * **アマゾン fireos**:[アマゾン火 OS SDK](https://developer.amazon.com/public/solutions/platforms/android-fireos/docs/building-and-testing-your-hybrid-app) -**注意**しない限り、Android SDK をインストールしておくと、上記のようにパスが更新され、このツールは動作しません。 さらに AmazonWebView SDK をインストールし、~/.cordova/lib/commonlibs フォルダーに**Mac の/Linux**システムまたは**Windows** %USERPROFILE%/.cordova/lib/coomonlibs に awv_interface.jar をコピーする必要があります。 Commonlibs フォルダーが存在しない場合は、1 つを作成します。
      * [Windows Phone SDK](http://dev.windowsphone.com/en-us/downloadsdk) -**注****システム パス**に`msbuild`がなければ、このツールは動作しない Windows Phone サポートが失敗するそれ以外の場合 (`msbuild.exe`一般にある`C:\Windows\Microsoft.NET\Framework\v4.0.30319`).
      * **ブラックベリー 10**: [10 ブラックベリー WebWorks SDK](http://developer.blackberry.com/html5/download/)。 パスに追加の SDK ディレクトリ内`dependencies/tools/bin`フォルダーを確認してください!
      * **iOS**: [iOS SDK](http://developer.apple.com)の最新の`Xcode`と`Xcode コマンド ライン ツール`
      * **Windows Phone**: [Windows Phone SDK](http://dev.windowsphone.com/en-us/downloadsdk) -**注****システム パス**に`msbuild`がなければ、このツールは動作しない Windows Phone サポートが失敗するそれ以外の場合 (`msbuild.exe`一般にある`C:\Windows\Microsoft.NET\Framework\v4.0.30319`).

`コルドバ cli`は、 **Mac OS X**、 **Linux**、 **Windows 7**、および**Windows 8**でテストされています.

いくつかのプラットフォームに OS の制限があることに注意してください。 たとえば、Windows 8 または Mac OS x、Windows Phone 8 ビルドすることはできませんも Windows で構築することができます。

# インストール

Ubuntu 13.10 (生意気) (現在のリリース) として (開発中) 14.04 (信頼できる) PPA の Ubuntu のパッケージがあります。

    sudo apt-add-repository ppa:cordova-ubuntu/ppa
    sudo apt-get update
    sudo apt-get install cordova-cli
    npm install -g cordova
    

Ubuntu のプラットフォームのアプリケーションをビルドするには、次の追加パッケージが必要です。

    sudo apt-get install cmake debhelper libx11-dev libicu-dev pkg-config qtbase5-dev qtchooser qtdeclarative5-dev qtfeedback5-dev qtlocation5-dev qtmultimedia5-dev qtpim5-dev qtsensors5-dev qtsystems5-dev
    

## マスターからのインストール

`Git`から[CLI](https://git-wip-us.apache.org/repos/asf/cordova-cli.git)と[Plugman](https://git-wip-us.apache.org/repos/asf/cordova-plugman.git)の両方をインストールする必要があります。 1 つの*故宮博物院のバージョン*と他の*(git) マスター バージョン*を実行する苦しみで終わりそうです。

Sudo を使用して避けるために、"を参照してください[須藤から得る: 故宮博物院ルートなし](http://justjs.com/posts/npm-link-developing-your-own-npm-modules-without-tears).

次のコマンドを実行します。

    git clone https://git-wip-us.apache.org/repos/asf/cordova-plugman.git
    cd cordova-plugman
    npm install
    sudo npm link
    cd ..
    git clone https://git-wip-us.apache.org/repos/asf/cordova-cli.git
    cd cordova-cli
    npm install
    sudo npm link
    npm link plugman
    

今`コルドバ`と`plugman`あなたのパスには、ローカルの git バージョンです。それらを最新に保つことを忘れないでください!

## Ubuntu にインストールします。

    apt-get install cordova-cli
    

# はじめに

`コルドバ cli`には、指定したディレクトリにコルドバの新しいプロジェクトを作成します`作成`1 つグローバルなコマンドがあります。 プロジェクトを作成したら、 `cd`にはさまざまなプロジェクト レベルのコマンドを実行できます。 完全に git のインターフェイスに触発されました。

## グローバル コマンド

  * すべての利用可能なコマンドのヘルプ ページを表示`help`
  * 省略可能な名前と id (パッケージ名、逆ドメイン スタイル) 新しいコルドバ プロジェクトを作成`create <directory> [<id> [<name>]]`します。

<a name="project_commands" />

## プロジェクト コマンド

  * `platform [ls | list]`プロジェクトはビルド対象のすべてのプラットフォームの一覧を表示
  * `platform add <platform> [<platform> ...]`プロジェクトのビルド ターゲットとして 1 つ (または複数) のプラットフォームを追加します。
  * `platform [rm | remove] <platform> [<platform> ...]`1 つ (または複数) のプラットフォーム ビルド ターゲットをプロジェクトから削除します。
  * `platform [up | update] <platform>` -特定のプラットフォームの使用コルドバのバージョンを更新
  * `plugin [ls | list]`プロジェクトに含まれるすべてのプラグインを一覧表示
  * `plugin add <path-to-plugin> [<path-to-plugin> ...]`プロジェクトに 1 つ (または複数) のプラグインを追加します。
  * `plugin [rm | remove] <plugin-name> [<plugin-name> ...]`プロジェクトからの 1 つ (または複数) のプラグインを削除します。
  * `plugin search [<keyword1> <keyword2> ...]`キーワードのリストに一致するプラグインのプラグイン レジストリを検索します。
  * 特定のプラットフォームまたはすべてのプラットフォームにファイルをコピー `prepare [platform...]`、 `Xcode`などによる建物の準備ができて、それになります。
  * アプリを対象プラットフォームごとにバイナリにコンパイルする`compile [platform...]`します。パラメーターなしで、すべてのプラットフォーム用のビルドとそれ以外の場合指定したプラットフォームのビルドします。
  * `build [<platform> [<platform> [...]]]` `コルドバの準備`の別名が`コルドバをコンパイル`続いて
  * `emulate [<platform> [<platform> [...]]]`エミュレーターを起動し、アプリケーションを展開します。 パラメーターなしでエミュレートをプロジェクトに追加するすべてのプラットフォームで、それ以外の場合、指定したプラットフォームをエミュレート
  * 指定されたポート (デフォルトは 8000) の各プラットフォームの www ディレクトリにアクセスすることを許可するローカル web サーバーを起動`serve [port]`します。

### オプションのフラグ

  * `-d`または`--verbose`、シェルにより冗長な出力をパイプします。 `Cordova.on ('log'、関数 {})`または`cordova.on ('warn'、関数 {})`を呼び出してノード モジュールとしてのかかる`コルドバ cli`のならは、また`log`および`warn`イベントを購読することができます。.
  * `-v`または`--verbose`、`コルドバ cli`のバージョンを印刷がインストールされます。

# プロジェクトのディレクトリ構造

`コルドバ cli`で構築されたコルドバ アプリケーションには、次のディレクトリ構造があります。

    myApp/
    |-- config.xml
    |-- hooks/
    |-- merges/
    | | |-- android/
    | | |-- blackberry10/
    | | `-- ios/
    |-- www/
    |-- platforms/
    | |-- android/
    | |-- blackberry10/
    | `-- ios/
    `-- plugins/
    

## hooks/

このディレクトリには、コルドバ cli コマンドをカスタマイズに使用するスクリプトが含まれます。 このディレクトリが`.cordova/フック`で存在して今プロジェクトのルートに移動されています。 これらのディレクトリに追加する任意のスクリプトは、する前に、ディレクトリ名に対応するコマンドの後に実行されます。 ビルド システムを統合またはバージョン管理システムと統合するために役立ちます。

詳細についてはフック ・ ガイド 』</a>を参照してください。

## merges/

プラットフォーム固有の web 資産 (HTML、CSS、JavaScript ファイル) は、このディレクトリ内の適切なサブフォルダーに含まれます。 これらは、`準備`するためのネイティブの適切なディレクトリに配置されます。 ファイルの下に配置`マージ/`内の対応するファイルよりも優先されます、 `www/`関連のプラットフォームのフォルダー。 簡単な例のプロジェクト構造を想定して:

    merges/
    |-- ios/
    | `-- app.js
    |-- android/
    | `-- android.js
    www/
    `-- app.js
    

Android と iOS のプロジェクトをビルドした後 Android アプリケーション`app.js`と`android.js`の両方が含まれます。 ただし、iOS アプリケーションにはのみが含まれます、 `app.js`と`merges/ios/app.js`、内にある「共通」 `app.js`をオーバーライドするから 1 時である`www/`.

## www/

.Html、.css と .js ファイルなど、プロジェクトの web アイテムが含まれています。これらは、メインのアプリケーション資産です。彼らは、`コルドバを準備`する各プラットフォームの www ディレクトリにコピーされます。

### あなたの毛布: config.xml

このファイルは、する必要があります編集、アプリケーションのメタデータを変更します。 コルドバ cli のすべてのコマンドを実行すればいつでもツールに`config.xml`の内容を見て、このファイルからすべての関連情報を使用して、ネイティブのアプリケーション情報を定義します。 コルドバ cli では、 `config.xml`ファイル内の次の要素を使用して、アプリケーションのデータを変更することをサポートしています。

  * ユーザー名は、 `< name >`要素のコンテンツを介して変更できます。
  * パッケージ名 (AKA のバンドル識別子またはアプリケーション id) は、最上位の`< widget >`要素の`id`属性によって変更できます。
  * バージョンは、最上位の`< version >`要素の`バージョン`属性によって変更できます。
  * ホワイト リストは、 `< access >`要素を使用して変更できます。 ようにしてください、 `< acess >`要素のポイントの`origin`の属性 ( `*`として使えますワイルドカード) の有効な URL。 ホワイト リストの構文の詳細については、 [docs.phonegap.com](http://docs.phonegap.com/en/2.2.0/guide_whitelist_index.md.html#Domain%20Whitelist%20Guide)を参照してください。 属性の`uri` ([ブラックベリー独自](https://developer.blackberry.com/html5/documentation/access_element_834677_11.html)) または`起源`のいずれかを使用できます ([標準に準拠した](http://www.w3.org/TR/widgets-access/#attributes)) ドメインを示すために。
  * `< preference >`タグを介してプラットフォームに固有の設定をカスタマイズできます。 使用できる設定の一覧については[docs.phonegap.com](http://docs.phonegap.com/en/2.3.0/guide_project-settings_index.md.html#Project%20Settings)を参照してください。
  * `< コンテンツ src >`要素 + 属性を介して、アプリケーションのエントリ/スタート ページを定義できます。

## platforms/

アプリケーションに追加のプラットフォームでお越しの際にもネイティブ アプリケーション プロジェクトの構造は、このディレクトリ内に置かれています。

## plugins/

任意の追加のプラグインを抽出またはこのディレクトリにコピーされます。

# Hooks

コルドバ cli によって作成されたプロジェクトが`before`と`after`それぞれの[プロジェクト コマンド](#project_commands)のフック.

フックの 2 種類があります: プロジェクトに固有のものとモジュール レベルのもの。どちらのフックの種類は、プロジェクトのルート フォルダーをパラメーターとして受け取ります。

## プロジェクト固有のフック

これらは、コルドバのプロジェクトのルートの`hooks`ディレクトリ下にあります。 これらのディレクトリに追加する任意のスクリプトは、適切なコマンドの前後に実行されます。 ビルド システムを統合またはバージョン管理システムと統合するために役立ちます。 **注意**: スクリプトを実行可能にします。 詳細については[フック ・ ガイド 』](http://cordova.apache.org/docs/en/edge/guide_appdev_hooks_index.md.html#Hooks%20Guide)を参照してください。

### 例

  * [dpogue](http://github.com/dpogue)礼儀の[`before_build`フック玉テンプレートをコンパイルするため](https://gist.github.com/4100866)の

## モジュール レベルのフック

大きい**Node**アプリケーション内のモジュールとしてコルドバ cli を使用する場合標準の`EventEmitter`メソッドを使用して、イベントにアタッチできます。 イベントには、 `before_build`、 `before_compile`、 `before_docs`、 `before_emulate`、 `before_run`、 `before_platform_add`、 `before_library_download`、 `before_platform_ls`、 `before_platform_rm`、 `before_plugin_add`、 `before_plugin_ls`、 `before_plugin_rm` 、 `before_prepare`が含まれます。 `Library_download`進行状況イベントもあります。 また、上記のすべてのイベントの`after_`味があります。

一度、 `require('cordova')`ノード プロジェクトで、あなたが通常`EventEmitter`利用できるメソッド`on`、`off` `removeListener`、 `removeAllListeners`と`emit`または`trigger`).

# 例

## コルドバの新しいプロジェクトを作成します。

この例では、iOS および Android プラットフォームのサポート付け KewlApp ゼロからプロジェクトを作成する方法を示していて、Kewlio という名前のプラグインが含まれています。プロジェクトが ~/KewlApp に住んでいます。

    cordova create ~/KewlApp KewlApp
    cd ~/KewlApp
    cordova platform add ios android
    cordova plugin add http://example.org/Kewlio-1.2.3.tar.gz
    cordova build
    

KewlApp のディレクトリ構造は、このようになります。

    KewlApp/
    |-- hooks/
    |-- merges/
    | |-- android/
    | `-- ios/
    |-- www/
    | `-- index.html
    |-- platforms/
    | |-- android/
    | | `-- …
    | `-- ios/
    |   `-- …
    `-- plugins/
      `-- Kewlio/
    

# 貢献

## テストを実行します。

    npm test
    

## テスト カバレッジ レポートを取得します。

    npm run cover
    

## To DO + 問題

[CLI コンポーネントとコルドバの問題](http://issues.cordova.io)を確認してください。 とても親切にしてください場合は、このツールを使って問題を見つけるなどの問題をデバッグするために必要な関連情報を含めます。

  * ご使用のオペレーティング システムおよびバージョン
  * アプリケーション名、ディレクトリの場所、および`作成`で使用される識別子
  * どのモバイルの Sdk をインストールしているとそのバージョンこれに関連して: `Xcode`バージョンの問題を提出する場合は iOS に関連
  * 受信した任意のエラーのスタック トレース

## 貢献者

貢献のためのみんなに感謝!関係者のリストは、 `package.json`ファイルを参照してください。

# 既知の問題とトラブルシューティング

## 任意の OS

### プロキシの設定

`コルドバ cli` `npm`のプロキシ設定が使用されます。 `故宮博物院`経由でコルドバ cli をダウンロード、プロキシの背後にある可能性がありますコルドバ cli は、それは最初の場所でそれらの設定を使用して、あなたのため動作するはずです。 `Https プロキシ`と`プロキシ`の故宮博物院構成変数が正しく設定されていることを確認します。 詳細については[故宮博物院の構成に関するドキュメント](https://npmjs.org/doc/config.html)を参照してください。

## Windows

### プラットフォームとして Android を追加できません。

次のエラー メッセージに実行する場合、Windows マシンにプラットフォームを追加しようとしたとき:「アンドロイド」のコルドバ ライブラリは既に存在します。 ダウンロードする必要はありません。 続けています。 プラットフォーム「android」が最小要件を渡す場合をチェック. Android の要件を確認しています. 「Android リスト ターゲット」(に従って出力) を実行しています。

    Error: The command `android` failed. Make sure you have the latest Android SDK installed, and the `android` command (inside the tools/ folder) added t
    o your path. Output:
    at C:\Users\me\AppData\Roaming\npm\node_modules\cordova\src\platform.js:185:42
    at C:\Users\me\AppData\Roaming\npm\node_modules\cordova\src\metadata\android_parser.js:50:13
    at C:\Users\me\AppData\Roaming\npm\node_modules\cordova\node_modules\shelljs\shell.js:1707:7
    at exithandler (child_process.js:633:7)
    at ChildProcess.errorhandler (child_process.js:649:5)
    at ChildProcess.EventEmitter.emit (events.js:95:17)
    at Process.ChildProcess._handle.onexit (child_process.js:787:12)
    

`android list target`のコマンドを実行します。 表示された場合。

    'xcopy' is not recognized as an internal or external command,
    operable program or batch file.
    

コマンドの出力の先頭、それは xcopy を含めるように Windows の Path 変数を修正する必要がありますを意味します。この場所は、C:\Windows\System32 の下で通常です。

## Windows 8

住んでいるアプリを表示する**Visual Studio**をオープンする必要がありますので、Windows 8 サポート起動/実行/エミュレート、する機能は含みません。 Windows8 で次のコマンドを使用するまだことができます。

  * `platform add windows8`
  * `platform remove windows8`
  * `prepare windows8`
  * `compile windows8`
  * `build windows8`

アプリを実行するのには、 **Visual Studio 2012**を使用して`platforms/windows8`フォルダー内`.sln`を開く必要があります。.

**Visual Studio**では、プロジェクトが読み込まれる間、上記のコマンドのいずれかを実行する場合は、プロジェクトを再読み込みするように指示されます。

## アマゾン火 OS

アマゾンの火の OS では、エミュレートする機能は含まれません。まだアマゾン火 OS で次のコマンドを使用することができます。

  * `platform add amazon-fireos`
  * `platform remove amazon-fireos`
  * `prepare amazon-fireos`
  * `compile amazon-fireos`
  * `build amazon-fireos`

## Ubuntu

コルドバ ubuntu の最初のリリースは、armhf デバイスの自動的にアプリケーションの構築をできません。アプリケーションを作成し、いくつかの手順でパッケージをクリックしても不可能です。

このバグ レポートそれのための問題およびソリューションのドキュメント: https://bugs.launchpad.net/ubuntu/+source/cordova-ubuntu/+bug/1260500 将来のリリース開発者クロス コンパイル armhf x 86 デスクトップから直接パッケージをクリックしてできるようになります。

## Firefox の OS

Firefox OS をエミュレートし、実行し、提供する機能は含みません。 建物後、すべての Firefox ブラウザーに付属している[WebIDE](https://developer.mozilla.org/docs/Tools/WebIDE)でアプリの`firefoxos`プラットフォーム ディレクトリを開く必要があります。 このウィンドウを開いたままでき、アプリのビルドが終了するたびに、「再生」ボタンをクリックします。