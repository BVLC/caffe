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

> 命令列工具來構建、 部署和管理[科爾多瓦](http://cordova.io)-基於的應用程式。

[Apache 科爾多瓦](http://cordova.io)允許生成本機使用 HTML、 CSS 和 JavaScript 的移動應用程式。 此工具説明管理的多平臺科爾多瓦應用以及科爾多瓦外掛程式的整合。

簽出[入門指南](http://cordova.apache.org/docs/en/edge/)有關如何處理科爾多瓦子專案的詳細資訊。

# 支援的科爾多瓦平臺

  * 亞馬遜火 OS
  * Android 系統
  * 黑莓 10
  * 火狐瀏覽器作業系統
  * iOS
  * Ubuntu
  * Windows Phone 8
  * Windows 8

# 要求

  * [Node.js](http://nodejs.org/)
  * 您希望支援每個平臺 Sdk: 
      * **安卓**: [Android SDK](http://developer.android.com) -**注意**此工具將不工作，除非你有絕對的最新更新 Android SDK 的所有元件。 此外你將需要 SDK 的`工具`，你的**系統路徑**否則為 Android 支援`平臺工具`目錄將失敗。
      * **亞馬遜 fireos**:[亞馬遜火 OS SDK](https://developer.amazon.com/public/solutions/platforms/android-fireos/docs/building-and-testing-your-hybrid-app) -**注意**此工具將不工作，除非你有安裝 Android SDK 和路徑更新如上文所述。 此外，你需要安裝 AmazonWebView SDK 並複製到 ~/.cordova/lib/commonlibs 資料夾中的**Mac/Linux**系統上或**Windows** %USERPROFILE%/.cordova/lib/coomonlibs awv_interface.jar。 如果 commonlibs 資料夾不存在則創建一個。
      * [Windows Phone SDK](http://dev.windowsphone.com/en-us/downloadsdk) -**注意**此工具將不工作，除非你對你的**系統路徑**的`msbuild`否則 Windows Phone 支援將失敗 (`msbuild.exe`一般都位於`C:\Windows\Microsoft.NET\Framework\v4.0.30319`).
      * **黑莓 10**:[黑莓 10 WebWorks SDK](http://developer.blackberry.com/html5/download/)。 請確保您有`依賴項，工具，bin`資料夾內的 SDK 目錄添加到您的路徑!
      * **iOS**: [iOS SDK](http://developer.apple.com)提供的最新`Xcode`和`Xcode 命令列工具`
      * **Windows Phone**: [Windows Phone SDK](http://dev.windowsphone.com/en-us/downloadsdk) -**注意**此工具將不工作，除非你對你的**系統路徑**的`msbuild`否則 Windows Phone 支援將失敗 (`msbuild.exe`一般都位於`C:\Windows\Microsoft.NET\Framework\v4.0.30319`).

`科爾多瓦 cli`已經過測試的**Mac OS X**、 **Linux**、 **Windows 7**， **Windows 8**.

請注意，某些平臺作業系統限制。 例如，為 Windows 8 或 Windows Phone 8 在 Mac OS X 上，你不能建立，也為 iOS 上 Windows，你還可以建立。

# 安裝

Ubuntu 的套裝軟體在 PPA 為 Ubuntu 13.10 (Saucy) (當前版本) 14.04 (可靠) (正在開發中)。

    sudo apt-add-repository ppa:cordova-ubuntu/ppa
    sudo apt-get update
    sudo apt-get install cordova-cli
    npm install -g cordova
    

若要生成 Ubuntu 平臺應用，以下額外套裝軟體是必需的:

    sudo apt-get install cmake debhelper libx11-dev libicu-dev pkg-config qtbase5-dev qtchooser qtdeclarative5-dev qtfeedback5-dev qtlocation5-dev qtmultimedia5-dev qtpim5-dev qtsensors5-dev qtsystems5-dev
    

## 從主安裝

你會需要從`git`安裝[CLI](https://git-wip-us.apache.org/repos/asf/cordova-cli.git)和[Plugman](https://git-wip-us.apache.org/repos/asf/cordova-plugman.git) 。 運行*npm 版本*的一個和*(git) 主版本*的其他有可能結束你的痛苦。

若要避免使用 sudo，請參閱[擺脫 sudo: 故宮無根](http://justjs.com/posts/npm-link-developing-your-own-npm-modules-without-tears).

運行以下命令:

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
    

現在的`科爾多瓦`和`plugman`在您的路徑是本地 git 版本。別忘了讓他們與時俱進!

## 在 Ubuntu 上安裝

    apt-get install cordova-cli
    

# 入門

`科爾多瓦 cli`有一個單一全球`創建`的命令，到指定的目錄中創建新的科爾多瓦專案。 一旦你創建了一個專案，`裁談會`到它，你可以執行各種專案級別的命令。 完全靈感 git 的介面。

## 全域命令

  * `help`顯示所有可用命令的説明頁面
  * `create <directory> [<id> [<name>]]`創建一個新的科爾多瓦專案可選名稱和 id (包名稱，反向域風格)

<a name="project_commands" />

## 專案命令

  * `platform [ls | list]`列出所有平臺，該專案將建設
  * `platform add <platform> [<platform> ...]`添加一個 (或多個) 的平臺，作為該專案的生成目標
  * `platform [rm | remove] <platform> [<platform> ...]`從專案中移除一個 (或多個) 平臺生成目標
  * `platform [up | update] <platform>` -更新用於給定平臺的科爾多瓦版本
  * `plugin [ls | list]`列出專案中包含的所有外掛程式
  * `plugin add <path-to-plugin> [<path-to-plugin> ...]`向專案中添加一個 (或多個) 的外掛程式
  * `plugin [rm | remove] <plugin-name> [<plugin-name> ...]`從專案中移除一個 (或多個) 的外掛程式。
  * `plugin search [<keyword1> <keyword2> ...]`匹配的關鍵字清單的外掛程式外掛程式註冊表中搜索
  * `prepare [platform...]`將檔案複製到指定的平臺或所有平臺。它是然後準備建設由`Eclipse`， `Xcode`，等等。
  * `compile [platform...]`將應用程式編譯為二進位檔案為每個目標平臺。不帶參數，生成的所有平臺上，否則生成指定的平臺。
  * `build [<platform> [<platform> [...]]]`別名`科爾多瓦準備`跟著`科爾多瓦編譯`
  * `emulate [<platform> [<platform> [...]]]`啟動模擬程式，並將應用程式部署到他們。 不帶參數類比添加到專案中的所有平臺，否則類比為指定的平臺
  * `serve [port]`啟動本地 web 伺服器允許您訪問給定的埠 (預設 8000) 上的每個平臺 www 目錄。

### 可選標誌

  * `-d`或`--verbose`將管出更詳細的輸出到您的 shell。 你也可以訂閱`log`和`warn`事件，如果你消費`cordova-cli`作為節點模組通過調用`cordova.on ('log'，{function()})`或`cordova.on ('warn'，{function()})`.
  * `-v`或`--version`將列印出的你的`科爾多瓦 cli`版本安裝。

# 專案目錄結構

構建與`科爾多瓦 cli`的科爾多瓦應用程式將具有以下目錄結構:

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

此目錄可能包含用於自訂科爾多瓦 cli 命令的腳本。 此目錄用於存在於`.cordova/hooks`，但現在已搬到專案根目錄。 之前和之後的目錄名稱所對應的命令，將執行任何腳本，您將添加到這些目錄。 用於您自己的生成系統集成或與版本控制系統集成。

有關詳細資訊，請參閱鉤指南</a>。

## merges/

特定于平臺的 web 資產 (HTML、 CSS 和 JavaScript 檔) 都包含在此目錄中的相應子資料夾內。 這些資產的部署期間`prepare`適當的本機目錄。 根據放置的檔`merges/`將覆蓋中的匹配檔`www/`資料夾相關的平臺。 一個簡單的例子，假設專案結構:

    merges/
    |-- ios/
    | `-- app.js
    |-- android/
    | `-- android.js
    www/
    `-- app.js
    

生成後的 Android 和 iOS 的專案，Android 的應用程式將包含`app.js`和`android.js`。 但是，它將一個來自`merges/ios/app.js`，重寫"常見的" `app.js`位於內，iOS 應用程式將只包含`app.js`， `www /`.

## www/

包含該專案的 Web 專案，例如.html，.css 和.js 檔。這些都是您的主應用程式資產。他們將在`科爾多瓦準備`到每個平臺 www 目錄複寫。

### 你的毯子: config.xml

此檔是您應該編輯修改您的應用程式的中繼資料。 任何時間你運行任何科爾多瓦 cli 命令，該工具將看的`config.xml`內容並使用從該檔的所有相關資訊來定義本機應用程式資訊。 科爾多瓦 cli 支援更改您的應用程式資料通過`config.xml`檔中的以下元素:

  * 面向使用者的名稱可以通過`< 名稱 >`元素的內容進行修改。
  * 可以通過從頂級`< widget >`元素的`id`屬性修改包名稱 (AKA 束識別碼或應用程式 id)。
  * 可以通過從頂級`<widget>`元素`version`屬性修改版本。
  * 可以使用`<access>`元素修改白名單。 請確保您`<access>`元素點的`origin`屬性到一個有效的 URL (你可以使用`*`作為萬用字元)。 白名單語法的詳細資訊，請參閱[docs.phonegap.com](http://docs.phonegap.com/en/2.2.0/guide_whitelist_index.md.html#Domain%20Whitelist%20Guide)。 您可以使用屬性`uri` ([黑莓專有](https://developer.blackberry.com/html5/documentation/access_element_834677_11.html)) 或`origin`([符合標準](http://www.w3.org/TR/widgets-access/#attributes)) 來表示域。
  * 通過`<preference>`標記，可以自訂平臺特定的首選項。 看到[docs.phonegap.com](http://docs.phonegap.com/en/2.3.0/guide_project-settings_index.md.html#Project%20Settings)的首選項，您可以使用清單。
  * 您的應用程式的入口/起始頁可以通過`<content src>`元素 + 屬性定義。

## platforms/

添加到您的應用程式的平臺會有專案結構此目錄內鋪設了本機應用程式。

## plugins/

將提取任何添加的外掛程式，或複製到此目錄。

# 鉤子

通過科爾多瓦 cli 創建的專案有`before`和`after`鉤為每個[專案命令](#project_commands).

有兩種類型的鉤子: 具體專案部分和模組層級的那些。這兩種類型的鉤子作為參數接收的專案根資料夾。

## 具體專案掛鉤

這些位於科爾多瓦專案的根目錄中的`hooks`目錄下。 您將添加到這些目錄的任何腳本將執行之前和之後的適當的命令。 用於您自己的生成系統集成或與版本控制系統集成。 **記住**: 使您的腳本可執行。 有關詳細資訊，請參閱[鉤指南](http://cordova.apache.org/docs/en/edge/guide_appdev_hooks_index.md.html#Hooks%20Guide)。

### 例子

  * 由[dpogue](http://github.com/dpogue) [ `before_build`鉤玉範本編制](https://gist.github.com/4100866)

## 模組層級別掛鉤

如果你使用科爾多瓦 cli 作為一個較大的**Node**應用程式中的一個模組，你也可以使用標準的`EventEmitter`方法將附加到事件。 這些事件包括`before_build`、 `before_compile`、 `before_docs`、 `before_emulate`、 `before_run`、 `before_platform_add`、 `before_library_download`、 `before_platform_ls`、 `before_platform_rm`、 `before_plugin_add`、 `before_plugin_ls`、 `before_plugin_rm`和`before_prepare`。 另外還有`library_download`進度事件。 此外，還有`after_`的所有上述事件的味道。

一旦你`require('cordova')`節點專案中，你將有通常`EventEmitter`可用的方法 (`on`、`off`或`removeListener`、 `removeAllListeners`，和`emit`或`trigger`).

# 例子

## 創建一個新的科爾多瓦專案

本示例演示如何從頭開始創建一個專案命名 KewlApp 與 iOS 和 Android 平臺的支援，並包括一個名為 Kewlio 的外掛程式。該專案將住在 ~/KewlApp

    cordova create ~/KewlApp KewlApp
    cd ~/KewlApp
    cordova platform add ios android
    cordova plugin add http://example.org/Kewlio-1.2.3.tar.gz
    cordova build
    

現在，KewlApp 的目錄結構如下所示:

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
    

# 貢獻

## 運行測試

    npm test
    

## 獲取測試覆蓋報告

    npm run cover
    

## 待辦事項 + 問題

請檢查[科爾多瓦與 CLI 元件的問題](http://issues.cordova.io)。 如果你找到這個工具的問題，請成這樣包括如調試問題所需的相關資訊:

  * 您的作業系統和版本
  * 應用程式名稱、 目錄位置和識別碼`創建`與使用
  * 您已安裝，哪些移動 Sdk 和它們的版本。與此相關的: 哪個`Xcode`的版本，如果您正在提交問題相關的 ios
  * 您收到任何錯誤堆疊追蹤

## 派遣國

感謝大家對作出貢獻!涉及人的清單，請參閱`package.json`檔。

# 已知的問題和疑難排解

## 任何作業系統

### 代理伺服器設置

`科爾多瓦 cli`將使用`npm`的代理設置。 如果您下載科爾多瓦 cli 通過`故宮`，在代理伺服器後面，機會是科爾多瓦 cli 應為你工作，因為它將使用這些設置在第一位。 請確保正確設置`HTTPs 代理伺服器`和`代理`的新公共管理組態變數。 請參閱[新公共管理的設定檔](https://npmjs.org/doc/config.html)的詳細資訊。

## Windows

### 加入 Android 作為平臺的麻煩

當試圖添加一個平臺 Windows 機器上，如果您遇到以下錯誤訊息:"安卓"的科爾多瓦庫已存在。 無需下載。 繼續。 正在檢查是否"android"平臺通過最低要求... 正在檢查 Android 要求... 運行"android 清單目標"(輸出跟隨)

    Error: The command `android` failed. Make sure you have the latest Android SDK installed, and the `android` command (inside the tools/ folder) added t
    o your path. Output:
    at C:\Users\me\AppData\Roaming\npm\node_modules\cordova\src\platform.js:185:42
    at C:\Users\me\AppData\Roaming\npm\node_modules\cordova\src\metadata\android_parser.js:50:13
    at C:\Users\me\AppData\Roaming\npm\node_modules\cordova\node_modules\shelljs\shell.js:1707:7
    at exithandler (child_process.js:633:7)
    at ChildProcess.errorhandler (child_process.js:649:5)
    at ChildProcess.EventEmitter.emit (events.js:95:17)
    at Process.ChildProcess._handle.onexit (child_process.js:787:12)
    

運行命令`android list target`。 如果您看到:

    'xcopy' is not recognized as an internal or external command,
    operable program or batch file.
    

在命令輸出的開頭，它意味著你將需要修復 Windows 路徑變數，包括 xcopy。此位置通常是在 C:\Windows\System32 下。

## Windows 8

Windows 8 支援不包括啟動/運行/類比的能力，所以您將需要打開**Visual Studio**看到你活著的應用程式。 您將仍然能夠與 windows8 使用以下命令:

  * `platform add windows8`
  * `platform remove windows8`
  * `prepare windows8`
  * `compile windows8`
  * `build windows8`

若要運行您的應用程式，您將需要在使用**Visual Studio 2012**的`平臺/windows8`資料夾中打開`.sln`.

**Visual Studio**會告訴你，如果你運行上面的命令的任何載入專案時重新載入專案。

## 亞馬遜火 OS

亞馬遜火 OS 不包括模仿的能力。您將仍然能夠使用以下命令與亞馬遜火 OS

  * `platform add amazon-fireos`
  * `platform remove amazon-fireos`
  * `prepare amazon-fireos`
  * `compile amazon-fireos`
  * `build amazon-fireos`

## Ubuntu

科爾多瓦 ubuntu 的初始版本不支援構建應用程式的 armhf 設備自動。它是可能生成的應用程式，點擊雖然包幾個步驟。

此 bug 報告檔的問題和解決方案為它: HTTPs://bugs.launchpad.net/ubuntu/+source/cordova-ubuntu/+bug/1260500 未來的版本會讓開發人員進行交叉編譯 armhf 按一下直接從 x 86 桌面套裝軟體。

## 火狐瀏覽器作業系統

火狐瀏覽器作業系統不包括模仿、 運行和服務的能力。 經過建設，你將不得不在[WebIDE](https://developer.mozilla.org/docs/Tools/WebIDE)附帶每個火狐瀏覽器中打開您的應用程式的`firefoxos`平臺目錄。 你可以讓這個視窗保持打開，點擊"播放"按鈕每次你完成構建您的應用程式。