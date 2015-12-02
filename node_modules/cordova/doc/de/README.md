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

> Das Kommandozeilen-Tool zum Erstellen, bereitstellen und Verwalten von [Cordova](http://cordova.io)-basierten Anwendungen.

[Apache-Cordova](http://cordova.io) ermöglicht native mobile Applikationen mit HTML, CSS und JavaScript erstellen. Dieses Tool hilft bei der Verwaltung von Multiplattform-Cordova-Anwendungen sowie Cordova Plugin Integration.

Schauen Sie sich die [Erste Schritte-Handbücher](http://cordova.apache.org/docs/en/edge/) für weitere Informationen zum Arbeiten mit Teilprojekte Cordova.

# Cordova unterstützte Plattformen

  * Amazon Fire OS
  * Android
  * BlackBerry 10
  * Firefox OS
  * iOS
  * Ubuntu
  * Windows Phone 8
  * Windows 8

# Anforderungen

  * [Node.js](http://nodejs.org/)
  * SDKs für jede Plattform, die Sie unterstützen möchten: 
      * **Android**: [Android SDK](http://developer.android.com) - **Hinweis** dieses Tool funktioniert nicht, es sei denn, Sie die absolut neuesten Updates für alle Android SDK-Komponenten haben. Außerdem benötigen Sie die SDK- `Werkzeuge` und `Plattform-Tools` -Verzeichnisse auf Ihrem **Systempfad** sonst Android Unterstützung schlägt fehl.
      * **Amazon-Fireos**: [Amazon Fire OS SDK](https://developer.amazon.com/public/solutions/platforms/android-fireos/docs/building-and-testing-your-hybrid-app) - **Hinweis** dieses Tool funktioniert nicht, es sei denn, Sie Android SDK installiert haben und wie bereits erwähnt, werden Pfade aktualisiert. Darüber hinaus müssen Sie AmazonWebView SDK installieren und kopieren awv_interface.jar auf **Mac/Linux** -System zu ~/.cordova/lib/commonlibs Ordner oder **Windows** %USERPROFILE%/.cordova/lib/coomonlibs. Dann erstellen Sie eine, wenn Commonlibs Ordner nicht vorhanden ist.
      * [Windows Phone SDK](http://dev.windowsphone.com/en-us/downloadsdk) - **Hinweis** dieses Tool nicht funktioniert, wenn Sie `Msbuild` auf Ihrem **Systempfad** haben sonst Windows Phone Unterstützung fehl (`msbuild.exe` im allgemeinen befindet sich im `C:\Windows\Microsoft.NET\Framework\v4.0.30319`).
      * **BlackBerry 10**: [10 BlackBerry WebWorks SDK](http://developer.blackberry.com/html5/download/). Stellen Sie sicher, dass Sie den `Abhängigkeiten/Werkzeuge/bin` Ordner im SDK-Verzeichnis auf Ihrem Pfad hinzugefügt haben!
      * **iOS**: [iOS SDK](http://developer.apple.com) mit den neuesten `Xcode` und `Xcode-Befehlszeilen-Tools`
      * **Windows Phone**: [Windows Phone SDK](http://dev.windowsphone.com/en-us/downloadsdk) - **Hinweis** dieses Tool nicht funktioniert, wenn Sie `Msbuild` auf Ihrem **Systempfad** haben sonst Windows Phone Unterstützung fehl (`msbuild.exe` im allgemeinen befindet sich im `C:\Windows\Microsoft.NET\Framework\v4.0.30319`).

`Cordova-Cli` wurde auf **Mac OS X**, **Linux**, **Windows 7**und **Windows 8** getestet.

Bitte beachten Sie, dass einige Plattformen OS Beschränkungen. Zum Beispiel für Windows 8 oder Windows Phone 8 unter Mac OS X kann man nicht bauen, noch können Sie für iOS unter Windows erstellen.

# Installieren

Ubuntu-Pakete stehen in einer PPA für Ubuntu 13,10 (Saucy) (aktuelle Version) sowie 14.04. (Trusty) (in Entwicklung).

    sudo apt-add-repository ppa:cordova-ubuntu/ppa
    sudo apt-get update
    sudo apt-get install cordova-cli
    npm install -g cordova
    

Um eine Anwendung für die Ubuntu-Plattform zu erstellen, werden die folgenden zusätzlichen Pakete benötigt:

    sudo apt-get install cmake debhelper libx11-dev libicu-dev pkg-config qtbase5-dev qtchooser qtdeclarative5-dev qtfeedback5-dev qtlocation5-dev qtmultimedia5-dev qtpim5-dev qtsensors5-dev qtsystems5-dev
    

## Installation vom master

Du musst [CLI](https://git-wip-us.apache.org/repos/asf/cordova-cli.git) und [Plugman](https://git-wip-us.apache.org/repos/asf/cordova-plugman.git) von `Git`installieren. Die *npm version* von One und *(git) master-Version* des anderen wird voraussichtlich mit ihnen leiden zu beenden.

Um zu vermeiden, mit Sudo, siehe [wegkommen von Sudo: Npm ohne Root](http://justjs.com/posts/npm-link-developing-your-own-npm-modules-without-tears).

Führen Sie die folgenden Befehle:

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
    

Jetzt sind die `Cordova` und `Plugman` in Ihrem Pfad der lokalen Git-Versionen. Vergessen Sie nicht, sie auf dem laufenden halten!

## Installation auf Ubuntu

    apt-get install cordova-cli
    

# Erste Schritte

`Cordova-Cli` hat einen einzigen globalen `Erstellen` Befehl, der neue Cordova-Projekte in einem angegebenen Verzeichnis erstellt. Sobald Sie ein Projekt erstellen, können `cd` in es und Sie eine Vielzahl von Projektebene Befehle ausführen. Git Schnittstelle inspiriert vollständig.

## Globale Befehle

  * `help` anzeigen eine Hilfeseite mit allen verfügbaren Befehle
  * `create <directory> [<id> [<name>]]` erstellen ein neues Cordova-Projekt mit optionaler Name und Id (Paketname, Rückwärtsgang-Domäne-Style)

<a name="project_commands" />

## Projekt-Befehle

  * `platform [ls | list]` Listen Sie alle Plattformen für das Erstellen des Projekts
  * `platform add <platform> [<platform> ...]` Fügen Sie eine (oder mehrere) Plattformen als Buildziel für das Projekt
  * `platform [rm | remove] <platform> [<platform> ...]` die Ziele von einem (oder mehreren) Plattform erstellen entfernt aus dem Projekt
  * `platform [up | update] <platform>` -aktualisiert die Cordova-Version für die Plattform verwendet
  * `plugin [ls | list]` Listen Sie alle Plugins, die Teil des Projekts
  * `plugin add <path-to-plugin> [<path-to-plugin> ...]` Fügen Sie eine (oder mehrere) Plugins zum Projekt
  * `plugin [rm | remove] <plugin-name> [<plugin-name> ...]` Entfernen Sie eine (oder mehrere) Plugins aus dem Projekt.
  * `plugin search [<keyword1> <keyword2> ...]` Suchen Sie die Plugin-Registrierung für Plugins passend die Liste von Schlüsselwörtern
  * `prepare [platform...]` kopiert die Dateien in den angegebenen Plattformen oder alle Plattformen. Es ist dann für den Aufbau von `Eclipse`, `Xcode`, etc. bereit.
  * `compile [platform...]` kompiliert die app in eine Binärdatei für jede gezielte Plattform. Ohne Parameter, Builds für alle Plattformen, ansonsten baut für die angegebenen Plattformen.
  * `build [<platform> [<platform> [...]]]` ein Alias für die `cordova prepare` gefolgt von `cordova compile`
  * `emulate [<platform> [<platform> [...]]]` Emulatoren starten und app für sie bereitstellen. Ohne Parameter emuliert für alle Plattformen, die dem Projekt hinzugefügt, ansonsten für die angegebenen Plattformen emuliert
  * `serve [port]` Starten Sie einen lokalen Webserver, so dass Sie Zugriff auf jede Plattform Www-Verzeichnis auf dem angegebenen Port (Standard 8000).

### Optionalen Flags

  * `-d` oder `--verbose` wird heraus ausführlichere Ausgabe Ihrer Shell Pfeife. Sie können auch Ereignisse `log` und `warn` abonnieren, wenn Sie raubend `Cordova-Cli` als Modul Knoten sind, durch Aufrufen von `cordova.on ("log", function (){}` oder `cordova.on ('warn', function () {}`.
  * `-v` oder `--version` wird Ausdrucken Ihrer `Cordova-Cli` -Version installieren.

# Verzeichnisstruktur des Projekts

Eine Anwendung für Cordova mit `Cordova-Cli` haben die folgende Verzeichnisstruktur:

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

Dieses Verzeichnis kann enthält Skripts zum Anpassen von Cordova-Cli-Befehle verwendet. Dieses Verzeichnis verwendet, um zu `.cordova/hooks`vorhanden sein, aber jetzt nach dem Projektstamm wurde verschoben. Alle Skripts, die Sie diese Verzeichnisse hinzufügen werden vor und nach den Befehlen, die den Namen des Verzeichnisses entspricht ausgeführt. Nützlich für die Integration Ihrer eigenen Build-Systeme oder Integration mit Versionskontrollsystemen.

Weitere Informationen finden Sie unter Haken Guide</a> .

## merges/

Plattformspezifische Web Vermögenswerte (HTML, CSS und JavaScript-Dateien) sind in den entsprechenden Unterordner in diesem Verzeichnis enthalten. Diese sind während einer `prepare` in das entsprechende Verzeichnis der native eingesetzt. Dateien unter `merges/` überschreibt die entsprechende Dateien im der `www /` Ordner für die entsprechende Plattform. Ein kurzes Beispiel, eine Projektstruktur des vorausgesetzt:

    merges/
    |-- ios/
    | `-- app.js
    |-- android/
    | `-- android.js
    www/
    `-- app.js
    

Nach dem Aufbau der Android und iOS-Projekte, wird die Android-Anwendung `app.js` und `android.js`enthalten. Jedoch die iOS-Anwendung enthält nur eine `app.js`, und es wird derjenige sein, der aus `merges/ios/app.js`, überschreiben die "gewöhnlichen" `app.js` innerhalb der `www /`.

## www/

Das Projekt Web Artefakte wie HTML, CSS und JS-Dateien enthält. Dies sind Ihre Hauptanwendung Vermögenswerte. Sie werden auf eine `cordova prepare` jede Plattform Www-Verzeichnis kopiert werden.

### Ihre Decke: "config.xml"

Diese Datei ist was Sie bearbeiten sollten um Ihre Anwendung Metadaten ändern. Jedesmal wenn Sie Cordova-Cli-Befehle ausführen das Tool betrachten Sie den Inhalt der `Datei config.xml` und alle relevanten Informationen aus dieser Datei verwenden, um native Anwendungsinformationen zu definieren. Cordova-Cli unterstützt Ihre Anwendungsdaten über die folgenden Elemente in der Datei `"config.xml"` ändern:

  * Über den Inhalt des Elements `<name>` kann der Benutzer-Namen geändert werden.
  * Der Paketname (AKA Bündel Bezeichner oder Anwendung Id) kann über das `Id` -Attribut aus dem Element der obersten Ebene `<widget>` verändert werden.
  * Die Version kann über das `version` -Attribut aus dem Element der obersten Ebene `<widget>` geändert werden.
  * Die Whitelist kann über die `<access>` Elemente geändert werden. Stellen Sie sicher das `origin` -Attribut des Ihre Element `<access>` punkte auf eine gültige URL ( `*` als Platzhalter können). Weitere Informationen über die Whitelist-Syntax finden Sie unter [docs.phonegap.com](http://docs.phonegap.com/en/2.2.0/guide_whitelist_index.md.html#Domain%20Whitelist%20Guide). Können Sie Attribut `Uri` ([BlackBerry-proprietäre](https://developer.blackberry.com/html5/documentation/access_element_834677_11.html)) oder der `origin` ([standardkonforme](http://www.w3.org/TR/widgets-access/#attributes)) um die Domäne zu bezeichnen.
  * Plattform-spezifische Einstellungen können über `<preference>` Tags angepasst werden. Siehe [docs.phonegap.com](http://docs.phonegap.com/en/2.3.0/guide_project-settings_index.md.html#Project%20Settings) für eine Liste der Einstellungen, die Sie verwenden können.
  * Die Eintrag/Startseite für Ihre Anwendung kann über die `<content src>` Element + Attribut definiert werden.

## platforms/

Plattformen zur Anwendung hinzugefügt werden die Ausgangsanwendung Projekt Strukturen innerhalb des Verzeichnisses angelegt haben.

## plugins/

Keine zusätzlichen Plugins werden extrahiert oder in dieses Verzeichnis kopiert.

# Hooks

Erstellt von Cordova-Cli Projekte haben `before` und `after` Haken für jeden [Befehl Projekt](#project_commands).

Es gibt zwei Arten von Haken: projektspezifische und auf Modulebene. Beide Typen der Haken erhalten den Stammordner des Projekts als Parameter.

## Projektspezifische Haken

Diese befinden sich unter dem `hooks` -Verzeichnis im Stammverzeichnis Ihres Projekts Cordova. Alle Skripts, die Sie diese Verzeichnisse hinzufügen werden vor und nach der entsprechenden Befehle ausgeführt werden. Nützlich für die Integration Ihrer eigenen Build-Systeme oder Integration mit Versionskontrollsystemen. **Denken Sie daran**: Ihre Skripte ausführbar zu machen. Weitere Informationen finden Sie unter [Haken Guide](http://cordova.apache.org/docs/en/edge/guide_appdev_hooks_index.md.html#Hooks%20Guide) .

### Beispiele

  * [ `Before_build` Haken zum Kompilieren von jade Vorlage](https://gist.github.com/4100866) mit freundlicher Genehmigung von [dpogue](http://github.com/dpogue)

## Auf Modulebene Haken

Wenn Sie Cordova-Cli als Modul innerhalb einer größeren **Node** -Anwendung verwenden, können Sie die Ereignisse an auch `EventEmitter` Standardmethoden. Die Veranstaltungen umfassen `Before_build`, `Before_compile`, `Before_docs`, `Before_emulate`, `Before_run`, `Before_platform_add`, `Before_library_download`, `Before_platform_ls`, `Before_platform_rm`, `Before_plugin_add`, `Before_plugin_ls`, `Before_plugin_rm` und `Before_prepare`. Es gibt auch eine `library_download` -Progress-Ereignis. Darüber hinaus gibt es `after_` Aromen aller oben genannten Ereignisse.

Einmal Sie `require('cordova')` in Ihrem Projekt Knoten, verfügen Sie über die üblichen `EventEmitter` verfügbaren Methoden (`on`, `off` oder `removeListener`, `RemoveAllListeners`, und `emit` oder `trigger`).

# Beispiele

## Erstellen eines neuen Projekts Cordova

In diesem Beispiel wird veranschaulicht, wie ein Projekt namens "KewlApp" mit iOS und Android-Plattform-Unterstützung erstellen und beinhaltet eine Plugin namens Kewlio. Das Projekt wird in ~/KewlApp Leben.

    cordova create ~/KewlApp KewlApp
    cd ~/KewlApp
    cordova platform add ios android
    cordova plugin add http://example.org/Kewlio-1.2.3.tar.gz
    cordova build
    

Die Verzeichnisstruktur des KewlApp sieht nun folgendermaßen aus:

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
    

# Einen Beitrag

## Ausführen von Tests

    npm test
    

## Testberichte-Abdeckung zu erhalten

    npm run cover
    

## To-do + Fragen

Bitte überprüfen Sie [Cordova Probleme mit der CLI-Komponente](http://issues.cordova.io). Wenn Sie Probleme mit diesem Tool finden, bitte seien Sie so nett, einschlägige Informationen zu Themen wie Debuggen benötigt:

  * Ihr Betriebssystem und version
  * Der Anwendungsname, Verzeichnis und Bezeichner verwendet mit `Erstellen`
  * Welche mobile SDKs, die Sie installiert haben und ihre Versionen. In diesem Zusammenhang: welche `Xcode` -Version, wenn Sie Fragen senden mit Bezug zu iOS
  * Jeder Fehler-Stack-Traces, die Sie erhalten

## Mitwirkende

Vielen Dank an alle für Ihren Beitrag! Eine Liste der beteiligten Personen finden Sie in der Datei `package.json` .

# Bekannte Probleme und Fehlerbehebung

## Alle OS

### Proxy-Einstellungen

`Cordova-Cli` wird `Npm`-Proxy-Einstellungen verwenden. Wenn Sie hinter einem Proxy sind und Cordova-Cli über `Npm` heruntergeladen, sind Wahrscheinlichkeiten Cordova-Cli für Sie arbeiten sollten, wie sie diese Einstellungen in erster Linie verwenden wird. Stellen Sie sicher, dass die `Https-Proxy` und `Proxy` Npm-Config-Variable korrekt gesetzt sind. Finden Sie weitere Informationen [des Npm Konfigurationsdokumentation](https://npmjs.org/doc/config.html) .

## Windows

### Probleme beim Hinzufügen von Android als Plattform

Wenn Sie versuchen, eine Plattform auf einem Windows-Computer hinzufügen, wenn Sie die folgende Fehlermeldung bekommen: Cordova Bibliothek für "Android" bereits vorhanden ist. Kein download erforderlich. Fort. Überprüfen, ob die Plattform "android" Mindestanforderungen übergibt... Kontrolle auf Android... Ausführen von "android Liste Ziel" (Ausgabe folgen)

    Error: The command `android` failed. Make sure you have the latest Android SDK installed, and the `android` command (inside the tools/ folder) added t
    o your path. Output:
    at C:\Users\me\AppData\Roaming\npm\node_modules\cordova\src\platform.js:185:42
    at C:\Users\me\AppData\Roaming\npm\node_modules\cordova\src\metadata\android_parser.js:50:13
    at C:\Users\me\AppData\Roaming\npm\node_modules\cordova\node_modules\shelljs\shell.js:1707:7
    at exithandler (child_process.js:633:7)
    at ChildProcess.errorhandler (child_process.js:649:5)
    at ChildProcess.EventEmitter.emit (events.js:95:17)
    at Process.ChildProcess._handle.onexit (child_process.js:787:12)
    

Führen Sie den Befehl `android list target`. Wenn Sie sehen:

    'xcopy' is not recognized as an internal or external command,
    operable program or batch file.
    

zu Beginn der Ausgabe des Befehls bedeutet dies, dass Sie benötigen, um Ihre Windows Path-Variable um Xcopy erweitert zu beheben. Dieser Ort ist in der Regel unter C:\Windows\System32.

## Windows 8

Windows-8-Unterstützung beinhaltet nicht die Fähigkeit zu Start/ausführen/emulieren, so Sie öffnen Sie **Visual Studio** um Ihre app Leben zu sehen müssen. Sie sind noch in der Lage, die folgenden Befehle mit windows8 zu verwenden:

  * `platform add windows8`
  * `platform remove windows8`
  * `prepare windows8`
  * `compile windows8`
  * `build windows8`

Um Ihre Anwendung auszuführen, müssen Sie die `sln` im `Plattformen/windows8` Ordner mithilfe von **Visual Studio 2012** öffnen.

**Visual Studio** erfahren Sie das Projekt neu zu laden, wenn Sie einen der obigen Befehle ausführen, während das Projekt geladen wird.

## Amazon Fire OS

Amazon Fire OS beinhaltet nicht die Fähigkeit zu emulieren. Sie sind noch in der Lage, die folgenden Befehle mit Amazon Fire OS zu verwenden

  * `platform add amazon-fireos`
  * `platform remove amazon-fireos`
  * `prepare amazon-fireos`
  * `compile amazon-fireos`
  * `build amazon-fireos`

## Ubuntu

Die erste Version von Cordova-Ubuntu unterstützt Gebäude keine Anwendungen für Armhf Geräte automatisch. Es ist möglich, Anwendungen und Pakete in wenigen Schritten aber auf.

Dieser Bug-Report dokumentiert das Problem und Lösungen dafür: https://bugs.launchpad.net/ubuntu/+source/cordova-ubuntu/+bug/1260500 einer zukünftigen Version wird Entwickler Kreuz-kompilieren Armhf Pakete direkt von einem X 86-Desktop klicken lassen.

## Firefox OS

Firefox-OS beinhaltet nicht die Fähigkeit zu emulieren, zu laufen und zu dienen. Nach Bau musst du das `Firefoxos` -Plattform-Verzeichnis Ihrer Anwendung in der [WebIDE](https://developer.mozilla.org/docs/Tools/WebIDE) zu öffnen, die mit jedem Firefox-Browser kommt. Sie können dieses Fenster offen halten und klicken Sie auf die Taste "play", jedes Mal, wenn Sie fertig bauen Ihre app.