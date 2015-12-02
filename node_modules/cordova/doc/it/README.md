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

> Lo strumento di riga di comando per creare, distribuire e gestire [Cordova](http://cordova.io)-applicazioni basate su.

[Apache Cordova](http://cordova.io) consente per la creazione di applicazioni mobile native utilizzando HTML, CSS e JavaScript. Questo strumento aiuta con la gestione di applicazioni di Cordova multi-piattaforma, nonché integrazione di plugin di Cordova.

Scopri le [guide introduttive](http://cordova.apache.org/docs/en/edge/) per maggiori dettagli su come lavorare con Cordova sotto-progetti.

# Piattaforme supportate Cordova

  * Amazon fuoco OS
  * Android
  * BlackBerry 10
  * Firefox OS
  * iOS
  * Ubuntu
  * Windows Phone 8
  * Windows 8

# Requisiti

  * [Node.js](http://nodejs.org/)
  * SDK per ogni piattaforma che si desidera supportare: 
      * **Android**: [Android SDK](http://developer.android.com) - **Nota** questo strumento non funzionerà a meno che non hai gli ultimi aggiornamenti assoluti per tutti i componenti del SDK di Android. Inoltre avrete bisogno di SDK `tools` e directory di `platform-tools` sul vostro sostegno altrimenti Android di **percorso di sistema** avrà esito negativo.
      * **Amazon-fireos**: [Amazon fuoco OS SDK](https://developer.amazon.com/public/solutions/platforms/android-fireos/docs/building-and-testing-your-hybrid-app) - **Nota** questo strumento non funzionerà a meno che non avete installato Android SDK e i percorsi sono aggiornati come accennato in precedenza. Inoltre è necessario installare AmazonWebView SDK e copiare awv_interface su sistema **Mac/Linux** ~/.cordova/lib/commonlibs cartella o su **Windows** %USERPROFILE%/.cordova/lib/coomonlibs. Se la cartella commonlibs non esiste quindi crearne uno.
      * [Windows Phone SDK](http://dev.windowsphone.com/en-us/downloadsdk) - **Nota** questo strumento non funzionerà a meno che non si dispone di `msbuild` sul tuo **percorso di sistema** altrimenti avrà esito negativo il supporto di Windows Phone (`msbuild.exe` generalmente si trova in `C:\Windows\Microsoft.NET\Framework\v4.0.30319`).
      * **BlackBerry 10**: [10 BlackBerry WebWorks SDK](http://developer.blackberry.com/html5/download/). Assicuratevi di che avere la cartella `dipendenze/strumenti/bin` all'interno della directory SDK aggiunta al tuo percorso!
      * **iOS**: [iOS SDK](http://developer.apple.com) con le ultime `Xcode` e `Strumenti della riga di comando di Xcode`
      * **Windows Phone**: [Windows Phone SDK](http://dev.windowsphone.com/en-us/downloadsdk) - **Nota** questo strumento non funzionerà a meno che non si dispone di `msbuild` sul tuo **percorso di sistema** altrimenti avrà esito negativo il supporto di Windows Phone (`msbuild.exe` generalmente si trova in `C:\Windows\Microsoft.NET\Framework\v4.0.30319`).

`Cordova-cli` è stato testato su **Mac OS X**, **Linux**, **Windows 7**e **Windows 8**.

Siete pregati di notare che alcune piattaforme hanno restrizioni di OS. Ad esempio, non si può costruire per Windows 8 o Windows Phone 8 su Mac OS X, né si può costruire per iOS su Windows.

# Installare

Pacchetti di Ubuntu sono disponibili in un PPA per Ubuntu 13.10 (Saucy) (versione corrente) come pure 14,04 (fidato) (in fase di sviluppo).

    sudo apt-add-repository ppa:cordova-ubuntu/ppa
    sudo apt-get update
    sudo apt-get install cordova-cli
    npm install -g cordova
    

Per compilare un'applicazione per la piattaforma di Ubuntu, sono necessari i seguenti pacchetti extra:

    sudo apt-get install cmake debhelper libx11-dev libicu-dev pkg-config qtbase5-dev qtchooser qtdeclarative5-dev qtfeedback5-dev qtlocation5-dev qtmultimedia5-dev qtpim5-dev qtsensors5-dev qtsystems5-dev
    

## L'installazione dal master

È necessario installare sia [CLI](https://git-wip-us.apache.org/repos/asf/cordova-cli.git) e [Plugman](https://git-wip-us.apache.org/repos/asf/cordova-plugman.git) da `git`. In esecuzione la *versione di npm* di uno e *versione master (git)* di altro è probabile che alla fine con sofferenza.

Per evitare l'uso di sudo, vedere [allontanarsi sudo: npm senza radice](http://justjs.com/posts/npm-link-developing-your-own-npm-modules-without-tears).

Eseguire i seguenti comandi:

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
    

Ora il `cordova` e `plugman` nel tuo percorso sono le versioni git locale. Non dimenticate di tenerli aggiornati!

## L'installazione su Ubuntu

    apt-get install cordova-cli
    

# Guida introduttiva

`Cordova-cli` ha un comando unico globale `create` che crea i nuovi progetti di Cordova in una directory specificata. Una volta che si crea un progetto, `cd` in esso e si può eseguire una varietà di comandi a livello di progetto. Completamente ispirato dall'interfaccia di git.

## Comandi globali

  * `help` visualizzare una pagina di aiuto con tutti i comandi disponibili
  * `create <directory> [<id> [<name>]]` creare un nuovo progetto di Cordova con opzionale nome e id (nome del pacchetto, stile retro-dominio)

<a name="project_commands" />

## Comandi del progetto

  * `platform [ls | list]` elencare tutte le piattaforme per le quali si baserà il progetto
  * `platform add <platform> [<platform> ...]` aggiungere uno (o più) piattaforme come una destinazione di generazione per il progetto
  * `platform [rm | remove] <platform> [<platform> ...]` Rimuove destinazioni di generazione di piattaforma di uno (o più) dal progetto
  * `platform [up | update] <platform>` -aggiorna la versione di Cordova utilizzata per la piattaforma specificata
  * `plugin [ls | list]` elencare tutti i plugin incluso nel progetto
  * `plugin add <path-to-plugin> [<path-to-plugin> ...]` aggiungere uno (o più) plugin per il progetto
  * `plugin [rm | remove] <plugin-name> [<plugin-name> ...]` rimuovere uno (o più) plugin dal progetto.
  * `plugin search [<keyword1> <keyword2> ...]` il registro dei plugin per plugin associando l'elenco di parole chiave di ricerca
  * `prepare [platform...]` copia file nelle piattaforme specificate, o tutte le piattaforme. Quindi è pronto per la costruzione di `Eclipse`, `Xcode`, ecc.
  * `compile [platform...]` compila l'app in un binario per ogni piattaforma mirata. Senza parametri, compilazioni per tutte le piattaforme, altrimenti compila per le piattaforme specificate.
  * `build [< platform > [< platform > [...]]]` un alias per `cordova prepare` seguita da `cordova compile`
  * `emulate [<platform> [<platform> [...]]]` avviare Emulatori e distribuire app a loro. Senza parametri emula per tutte le piattaforme aggiungere al progetto, in caso contrario emula per le piattaforme specificate
  * `serve [port]` avviare un server web locale consente di accedere a directory di ogni piattaforma www la porta fornita (8000 di default).

### Flag facoltativo

  * `-d` o `--verbose` sarà tubo fuori un output più dettagliato nel tuo guscio. Puoi anche iscriverti agli eventi `log` e `warn` se sei termini di `cordova-cli` come modulo nodo chiamando `cordova.on ('log', Function () {})` o `cordova.on ('warn', Function () {})`.
  * `-v` o `--version` stampa la versione del tuo `cordova-cli` installerà.

# Struttura di Directory del progetto

Un'applicazione di Cordova costruita con `cordova-cli` avrà la seguente struttura di directory:

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

Questa directory può contiene gli script utilizzati per personalizzare i comandi cli cordova. Questa directory utilizzato per esistere a `.cordova/hooks`, ma ora è stata spostata alla radice del progetto. Qualsiasi script che si aggiunge a queste directory verrà eseguito prima e dopo i comandi che corrispondono al nome della directory. Utile per integrare i propri sistemi di compilazione o di integrazione con sistemi di controllo di versione.

Fare riferimento alla Guida di ganci</a> per ulteriori informazioni.

## merges/

Risorse specifiche della piattaforma web (file HTML, CSS e JavaScript) sono contenuti all'interno di sottocartelle appropriate in questa directory. Queste vengono distribuite durante un `prepare` nella directory appropriata nativo. I file inseriti sotto `merges /` sovrascriverà i file corrispondenti nella `www /` cartella per la piattaforma pertinente. Un rapido esempio, assumendo una struttura di progetto di:

    merges/
    |-- ios/
    | `-- app.js
    |-- android/
    | `-- android.js
    www/
    `-- app.js
    

Dopo la compilazione dei progetti di Android e iOS, l'applicazione Android conterrà sia `JS` e `android.js`. Tuttavia, l'applicazione iOS conterrà solo un `JS`, e sarà quello da `merges/ios/app.js`, si esegue l'override il "comune" `JS` situato all'interno di `www /`.

## www/

Contiene gli artefatti web del progetto, ad esempio i file HTML, CSS e js. Queste sono le risorse dell'applicazione principale. Verranno copiati su un `cordova preparare` alla directory di www di ogni piattaforma.

### La coperta: config. XML

Questo file è quello che si dovrebbe modificare per modificare i metadati dell'applicazione. Ogni volta che si eseguono tutti i comandi di cordova-cli, lo strumento verrà esaminare il contenuto del `file config. XML` e utilizzare tutte le informazioni pertinenti da questo file per definire le informazioni di applicazione nativa. Cordova-cli supporta la modifica dei dati dell'applicazione tramite i seguenti elementi all'interno del file `config. XML` :

  * Il nome utente esposto può essere modificato tramite il contenuto dell'elemento `< name >` .
  * Il nome del pacchetto (id identificatore o applicazione pacchetto AKA) può essere modificato tramite l'attributo `id` dall'elemento di primo livello `< widget >` .
  * La versione può essere modificata tramite l'attributo di `version` dall'elemento di primo livello `< widget >` .
  * La whitelist può essere modificato utilizzando gli elementi `< access >` . Assicurarsi che l'attributo di `origin` tuo < > elemento di punti di `access` a un URL valido (è possibile utilizzare `*` come carattere jolly). Per ulteriori informazioni sulla sintassi di whitelisting, vedere [docs.phonegap.com](http://docs.phonegap.com/en/2.2.0/guide_whitelist_index.md.html#Domain%20Whitelist%20Guide). È possibile utilizzare attributo `uri` ([BlackBerry proprietari](https://developer.blackberry.com/html5/documentation/access_element_834677_11.html)) o `origine` ([standard](http://www.w3.org/TR/widgets-access/#attributes)) per indicare il dominio.
  * Preferenze specifiche della piattaforma possono essere personalizzate tramite tag `< preference >` . Vedere [docs.phonegap.com](http://docs.phonegap.com/en/2.3.0/guide_project-settings_index.md.html#Project%20Settings) per un elenco di preferenze che è possibile utilizzare.
  * La pagina di ingresso/iniziale per l'applicazione può essere definita tramite l'elemento `< content src >` + attributo.

## platforms/

Piattaforme aggiunti all'applicazione avranno l'applicazione nativa di strutture all'interno di questa directory di progetto.

## plugins/

Qualsiasi plugin aggiunto sarà estratta o copiati in questa directory.

# Hooks

I progetti creati da cordova-cli hanno `before` e `after` ganci per ogni [comando di progetto](#project_commands).

Ci sono due tipi di ganci: specifiche del progetto e quelli a livello di modulo. Entrambi questi tipi di ganci ricevere la cartella radice del progetto come parametro.

## Ganci specifici del progetto

Questi si trovano sotto i `ganci` della directory principale del vostro progetto di Cordova. Qualsiasi script che si aggiunge a queste directory verrà eseguito prima e dopo i comandi appropriati. Utile per integrare i propri sistemi di compilazione o di integrazione con sistemi di controllo di versione. **Remember**: rendere il vostro script eseguibile. Fare riferimento alla [Guida di ganci](http://cordova.apache.org/docs/en/edge/guide_appdev_hooks_index.md.html#Hooks%20Guide) per ulteriori informazioni.

### Esempi

  * [ `before_build` gancio per la compilazione del modello giada](https://gist.github.com/4100866) per gentile concessione di [dpogue](http://github.com/dpogue)

## Livello di modulo ganci

Se si utilizza cordova-cli come un modulo all'interno di un'applicazione di **node** più grande, è anche possibile utilizzare i metodi standard di `EventEmitter` per associare agli eventi. Gli eventi includono `before_build`, `before_compile`, `before_docs`, `before_emulate`, `before_run`, `before_platform_add`, `before_library_download`, `before_platform_ls`, `before_platform_rm`, `before_plugin_add`, `before_plugin_ls`, `before_plugin_rm` e `before_prepare`. C'è anche un evento progress `library_download` . Inoltre, ci sono sapori di `after_` di tutti gli eventi di cui sopra.

Una volta si `require('cordova')` nel progetto nodo, avrete i metodi `EventEmitter` usuali disponibili (`on`, `off` o `removeListener`, `removeAllListeners`ed `emit` o `trigger`).

# Esempi

## Creazione di un nuovo progetto di Cordova

In questo esempio viene illustrato come creare un progetto da zero denominato KewlApp con supporto piattaforma Android e iOS e include un plugin chiamato Kewlio. Il progetto vivrà in ~/KewlApp

    cordova create ~/KewlApp KewlApp
    cd ~/KewlApp
    cordova platform add ios android
    cordova plugin add http://example.org/Kewlio-1.2.3.tar.gz
    cordova build
    

La struttura di directory di KewlApp ora assomiglia a questo:

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
    

# Contribuendo

## Esecuzione di test

    npm test
    

## Ottenere rapporti di copertura di prova

    npm run cover
    

## To-do + problemi

Si prega di controllare [problemi di Cordova con il componente di CLI](http://issues.cordova.io). Se si riscontrano problemi con questo strumento, si prega di essere così gentile da includere le informazioni pertinenti necessarie per eseguire il debug problemi come:

  * Il sistema operativo e versione
  * Il nome dell'applicazione, percorso della directory e identificatore utilizzato con `create`
  * Quale mobile SDK è installato e le relative versioni. Relativi a questo: quale versione di `Xcode` , se intendi presentare problemi relativi a iOS
  * Eventuali tracce di stack di errore che hai ricevuto

## Contributori

Grazie a tutti per aver contribuito! Per un elenco delle persone coinvolte, vedere il file `JSON` .

# Problemi noti e risoluzione dei problemi

## Qualsiasi sistema operativo

### Impostazioni del proxy

`Cordova-cli` utilizzerà le impostazioni del proxy di `npm`. Se hai scaricato cordova-cli tramite `npm` e sono dietro un proxy, le probabilità sono di cordova-cli dovrebbe funzionare per voi come utilizzerà tali impostazioni in primo luogo. Assicurarsi che le variabili di configurazione di npm `https proxy` e `proxy` siano impostate correttamente. Vedere la [documentazione di configurazione di npm](https://npmjs.org/doc/config.html) per ulteriori informazioni.

## Windows

### Difficoltà ad aggiungere Android come piattaforma

Quando si tenta di aggiungere una piattaforma su una macchina Windows, se si verifica il seguente messaggio di errore: libreria di Cordova per "android" esiste già. Non c'è bisogno di scaricare. Continuando. Verifica se la piattaforma "android" passa requisiti minimi... Verifica requisiti di Android... Eseguire "android elenco destinazione" (uscita a seguire)

    Error: The command `android` failed. Make sure you have the latest Android SDK installed, and the `android` command (inside the tools/ folder) added t
    o your path. Output:
    at C:\Users\me\AppData\Roaming\npm\node_modules\cordova\src\platform.js:185:42
    at C:\Users\me\AppData\Roaming\npm\node_modules\cordova\src\metadata\android_parser.js:50:13
    at C:\Users\me\AppData\Roaming\npm\node_modules\cordova\node_modules\shelljs\shell.js:1707:7
    at exithandler (child_process.js:633:7)
    at ChildProcess.errorhandler (child_process.js:649:5)
    at ChildProcess.EventEmitter.emit (events.js:95:17)
    at Process.ChildProcess._handle.onexit (child_process.js:787:12)
    

eseguire il comando `android list target`. Se vedete:

    'xcopy' is not recognized as an internal or external command,
    operable program or batch file.
    

all'inizio dell'output del comando, significa che sarà necessario correggere la variabile Windows Path per includere xcopy. Questa posizione è in genere sotto C:\Windows\System32.

## Windows 8

Supporto a Windows 8 non include la capacità di avvio/Esegui/emulare, quindi sarà necessario aprire **Visual Studio** per vedere l'app dal vivo. Siete ancora in grado di utilizzare i seguenti comandi con windows8:

  * `platform add windows8`
  * `platform remove windows8`
  * `prepare windows8`
  * `compile windows8`
  * `build windows8`

Per eseguire l'app, sarà necessario aprire il `. sln` nella cartella `piattaforme/windows8` utilizzando **Visual Studio 2012**.

**Visual Studio** vi dirà di ricaricare il progetto, se si esegue uno dei comandi di cui sopra, mentre il progetto viene caricato.

## Amazon fuoco OS

Amazon fuoco OS non include la capacità di emulare. Siete ancora in grado di utilizzare i seguenti comandi con Amazon fuoco OS

  * `platform add amazon-fireos`
  * `platform remove amazon-fireos`
  * `prepare amazon-fireos`
  * `compile amazon-fireos`
  * `build amazon-fireos`

## Ubuntu

La versione iniziale di cordova-ubuntu non supporta creazione di applicazioni per dispositivi armhf automaticamente. È possibile produrre applicazioni e fare clic su pacchetti in pochi passi però.

Questo bug report documenta il problema e le soluzioni per esso: https://bugs.launchpad.net/ubuntu/+source/cordova-ubuntu/+bug/1260500 una versione futura permetterà agli sviluppatori cross-compilare armhf scegliere pacchetti direttamente da un desktop di 86 x.

## Firefox OS

Firefox OS non include la capacità di emulare, eseguire e servire. Dopo la costruzione, si dovrà aprire la directory di piattaforma `firefoxos` della tua app in [WebIDE](https://developer.mozilla.org/docs/Tools/WebIDE) che viene fornito con ogni browser Firefox. Si può tenere aperta questa finestra e fare clic sul pulsante "play" ogni volta che finito di costruire la tua app.