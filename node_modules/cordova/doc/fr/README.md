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

> L'outil de ligne de commande pour construire, déployer et gérer [Cordova](http://cordova.io)-applications basées sur.

[Cordova Apache](http://cordova.io) permet de construire des applications mobiles natives à l'aide de HTML, CSS et JavaScript. Cet outil aide à la gestion des demandes de Cordova multi-plateformes ainsi que l'intégration du plugin Cordova.

Découvrez les [guides de mise en route](http://cordova.apache.org/docs/en/edge/) pour plus de détails sur l'utilisation des sous-projets de Cordova.

# Plates-formes prises en charge Cordova

  * Amazon Fire OS
  * Android
  * BlackBerry 10
  * Firefox OS
  * iOS
  * Ubuntu
  * Windows Phone 8
  * Windows 8

# Exigences

  * [Node.js](http://nodejs.org/)
  * Kits de développement logiciel pour chaque plate-forme, vous souhaitez soutenir : 
      * **Android**: [Android SDK](http://developer.android.com) - **NOTE** cet outil ne fonctionnera pas à moins d'avoir les mises à jour plus récentes absolues pour tous les composants SDK Android. Aussi, vous aurez besoin du SDK `Outils` et `plateforme-outils` des répertoires sur votre soutien sinon Android de **chemin d'accès système** échouera.
      * **Amazon-fireos**: [Amazon Fire OS SDK](https://developer.amazon.com/public/solutions/platforms/android-fireos/docs/building-and-testing-your-hybrid-app) - **NOTE** cet outil ne fonctionnera pas sauf si vous avez Android SDK installé et chemins sont mises à jour comme indiqué ci-dessus. En outre, vous devrez installer SDK AmazonWebView et copier awv_interface.jar sur ~/.cordova/lib/commonlibs dossier système **Mac/Linux** ou **Windows** %USERPROFILE%/.cordova/lib/coomonlibs. Si commonlibs dossier n'existe pas alors en créer un.
      * [SDK Windows Phone](http://dev.windowsphone.com/en-us/downloadsdk) - **NOTE** cet outil ne fonctionnera pas à moins d'avoir `msbuild` sur votre **chemin d'accès système** sinon support Windows Phone échouera (`msbuild.exe` est généralement situé dans `C:\Windows\Microsoft.NET\Framework\v4.0.30319`).
      * **BlackBerry 10**: [SDK WebWorks BlackBerry 10](http://developer.blackberry.com/html5/download/). Assurez-vous que vous avez le dossier `dependencies/toolss/bin` dans le répertoire SDK ajouté à votre chemin !
      * **iOS**: [iOS SDK](http://developer.apple.com) avec la dernière `Xcode` et les `Outils de ligne de commande de Xcode`
      * **Windows Phone**: [Windows Phone SDK](http://dev.windowsphone.com/en-us/downloadsdk) - **NOTE** cet outil ne fonctionnera pas à moins d'avoir `msbuild` sur votre **chemin d'accès système** sinon support Windows Phone échouera (`msbuild.exe` est généralement situé dans `C:\Windows\Microsoft.NET\Framework\v4.0.30319`).

`Cordova-cli` a été testé sur **Mac OS X**, **Linux**, **Windows 7**et **Windows 8**.

Veuillez noter que certaines plateformes ont des restrictions de l'OS. Par exemple, vous ne pouvez pas construire pour Windows 8 ou Windows Phone 8 sur Mac OS X, ni pouvez-vous construire pour iOS sur Windows.

# Installer

Paquets Ubuntu sont disponibles dans un PPA pour Ubuntu 13.10 (Saucy) (la version actuelle), mais aussi 14.04 (fidèle) (en cours d'élaboration).

    sudo apt-add-repository ppa:cordova-ubuntu/ppa
    sudo apt-get update
    sudo apt-get install cordova-cli
    npm install -g cordova
    

Pour générer une application pour la plateforme Ubuntu, les paquets supplémentaires suivants sont requis :

    sudo apt-get install cmake debhelper libx11-dev libicu-dev pkg-config qtbase5-dev qtchooser qtdeclarative5-dev qtfeedback5-dev qtlocation5-dev qtmultimedia5-dev qtpim5-dev qtsensors5-dev qtsystems5-dev
    

## Installation du maître

Vous aurez besoin d'installer de la [CLI](https://git-wip-us.apache.org/repos/asf/cordova-cli.git) et [Plugman](https://git-wip-us.apache.org/repos/asf/cordova-plugman.git) de `git`. Exécutant la *version NGP* d'un et *version principale (git)* de l'autre est susceptible de se terminer par vous souffrant.

Pour éviter d'utiliser sudo, voir [sortir de sudo : NGP sans racine](http://justjs.com/posts/npm-link-developing-your-own-npm-modules-without-tears).

Exécutez les commandes suivantes :

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
    

Maintenant le `cordova` et `plugman` dans votre chemin d'accès sont les versions git local. N'oubliez pas de les garder à jour !

## Installation sur Ubuntu

    apt-get install cordova-cli
    

# Mise en route

`Cordova-cli` a une commande globale unique `créer` qui crée de nouveaux projets de Cordova dans un répertoire spécifié. Une fois que vous créez un projet, `cd` dedans et vous pouvez exécuter une variété de commandes au niveau du projet. Totalement inspiré par interface de git.

## Commandes globales

  * `help` afficher une page d'aide avec toutes les commandes disponibles
  * `create <directory> [<id> [<name>]]` créer un nouveau projet de Cordoue avec option name et id (nom du package, style inverse-domaine)

<a name="project_commands" />

## Commandes de projet

  * `platform [ls | list]` la liste de toutes les plateformes pour lesquelles le projet s'appuiera
  * `platform add <platform> [<platform> ...]` Ajouter une (ou plusieurs) plateformes comme une cible de génération du projet
  * `platform [rm | remove] <platform> [<platform> ...]` supprime une (ou plusieurs) cibles de génération de plate-forme de projet
  * `platform [up | update] <platform>` -met à jour la version de Cordova, utilisée pour la plateforme donnée
  * `plugin [ls | list]` la liste de tous les plugins inclus dans le projet
  * `plugin add <path-to-plugin> [<path-to-plugin> ...]` Ajoutez au projet un (ou plusieurs) des plugins
  * `plugin [rm | remove] <plugin-name> [<plugin-name> ...]` supprimer un (ou plusieurs) des plugins du projet.
  * `plugin search [<keyword1> <keyword2> ...]` Rechercher dans le registre plugin pour les plugins correspondant à la liste de mots clés
  * `prepare [platform...]` copie les fichiers dans les plates-formes spécifiés, ou toutes les plateformes. Il est alors prêt pour la construction par `Eclipse`, `Xcode`, etc.
  * `compile [platform...]` compile l'application dans un fichier binaire pour chaque plateforme ciblée. Sans paramètres, construit pour toutes les plates-formes, sinon génère pour les plates-formes spécifiés.
  * `build [<platform> [<platform> [...]]]` suivie d'un alias pour `préparer de cordova` `cordova compiler`
  * `emulate [<platform> [<platform> [...]]]` lancer des émulateurs et déployer des app sur eux. Sans paramètre émule pour toutes les plates-formes, ajoutés au projet, sinon émule les plates-formes spécifié
  * `serve [port]` , lancer un serveur web local vous permettant d'accéder au répertoire www de la plate-forme sur le port donné (par défaut 8000).

### Indicateurs facultatifs

  * `-d` ou `--verbose` se diriger sur une sortie plus bavarde à votre shell. Vous pouvez également vous abonner à des événements de `log` et de `warn` en garde si vous êtes de votre `cordova-cli` comme un module de nœud en appelant `cordova.on ('log', function() {})` ou `cordova.on ('warn', function() {})`.
  * `-v` ou `--version` installera imprimer la version de votre `cordova-cli` .

# Structure de répertoires du projet

Une application de Cordova construite avec `cordova-cli` aura la structure de répertoire suivante :

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

Ce répertoire peut contenir des scripts pour personnaliser les commandes cordova-cli. Ce répertoire existait à `.cordova/hooks`, mais a maintenant été déplacé vers la racine du projet. Tous les scripts que vous ajoutez à ces répertoires seront exécutés avant et après les commandes correspondant au nom de répertoire. Utile pour intégrer vos propres systèmes de construction ou d'intégration avec les systèmes de contrôle de version.

Reportez-vous au Guide de crochets</a> pour plus d'informations.

## merges/

Actifs spécifiques à la plateforme web (fichiers HTML, CSS et JavaScript) sont contenus dans les sous-dossiers appropriés dans ce répertoire. Ils sont déployés durant une `prepare` dans le répertoire approprié du natif. Fichiers placés sous `merges/` remplace les fichiers correspondants dans le `www /` dossier pour la plateforme pertinente. Un petit exemple, en supposant qu'une structure de projet de :

    merges/
    |-- ios/
    | `-- app.js
    |-- android/
    | `-- android.js
    www/
    `-- app.js
    

Après avoir construit les projets iOS et Android, l'application Android contiendra fois `app.js` et `android.js`. Toutefois, l'application iOS ne contiendra qu'un `app.js`, et il sera celle de la `merges/ios/app.js`, la substitution de "commune" `app.js` situé à l'intérieur `www /`.

## www/

Contient des artefacts de web du projet, tels que des fichiers .html, .css et .js. Ce sont vos ressources de l'application principale. Ils seront copiés sur un `cordova préparer` au répertoire www de chaque plate-forme.

### Votre couverture : config.xml

Ce fichier est ce que vous devriez être édition pour modifier les métadonnées de votre application. Toute fois que vous exécutez des commandes de cordova-cli, l'outil va regarder le contenu du `fichier config.xml` et utiliser toutes les informations pertinentes de ce fichier pour définir les informations de l'application native. Cordova-cli prend en charge la modification des données de votre application via les éléments suivants dans le fichier `config.xml` :

  * Le nom d'utilisateur peut être modifié via le contenu de l'élément `<name>` .
  * Le nom du package (AKA bundle identificateur ou application id) peut être modifié par l'intermédiaire de l'attribut `id` de l'élément de niveau supérieur `< widget >` .
  * La version modifiable via l'attribut de `version` de l'élément de niveau supérieur `< widget >` .
  * La liste blanche peut être modifiée en utilisant les éléments `< access >` . Assurez-vous que l'attribut `origin` votre élément de points `<access>` à une URL valide (vous pouvez utiliser `*` comme Joker). Pour plus d'informations sur la syntaxe de la liste blanche, consultez le [docs.phonegap.com](http://docs.phonegap.com/en/2.2.0/guide_whitelist_index.md.html#Domain%20Whitelist%20Guide). Vous pouvez utiliser de l'attribut `uri` ([BlackBerry propriétaires](https://developer.blackberry.com/html5/documentation/access_element_834677_11.html)) ou `origin` ([conforme aux normes](http://www.w3.org/TR/widgets-access/#attributes)) pour désigner le domaine.
  * Spécifique à la plateforme préférences peuvent être personnalisés par l'intermédiaire de balises `< preference >` . Voir [docs.phonegap.com](http://docs.phonegap.com/en/2.3.0/guide_project-settings_index.md.html#Project%20Settings) pour obtenir la liste des préférences, que vous pouvez utiliser.
  * La page d'entrée/départ pour votre application peut être définie via l'élément `<content src>` + attribut.

## platforms/

Plates-formes ajoutées à votre application aura l'application native du projet structures disposées dans ce répertoire.

## plugins/

Les plug-ins supplémentaires seront extraites ou copiés dans ce répertoire.

# Hooks

Les projets créés par cordova-cli ont `before` et `after` les crochets pour chaque [commande de projet](#project_commands).

Il existe deux types de crochets : spécifiques au projet et au niveau du module. Ces deux types d'hameçons recevoir le dossier racine du projet en tant que paramètre.

## Crochets spécifiques au projet

Ceux-ci se trouvent sous le répertoire de `hooks` à la racine de votre projet de Cordova. Tous les scripts que vous ajoutez à ces répertoires seront exécutés avant et après les commandes appropriées. Utile pour intégrer vos propres systèmes de construction ou d'intégration avec les systèmes de contrôle de version. **N'oubliez pas**: faites vos scripts exécutables. Reportez-vous au [Guide de crochets](http://cordova.apache.org/docs/en/edge/guide_appdev_hooks_index.md.html#Hooks%20Guide) pour plus d'informations.

### Exemples

  * [ `before_build` crochet pour la compilation du modèle jade](https://gist.github.com/4100866) gracieuseté de [dpogue](http://github.com/dpogue)

## Crochets de niveau module

Si vous utilisez cordova-cli comme un module dans une plus grande demande de **node** , vous pouvez également utiliser les méthodes standards de `EventEmitter` pour attacher aux événements. Ces événements comprennent `before_build`, `before_compile`, `before_docs`, `before_emulate`, `before_run`, `before_platform_add`, `before_library_download`, `before_platform_ls`, `before_platform_rm`, `before_plugin_add`, `before_plugin_ls`, `before_plugin_rm` et `before_prepare`. Il y a également un événement de progression `library_download` . En outre, il y a des saveurs `after_` de tous les événements ci-dessus.

Une fois que vous `require('cordova')` dans votre projet de nœud, vous aurez les `EventEmitter` méthodes habituelles disponibles (`on`, `off` ou `removeListener`, `removeAllListeners`et `emit` ou `trigger`).

# Exemples

## Créer un nouveau projet de Cordova

Cet exemple montre comment créer un projet à partir de zéro, nommé KewlApp avec iOS et le soutien de la plateforme Android et inclut un plugin nommé Kewlio. Le projet va vivre en ~/KewlApp

    cordova create ~/KewlApp KewlApp
    cd ~/KewlApp
    cordova platform add ios android
    cordova plugin add http://example.org/Kewlio-1.2.3.tar.gz
    cordova build
    

La structure de répertoires de KewlApp ressemble maintenant à ceci :

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
    

# Qui contribuent

## Exécution de Tests

    npm test
    

## Obtenir des rapports de couverture de test

    npm run cover
    

## To-do + questions

S'il vous plaît vérifier [Cordova questions avec le composant CLI](http://issues.cordova.io). Si vous trouvez des problèmes avec cet outil, veuillez avoir la gentillesse pour inclure des informations pertinentes nécessaires pour déboguer des problèmes tels que :

  * Votre système d'exploitation et version
  * Le nom de l'application, l'emplacement du répertoire et identificateur utilisé avec `create`
  * Quel SDK mobiles vous avez installé et leurs versions. Lié à cela : quelle version de `Xcode` , si vous présentez des problèmes associés à iOS
  * Toute trace de pile d'erreur que vous avez reçu

## Contributeurs

Merci à tous pour leur contribution ! Pour obtenir la liste des personnes concernées, veuillez consulter le fichier `package.json` .

# Problèmes connus et dépannage

## N'importe quel OS

### Paramètres du proxy

`Cordova-cli` utilisera les paramètres de proxy de la `NGP`. Si vous avez téléchargé cordova-cli via `NGP` et derrière un proxy, les chances sont cordova-cli devrait fonctionner pour vous, qu'il utilisera ces paramètres en premier lieu. Assurez-vous que les variables de configuration NGP `https proxy` et le `proxy` sont corrects. Consultez la [documentation de configuration de la NGP](https://npmjs.org/doc/config.html) pour plus d'informations.

## Windows

### Problème ajout d'Android comme une plateforme

Lorsque vous essayez d'ajouter une plateforme sur une machine Windows, si vous rencontrez le message d'erreur suivant : bibliothèque de Cordova pour « android » existe déjà. Pas besoin de télécharger. Continue. Vérifier si la plate-forme « android » passe exigences minimales... Vérification des exigences Android... Exécuter "android liste cible" (sortie à suivre)

    Error: The command `android` failed. Make sure you have the latest Android SDK installed, and the `android` command (inside the tools/ folder) added t
    o your path. Output:
    at C:\Users\me\AppData\Roaming\npm\node_modules\cordova\src\platform.js:185:42
    at C:\Users\me\AppData\Roaming\npm\node_modules\cordova\src\metadata\android_parser.js:50:13
    at C:\Users\me\AppData\Roaming\npm\node_modules\cordova\node_modules\shelljs\shell.js:1707:7
    at exithandler (child_process.js:633:7)
    at ChildProcess.errorhandler (child_process.js:649:5)
    at ChildProcess.EventEmitter.emit (events.js:95:17)
    at Process.ChildProcess._handle.onexit (child_process.js:787:12)
    

Exécutez la commande `android list target`. Si vous voyez :

    'xcopy' is not recognized as an internal or external command,
    operable program or batch file.
    

au début de la sortie de la commande, cela signifie que vous devrez fixer votre variable de chemin d'accès Windows pour inclure xcopy. Cet emplacement se trouve généralement sous C:\Windows\System32.

## Windows 8

Support Windows 8 n'inclut pas la capacité de lancement/run/émuler, donc vous aurez besoin ouvrir **Visual Studio** pour voir votre application en direct. Vous êtes toujours en mesure d'utiliser les commandes suivantes avec windows8 :

  * `platform add windows8`
  * `platform remove windows8`
  * `prepare windows8`
  * `compile windows8`
  * `build windows8`

Pour exécuter votre application, vous devrez ouvrir le `.sln` dans le dossier de `platforms/windows8` , à l'aide de **Visual Studio 2012**.

**Visual Studio** vous le diront pour recharger le projet si vous exécutez une des commandes ci-dessus, alors que le projet est chargé.

## Amazon Fire OS

Amazon Fire OS n'inclut pas la capacité d'imiter. Vous êtes toujours en mesure d'utiliser les commandes suivantes avec Amazon Fire OS

  * `platform add amazon-fireos`
  * `platform remove amazon-fireos`
  * `prepare amazon-fireos`
  * `compile amazon-fireos`
  * `build amazon-fireos`

## Ubuntu

La version initiale de cordova-ubuntu ne supporte pas automatiquement créer des applications pour les dispositifs de Portage. Il est possible de produire des applications, puis cliquez sur les paquets en quelques étapes bien.

Ce rapport de bogue documente le problème et les solutions pour cela : https://bugs.launchpad.net/ubuntu/+source/cordova-ubuntu/+bug/1260500 une version ultérieure permettra de développeurs cross-compiler Portage cliquez sur paquets directement depuis un ordinateur de bureau x 86.

## Firefox OS

OS de Firefox n'inclut pas la capacité d'imiter, d'exécuter et de servir. Après construction, vous devrez ouvrir le répertoire de plateforme de `firefoxos` de votre application dans l' [WebIDE](https://developer.mozilla.org/docs/Tools/WebIDE) qui vient avec tous les navigateurs Firefox. Vous pouvez ouvrir cette fenêtre et cliquez sur le bouton « jouer » chaque fois que vous terminé la construction de votre application.