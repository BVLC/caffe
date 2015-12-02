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

> La herramienta de línea de comandos para crear, implementar y administrar [Cordova](http://cordova.io)-aplicaciones basadas en.

[Cordova de Apache](http://cordova.io) permite construir aplicaciones móviles nativas usando HTML, CSS y JavaScript. Esta herramienta ayuda a gestión de aplicaciones en múltiples plataformas Córdoba así como la integración de plugin de Córdoba.

Compruebe hacia fuera las [guías de introducción](http://cordova.apache.org/docs/en/edge/) para obtener más información sobre cómo trabajar con sub-proyectos Córdoba.

# Plataformas soportadas Cordova

  * Amazon fire OS
  * Android
  * BlackBerry 10
  * Firefox OS
  * iOS
  * Ubuntu
  * Windows Phone 8
  * Windows 8

# Requisitos

  * [Node.js](http://nodejs.org/)
  * SDK para cada plataforma que desea apoyar: 
      * **Android**: [Android SDK](http://developer.android.com) - **Nota** esta herramienta no funcionará a menos que tengas las últimas actualizaciones absolutas para todos los componentes del SDK de Android. También necesitará del SDK `tools` y `platform-tools` directorios en su **ruta del sistema** de lo contrario Android soporte fallará.
      * **fireos Amazon**: [Amazon fuego OS SDK](https://developer.amazon.com/public/solutions/platforms/android-fireos/docs/building-and-testing-your-hybrid-app) - **Nota** esta herramienta no funcionará si no tienes instalado el SDK de Android y caminos se actualizan como se mencionó anteriormente. Además deberás instalar SDK de AmazonWebView y copiar awv_interface.jar en sistema **Mac/Linux** a la carpeta ~/.cordova/lib/commonlibs o en %USERPROFILE%/.cordova/lib/coomonlibs **Windows** . Si no existe la carpeta commonlibs crear uno.
      * [Windows Phone SDK](http://dev.windowsphone.com/en-us/downloadsdk) - **Nota** esta herramienta no funcionará a menos que tengas `msbuild` en su **ruta del sistema** de lo contrario dejará de soporte de Windows Phone (`msbuild.exe` está generalmente situado en `C:\Windows\Microsoft.NET\Framework\v4.0.30319`).
      * **BlackBerry 10**: [BlackBerry 10 WebWorks SDK](http://developer.blackberry.com/html5/download/). Asegúrese de que tener la carpeta de `dependencias/herramientas/bin` dentro del directorio SDK añade a tu camino!
      * **iOS**: [iOS SDK](http://developer.apple.com) y el último `Xcode` `Xcode herramientas de línea de comandos`
      * **Windows Phone**: [Windows Phone SDK](http://dev.windowsphone.com/en-us/downloadsdk) - **Nota** esta herramienta no funcionará a menos que tengas `msbuild` en su **ruta del sistema** de lo contrario dejará de soporte de Windows Phone (`msbuild.exe` está generalmente situado en `C:\Windows\Microsoft.NET\Framework\v4.0.30319`).

`Cordova-cli` se ha probado en **Mac OS X**, **Linux**, **Windows 7**y **Windows 8**.

Tenga en cuenta que algunas plataformas de tengan restricciones de sistema operativo. Por ejemplo, se puede construir para Windows 8 y Windows Phone 8 en Mac OS X, tampoco puede construir para iOS en Windows.

# Instalar

Paquetes de Ubuntu están disponibles en un PPA para Ubuntu 13.10 (Saucy) (la versión actual), así como 14.04 (fiel) (en desarrollo).

    sudo apt-add-repository ppa:cordova-ubuntu/ppa
    sudo apt-get update
    sudo apt-get install cordova-cli
    npm install -g cordova
    

Para construir una aplicación para la plataforma de Ubuntu, son necesarios los siguientes paquetes adicionales:

    sudo apt-get install cmake debhelper libx11-dev libicu-dev pkg-config qtbase5-dev qtchooser qtdeclarative5-dev qtfeedback5-dev qtlocation5-dev qtmultimedia5-dev qtpim5-dev qtsensors5-dev qtsystems5-dev
    

## Instalación de maestro

Usted necesitará instalar la [CLI](https://git-wip-us.apache.org/repos/asf/cordova-cli.git) y [Plugman](https://git-wip-us.apache.org/repos/asf/cordova-plugman.git) de `git`. Ejecuta la *versión del npm* de uno y *versión principal (git)* de la otra es probable que al terminar usted sufren.

Para evitar el uso de sudo, ver [alejarse de sudo: npm sin raíz](http://justjs.com/posts/npm-link-developing-your-own-npm-modules-without-tears).

Ejecute los comandos siguientes:

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
    

Ahora el `cordova` y `plugman` en tu camino son las versiones de git local. No te olvides de mantenerlas hasta la fecha.

## Instalar en Ubuntu

    apt-get install cordova-cli
    

# Para empezar

`Cordova-cli` tiene un comando global solo `create` que crea nuevos proyectos de Córdoba en un directorio especificado. Una vez que se crea un proyecto, `cd` y se puede ejecutar una variedad de comandos de nivel de proyecto. Totalmente inspirado en la interfaz de git.

## Comandos globales

  * `help` muestra una página de ayuda con todos los comandos disponibles
  * `create <directory> [<id> [<name>]]` crea un nuevo proyecto de Cordova con nombre opcional y el identificador (nombre del paquete, estilo de dominio reverso)

<a name="project_commands" />

## Comandos de proyecto

  * `platform [ls | list]` lista de todas las plataformas para que el proyecto se basará
  * `platform add <platform> [<platform> ...]` Añadir plataformas uno (o más) como un objetivo de construcción del proyecto
  * `platform [rm | remove] <platform> [<platform> ...]` quita uno (o más) objetivos de construcción de plataforma del proyecto
  * `platform [up | update] <platform>` -actualiza la versión de Córdoba utilizada la plataforma dada
  * `plugin [ls | list]` lista todos los plugins incluidos en el proyecto
  * `plugin add <path-to-plugin> [<path-to-plugin> ...]` Añadir plugins uno (o más) para el proyecto
  * `plugin [rm | remove] <plugin-name> [<plugin-name> ...]` quitar plugins uno (o más) del proyecto.
  * `plugin search [<keyword1> <keyword2> ...]` busca el registro de plugin para plugins que empareja la lista de palabras clave de búsqueda
  * `prepare [platform...]` copia los archivos en las plataformas especificadas, o todas las plataformas. Entonces está listo para edificio por `Eclipse`, `Xcode`, etc.
  * `compile [platform...]` compila la aplicación en un archivo binario para cada plataforma dirigido. Sin parámetros, estructuras para todas las plataformas, de lo contrario construye para las plataformas especificadas.
  * `construir [< plataforma > [< plataforma > [...]]]` seguida de un alias para `cordova prepare` `cordova compile`
  * `emulate [<platform> [<platform> [...]]]` lanzamiento de emuladores y desplegar la aplicación en ellos. Sin parámetros emula para todas las plataformas ha añadido al proyecto, emula lo contrario para las plataformas especificadas
  * `serve [port]` en marcha un servidor web local lo que le permite acceder a directorio de www de cada plataforma en el puerto determinado (por defecto 8000).

### Banderas opcionales

  * `-d` o `--verbose` envía una salida más verbosa a su shell. También puede suscribirse a eventos de `log` y `advertir` si eres consumidor `cordova-cli` como módulo nodo llamando a `cordova.on ('log', function() {})` o `cordova.on ('warn', function() {})`.
  * `-v` o `--version` instalará imprimir la versión de su `Córdoba-cli` .

# Estructura de directorios del proyecto

Una aplicación de Cordova con `cordova-cli` tendrá la siguiente estructura de directorios:

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

Este directorio puede contiene secuencias de comandos se usa para personalizar los comandos cli de Córdoba. Este directorio solía existir en los `.cordova/hooks`, pero ahora se ha movido a la raíz del proyecto. Cualquier scripts que añadir a estos directorios se ejecutarán antes y después de los comandos correspondientes a nombre del directorio. Útil para integrar su propio sistema de construcción o integración con sistemas de control de versiones.

Para más información, consulte Guía de ganchos</a> .

## merges/

Activos específicos de la plataforma web (archivos HTML, CSS y JavaScript) están contenidos en las subcarpetas correspondientes en este directorio. Estos se despliegan durante un `prepare` el directorio nativo apropiado. Archivos colocados bajo `merges/` sobrescribirá los archivos coincidentes en el `www /` carpeta para la plataforma correspondiente. Un ejemplo rápido, si se asume que una estructura de proyecto de:

    merges/
    |-- ios/
    | `-- app.js
    |-- android/
    | `-- android.js
    www/
    `-- app.js
    

Después de construir los proyectos de Android y iOS, la aplicación Android contendrá `app.js` y `android.js`. Sin embargo, la aplicación de iOS sólo contendrá un `app.js`, y será uno de `merges/ios/app.js`, reemplazando el "común" `app.js` ubicado dentro de `www /`.

## www/

Contiene artefactos de la web del proyecto, tales como ficheros .html, .css y .js. Estos son los activos de principal de la aplicación. Se copiarán en una `Córdoba prepara` al directorio www de cada plataforma.

### Su manta: config.xml

Este archivo es usted debe editar para modificar los metadatos de la aplicación. Cualquier momento que ejecutar cualquier comando de cordova-cli, la herramienta Buscar en el contenido del `archivo config.xml` y utilice toda la información relevante de este archivo para definir información de aplicación nativa. Cordova-cli admite cambiar datos de su aplicación a través de los siguientes elementos dentro del archivo `config.xml` :

  * El nombre de usuario puede modificarse mediante el contenido del elemento `<name>` .
  * El nombre del paquete (AKA paquete identificador o aplicación id) puede modificarse mediante el atributo `id` del elemento de nivel superior `< widget >` .
  * La versión puede ser modificada mediante el atributo de `version` del elemento de nivel superior `< widget >` .
  * La lista blanca puede ser modificada utilizando los elementos de `acceso < a >` . Asegúrese de que el atributo de `origin` sus < > elemento de puntos de `access` a una dirección URL válida (usted puede utilizar `*` como comodín). Para obtener más información sobre la sintaxis de las listas blancas, vea [docs.phonegap.com](http://docs.phonegap.com/en/2.2.0/guide_whitelist_index.md.html#Domain%20Whitelist%20Guide). Usted puede utilizar el atributo `uri` ([propietaria de BlackBerry](https://developer.blackberry.com/html5/documentation/access_element_834677_11.html)) u `origin` ([estándares](http://www.w3.org/TR/widgets-access/#attributes)) para denotar el dominio.
  * Preferencias específicas de la plataforma pueden ser personalizados via etiquetas `<preference>` . Consulte [docs.phonegap.com](http://docs.phonegap.com/en/2.3.0/guide_project-settings_index.md.html#Project%20Settings) para obtener una lista de preferencias que se puede utilizar.
  * La página de entrada/salida para su aplicación puede definirse mediante el elemento `<content src>` + atributo.

## platforms/

Añadido a la aplicación de plataformas tendrá la aplicación nativa proyecto estructuras establecidas dentro de este directorio.

## plugins/

Cualquier añadidos plugins serán extraídos o copiados en este directorio.

# Hooks

Proyectos creados por cordova-cli tienen `antes` y `después de` ganchos para cada [comando proyecto](#project_commands).

Hay dos tipos de ganchos: los proyectos específicos y los de nivel de módulo. Ambos de estos tipos de ganchos reciben la carpeta raíz del proyecto como parámetro.

## Hooks de proyectos específicos

Estos están ubicados en el directorio de `ganchos` en la raíz de su proyecto de Cordova. Cualquier scripts que añadir a estos directorios se ejecutarán antes y después de los comandos apropiados. Útil para integrar su propio sistema de construcción o integración con sistemas de control de versiones. **Recuerde**: hacer sus scripts ejecutables. Para más información, consulte [Guía de ganchos](http://cordova.apache.org/docs/en/edge/guide_appdev_hooks_index.md.html#Hooks%20Guide) .

### Ejemplos

  * [ `before_build` gancho para plantilla jade compilación](https://gist.github.com/4100866) cortesía de [dpogue](http://github.com/dpogue)

## Hooks de nivel de módulo

Si usas cordova-cli como un módulo dentro de una mayor aplicación de **node** , puede usar los métodos estándar de `EventEmitter` para sujetar a los acontecimientos. Los eventos incluyen `before_build`, `before_compile`, `before_docs`, `before_emulate`, `before_run`, `before_platform_add`, `before_library_download`, `before_platform_ls`, `before_platform_rm`, `before_plugin_add`, `before_plugin_ls`, `before_plugin_rm` y `before_prepare`. También es un evento de progreso `library_download` . Además, hay `after_` de sabores de todos los eventos antes mencionados.

Una vez `require('cordova')` en su proyecto de nodo, tendrá los habituales `EventEmitter` métodos disponibles (`on`, `off` o `removeListener`, `removeAllListeners`y `emit` o `trigger`).

# Ejemplos

## Crear un nuevo proyecto de Cordova

Este ejemplo muestra cómo crear un proyecto desde cero llamado KewlApp con el apoyo de la plataforma Android y iOS e incluye un plugin llamado Kewlio. El proyecto va a vivir a ~/KewlApp

    cordova create ~/KewlApp KewlApp
    cd ~/KewlApp
    cordova platform add ios android
    cordova plugin add http://example.org/Kewlio-1.2.3.tar.gz
    cordova build
    

La estructura de directorios de KewlApp ahora se ve así:

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
    

# Contribuyendo

## Ejecución de pruebas

    npm test
    

## Obtener informes de cobertura de la prueba

    npm run cover
    

## Tareas + temas

Por favor compruebe [Cordova aspectos con el componente de CLI](http://issues.cordova.io). Si encuentras problemas con esta herramienta, por favor sea tan amable como para incluir información relevante necesaria para depurar problemas tales como:

  * El sistema operativo y versión
  * El nombre de la aplicación, directorio e identificador usado con `create`
  * Que SDK móvil que ha instalado y sus versiones. Relacionado con esto: que versión de `Xcode` si están presentando temas relacionados con iOS
  * Cualquier rastro de pila de error que recibió

## Colaboradores

Gracias a todos por su contribución! Para una lista de personas involucradas, por favor, consulte el archivo `package.json` .

# Problemas y solución de problemas

## Cualquier sistema operativo

### Configuración de proxy

`Cordova-cli` utilizar configuración de proxy del `NPM`. Si descargaste cordova-cli mediante `NPM` y están detrás de un proxy, lo más probable es cordova-cli debe trabajar para usted ya que utilizará los valores en primer lugar. Asegúrese de que el npm config variables `proxy` y `proxy https` están ajustadas correctamente. Consulte la [documentación de configuración del NPM](https://npmjs.org/doc/config.html) para obtener más información.

## Windows

### Problemas al agregar Android como una plataforma

Al intentar agregar una plataforma en una máquina Windows si ejecuta en el siguiente mensaje de error: biblioteca Córdoba para "android" ya existe. No hay necesidad de descargar. Continuando. Comprobar si la plataforma "android" pasa requisitos mínimos... Comprobación de requisitos de Android... Funcionamiento "target android lista" (salida a seguir)

    Error: The command `android` failed. Make sure you have the latest Android SDK installed, and the `android` command (inside the tools/ folder) added t
    o your path. Output:
    at C:\Users\me\AppData\Roaming\npm\node_modules\cordova\src\platform.js:185:42
    at C:\Users\me\AppData\Roaming\npm\node_modules\cordova\src\metadata\android_parser.js:50:13
    at C:\Users\me\AppData\Roaming\npm\node_modules\cordova\node_modules\shelljs\shell.js:1707:7
    at exithandler (child_process.js:633:7)
    at ChildProcess.errorhandler (child_process.js:649:5)
    at ChildProcess.EventEmitter.emit (events.js:95:17)
    at Process.ChildProcess._handle.onexit (child_process.js:787:12)
    

Ejecute el comando `target list android`. Si ves:

    'xcopy' is not recognized as an internal or external command,
    operable program or batch file.
    

al principio de la salida del comando, que significa que usted necesitará fijar la variable Path de Windows para incluir xcopy. Esta ubicación suele estar en C:\Windows\System32.

## Windows 8

Soporte Windows 8 no incluye la capacidad de lanzamiento/run/emular, por lo que necesitarás abrir **Visual Studio** para ver su aplicación en vivo. Eres todavía capaz de usar los siguientes comandos con windows8:

  * `platform add windows8`
  * `platform remove windows8`
  * `prepare windows8`
  * `compile windows8`
  * `build windows8`

Para ejecutar la aplicación, usted necesitará abrir `.sln` en la carpeta de `platforms/windows8` usando **Visual Studio 2012**.

**Visual Studio** le dirá que vuelva a cargar el proyecto si se ejecuta cualquiera de los comandos anteriores mientras se carga el proyecto.

## Amazon fire OS

Amazon Fire OS no incluyen la capacidad de emular. Todavía puede utilizar los siguientes comandos con Amazon fuego OS

  * `platform add amazon-fireos`
  * `platform remove amazon-fireos`
  * `prepare amazon-fireos`
  * `compile amazon-fireos`
  * `build amazon-fireos`

## Ubuntu

La versión inicial de cordova-ubuntu no permite desarrollo de aplicaciones para dispositivos armhf automáticamente. Es posible crear aplicaciones y haga clic en paquetes en unos pocos pasos sin embargo.

Este informe documenta el problema y las soluciones para: https://bugs.launchpad.net/ubuntu/+source/cordova-ubuntu/+bug/1260500 una versión futura permitirá desarrolladores cross-compilar armhf haga clic en paquetes directamente desde un escritorio de x 86.

## Firefox OS

Firefox OS no incluyen la capacidad de emular, ejecutar y servir. Después de la construcción, tienes que abrir el directorio de plataforma `firefoxos` de su aplicación en el [WebIDE](https://developer.mozilla.org/docs/Tools/WebIDE) que viene con cada navegador Firefox. Puede mantener abierta esta ventana y haga clic en el botón "play" cada vez que terminado de construir tu aplicación.