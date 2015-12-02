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

# Bash-Shell-Unterstützung

Cordova CLI kommt zusammengerollt mit einem Skript, die Bash Befehlszeile Befehlszeilenergänzung vorsieht. Wenn Sie ein ausreichend Unix-y Betriebssystem (Linux, BSD, OS X) ausführen können Sie damit tippen Cordova-Befehlszeilen erleichtern installieren.

## Installation

### Linux

Um auf einem Linux- oder BSD-System zu installieren, kopieren Sie die `scripts/cordova.completion` Datei in Ihr `/etc/bash_completion.d` Verzeichnis. Dies wird beim nächsten Start eine neue Shell gelesen.

### OS X

Unter OS X setzen die `scripts/cordova.completion` Datei überall lesbar, und fügen Sie folgende Zeile an das Ende Ihrer `~/.bashrc` Datei:

    source <path to>/cordova.completion
    

Dies wird beim nächsten Start eine neue Shell gelesen.

## Verwendung

Es ist einfach! Solange Ihre Befehlszeile mit einer ausführbaren Datei namens 'Cordoba' beginnt, drücken Sie einfach `<TAB>` zu jedem Zeitpunkt, eine Liste der gültigen Ergänzungen zu sehen.

Beispiele:

    $ cordova <TAB>
    build     compile   create    emulate   platform  plugin    prepare   serve
    
    $ cordova pla<TAB>
    
    $ cordova platform <TAB>
    add ls remove rm
    
    $ cordova platform a<TAB>
    
    $ cordova platform add <TAB>
    android     blackberry  ios         wp8         www
    
    $ cordova plugin rm <TAB>
    
    $ cordova plugin rm org.apache.cordova.<TAB>
    org.apache.cordova.file    org.apache.cordova.inappbrowser