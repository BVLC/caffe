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

# Powłoki bash

Cordova CLI przyjechał wiązany rezygnować skrypt, który zawiera kartę zakończenia wiersza poleceń dla Bash. Jeśli używasz wystarczająco Unix-y system operacyjny (Linux, BSD, OS X) można zainstalować ten ułatwia pisanie wierszy polecenia cordova.

## Instalacja

### Linux

Aby zainstalować na systemie Linux i BSD, skopiuj `scripts/cordova.completion` pliku do swojego `/etc/bash_completion.d` katalogu. To będzie odczytywane przy następnym uruchomieniu nowej powłoki.

### OS X

Na OS X, `scripts/cordova.completion` plik w dowolnym miejscu czytelny i dodać następującą linię na końcu swojej `~/.bashrc` pliku:

    source <path to>/cordova.completion
    

To będzie odczytywane przy następnym uruchomieniu nowej powłoki.

## Użycie

To proste! Tak długo, jak twój wiersz poleceń zaczyna się plik wykonywalny o nazwie "cordova", po prostu wciskamy `<TAB>` w dowolnym momencie, aby zobaczyć listę ważnych uzupełnień.

Przykłady:

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