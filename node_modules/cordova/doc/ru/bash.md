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

# Поддержка оболочки командной строки bash

Cordova CLI поставляется в комплекте со скриптом, который обеспечивает автодополнение в командной строке по клавише Tab для Bash. Если вы работаете достаточно Unix подобную операционную систему (Linux, BSD, OS X) вы можете установить этот скрипт для упрощения ввода командных строк с cordova.

## Установка

### Linux

Для установки в системе Linux или BSD, скопируйте `scripts/cordova.completion` файла в ваш каталог `/etc/bash_completion.d`. Этот файл будет прочитан в следующий раз когда вы запустите новое окно терминала.

### OS X

На OS X, положите файл `scripts/cordova.completion` где он будет доступен для чтения и добавьте следующую строку в конец вашего `~/.bashrc` файл:

    source <path to>/cordova.completion
    

Этот файл будет прочитан в следующий раз когда вы запустите новое окно терминала.

## Применение

Это очень просто! При условии что ваша командная строка начинается с исполняемого файла под названием «cordova», просто нажмите `<TAB>` в любой момент, чтобы просмотреть список допустимых вариантов.

Примеры:

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