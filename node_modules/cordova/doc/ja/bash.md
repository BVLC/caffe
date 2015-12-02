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

# Bash シェルのサポート

コルドバの CLI は、Bash のコマンドラインのタブ補完を提供するスクリプトが付属しています。 十分に Unix y オペレーティング システム （Linux、BSD、OS X） を実行している場合入力コルドバ コマンド ・ ラインを容易にこれをインストールできます。

## インストール

### Linux

Linux や BSD のシステムをインストールするコピー、 `scripts/cordova.completion` ファイルを `/etc/bash_completion.d` ディレクトリ。これは次に新しいシェルを起動したときに読み取られます。

### OS X

OS X に入れて、 `scripts/cordova.completion` ファイルのどこにでも読みやすいとの末尾に次の行を追加、 `~/.bashrc` ファイル。

    source <path to>/cordova.completion
    

これは次に新しいシェルを起動したときに読み取られます。

## 使い方

それは簡単です ！コマンド ライン 'コルドバ' と呼ばれる実行可能ファイルで始まる限り、ちょうどヒット `<TAB>` 有効な入力候補の一覧を表示する任意の時点で。

例:

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