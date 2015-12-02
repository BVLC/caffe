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

# Bash shell 支援

科爾多瓦 CLI 捆綁了一個腳本，為 Bash 提供了命令列選項卡完成。 如果您正在運行一個足夠 Unix-y 的作業系統 （Linux，BSD，OS X) 你可以安裝此包以方便打字科爾多瓦命令列。

## 安裝

### Linux

若要在一個 Linux 或 BSD 系統上安裝，請複製 `scripts/cordova.completion` 檔到你 `/etc/bash_completion.d` 目錄。這將會讀取下次你開始一個新的 shell。

### OS X

在 OS X 上，把 `scripts/cordova.completion` 檔在任何地方可讀的並將以下行添加到年底你 `~/.bashrc` 檔：

    source <path to>/cordova.completion
    

這將會讀取下次你開始一個新的 shell。

## 用法

是容易的 ！只要您的命令列與可執行檔叫做 '科爾多瓦' 開始，剛打了 `<TAB>` 在任何點要查看清單的有效完成。

例子：

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