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

# Bash 쉘 지원

코르 도우 바 CLI Bash에 대 한 명령줄 탭 완성 기능을 제공 하는 스크립트와 함께 번들로 제공. 충분히 유닉스 y 운영 체제 (리눅스, BSD, OS X)을 실행 하는 경우 입력 코르도바 커맨드 라인을 쉽게 하기 위해이 설치할 수 있습니다.

## 설치

### 리눅스

리눅스 또는 BSD 시스템에 설치 하려면 복사는 `scripts/cordova.completion` 파일을 당신의 `/etc/bash_completion.d` 디렉터리. 이 다음에 새 셸을 시작할 때 읽을 것 이다.

### 운영 체제 X

OS X에는 `scripts/cordova.completion` 파일을 어디서 나 읽을 수 있는, 그리고의 끝에 다음 줄을 추가 `~/.bashrc` 파일:

    source <path to>/cordova.completion
    

이 다음에 새 셸을 시작할 때 읽을 것 이다.

## 사용

그것은 쉽게! 명령줄 '코르도바' 라는 실행 파일을 시작으로 그냥 누르면 `<TAB>` 유효한 완료 목록을 보려면 언제 든 지.

예:

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