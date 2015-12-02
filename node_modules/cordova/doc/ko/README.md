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

> 커맨드 라인 도구를 빌드, 배포 및 관리 [코르도바](http://cordova.io)-기반 응용 프로그램.

[아파치 코르도바](http://cordova.io) HTML, CSS, 자바 스크립트를 사용 하 여 네이티브 모바일 응용 프로그램을 빌드할 수 있습니다. 이 도구는 멀티 플랫폼 코르도바 응용 프로그램으로 코르도바 플러그인 통합 관리 도움이 됩니다.

코르도바 하위 프로젝트 작업 방법에 대 한 자세한 내용은 [시작 가이드](http://cordova.apache.org/docs/en/edge/) 를 확인 하십시오.

# 지원된 코르도바 플랫폼

  * 아마존 화재 운영 체제
  * 안 드 로이드
  * 블랙베리 10
  * Firefox 운영 체제
  * iOS
  * 우분투
  * Windows Phone 8
  * 윈도우 8

# 요구 사항

  * [Node.js](http://nodejs.org/)
  * 지원 하려는 각 플랫폼 Sdk: 
      * **안 드 로이드**: [안 드 로이드 SDK](http://developer.android.com) - **참고** 이 도구는 모든 안 드 로이드 SDK 구성 요소에 대 한 절대 최신 업데이트 하지 않은 경우 작동 하지 것입니다. 또한 SDK의 `tools` 필요 합니다 및 **platform-tools** 그렇지 않으면 안 드 로이드 지원에 `플랫폼 도구` 디렉터리 실패 합니다.
      * **아마존 fireos**: [아마존 화재 OS SDK](https://developer.amazon.com/public/solutions/platforms/android-fireos/docs/building-and-testing-your-hybrid-app) - **참고** 해야 안 드 로이드 SDK 설치 하 고 경로 위에서 언급 한 업데이트는이 도구가 작동 하지 것입니다. 또한 AmazonWebView SDK를 설치 하 고 awv_interface.jar **Windows** %USERPROFILE%/.cordova/lib/coomonlibs 또는 ~/.cordova/lib/commonlibs 폴더를 **맥/리눅스** 시스템에 복사 해야 합니다. Commonlibs 폴더는 존재 하지 않는 경우 다음 만듭니다.
      * [Windows Phone SDK](http://dev.windowsphone.com/en-us/downloadsdk) - **참고** 이 도구는 `msbuild` **시스템 경로** 에 있으면 않는 한 작동 하지 것입니다 그렇지 않으면 Windows Phone 지원 실패 합니다 (`msbuild.exe` 일반적으로에 있는 `C:\Windows\Microsoft.NET\Framework\v4.0.30319`).
      * **블랙베리 10**: [10 블랙베리 WebWorks SDK](http://developer.blackberry.com/html5/download/). 당신은 당신의 경로에 추가 SDK 디렉토리 안에 `종속성/도구/bin` 폴더 다는 것을 확인 하십시오!
      * **iOS**: [iOS SDK](http://developer.apple.com) 최신 `Xcode` 와 `Xcode 명령줄 도구`
      * **Windows Phone**: [Windows Phone SDK](http://dev.windowsphone.com/en-us/downloadsdk) - **참고** 이 도구는 `msbuild` **시스템 경로** 에 있으면 않는 한 작동 하지 것입니다 그렇지 않으면 Windows Phone 지원 실패 합니다 (`msbuild.exe` 일반적으로에 있는 `C:\Windows\Microsoft.NET\Framework\v4.0.30319`).

`코르도바-cli가` **맥 OS X**, **리눅스**, **윈도우 7**, **윈도우 8** 에서 테스트 되었습니다..

일부 플랫폼에 OS 제한이 note 하시기 바랍니다. 예를 들어 윈도우 8 또는 Windows Phone 8 맥 OS X에 대 한 만들 수 없습니다 없으며 윈도우에서 iOS 용을 만들 수 없습니다.

# 설치

우분투 패키지 우분투 13.10 (Saucy) (현재 버전) (개발) 중인 14.04 (믿을 함)에 대 한 PPA에서 사용할 수 있습니다.

    sudo apt-add-repository ppa:cordova-ubuntu/ppa
    sudo apt-get update
    sudo apt-get install cordova-cli
    npm install -g cordova
    

우분투 플랫폼에 대 한 신청서를 작성 하려면 다음 추가 패키지가 필요 합니다.

    sudo apt-get install cmake debhelper libx11-dev libicu-dev pkg-config qtbase5-dev qtchooser qtdeclarative5-dev qtfeedback5-dev qtlocation5-dev qtmultimedia5-dev qtpim5-dev qtsensors5-dev qtsystems5-dev
    

## 마스터 설치

`Git`에서 [CLI](https://git-wip-us.apache.org/repos/asf/cordova-cli.git) 와 [Plugman](https://git-wip-us.apache.org/repos/asf/cordova-plugman.git) 둘 다를 설치 해야 합니다. 하나의 *npm 버전* 및 다른 *(git) 마스터 버전* 실행 고통을 당신과 함께 끝낼 것입니다.

Sudo를 사용 하지 않도록, 참조 [sudo 도망: 루트 없이 npm](http://justjs.com/posts/npm-link-developing-your-own-npm-modules-without-tears).

다음 명령을 실행 합니다.

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
    

지금 `코르도바` 및 `plugman` 경로에 로컬 git 버전을 있습니다. 그들을 최신 상태로 유지 하는 것을 잊지 마세요!

## 우분투에 설치

    apt-get install cordova-cli
    

# 시작 하기

`코르도바-cli` 는 지정된 된 디렉터리에 새 코르도바 프로젝트를 만드는 단일 글로벌 `만들` 명령. 프로젝트를 만들면 `cd` 그것으로 다양 한 프로젝트 수준 명령 실행할 수 있습니다. 완전히 git의 인터페이스에 의해 영감을.

## 글로벌 명령

  * 모든 사용 가능한 명령 도움말 페이지를 표시 `help`
  * 옵션 이름 및 id (패키지 이름, 리버스 도메인 스타일) 새로운 코르도바 프로젝트를 생성 `만들기 < 디렉토리 > [< id > [< 이름 >]]`

<a name="project_commands" />

## 프로젝트 명령

  * `플랫폼 [ls | 목록]` 프로젝트를 구축 하는 모든 플랫폼을 나열
  * `플랫폼 추가 < 플랫폼 > [< 플랫폼 >...]` 프로젝트에 대 한 빌드 대상으로 하나 (혹은 이상) 플랫폼 추가
  * `플랫폼 [rm | 제거] < 플랫폼 > [< 플랫폼 >...]` (이상의) 플랫폼 빌드 대상 프로젝트에서 제거
  * `플랫폼 [최대 | 업데이트] < 플랫폼 >` -특정된 플랫폼에 대 한 사용 하는 코르도바 버전 업데이트
  * `플러그인 [ls | 목록]` 는 프로젝트에 포함 된 모든 플러그인 목록
  * `플러그인 추가 < 경로-을-플러그인 > [< 경로-을-플러그인 >...]` 프로젝트를 하나 (혹은 이상) 플러그인 추가
  * `플러그인 [rm | 제거] < 플러그인-이름 > [< 플러그인->...]` 프로젝트에서 하나 (혹은 이상) 플러그인을 제거 합니다.
  * `플러그인 검색 [< keyword1 >< keyword2 >...]` 키워드의 목록을 플러그인 플러그인 레지스트리 검색
  * 특정된 플랫폼 이나 모든 플랫폼에 파일을 복사 하는 `[platform...]를 준비` . 그것은 `이클립스`, `Xcode`, 등등에 의해 건물에 대 한 준비 다음입니다.
  * `[platform...]를 컴파일` 각 타겟 플랫폼에 대 한 바이너리에 응용 프로그램을 컴파일합니다. 매개 변수 없이, 모든 플랫폼에 대 한 빌드 그렇지 않으면 지정 된 플랫폼에 대 한 빌드합니다.
  * `빌드 [< 플랫폼 > [< 플랫폼 > [...]]]` `코르도바 컴파일` 뒤에 `코르도바 준비` 에 대 한 별칭
  * `에뮬레이션 [< 플랫폼 > [< 플랫폼 > [...]]]` 에뮬레이터를 시작 하 고 그들에 게 응용 프로그램을 배포. 매개 변수 없이 프로젝트에 추가 하는 모든 플랫폼에 대 한 에뮬레이션, 그렇지 않으면 지정 된 플랫폼에 대 한 에뮬레이션
  * `[port] 서브` 각 플랫폼의 www 디렉토리 (기본 8000) 특정된 포트에 액세스할 수 있도록 로컬 웹 서버를 시작 합니다.

### 옵션 플래그

  * `-d` 또는 `-자세한` 껍질에 더 자세한 출력을 파이프 됩니다. 만약 당신이 소모 `코르도바 cli` 로 노드 모듈 `cordova.on ('로그', function() {})` 또는 `cordova.on ('경고', function() {})를` 호출 하 여 또한 `로그` 및 `경고` 이벤트를 구독할 수 있습니다..
  * `-v` 또는 `-버전` 인쇄 `코르도바 cli` 의 버전을 설치 합니다.

# 프로젝트 디렉터리 구조

`코르도바-cli` 로 빌드된 코르도바 응용 프로그램 다음과 같은 디렉터리 구조를 갖습니다.

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

이 디렉터리 수 있습니다 코르도바 cli 명령을 사용자 지정 하는 데 사용 하는 스크립트가 포함 되어 있습니다. 이 디렉터리에서 `.cordova/후크`, 존재 하는 데 사용 하지만 지금 프로젝트 루트에 이동 되었다. 이러한 디렉터리에 추가한 모든 스크립트 디렉터리 이름에 해당 하는 명령을 전후에 실행 됩니다. 자신의 빌드 시스템 통합 또는 버전 제어 시스템 통합에 유용 합니다.

자세한 내용은 후크 가이드</a> 를 참조 하십시오.

## merges/

이 디렉터리에 적절 한 하위 플랫폼 관련 웹 자산 (HTML, CSS 및 JavaScript 파일) 포함 됩니다. 이는 `준비` 는 적절 한 기본 디렉터리를 하는 동안 배포 됩니다. 아래에 배치 하는 파일 `병합 /` 에 일치 하는 파일을 무시 합니다는 `www /` 관련 플랫폼에 대 한 폴더. 빠른 예의 프로젝트 구조를 가정:

    merges/
    |-- ios/
    | `-- app.js
    |-- android/
    | `-- android.js
    www/
    `-- app.js
    

안 드 로이드 및 iOS 프로젝트를 빌드한 후 안 드 로이드 응용 프로그램 `app.js` 와 `android.js`모두 포함 됩니다. 그러나, iOS 응용 프로그램은는 `app.js`를 포함 하 고 그것은 `merges/ios/app.js`, 안쪽에 있는 "일반적인" `app.js` 재정의에서 하나 될 것입니다 `www /`.

## www/

프로젝트의 웹 아티팩트,.html,.css 및.js 파일 등을 포함합니다. 이들은 기본 응용 프로그램 자산입니다. 그들은 한 `코르도바 준비` 를 각 플랫폼의 www 디렉토리에 복사 됩니다.

### 당신의 담요: config.xml

이 파일은 무엇 당신이 해야 편집 응용 프로그램의 메타 데이터를 수정할 수 있습니다. 언제 든 지 어떤 코르도바 cli 명령을 실행할 도구 `config.xml` 의 내용을 보고 하 고이 파일에서 모든 관련 정보를 사용 하 여 네이티브 응용 프로그램 정보를 정의 것입니다. 코르도바-cli 지원 `config.xml` 파일 내에 다음과 같은 요소를 통해 응용 프로그램의 데이터를 변경 합니다.

  * 사용자 이름 `< 이름 >` 요소의 내용을 통해 수정할 수 있습니다.
  * 패키지 이름 (일명 번들 식별자 또는 응용 프로그램 id) 최상위 `< 위젯 >` 요소의 `id` 특성을 통해 수정할 수 있습니다.
  * 버전 최상위 `< 위젯 >` 요소에서 `버전` 특성을 통해 수정할 수 있습니다.
  * `< 액세스 >` 요소를 사용 하는 허용을 수정할 수 있습니다. 있는지 확인 하십시오 당신의 `< >` 요소 접근점의 `근원` 특성 (사용할 수 있습니다 `*` 와일드 카드) 유효한 URL을. 허용 된 구문에 [docs.phonegap.com](http://docs.phonegap.com/en/2.2.0/guide_whitelist_index.md.html#Domain%20Whitelist%20Guide)를 참조 하십시오. 특성 `uri` ([블랙베리 독점](https://developer.blackberry.com/html5/documentation/access_element_834677_11.html)) 또는 `원본` 사용할 수 있습니다 ([표준 규격](http://www.w3.org/TR/widgets-access/#attributes))을 도메인을 나타내기 위해.
  * 플랫폼 관련 환경 설정 `< 기본 설정 >` 태그를 통해 사용자 지정할 수 있습니다. 사용할 수 있는 기본 설정의 목록에 대 한 [docs.phonegap.com](http://docs.phonegap.com/en/2.3.0/guide_project-settings_index.md.html#Project%20Settings) 를 참조 하십시오.
  * `< 콘텐츠 src >` 요소 + 특성을 통해 응용 프로그램에 대 한 항목/시작 페이지를 정의할 수 있습니다.

## platforms/

플랫폼 응용 프로그램에 추가이 디렉터리 안에 배치 하는 구조를 프로젝트 네이티브 응용 프로그램을 가질 것 이다.

## plugins/

어떤 추가 플러그인 추출 하거나이 디렉터리에 복사 될 것입니다.

# Hooks

코르도바-cli에서 만든 프로젝트는 `전에` 그리고 `후` 각 [프로젝트 명령](#project_commands) 에 대 한 후크.

걸이의 두 가지 유형이 있다: 프로젝트 특정 사람 및 모듈 수준 것 들. 후크 이러한 유형의 모두 프로젝트 루트 폴더 매개 변수로 받을 수 있습니다.

## 프로젝트 특정 후크

이러한 `후크` 디렉터리 코르도바 프로젝트의 루트에 있습니다. 적절 한 명령을 전후 이러한 디렉터리에 추가한 모든 스크립트를 실행 됩니다. 자신의 빌드 시스템 통합 또는 버전 제어 시스템 통합에 유용 합니다. **기억**: 스크립트 실행 확인. 자세한 내용은 [후크 가이드](http://cordova.apache.org/docs/en/edge/guide_appdev_hooks_index.md.html#Hooks%20Guide) 를 참조 하십시오.

### 예제

  * [dpogue](http://github.com/dpogue) 의 [ `before_build` 옥 템플릿 편집에 대 한 연결](https://gist.github.com/4100866)

## 모듈 수준 후크

코르도바-cli 모듈 큰 **노드** 응용 프로그램 내에서 사용 하는 경우 이벤트를 연결 하려면 표준 `EventEmitter` 메서드 또한 사용할 수 있습니다. 이벤트는 `before_build`, `before_compile`, `before_docs`, `before_emulate`, `before_run`, `before_platform_add`, `before_library_download`, `before_platform_ls`, `before_platform_rm`, `before_plugin_add`, `before_plugin_ls`, `before_plugin_rm` 및 `before_prepare`을 포함합니다. `Library_download` 진행 이벤트 이기도합니다. 또한, 위의 모든 이벤트의 `after_` 풍미 있다.

일단 당신이 `require('cordova')` 노드 프로젝트에, 당신은 것입니다 일반적인 `EventEmitter` 방법 사용할 수 (`에`, `떨어져` 또는 `removeListener`, `removeAllListeners`및 `방출` 또는 `트리거`).

# 예제

## 새로운 코르도바 프로젝트 만들기

이 예제에서는 iOS와 안 드 로이드 플랫폼 지원, KewlApp 라는 처음부터 프로젝트를 만드는 방법을 보여 줍니다 고 Kewlio 라는 플러그인을 포함 한다. 프로젝트는 ~/KewlApp에 살 것 이다

    cordova create ~/KewlApp KewlApp
    cd ~/KewlApp
    cordova platform add ios android
    cordova plugin add http://example.org/Kewlio-1.2.3.tar.gz
    cordova build
    

KewlApp의 디렉토리 구조는 지금이 같이 보입니다.

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
    

# 기여

## 테스트 실행

    npm test
    

## 테스트 범위 보고서

    npm run cover
    

## 할 일 + 문제

[CLI 구성 요소와 코르도바 문제](http://issues.cordova.io)를 확인 하시기 바랍니다. 이 도구로 문제를 찾을 경우 하시기 바랍니다 너무 친절 할로 같은 문제를 디버깅 하는 데 필요한 관련 정보를 포함:

  * 운영 체제와 버전
  * 응용 프로그램 이름, 디렉토리 위치, 및 `만들기` 와 함께 사용 하는 식별자
  * 어떤 모바일 Sdk 설치, 그리고 그들의 버전. 이에 관련 된: iOS와 관련 된 문제를 전송 하는 경우 `Xcode` 버전
  * 받은 오류 스택 추적

## 참여자

기여에 대 한 모든 사람에 게 감사! 관련 된 사람들의 목록에 대 한 `package.json` 파일을 참조 하십시오.

# 알려진된 문제 및 문제 해결

## 어떤 운영 체제

### 프록시 설정

`코르도바-cli` `npm`의 프록시 설정을 사용 합니다. 코르도바-cli `npm` 을 통해 다운로드 하는 프록시 뒤에 있는 경우 기회는 코르도바 cli 당신을 위해 작동 해야 첫 번째 장소에서 이러한 설정을 사용 합니다. `Https 프록시` 및 `프록시` npm config 변수 제대로 설정 되어 있는지 확인 합니다. 자세한 내용은 [고궁의 구성 설명서](https://npmjs.org/doc/config.html) 를 참조 하십시오.

## 윈도우

### 안 드 로이드 플랫폼으로 추가 문제

다음과 같은 오류 메시지가 발생 하면 Windows 시스템에 플랫폼을 추가 하려고 할 때: 이미 존재 하는 "안 드 로이드" 코르도바 라이브러리. 다운로드 필요 없음입니다. 계속. "안 드 로이드" 플랫폼 최소 요구 사항을 전달 확인... 안 드 로이드 요구 사항 확인... "안 드 로이드 목록 대상" (따라 출력)를 실행

    Error: The command `android` failed. Make sure you have the latest Android SDK installed, and the `android` command (inside the tools/ folder) added t
    o your path. Output:
    at C:\Users\me\AppData\Roaming\npm\node_modules\cordova\src\platform.js:185:42
    at C:\Users\me\AppData\Roaming\npm\node_modules\cordova\src\metadata\android_parser.js:50:13
    at C:\Users\me\AppData\Roaming\npm\node_modules\cordova\node_modules\shelljs\shell.js:1707:7
    at exithandler (child_process.js:633:7)
    at ChildProcess.errorhandler (child_process.js:649:5)
    at ChildProcess.EventEmitter.emit (events.js:95:17)
    at Process.ChildProcess._handle.onexit (child_process.js:787:12)
    

`android list target`명령을 실행 합니다. 당신이 본다면:

    'xcopy' is not recognized as an internal or external command,
    operable program or batch file.
    

명령 출력의 시작 부분에서 xcopy를 포함 하도록 Windows Path 변수를 수정 해야 합니다 의미 합니다. 이 위치는 일반적으로 C:\Windows\System32 아래.

## 윈도우 8

**Visual Studio** 응용 프로그램 라이브를 볼 수를 열 필요가 것입니다 그래서 윈도우 8 지원 시작/실행/에뮬레이션, 수를 포함 하지 않습니다. 여전히 windows8 다음 명령을 사용할 수 있습니다.

  * `platform add windows8`
  * `platform remove windows8`
  * `prepare windows8`
  * `compile windows8`
  * `build windows8`

응용 프로그램을 실행 하려면 **Visual Studio 2012** 를 사용 하 여 `플랫폼/windows8` 폴더에서 `.sln` 을 열어야 할 것 이다.

**Visual Studio** 프로젝트를 로드 하는 동안 위의 명령을 실행 하는 경우 프로젝트를 다시 로드 하 말할 것 이다.

## 아마존 화재 운영 체제

아마존 화재 운영 체제 에뮬레이션 하는 기능을 포함 하지 않습니다. 여전히 아마존 화재 운영 체제와 함께 다음 명령을 사용할 수 있습니다.

  * `platform add amazon-fireos`
  * `platform remove amazon-fireos`
  * `prepare amazon-fireos`
  * `compile amazon-fireos`
  * `build amazon-fireos`

## 우분투

코르도바-우분투의 초기 릴리스에서 지원 하지 않습니다 응용 프로그램 빌드 armhf 장치에 대 한 자동으로. 그것은 응용 프로그램을 생산 하 여 비록 몇 단계에서 패키지를 클릭 합니다.

이 버그 리포트 문서 그것에 대 한 문제 및 솔루션: https://bugs.launchpad.net/ubuntu/+source/cordova-ubuntu/+bug/1260500 향후 릴리스에서 개발자 크로스 컴파일 armhf x 86 데스크톱에서 직접 패키지를 클릭 하 여 드릴 것입니다.

## Firefox 운영 체제

Firefox 운영 체제는 에뮬레이션 하 고, 실행 하 고 제공 하는 기능을 포함 하지 않습니다. 건물, 후 모든 파이어 폭스 브라우저와 함께 제공 되는 [WebIDE](https://developer.mozilla.org/docs/Tools/WebIDE) 에 `firefoxos` 플랫폼 디렉터리의 응용 프로그램을 열고 해야 합니다. 이 창을 열어 하 고 때마다 귀하의 응용 프로그램을 구축 완료 "재생" 버튼을 클릭할 수 있습니다.