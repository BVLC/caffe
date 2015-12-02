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

> The command line tool to build, deploy and manage [Cordova](http://cordova.io)-based applications.

[Apache Cordova](http://cordova.io) allows for building native mobile applications using HTML, CSS and JavaScript. This tool helps with management of multi-platform Cordova applications as well as Cordova plugin integration.

Check out the [Getting Started guides](http://cordova.apache.org/docs/en/edge/) for more details on how to work with Cordova sub-projects.

# Supported Cordova Platforms

- Amazon Fire OS
- Android
- BlackBerry 10
- Firefox OS
- iOS
- Ubuntu
- Windows Phone 8
- Windows 8

# Requirements

* [Node.js](http://nodejs.org/)
* SDKs for each platform you wish to support:
  - **Android**: [Android SDK](http://developer.android.com) - **NOTE** This tool
    will not work unless you have the absolute latest updates for all
    Android SDK components. Also you will need the SDK's `tools` and `platform-tools` directories on your __system path__ otherwise Android support will fail.
  - **Amazon Fire OS**: [Amazon Fire OS SDK](https://developer.amazon.com/public/solutions/platforms/android-fireos/docs/building-and-testing-your-hybrid-app) - **NOTE** This tool will not work unless you have Android SDK installed and paths are updated as mentioned above. In addition you need to install AmazonWebView SDK and copy awv_interface.jar on **Mac/Linux** system to ~/.cordova/lib/commonlibs folder or on **Windows** %USERPROFILE%/.cordova/lib/coomonlibs. If commonlibs folder does not exist then create one.
  - **BlackBerry 10**: [BlackBerry 10 WebWorks SDK](http://developer.blackberry.com/html5/download/). Make sure you have the `dependencies/tools/bin` folder inside the SDK directory added to your path!
  - **iOS**: [iOS SDK](http://developer.apple.com) with the latest `Xcode` and `Xcode Command Line Tools`
  - **Windows Phone**: [Windows Phone SDK](http://dev.windowsphone.com/en-us/downloadsdk) - **NOTE** This tool will not work unless you have `msbuild` on your __system path__ otherwise Windows Phone support will fail (`msbuild.exe` is generally located in `C:\Windows\Microsoft.NET\Framework\v4.0.30319`).

`cordova-cli` has been tested on **Mac OS X**, **Linux**, **Windows 7**, and **Windows 8**.

Please note that some platforms have OS restrictions.  For example, you cannot build for Windows 8 or Windows Phone 8 on Mac OS X, nor can you build for iOS on Windows.

# Install

Ubuntu packages are available in a PPA for Ubuntu 13.10 (Saucy) (the current release) as well as 14.04 (Trusty) (under development).

    sudo apt-add-repository ppa:cordova-ubuntu/ppa
    sudo apt-get update
    sudo apt-get install cordova-cli
    npm install -g cordova

To build an application for the Ubuntu platform, the following extra packages are required:

    sudo apt-get install cmake debhelper libx11-dev libicu-dev pkg-config qtbase5-dev qtchooser qtdeclarative5-dev qtfeedback5-dev qtlocation5-dev qtmultimedia5-dev qtpim5-dev qtsensors5-dev qtsystems5-dev


## Installing from master

You'll need to install both [CLI](https://git-wip-us.apache.org/repos/asf/cordova-cli.git) and [Plugman](https://git-wip-us.apache.org/repos/asf/cordova-plugman.git) from `git`. Running the *npm version* of one and *(git) master version* of the other is likely to end with you suffering.

To avoid using sudo, see [Get away from sudo: npm without root](http://justjs.com/posts/npm-link-developing-your-own-npm-modules-without-tears).

Run the following commands:

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

Now the `cordova` and `plugman` in your path are the local git versions. Don't forget to keep them up to date!

## Installing on Ubuntu

    apt-get install cordova-cli


# Getting Started

`cordova-cli` has a single global `create` command that creates new Cordova projects into a specified directory. Once you create a project, `cd` into it and you can execute a variety of project-level commands. Completely inspired by git's interface.

## Global Commands

- `help` display a help page with all available commands
- `create <directory> [<id> [<name>]]` create a new Cordova project with optional name and id (package name, reverse-domain style)

<a name="project_commands" />
## Project Commands

- `platform [ls | list]` list all platforms for which the project will build
- `platform add <platform> [<platform> ...]` add one (or more) platforms as a build target for the project
- `platform [rm | remove] <platform> [<platform> ...]` removes one (or more) platform build targets from the project
- `platform [up | update] <platform> ` - updates the Cordova version used for the given platform
- `plugin [ls | list]` list all plugins included in the project
- `plugin add <path-to-plugin> [<path-to-plugin> ...]` add one (or more) plugins to the project
- `plugin [rm | remove] <plugin-name> [<plugin-name> ...]` remove one (or more) plugins from the project.
- `plugin search [<keyword1> <keyword2> ...]` search the plugin registry for plugins matching the list of keywords
- `prepare [platform...]` copies files into the specified platforms, or all platforms. It is then ready for building by `Eclipse`, `Xcode`, etc.
- `compile [platform...]` compiles the app into a binary for each targetted platform. With no parameters, builds for all platforms, otherwise builds for the specified platforms.
- `build [<platform> [<platform> [...]]]` an alias for `cordova prepare` followed by `cordova compile`
- `emulate [<platform> [<platform> [...]]]` launch emulators and deploy app to them. With no parameters emulates for all platforms added to the project, otherwise emulates for the specified platforms
- `serve [port]` launch a local web server allowing you to access each  platform's www directory on the given port (default 8000).

### Optional Flags

- `-d` or `--verbose` will pipe out more verbose output to your shell. You can also subscribe to `log` and `warn` events if you are consuming `cordova-cli` as a node module by calling `cordova.on('log', function() {})` or `cordova.on('warn', function() {})`.
- `-v` or `--version` will print out the version of your `cordova-cli` install.
- `--no-update-notifier` will disable updates check. Alternatively set `"optOut": true` in `~/.config/configstore/update-notifier-cordova.json` or set `NO_UPDATE_NOTIFIER` environment variable with any value (see details in [update-notifier docs](https://www.npmjs.com/package/update-notifier#user-settings)).

# Project Directory Structure
A Cordova application built with `cordova-cli` will have the following directory structure:

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
This directory may contains scripts used to customize cordova-cli commands. This
directory used to exist at `.cordova/hooks`, but has now been moved to the
project root. Any scripts you add to these directories will be executed before
and after the commands corresponding to the directory name. Useful for
integrating your own build systems or integrating with version control systems.

Refer to [Hooks Guide](http://cordova.apache.org/docs/en/edge/guide_appdev_hooks_index.md.html#Hooks%20Guide) for more information.

## merges/
Platform-specific web assets (HTML, CSS and JavaScript files) are contained within appropriate subfolders in this directory. These are deployed during a `prepare` to the appropriate native directory.  Files placed under `merges/` will override matching files in the `www/` folder for the relevant platform. A quick example, assuming a project structure of:

    merges/
    |-- ios/
    | `-- app.js
    |-- android/
    | `-- android.js
    www/
    `-- app.js

After building the Android and iOS projects, the Android application will contain both `app.js` and `android.js`. However, the iOS application will only contain an `app.js`, and it will be the one from `merges/ios/app.js`, overriding the "common" `app.js` located inside `www/`.

## www/

Contains the project's web artifacts, such as .html, .css and .js files. These are your main application assets. They will be copied on a `cordova prepare` to each platform's www directory.

### Your Blanket: config.xml

This file is what you should be editing to modify your application's metadata. Any time you run any cordova-cli commands, the tool will look at the contents of `config.xml` and use all relevant info from this file to define native application information. cordova-cli supports changing your application's data via the following elements inside the `config.xml` file:

- The user-facing name can be modified via the contents of the `<name>` element.
- The package name (AKA bundle identifier or application id) can be modified via the `id` attribute from the top-level `<widget>` element.
- The version can be modified via the `version` attribute from the top-level `<widget>` element.
- The whitelist can be modified using the `<access>` elements. Make sure the `origin` attribute of your `<access>` element points to a valid URL (you can use `*` as wildcard). For more information on the whitelisting syntax, see the [docs.phonegap.com](http://docs.phonegap.com/en/2.2.0/guide_whitelist_index.md.html#Domain%20Whitelist%20Guide). You can use either attribute `uri` ([BlackBerry-proprietary](https://developer.blackberry.com/html5/documentation/access_element_834677_11.html)) or `origin` ([standards-compliant](http://www.w3.org/TR/widgets-access/#attributes)) to denote the domain.
- Platform-specific preferences can be customized via `<preference>` tags. See [docs.phonegap.com](http://docs.phonegap.com/en/2.3.0/guide_project-settings_index.md.html#Project%20Settings) for a list of preferences you can use.
- The entry/start page for your application can be defined via the `<content src>` element + attribute.

## platforms/
Platforms added to your application will have the native application project structures laid out within this directory.

## plugins/
Any added plugins will be extracted or copied into this directory.

# Hooks

Projects created by cordova-cli have `before` and `after` hooks for each [project command](#project_commands).

There are two types of hooks: project-specific ones and module-level ones. Both of these types of hooks receive the project root folder as a parameter.

## Project-specific Hooks

These are located under the `hooks` directory in the root of your Cordova project. Any scripts you add to these directories will be executed before and after the appropriate commands. Useful for integrating your own build systems or integrating with version control systems. __Remember__: make your scripts executable.
Refer to [Hooks Guide](http://cordova.apache.org/docs/en/edge/guide_appdev_hooks_index.md.html#Hooks%20Guide) for more information.

### Examples

- [`before_build` hook for jade template compiling](https://gist.github.com/4100866) courtesy of [dpogue](http://github.com/dpogue)

## Module-level Hooks

If you are using cordova-cli as a module within a larger **Node** application, you can also use the standard `EventEmitter` methods to attach to the events. The events include `before_build`, `before_compile`, `before_docs`, `before_emulate`, `before_run`, `before_platform_add`, `before_library_download`, `before_platform_ls`, `before_platform_rm`, `before_plugin_add`, `before_plugin_ls`, `before_plugin_rm` and `before_prepare`. There is also a `library_download` progress event. Additionally, there are `after_` flavours of all the above events.

Once you `require('cordova')` in your Node project, you will have the usual `EventEmitter` methods available (`on`, `off` or `removeListener`, `removeAllListeners`, and `emit` or `trigger`).

# Examples

## Creating a new Cordova project
This example shows how to create a project from scratch named KewlApp with iOS and Android platform support, and includes a plugin named Kewlio. The project will live in ~/KewlApp

    cordova create ~/KewlApp KewlApp
    cd ~/KewlApp
    cordova platform add ios android
    cordova plugin add http://example.org/Kewlio-1.2.3.tar.gz
    cordova build

The directory structure of KewlApp now looks like this:

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

# Contributing

## Running Tests

    npm test

## Get test coverage reports

    npm run cover

## TO-DO + Issues

Please check [Cordova issues with the CLI Component](http://issues.cordova.io). If you find issues with this tool, please be so kind as to include relevant information needed to debug issues such as:

- Your operating system and version
- The application name, directory location, and identifier used with `create`
- Which mobile SDKs you have installed, and their versions. Related to this: which `Xcode` version if you are submitting issues related to iOS
- Any error stack traces you received

## Contributors

Thanks to everyone for contributing! For a list of people involved, please see the `package.json` file.


# Known Issues and Troubleshooting

## Any OS

### Proxy Settings

`cordova-cli` will use `npm`'s proxy settings. If you downloaded cordova-cli via `npm` and are behind a proxy, chances are cordova-cli should work for you as it will use those settings in the first place. Make sure that the `https-proxy` and `proxy` npm config variables are set properly. See [npm's configuration documentation](https://npmjs.org/doc/config.html) for more information.

## Windows

### Trouble Adding Android as a Platform

When trying to add a platform on a Windows machine if you run into the following error message:
    Cordova library for "android" already exists. No need to download. Continuing.
    Checking if platform "android" passes minimum requirements...
    Checking Android requirements...
    Running "android list target" (output to follow)

    Error: The command `android` failed. Make sure you have the latest Android SDK installed, and the `android` command (inside the tools/ folder) added t
    o your path. Output:
    at C:\Users\me\AppData\Roaming\npm\node_modules\cordova\src\platform.js:185:42
    at C:\Users\me\AppData\Roaming\npm\node_modules\cordova\src\metadata\android_parser.js:50:13
    at C:\Users\me\AppData\Roaming\npm\node_modules\cordova\node_modules\shelljs\shell.js:1707:7
    at exithandler (child_process.js:633:7)
    at ChildProcess.errorhandler (child_process.js:649:5)
    at ChildProcess.EventEmitter.emit (events.js:95:17)
    at Process.ChildProcess._handle.onexit (child_process.js:787:12)

run the command `android list target`.  If you see:

    'xcopy' is not recognized as an internal or external command,
    operable program or batch file.

at the beginning of the command output, it means you will need to fix your Windows Path variable to include xcopy. This location is typically under C:\Windows\System32.

## Windows 8

Windows 8 support does not include the ability to launch/run/emulate, so you will need to open **Visual Studio** to see your app live.  You are still able to use the following commands with windows8:

- `platform add windows8`
- `platform remove windows8`
- `prepare windows8`
- `compile windows8`
- `build windows8`

To run your app, you will need to open the `.sln` in the `platforms/windows8` folder using **Visual Studio 2012**.

**Visual Studio** will tell you to reload the project if you run any of the above commands while the project is loaded.

## Amazon Fire OS

Amazon Fire OS does not include the ability to emulate. You are still able to use the following commands with Amazon Fire OS

- `platform add amazon-fireos`
- `platform remove amazon-fireos`
- `prepare amazon-fireos`
- `compile amazon-fireos`
- `build amazon-fireos`

## Ubuntu

The initial release of cordova-ubuntu does not support building applications for armhf devices automatically. It is possible to produce applications and click packages in a few steps though.

This bug report documents the issue and solutions for it: https://bugs.launchpad.net/ubuntu/+source/cordova-ubuntu/+bug/1260500 A future release will let developers cross-compile armhf click packages directly from an x86 desktop.

## Firefox OS

Firefox OS does not include the ability to emulate, run and serve. After building, you will have to open the `firefoxos` platform directory of your app in the [WebIDE](https://developer.mozilla.org/docs/Tools/WebIDE) that comes with every Firefox browser. You can keep this window open and click on the "play" button every time you finished building your app.
