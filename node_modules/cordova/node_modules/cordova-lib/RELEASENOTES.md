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
# Cordova-lib Release Notes

### 5.4.1 (Nov 19, 2015)
* CB-9976 Reinstall plugins for platform if they were installed with `cordova@<5.4.0`. 
* CB-9981 `path.parse` only available on `node 0.12+`.
* CB-9987 Adds compatibility layer for `cordova.raw.*` methods
* CB-9975 Fix issue with using `all" as orientation for **iOS**
* CB-9984 Bumps `plist` version and fixes failing `cordova-common` test

### 5.4.0 (Oct 30, 2015)
* CB-9935 Cordova CLI silently fails on node.js v5
* CB-9834 Introduce compat map for hook requires
* CB-9902 Fix broken `cordova run --list`
* CB-9872 Fixed save.spec.11 failure
* CB-9800 Fixing contribute link.
* CB-9736 Extra main activity generated when an android package name is specified
* CB-9675 OSX App Icons are not properly copied.
* CB-9758 Mobilespec crashes adding plugins on OS X
* CB-9782 Update create/update signatures for PlatformApi polyfill
* CB-9815 Engine name="cordova" should check tools version, not platforms. 
* CB-9824 removed plugin download counter code from lib
* CB-9821 Fix EventEmitter incorrect trace level usages
* CB-9813 Keep module-to-plugin mapping at hand.
* CB-9598 Fixes broken `require` for FFOS plugin handler
* Update 'serve' to use 'express' implementation of cordova-serve.
* CB-9712 CLI 5.3 breaks with node 3.3.3
* CB-9598 Fixies broken require calls that aren't covered by tests
* CB-9589 added more warnings and added conversion step to fetch.js
* CB-9589 auto convert old plugin ids to new npm ids using [registry-mapper](https://github.com/stevengill/cordova-registry-mapper)
* Pick ConfigParser changes from apache@0c3614e
* CB-9743 Removes system frameworks handling from ConfigChanges
* CB-9598 Cleans out code which has been moved to `cordova-common`
* CB-9598 Switches LIB to use `cordova-common`
* CB-9569 Support <access> and <allow-navigation> tag translation to Application Transport Security (ATS) Info.plist directives.
* CB-9737 (save flag) unit test failures for spec.14
* CB-8914 when project is renamed, remove userdata otherwise project is un-usable in xcode
* CB-9665 Support .xcassets for icons and splashscreens in the CLI
* CB-9407 Fixes incorrect applying of plugin-provided config changes.
* CB-8198 Unified console output logic for core platforms
* CB-9408 Added support for `windows-packageVersion` on `<widget>`
* CB-9588 Plugman. Add support for <resource-file> on Windows
* CB-8615 Improves plugman tests for Windows
* CB-8615 **Windows** .winmd files with the same names are not added properly when using framework tag with target attribute
* CB-9297 Parse xcode project syncronously to avoid issues with node v4
* CB-9617 Do not restore plugins after plugin removal.
* CB-9631 Save plugin to config.xml only if installation succeeds
* CB-9601 Fix <framework>.versions support on Windows after semver update
* CB-9617 Fixes incorrect project state after adding/removing plugins
* CB-9560 Issue using plugin restore for plugins with common dependencies 
* CB-8993 Plugin restore ignores search path
* CB-9587 Check if browser platform added properly before creating parser. 
* CB-9604 Fix error adding browser platform with PlatformApi polyfill.
* CB-9597 Initial Implementation of PlatformApiPoly
* CB-9354 Fix array merging with complex items
* CB-9556 Don't uninstall dependent plugin if it was installed as a top-level after

### 5.3.0 (Aug 28, 2015)
* pinned blackberry@3.8.0 in prepartion for its release
* pinned browser@4.0.0 and windows@4.1.0 
* CB-9559 Adding a plugin with caret in version results in an error
* Update cordova-serve required version to 0.1.3.
* CB-6506 RTC: Add support for OSX (closes #278)
* CB-9517 Adding a plugin on iOS/OSX that uses a private framework does not work (closes #281)
* CB-9549 Removes excess JS files from browserified app
* CB-9505 Correct plugin modules loading within browserify flow
* CB-8532 Adding Windows Plugin Failed with "Cannot read property 'text' of null" Updated elementtree API according 0.1.6 release. This closes #277

### 5.2.0 (Aug 06, 2015)
* CB-9436 Removes `require-tr` bundle transformation
* updated pinned ios version to ~3.9.0
* CB-9278: Restoring multiple platforms fails. This closes #266
* updated pinned android to ~4.1.0
* CB-9421 Added a test for plugin fetch with searchpath parameter
* CB-9421 Fixed searchpath parameter being ignored. This closes #269
* Update xcode dependency to latest stable version. This closes #272
* CB-9420 Fixes malformed require calls in browserify bundle. This closes #270
* CB-9405 limit author/description to 256 char per WMAppManifest schema
* CB-9414 plugin fetching now defaults to npm, CPR fallback
* CB-9384 Added tests that test plugin fetch from github branch|tag|sha
* added comment outlining the types of things git_ref can be : commit SHA | branch | tag
* actually checkout git_ref because it may be a branch OR a commit SHA
* CB-9332 Upgrade npm and semver to actual versions
* CB-9330 updated wording for warning messages for removal of publish/unpublish commands
* Adds stubs for `publish`/`unpublish` commands. This closes #254
* CB-9330 Removes 'plugman publish' related functionality
* CB-9335: Windows quality-of-life improvements.  To align with the change in Cordova-Windows which removes the Windows 8 project from the solution file used by Windows 8.1 and Windows 10, the same is done in the spec.
* Fix prepare to wait the promise from plugman prepare.
* CB-9362 Don't fail if superspawn can't chmod a file
* CB-9122 Added tests for platform/plugin add/rm/update with --save flag. This closes #246
* Fixed ios node-xcode related tests failing on Windows according to version update
* Added webOS parsers for project creation/manipulation
* CB-8965 Prevent cli from copying cordova.js and cordova-js-src/ multiple times
* CB-9114: Log deprecation message when --usegit flag is used. This closes #234
* CB-9126 Fix ios pbxproj' resources paths when adding ios platform on non-OSX environment. This closes #237
* CB-9221 Updates `cordova serve` command to use cordova-serve module.
* CB-9225 Add windows platform support to `plugman platform add`
* CB-9163 when engine check isn't satisfied, skip that plugin install
* CB-9162 Adds support for default values for plugin variables.
* CB-9188 Confusing error after delete plugin folder then prepare.
* CB-9145 prepare can lose data during config munge
* CB-9177 Use tilde instead of caret when save to config.xml.
* CB-9147 Adding a platform via caret version adds latest rather than the latest matching.
* CB-5578 Adds `clean` module to cordova. This closes #241
* CB-9124 Makes network-related errors' messages more descriptive.
* CB-9067 fixed plugman config set registry and adduser
* CB-8993 Plugin restore ignores search path. This closes #224
* CB-9087 updated pinned windows platform to 4.0.0
* CB-9108 Handle version ranges when add platform with --usegit.
* CB-8898 Makes error message descriptive when `requirements` is called outside of cordova project.
* CB-8007 Two cordova plugins modifying “*-Info.plist” CFBundleURLTypes
* CB-9065 Allow removing plugins by short name.
* CB-9001 Set WMAppManifest.xml Author, Description and Publisher attributes based on config.xml
* CB-9073 Allow to add platform if project path contains `&` symbol

### 5.1.1 (June 4, 2015)
* CB-9087 Updated pinned version of cordova-windows to 4.0.0
* CB-9108 Handle version ranges when add platform with --usegit.
* CB-8898 Makes error message descriptive when `requirements` is called outside of cordova project.
* Fix four failing tests on Windows.
* CB-8007 Two cordova plugins modifying “*-Info.plist” CFBundleURLTypes
* CB-9065 Allow removing plugins by short name.
* CB-9001 Set WMAppManifest.xml Author, Description and Publisher attributes based on config.xml
* CB-9073 Allow to add platform if project path contains `&` symbol
* CB-8783 - Revert 'all' as a global preference value for Orientation (specific to iOS for now)
* CB-8783 - 'default' value for Orientation does not support both landscape and portrait orientations. (new 'all' value)
* CB-9075 pinned platforms will include patch updates without new tools release
* CB-9051 Plugins don't get re-added if platforms folder deleted.
* CB-9025 Call windows `prepare` logic on as part of cordova-lib `prepare`. This closes #217
* CB-9048 Clean up git cloned directories (close #222)
* CB-8965 readded browserify transform
* CB-8965 copy platform specific js into platform_www when adding new platforms for browserify workflow
* CB-8965 passing platform as argument when getting symbolList
* CB-8965 copy platform specific js into platform_www when adding new platforms for browserify workflow
* Add support to specify a build config file. If none is specified `build.json` in the project root is used as a default This closes #215
* CB-9030: Modifies superspawn to support a "chmod" option. When truthy, attempts to set the target file mode to 755 before executing.  Specifies this argument as truthy for common CLI operations (compile, run, and steps in plugman).  Didn't add it for hooks runner since that particular mode is in legacy support.
* CB-8989 - cordova-lib jasmine tests are failing on older hardware
* CB-6462 CB-6026 - Orientation preference now updates `UISupportedInterfaceOrientations~ipad` too.
* CB-8898 Introduces `requirements` cordova module
* Update elementtree dependency to 0.1.6. Note it has a breaking API change. https://github.com/racker/node-elementtree/issues/24 (closes #209)
* CB-8757 Resolve symlinks in order to avoid relative path issues (close #212)
* CB-8956 Remove hardcoded reference to registry.npmjs.org
* CB-8934 fixed regression with projects config.json not being used in cordova create
* CB-8908 Make fetching via git faster via --depth=1
* CB-8897 Make default icon/splash on Android map to mdpi

### 5.0.0 (Apr 16, 2015)
* CB-8865 fixed plugman.help()
* Pinned Cordova-Android version 4.0.0
* CB-8775 updated warning message to be more descriptive
* Fix getPlatformVersion fails for paths with spaces
* CB-8799 Save plugin/platform src and version to 'spec' attribute.
* CB-8807 Platform Add fails to add plugins with variables.
* CB-8832 Fix iOS icon copying logic to not use default for every size
* Updated pinned versions of windows and wp8
* CB-8775 adding a plugin will still copy it to plugins folder, except if the plugin's new or old id is already installed.
* CB-8775 removed failing test
* Fix setGlobalPreference() in ConfigParser
* removed mostly unused relativePath checking and added missing cases for isAbsolutePath
* use string method for clarity
* CB-8775 new style plugins won't install if same RDS plugin is installed and vice versa
* CB-8791 Recognize UAP as a valid TargetPlatformIdentifier
* CB-8784 Prepare with no platforms should restore all platforms.
* Fix plugman install failure on iOS containing &
* CB-8703: Test failure after merge to head.
* CB-8703: Add support for semver and device-specific targeting of config-file to Windows
* CB-8596 Expose APIs to retrieve platforms and plugins saved in config.xml.
* CB-8741 Make plugin --save work more like npm install
* CB-8755 Plugin --save: Multiple config.xml entries don't get removed
* CB-8754 Auto-restoring a plugin fails when adding a platform.
* CB-8651 Restoring platforms causes plugin install to be triggered twice  (close #196)
* CB-8731 updated app hello world dependency to 3.9.0
* CB-8757 ios: Make paths with --link relative to the real project path (close #192)
* CB-8286 Fix regression from e70432f2: Never want to link to app-hello-world
* CB-8737 Available platforms list includes extraneous values
* Bugfix to json.parse before using cfg
* Add merges/ by default, now all tests pass
* Move cordova-app-hello-world dependency to cordova-lib
* Support the old 4-argument version of create again
* [CB-8286] Update create.js to always require passing in a www
* Show npm failure message when plugin fetch fails
* CB-8725 Fix plugin add from npm when authenticated to CPR
* CB-8499 Remove project_dir from (un)installers signature
* Add addElement() to ConfigParser
* CB-8696 Fix fetching of dependencies with semver constraints rather than exact versions
* CB-7747 Add `<allow-intent>` for App Store on iOS
* Export PlatformProjectAdapter from platforms.js
* Allow subdirs for icons on BB10
* CB-8670 Error when set engine name to "cordova-windows" in plugin.xml
* Allow hyphen in platform name
* CB-8521 Cleans up plugin metadata save method
* CB-8521 Adds `cordova plugin save` which saves all installed plugins to config.xml
* CB-7698 BugFix: For plugins which require variables, 'cordova plugin add FOO' should fail when no variables specified.
* Add setGlobalPreference() to ConfigParser
* CB-8499 Merge platforms.js from cordova and plugman
* rename references to feature to plugin
* Deprecate the old feature syntax from config.xml
* CB-8634 Fixes missed merge/rebase issue
* CB-8634 Adds support for custom branches for `cordova platform add`
* CB-8633 BugFix: Support for urls to tarballs was broken
* CB-8499 `cordova platform save`: save installed platforms and their sources (versions/git_urls/folders) into config.xml
* CB-8499 When deleting a platform, remove it from platforms.json
* CB-8499 When adding a platform, capture version/folder/url being added to allow us to be able to save all installed platforms and their versions later on by doing 'cordova platform save'
* CB-8602 plugman: publish fail early if unsupported npm is active
* CB-7747 Add `<allow-intent>`s to default template
* CB-8616 Support 9-patch images for default android splashscreen
* CB-8551 fixed regex in isValidCprName
* CB-8551 updated version of registry mapper and cordova plugin rm code
* CB-8551 merged fetchNPM and fetchPlugReg into fetchPlugin
* CB-8551 updated regex in isValidCprName to exclude matching @version
* CB-8551 split up changePluginId into two functions
* CB-8457 Ignore version specifier when running hooks (close #165)
* CB-8578 `cordova plugin add <plugin>` should be able to restore urls and folders in addition to versions. (close #173)
* CB-7827 Add support for `android-activityName` within `config.xml` (close #171)
* Add org.apache.cordova.test-framework to plugman publish whitelist
* CB-8577 - Read plugin variables from correct tag
* CB-8555 Incremented package version to -dev
* CB-8551 added plugin-name support for removing plugins.
* CB-8551 Skip CPR if pluginID isn't reverse domain name style
* CB-8551 added npm fetching as fallback

### 4.3.0 (Feb 27, 2015)
* updated pinned versions of ios to 3.8.0 and android to 3.7.1
* CB-8524 Switched to the latest Windows release
* changed createpackage.json keyword to ecosystem:cordova
* CB-8448 add support for activities
* CB-8482 rename: platformId -> platformName
* CB-8482: Update engine syntax within config.xml
* Organize save logic some more
* --save flag for plugins
* fix for test after prepare changes
* restore plugins and platforms on prepare
* CB-8472 Can't find config.xml error installing browser platform after plugin.  (close #167)
* CB-8469 android: Call into platform's build.js after `plugin add` so that Android Studio will work without needing an explicit command-line build first
* CB-8123 Fix JSHINT issue.
* CB-8123 Fix path handling so tests work on any platform.
* CB-8123 Rename further windows platform related files.
* CB-8123 Rename windows platform related files.
* CB-8123 Plugin references can target specific windows platforms.
* CB-8420 Make `cordova plugin add FOO` use version from config.xml (close #162)
* CB-8239 Fix `cordova platform add PATH` when PATH is relative and CWD != project root
* CB-8227 CB8237 CB-8238 Add --save flag and autosave to 'cordova platform add', 'cordova platform remove' and 'cordova platform update'
* CB-8409 compile: bubble failures
* CB-8239 Fix "platform update" should ignore `<cdv:engine>` (close #159)
* CB-8390 android: Make `<framework custom=false>` work with Gradle
* CB-8416 updated plugman publish to temporarily rename existing package.json files
* CB-8416: added `plugman createpackagejson .` command to create a package.json from plugin.xml
* CB-6973 add spec-plugman to npm run jshint
* CB-6973 fix spec-plugman jshint failures
* CB-6973 have base rules in jshintrc for spec-plugman
* CB-8377 Fixed <runs> tag parsing (close #156)
* CB-5696 find ios project directory using the xcode project file (close #151)
* CB-8373 android: Add gradle references to project.properties rather than build.gradle
* CB-8370 Make "plugman publish" without args default to CWD
* Fix publish type-error introduced in recent commit 15adc1b9fcc069438f5
* CB-8366 android: Remove empty `<framework>` directory upon uninstall
* CB-6973 Enable JSHint for spec-cordova
* CB-8239 Add support for git urls to 'cordova platform add' (close #148)
* CB-8358 Add `--link` for `platform add` and `platform update`
* CB-6973 remove base rules from individual files in src
* CB-6973 have base rules in .jshintrc file
* Add shims to undo breaking change in a20b3ae3 (didn't realize PluginInfo was exported)
* CB-8354 Add --link support for iOS source and header files
* Make all ad-hoc plugin.xml parsing use PluginInfo instead
* Make all usages of PluginInfo use PluginInfoProvider instead
* Add PluginInfoProvider for better caching of PluginInfo
* CB-8284 revert npm dependency due to issues with registry
* CB-8223 Expose config.xml in the Browser platform (close #149)
* CB-8168 --list support for cordova-lib (close #145)
* [Amazon] Improve error message when `<source-file>` is missing `target-dir`
* refactor: Make addUninstalledPluginToPrepareQueue take pluginId rather than dirName
* Chnage plugman test plugins to have IDs as directory names
* Make all test plugin IDs unique
* Empty out contents of plugin test files (and delete some unused ones)
* CB-4789 refactor: Remove config_changes.get/set_platform_json in favour of PlatformJson
* CB-8319 Remove config_changes module from plugman's public API
* CB-8314 Speed up Travis CI (close #150)
* refactor: Extract PlatformJson and munge-util into separate modules
* refactor: Move ConfigFile and ConfigKeeper into their own files
* CB-8285 Fix regression caused by c49eaa86c92b (PluginInfo's are cached, don't change them)
* CB-8208 Made CI systems to get cordova-js dependency from gihub (close #146)
* CB-8285 Don't create .fetch.json files within plugin directories
* CB-8286 Never persist value of create --link-to within .cordova/config.json
* CB-8288 Don't set config.setAutoPersist() in cordova.create
* Fix create spec sometimes failing because it's deleted its own tmp directory
* CB-8153 generate cordova_plugins.json for browserify based projects
* CB-8043 CB-6462 CB-6105 Refactor orientation preference support (close #128)
* FirefoxOS parser: allow passing in a ConfigParser object
* Parsers: extend base parser with helper functions
* CB-8244 android: Have `plugin add --link` create symlinks for `<source-file>`, `<framework>`, etc 
* CB-8244 Pass options object to platform handlers in plugman (commit attempt #2)
* CB-8226 'cordova platform add' : Look up version in config.xml if no version specified
* Delete root .npmignore, since there's no node module there

### 4.2.0 (Jan 06, 2015)
* `ConfigParser`: refactor `getPreference()`
* Parsers: add base parser (parser.js) and make platform parsers inherit from it
* Parsers: assign methods without overriding the prototype
* CB-8225 Add Unit Tests for `platform.js/add` function (closes #138)
* CB-8230 Make `project.properties` optional for Android sub-libraries
* CB-8215 Improve error message when `<source-file>` is missing `target-dir` on android
* CB-8217 Fix plugin add --link when plugin given as relative path
* CB-8216 Resolve plugin paths relative to original CWD
* CB-7311 Fix tests on windows for iOS parser
* CB-7803 Allow adding any platform on any host OS (close #126)
* CB-8155 Do not fail plugin installation from git url with --link (close #129)
* Updates README with description of npm commands for this package
* CB-8129 Adds 'npm run cover' command to generate tests coverage report (close #131)
* CB-8114 Specify a cache-min-time for plugins (closes #133)
* CB-8190 Make plugman config/cache directory to be customizable via PLUGMAN_HOME (close #134)
* CB-7863 Fixed broken test run on Windows 8.1 caused by incorrect use of promises (close #132, close #112)
* CB-7610 Fix `cordova plugin add d:\path` (or any other non-c: path) (close #135)
* CB-8179 Corrected latest wp8 version
* CB-8158 added hasModule check to browserify code
* CB-8173 Point to the latest ubuntu version
* CB-8179 Point to the latest wp8 version
* CB-8158 adding symbolList to cordova.js
* CB-8154 Fix errors adding platforms or plugins
* browserify: updated require to use symbollist
* Amazon related changes. Added a type named "gradleReference" in framework according to https://git-wip-us.apache.org/repos/asf?p=cordova-lib.git;a=commit;h=02a96d757acc604610eb403cf11f79513ead4ac5
* CB-7736 Update npm dep to promote qs module to 1.0
* Added a missing "else" keyword.
* CB-8086 Fixed framework tests.
* CB-8086 Added an explanatory comment.
* CB-8086 Prefixed subprojects with package name.
* CB-8067 externalized valid-identifier it is it's own module
* Added identifier checking for app id, searches for java+C# reserved words
* [CB-6472] Adding content to -Info.plist - Unexpected behaviour
* CB-8053: Including a project reference in a plugin fails on Windows platform.
* Pass the searchpath when installing plugins
* Add a type named "gradleReference" in framework

### 4.1.2 (Nov 13, 2014)
* CB-7079 Allow special characters and digits in id when publishing to plugins registry
* CB-7988: Update platform versions for iOS, wp8 & Windows to 3.7.0
* CB-7846 Fix plugin deletion when dependency plugin does not exist
* CB-6992 Fix build issue on iOS when app name contains accented characters
* CB-7890 validate file copy operations in plugman
* CB-7884 moved platform metadata to platformsConfig.json
* Amazon Specific changes: Added support for SdkVersion
* Expose PluginInfo from cordova-lib
* CB-7839 android: Fix versionCode logic when version is less than 3 digits
* CB-7033 Improve cordova platform check
* CB-7311 Fix xcode project manipulation on Windows host
* CB-7820 Make cordova platfrom restore not stop if a platforms fails to restore
* CB-7649 Support iPhone 6 Plus Icon in CLI config.xml
* CB-7647 Support new iPhone 6 and 6 Plus Images in the CLI config.xml
* CB-7909 "plugman platform add" fixes
* Enable platform-specific id for android and ios
* Check for a CORDOVA_HOME environment variable to create a global config path

### 4.0.0 (Oct 10, 2014)
* Bumped version to 4.0.0 to be semVer complient and to match cli version
* Pinned dependencies in package.json
* updated platforms.js for 3.6.4
* CB-5390 Uninstall - recursively remove dependencies of dependencies
* fixes HooksRunner test - should run before_plugin_uninstall
* CB-6481 getPluginsHookScripts to work if plugin platform not defined
* CB-6481 Context opts should copy not reference
* CB-6481 Fixed tests - removed output
* CB-6481 Fixed HooksRunner and tests Avoided issue with parallel tests running Added checks for handling mocked config.xml and package.json in HooksRunner and scriptsFinder Addressed jshint issues Renamed ScriptsFinder to scriptsFinder
* CB-6481 Addressed community review notes: Removed commonModules from Context Renamed Hooker and subclasses to HooksRunner and scriptsFinder Moved scriptsRunner code into HooksRunner
* CB-6481 Replaced CordovaError throwings with Error per @kamrik review Extracted prepareOptions Hooker method
* CB-6481 Docs: deprecated .cordova/hooks + other minor updates
* CB-6481 Updated hooks documentation
* CB-6481 Added unified hooks support for cordova app and plugins
* CB-7572 Serve - respond with 304 when resource not modified
* computeCommitId for browserify workflow fixed to handle cli and non cli workflows:q
* CB-7219 prepare-browserify now supports commitId and platformVersion for cordovajs
* CB-7219: initial work for cordova.js platformVersion
* CB-7219 prepare-browserify now supports commitId and platformVersion for cordovajs
* CB-7219: initial work for cordova.js platformVersion
* CB-7383 Updated version and RELEASENOTES.md for release 0.21.13
* Fix CB-7615 Read config.xml after pre-prepare hooks fire
* CB-7578 Windows. Fix platform name reported by pre_package hook
* CB-7576 Support 'windows' merges folder for Windows platform
* Revert "Merge branch 'browserPlatform' of https://github.com/surajpindoria/cordova-lib"
* Added tests for browser platform

### 0.21.13
* remove shrinkwrap

### 0.21.12
* CB-7383: depend on a newer version of cordova-js, bump self version

### 0.21.11
* bump version numbers of platforms to 3.6.3

### 0.21.10 (Sep 05, 2014)
* CB-7457 - cordova plugin add --searchpath does not recurse through subfolders when a plugin.xml is malformed in one of them
* CB-7457 - Add malformed plugin for tests
* [Windows8] Fix failing test to match updated functionality
* CB-7420 Windows. Plugin <resource-file>s are removed from platform during prepare
* Windows helper. Removes unnecessary $(MSBuildThisFileDirectory)
* updated Releasenotes.md
* updated version to 0.21.10-dev
* CB-7457 - cordova plugin add --searchpath does not recurse through subfolders when a plugin.xml is malformed in one of them
* CB-7457 - Add malformed plugin for tests
* [Windows8] Fix failing test to match updated functionality
* updated Releasenotes.md
* updated version to 0.21.10-dev
* updated version, updated ffos to use 3.6.1, updated cordova-js dependency to be strcit
* CB-7383 Incremented package version to -dev
* updated platforms.js to use 3.6.0
*  Updated version and RELEASENOTES.md for release 0.21.8
* CB-5535: Remove "--arc" from ios platform creation args
* Windows helper. Removes unnecessary $(MSBuildThisFileDirectory)
* CB-7420 Windows. Plugin <resource-file>s are removed from platform during prepare
* CB-7416 Fixes file path reference when adding new source file
* CB-7416 handleInstall tests for null platformTag. removed uncalled 'hasPlatformSection' from PluginInfo.js
* Remove use of path.join for manifest.launch_path
* CB-7347 Improve cordova platform add /path/to handling
* CB-7118 (fix jshint warnings)
* CB-7114 Android: add support of min/max/target SDK to config.xml
* CB-7118 Use updated version of node-xcode
* CB-7118 iOS: add target-device and MinimumOSVersion support to config.xml
* ubuntu: support incremental builds
* ubuntu: support target-dir for resource-file
* ubuntu: use common.copyFile
* ubuntu: check icon existence
* ffos: Make author url optional
* CB-7142 Add <variable> to <feature> for "plugin restore" command
* Set git clone depth to 10 for Travis to make it faster
* windows: update as per changed manifest file names
* Don't spy and expect it to call the other spy ...
* Well that looks like an error
* Fixing failing tests: update_proj should be update_project
* Fix failing tests. update_jsproj and update_csproj are now just update_proj
* Fix jshint errors in amazon_fireos_parser : mixed single/double quotes
* CB-6699 Include files from www folder via single element (use ** glob pattern)
* Taking care of dashes in amazon-fireos platform name.
* Upleveled amazon-fireos changes.
* Fix link/copy parent check for windows
* Style fixes - comments
* Fix error in comments for munge functions
* Add link to BuildBot at ci.cordova.io in README
* CB-7255 Fixed writing plist unescaped
* Allow plugin modules to be .json files
* Style fixes - white space only
* Add JSCS config file
* CB-7260 Get cordova-android 3.5.1 instead of 3.5.0
* CB-7228: Fixed issue with "cordova prepare --browserify"
* CB-7234 added better outputs for plugin registry workflows
* CB-7100: Use npm based lazy-load by default
* CB-7091: Remove check_requirements() funcs from platform parsers
* CB-7091: Remove check_requirements() funcs from platform parsers
* CB-7140 Check plugin versions in local search path
* CB-7001: Create a --browserify option for run action
* CB-7228: Cordova prepare --browserify runs on all installed plugins
* CB-7190: Add browserify support in cordova-lib/cordova-cli
* Remove references to "firefoxos"
* Browser platform is now being created from cli
* Created new files for browser

### 0.21.8 (Aug 29, 2014)
* CB-5535: Remove "--arc" from ios platform creation args
* CB-7416 Fixes file path reference when adding new source file
* CB-7416 handleInstall tests for null platformTag. removed uncalled 'hasPlatformSection' from PluginInfo.js
* Remove use of path.join for manifest.launch_path
* CB-7347 Improve cordova platform add /path/to handling
* CB-7118 (fix jshint warnings)
* CB-7114 Android: add support of min/max/target SDK to config.xml
* CB-7118 Use updated version of node-xcode
* CB-7118 iOS: add target-device and MinimumOSVersion support to config.xml
* ubuntu: support incremental builds
* ubuntu: support target-dir for resource-file
* ubuntu: use common.copyFile
* ubuntu: check icon existence
* ffos: Make author url optional
* CB-7142 Add <variable> to <feature> for "plugin restore" command
* Set git clone depth to 10 for Travis to make it faster
* windows: update as per changed manifest file names
* Don't spy and expect it to call the other spy ...
* Well that looks like an error
* Fixing failing tests: update_proj should be update_project
* Fix failing tests. update_jsproj and update_csproj are now just update_proj
* Fix jshint errors in amazon_fireos_parser : mixed single/double quotes
* CB-6699 Include files from www folder via single element (use ** glob pattern)
* Allow plugin modules to be .json files
* Taking care of dashes in amazon-fireos platform name.
* Upleveled amazon-fireos changes.
* Fix link/copy parent check for windows
* Style fixes - comments
* Fix error in comments for munge functions
* Add link to BuildBot at ci.cordova.io in README
* CB-7255 Fixed writing plist unescaped
* Style fixes - white space only
* Add JSCS config file
* CB-7228: Fixed issue with "cordova prepare --browserify"
* CB-7001: Create a --browserify option for run action
* CB-7228: Cordova prepare --browserify runs on all installed plugins
* CB-7190: Add browserify support in cordova-lib/cordova-cli
* CB-7260 Get cordova-android 3.5.1 instead of 3.5.0
* CB-7001: Create a --browserify option for run action
* CB-7228: Cordova prepare --browserify runs on all installed plugins
* CB-7190: Add browserify support in cordova-lib/cordova-cli
* CB-7234 added better outputs for plugin registry workflows
* CB-7100: Use npm based lazy-load by default
* CB-7091: Remove check_requirements() funcs from platform parsers
* CB-7091: Remove check_requirements() funcs from platform parsers
* CB-7140 Check plugin versions in local search path
* small refactor for missing code block after conditional statement
* CB-7203 isRelativePath needs to pass path through
* CB-7199 control git/npm using platform.js
* CB-7199 control git/npm using platform.js
* Fix style errors - make jshint happy
* CB-6756 Adds save and restore command for platforms.
* Add VERSION files to fix failing tests (forgot to git add in b7781cb)
* CB-7132 Fix regression regarding default resources
* CB-7187 Make CoreLocation a required library only for cordova-ios < 3.6.0
* Add AppVeyor badge to README
* Add Travis and npm badges to README.md
* fix(tests): cordova/lazy_load spec on Windows
* Fix plugman/install spec
* build configuration for AppVeyor
* build configurations for Travis
* CB-7124 Wrap the cordova platform string in Platform object
* CB-7140: Switch to using PluginInfo in plugman/fetch.js
* Minor style fixes in fetch.js
* CB-7078: Disable serve.spec.js
* CB-6512: platform add <path> was using wrong www/cordova.js
* CB-7083 Missing SDKReference support on Windows Phone
* CB-6874 Consolidate <Content> tag additions into 1 ItemGroup
* CB-7100: Use npm based lazy-load by default
* CB-7091: Remove check_requirements() funcs from platform parsers
* CB-7091: Don't call check_requirements during platform add
* Fix typo in comment.
* CB-7087 Retire blackberry10/ directory
* CB-6776: Fix uri/url renaming bug
* Remove npm-shrinkwrap.json


### 0.21.4 (Jun 23, 2014)
* CB-3571, CB-2606: support for splashscreens
* CB-6976 Add support for Windows Universal apps (Windows 8.1 and WP 8.1)
* Use Plugininfo module to determine plugin id and version
* Fix plugin check error, when plugin dependency with specific version is given
* CB-6709 Do not create merges/ folder when adding a platform
* CB-6140 Don't allow deletion of platform dependencies
* CB-6698: Fix 'android update lib-project' to work with paths containing spaces
* CB-6973: Run JSHint on all code in src/ via npm test
* CB-6542: Delay creating project until there's some chance that it will succeed
* folder_contents() now ignores .svn folders
* CB-6970 Share win project files manipulation code between cordova and plugman
* CB-6954: Share events.js between cordova and plugman
* CB-6698 Automatically copy sub-libraries to project's directory
* Revert "CB-6698 Resolve android <framework> relative to plugin_dir when custom=true"
* CB-6942 Describe running hooks only in verbose mode.
* CB-6512: Allow "cordova platform add /path/to/platform/files"
* Update hooks-README.md - shebang line in hooks on Windows.
* CB-6895 Add more config properties into manifest
* Allow "cordova platform add platform@version"
* Add util func for chaining promises
* removing doWrap from prepare
* adding configurable attribute
* cleaning up plugman.js for uninstall
* adding param to uninstall
* adding support for prepare flag
* adding prepare-browserify
* adding options to prepare
* adding and freezing cordova-js
* [CB-6879] config parser breakout into a cordova level module
* CB-6698 Resolve android <framework> relative to plugin_dir when custom=true
* Fix tests on node 0.11.x
* Fix android <framework> unit tests to not expect end of line.
* CB-6024: Accept cli vars as part of opts param
* Refer properties-parser package from NPM.
* CB-6859 Removed all wp7 references, tests still passing
* Extract AndroidProject class into a separate .js file
* CB-6698: Support library references for Android via the framework tag
* CB-6854 Strip BOM when adding cordova.define() to js-modules
* Add npm cache based downloading to lazy_load
* Use PluginInfo in plugman/install.js
* Extend PluginInfo to parse more of plugin.xml
* CB-6772 Provide a default for AndroidLaunchMode
* CB-6711: Use parseProjectFile when working with XCode projects.
* Start using PluginInfo object in plugman/install.js
* CB-6709 Remove merges/ folder for default apps
* support for shrinkwrap flag
* Initial implementation for restore and save plugin
* CB-6668: Use <description> for "plugin ls" when <name> is missing.
* Add --noregstry flag for disabling plugin lookup in the registry
* Remove --force from default npm settings for plugin registry
* Use "npm info" for fetching plugin metadata
* Use "npm cache add" for downloading plugins
* CB-6691: Change some instances of Error() to CordovaError()


### 0.21.1
Initial release v0.21.1 (picks up from the same version number as plugman was).
