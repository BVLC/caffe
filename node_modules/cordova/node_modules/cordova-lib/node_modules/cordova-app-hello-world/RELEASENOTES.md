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
# Release Notes

### 3.10.0 (Oct 26, 2015)
* CB-9712 CLI 5.3 breaks with node 3.3.3
* don't duplicate hooks guide
* CB-8896 updated whitelist dependency to use spec

### 3.9.0 (Mar 23, 2015)
* CB-8716 Add Whitelist plugin to config.xml

### 3.8.0 (Mar 10, 2015)
* Add unsafe-eval to default CSP since many frameworks run faster with it on
* Remove path from CSP string (since CSP ignores paths). Add CSP rationale within commen 
* CB-8295 Fix CSP string, which had an invalid : in it
* added license header to config.xml
* added releasenotes.md
* removed version file, package.json can keep track of version
* Adding hooks to default app
* [CB-8597] First attempt at importing cordova-app-hello-world via npm dependency
* Remove jasmine ref
* Tweak CSP string
* CB-8295 Add content-security-policy `<meta>` to template
* Remove target-density and height=device-height from `<viewport>`
* Remove self-closing slashes from `<meta>` since this isn't xhtml
