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

# cordova-lib
Contains npm modules used primarily by [cordova](https://github.com/apache/cordova-cli/) and [plugman](https://github.com/apache/cordova-plugman/).

## Setup
* Clone this repository onto your local machine. 
    `git clone https://git-wip-us.apache.org/repos/asf/cordova-lib.git`
* In terminal, navigate to the inner cordova-lib directory.
    `cd cordova-lib/cordova-lib`
* Install dependencies and npm-link
    `npm install && npm link`
* Navigate to CLI and Plugman directories and link cordova-lib
    `cd ../../cordova-cli && npm link cordova-lib && npm install`
    `cd ../../cordova-plugman && npm link cordova-lib && npm install`
