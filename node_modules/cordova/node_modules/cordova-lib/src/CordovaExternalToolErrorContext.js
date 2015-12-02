/**
 Licensed to the Apache Software Foundation (ASF) under one
 or more contributor license agreements.  See the NOTICE file
 distributed with this work for additional information
 regarding copyright ownership.  The ASF licenses this file
 to you under the Apache License, Version 2.0 (the
 "License"); you may not use this file except in compliance
 with the License.  You may obtain a copy of the License at

 http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing,
 software distributed under the License is distributed on an
 "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 KIND, either express or implied.  See the License for the
 specific language governing permissions and limitations
 under the License.
 */

/* jshint proto:true */

var path = require('path');

/**
 * @param {String} cmd Command full path
 * @param {String[]} args Command args
 * @param {String} [cwd] Command working directory
 * @constructor
 */
function CordovaExternalToolErrorContext(cmd, args, cwd) {
    this.cmd = cmd;
    // Helper field for readability
    this.cmdShortName = path.basename(cmd);
    this.args = args;
    this.cwd = cwd;
}

CordovaExternalToolErrorContext.prototype.toString = function(isVerbose) {
    if(isVerbose) {
        return 'External tool \'' + this.cmdShortName + '\'' +
            '\nCommand full path: ' + this.cmd + '\nCommand args: ' + this.args +
            (typeof this.cwd !== 'undefined' ? '\nCommand cwd: ' + this.cwd : '');
    }

    return this.cmdShortName;
};

module.exports = CordovaExternalToolErrorContext;
