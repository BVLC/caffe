/*
 *
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 *
*/

var path        = require('path');
var util        = require('util');
var fs          = require('fs');
var licensePath = path.join(__dirname, '..', 'templates', 'LICENSE-for-js-file.txt');

module.exports = function(outStream, platform, commitId, platformVersion, symbolList) {
  // some poppycock 
  var licenseText = util.format("/*\n *%s\n */\n", fs.readFileSync(licensePath, 'utf8').replace(/\n/g, "\n *"));

  outStream.write("// Platform: " + platform + "\n", 'utf8');
  outStream.write("// "  + commitId + "\n", 'utf8');
  outStream.write("// browserify" + "\n", 'utf8');
  outStream.write(licenseText, 'utf8');
  outStream.write("var PLATFORM_VERSION_BUILD_LABEL = '"  + platformVersion + "';\n", 'utf8');
  outStream.write("var define = {moduleMap: []};\n", 'utf8');
  //outStream.write(util.format("var symbolList = %s", JSON.stringify(symbolList)), 'utf8');

}
