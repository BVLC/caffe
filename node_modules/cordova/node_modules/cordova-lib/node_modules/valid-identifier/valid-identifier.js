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

var reserved = [
    "abstract","as","base","bool ","break","byte","case","catch","char",
    "checked","class","const","continue","decimal","default","delegate",
    "do","double","else","enum","event","explicit","extern","false","finally",
    "fixed","float","for","foreach","goto","if","implicit","in","int",
    "interface","internal","is","lock","long","namespace","new","null",
    "object","operator","out","override","params","private","protected",
    "public","readonly","ref","return","sbyte","sealed","short","sizeof",
    "stackalloc","static","string","struct","switch","this","throw","true",
    "try","typeof","uint","ulong","unchecked","unsafe","ushort","using",
    "virtual","void","volatile","while","assert","package","synchronized",
    "boolean","implements","import"];

var regX = /([a-zA-Z_$][a-zA-Z\d_$]*\.)*[a-zA-Z_$][a-zA-Z\d_$]*/;

module.exports = function (nsPackage) {
    var match = nsPackage.match(regX);
    var isValid = match ? match[0] == nsPackage : false;
    if(isValid) {
        nsPackage.split(".").every(function(val){
            if(reserved.indexOf(val) > -1) {
                isValid = false;
            }
            return isValid;
        });
    }
    return isValid;
};
