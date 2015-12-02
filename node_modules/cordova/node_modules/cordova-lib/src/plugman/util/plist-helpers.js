/*
 *
 * Copyright 2013 Brett Rudd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
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

// contains PLIST utility functions
var __     = require('underscore');
var plist = require('plist');

// adds node to doc at selector
module.exports.graftPLIST = graftPLIST;
function graftPLIST(doc, xml, selector) {
    var obj = plist.parse('<plist>'+xml+'</plist>');

    var node = doc[selector];
    if (node && Array.isArray(node) && Array.isArray(obj)){
        node = node.concat(obj);
        for (var i =0;i<node.length; i++){
            for (var j=i+1; j<node.length; ++j) {
              if (nodeEqual(node[i], node[j]))
                    node.splice(j--,1);
            }
        }
        doc[selector] = node;
    } else {
        //plist uses objects for <dict>. If we have two dicts we merge them instead of
        // overriding the old one. See CB-6472
        if (node && __.isObject(node) && __.isObject(obj) && !__.isDate(node) && !__.isDate(obj)){//arrays checked above
            __.extend(obj,node);
        }
        doc[selector] = obj;
    }

    return true;
}

// removes node from doc at selector
module.exports.prunePLIST = prunePLIST;
function prunePLIST(doc, xml, selector) {
    var obj = plist.parse('<plist>'+xml+'</plist>');

    pruneOBJECT(doc, selector, obj);

    return true;
}

function pruneOBJECT(doc, selector, fragment) {
    if (Array.isArray(fragment) && Array.isArray(doc[selector])) {
        var empty = true;
        for (var i in fragment) {
            for (var j in doc[selector]) {
                empty = pruneOBJECT(doc[selector], j, fragment[i]) && empty;
            }
        }
        if (empty)
        {
            delete doc[selector];
            return true;
        }
    }
    else if (nodeEqual(doc[selector], fragment)) {
        delete doc[selector];
        return true;
    }

    return false;
}

function nodeEqual(node1, node2) {
    if (typeof node1 != typeof node2)
        return false;
    else if (typeof node1 == 'string') {
        node2 = escapeRE(node2).replace(new RegExp('\\$[a-zA-Z0-9-_]+','gm'),'(.*?)');
        return new RegExp('^' + node2 + '$').test(node1);
    }
    else {
        for (var key in node2) {
            if (!nodeEqual(node1[key], node2[key])) return false;
        }
        return true;
    }
}

// escape string for use in regex
function escapeRE(str) {
    return str.replace(/[\-\[\]\/\{\}\(\)\*\+\?\.\\\^\$\|]/g, '$&');
}
