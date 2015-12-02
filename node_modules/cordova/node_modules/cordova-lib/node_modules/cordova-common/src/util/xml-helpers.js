/*
 *
 * Copyright 2013 Anis Kadri
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

/* jshint sub:true, laxcomma:true */

/**
 * contains XML utility functions, some of which are specific to elementtree
 */

var fs = require('fs')
  , path = require('path')
  , _ = require('underscore')
  , et = require('elementtree')
  ;

module.exports = {
    // compare two et.XML nodes, see if they match
    // compares tagName, text, attributes and children (recursively)
    equalNodes: function(one, two) {
        if (one.tag != two.tag) {
            return false;
        } else if (one.text.trim() != two.text.trim()) {
            return false;
        } else if (one._children.length != two._children.length) {
            return false;
        }

        var oneAttribKeys = Object.keys(one.attrib),
            twoAttribKeys = Object.keys(two.attrib),
            i = 0, attribName;

        if (oneAttribKeys.length != twoAttribKeys.length) {
            return false;
        }

        for (i; i < oneAttribKeys.length; i++) {
            attribName = oneAttribKeys[i];

            if (one.attrib[attribName] != two.attrib[attribName]) {
                return false;
            }
        }

        for (i; i < one._children.length; i++) {
            if (!module.exports.equalNodes(one._children[i], two._children[i])) {
                return false;
            }
        }

        return true;
    },

    // adds node to doc at selector, creating parent if it doesn't exist
    graftXML: function(doc, nodes, selector, after) {
        var parent = resolveParent(doc, selector);
        if (!parent) {
            //Try to create the parent recursively if necessary
            try {
                var parentToCreate = et.XML('<' + path.basename(selector) + '>'),
                    parentSelector = path.dirname(selector);

                this.graftXML(doc, [parentToCreate], parentSelector);
            } catch (e) {
                return false;
            }
            parent = resolveParent(doc, selector);
            if (!parent) return false;
        }

        nodes.forEach(function (node) {
            // check if child is unique first
            if (uniqueChild(node, parent)) {
                var children = parent.getchildren();
                var insertIdx = after ? findInsertIdx(children, after) : children.length;

                //TODO: replace with parent.insert after the bug in ElementTree is fixed
                parent.getchildren().splice(insertIdx, 0, node);
            }
        });

        return true;
    },

    // removes node from doc at selector
    pruneXML: function(doc, nodes, selector) {
        var parent = resolveParent(doc, selector);
        if (!parent) return false;

        nodes.forEach(function (node) {
            var matchingKid = null;
            if ((matchingKid = findChild(node, parent)) !== null) {
                // stupid elementtree takes an index argument it doesn't use
                // and does not conform to the python lib
                parent.remove(matchingKid);
            }
        });

        return true;
    },

    parseElementtreeSync: function (filename) {
        var contents = fs.readFileSync(filename, 'utf-8');
        if(contents) {
            //Windows is the BOM. Skip the Byte Order Mark.
            contents = contents.substring(contents.indexOf('<'));
        }
        return new et.ElementTree(et.XML(contents));
    }
};

function findChild(node, parent) {
    var matchingKids = parent.findall(node.tag)
      , i, j;

    for (i = 0, j = matchingKids.length ; i < j ; i++) {
        if (module.exports.equalNodes(node, matchingKids[i])) {
            return matchingKids[i];
        }
    }
    return null;
}

function uniqueChild(node, parent) {
    var matchingKids = parent.findall(node.tag)
      , i = 0;

    if (matchingKids.length === 0) {
        return true;
    } else  {
        for (i; i < matchingKids.length; i++) {
            if (module.exports.equalNodes(node, matchingKids[i])) {
                return false;
            }
        }
        return true;
    }
}

var ROOT = /^\/([^\/]*)/,
    ABSOLUTE = /^\/([^\/]*)\/(.*)/;

function resolveParent(doc, selector) {
    var parent, tagName, subSelector;

    // handle absolute selector (which elementtree doesn't like)
    if (ROOT.test(selector)) {
        tagName = selector.match(ROOT)[1];
        // test for wildcard "any-tag" root selector
        if (tagName == '*' || tagName === doc._root.tag) {
            parent = doc._root;

            // could be an absolute path, but not selecting the root
            if (ABSOLUTE.test(selector)) {
                subSelector = selector.match(ABSOLUTE)[2];
                parent = parent.find(subSelector);
            }
        } else {
            return false;
        }
    } else {
        parent = doc.find(selector);
    }
    return parent;
}

// Find the index at which to insert an entry. After is a ;-separated priority list
// of tags after which the insertion should be made. E.g. If we need to
// insert an element C, and the rule is that the order of children has to be
// As, Bs, Cs. After will be equal to "C;B;A".
function findInsertIdx(children, after) {
    var childrenTags = children.map(function(child) { return child.tag; });
    var afters = after.split(';');
    var afterIndexes = afters.map(function(current) { return childrenTags.lastIndexOf(current); });
    var foundIndex = _.find(afterIndexes, function(index) { return index != -1; });

    //add to the beginning if no matching nodes are found
    return typeof foundIndex === 'undefined' ? 0 : foundIndex+1;
}

var BLACKLIST = ['platform', 'feature','plugin','engine'];
var SINGLETONS = ['content', 'author'];
function mergeXml(src, dest, platform, clobber) {
    // Do nothing for blacklisted tags.
    if (BLACKLIST.indexOf(src.tag) != -1) return;

    //Handle attributes
    Object.getOwnPropertyNames(src.attrib).forEach(function (attribute) {
        if (clobber || !dest.attrib[attribute]) {
            dest.attrib[attribute] = src.attrib[attribute];
        }
    });
    //Handle text
    if (src.text && (clobber || !dest.text)) {
        dest.text = src.text;
    }
    //Handle platform
    if (platform) {
        src.findall('platform[@name="' + platform + '"]').forEach(function (platformElement) {
            platformElement.getchildren().forEach(mergeChild);
        });
    }

    //Handle children
    src.getchildren().forEach(mergeChild);

    function mergeChild (srcChild) {
        var srcTag = srcChild.tag,
            destChild = new et.Element(srcTag),
            foundChild,
            query = srcTag + '',
            shouldMerge = true;

        if (BLACKLIST.indexOf(srcTag) === -1) {
            if (SINGLETONS.indexOf(srcTag) !== -1) {
                foundChild = dest.find(query);
                if (foundChild) {
                    destChild = foundChild;
                    dest.remove(destChild);
                }
            } else {
                //Check for an exact match and if you find one don't add
                Object.getOwnPropertyNames(srcChild.attrib).forEach(function (attribute) {
                    query += '[@' + attribute + '="' + srcChild.attrib[attribute] + '"]';
                });
                var foundChildren = dest.findall(query);
                for(var i = 0; i < foundChildren.length; i++) {
                    foundChild = foundChildren[i];
                    if (foundChild && textMatch(srcChild, foundChild) && (Object.keys(srcChild.attrib).length==Object.keys(foundChild.attrib).length)) {
                        destChild = foundChild;
                        dest.remove(destChild);
                        shouldMerge = false;
                        break;
                    }
                }
            }

            mergeXml(srcChild, destChild, platform, clobber && shouldMerge);
            dest.append(destChild);
        }
    }
}

// Expose for testing.
module.exports.mergeXml = mergeXml;

function textMatch(elm1, elm2) {
    var text1 = elm1.text ? elm1.text.replace(/\s+/, '') : '',
        text2 = elm2.text ? elm2.text.replace(/\s+/, '') : '';
    return (text1 === '' || text1 === text2);
}
