var assert = require("assert");
var getFieldValue = require("./types").getFieldValue;
var sourceMap = require("source-map");
var SourceMapConsumer = sourceMap.SourceMapConsumer;
var SourceMapGenerator = sourceMap.SourceMapGenerator;
var hasOwn = Object.prototype.hasOwnProperty;

function getUnionOfKeys() {
    var result = {};
    var argc = arguments.length;
    for (var i = 0; i < argc; ++i) {
        var keys = Object.keys(arguments[i]);
        var keyCount = keys.length;
        for (var j = 0; j < keyCount; ++j) {
            result[keys[j]] = true;
        }
    }
    return result;
}
exports.getUnionOfKeys = getUnionOfKeys;

function comparePos(pos1, pos2) {
    return (pos1.line - pos2.line) || (pos1.column - pos2.column);
}
exports.comparePos = comparePos;

exports.composeSourceMaps = function(formerMap, latterMap) {
    if (formerMap) {
        if (!latterMap) {
            return formerMap;
        }
    } else {
        return latterMap || null;
    }

    var smcFormer = new SourceMapConsumer(formerMap);
    var smcLatter = new SourceMapConsumer(latterMap);
    var smg = new SourceMapGenerator({
        file: latterMap.file,
        sourceRoot: latterMap.sourceRoot
    });

    var sourcesToContents = {};

    smcLatter.eachMapping(function(mapping) {
        var origPos = smcFormer.originalPositionFor({
            line: mapping.originalLine,
            column: mapping.originalColumn
        });

        var sourceName = origPos.source;
        if (sourceName === null) {
            return;
        }

        smg.addMapping({
            source: sourceName,
            original: {
                line: origPos.line,
                column: origPos.column
            },
            generated: {
                line: mapping.generatedLine,
                column: mapping.generatedColumn
            },
            name: mapping.name
        });

        var sourceContent = smcFormer.sourceContentFor(sourceName);
        if (sourceContent && !hasOwn.call(sourcesToContents, sourceName)) {
            sourcesToContents[sourceName] = sourceContent;
            smg.setSourceContent(sourceName, sourceContent);
        }
    });

    return smg.toJSON();
};
