var pbxProj = require('./pbxProject'),
    util = require('util'),
    f = util.format,
    INDENT = '\t',
    COMMENT_KEY = /_comment$/,
    QUOTED = /^"(.*)"$/,
    EventEmitter = require('events').EventEmitter

// indentation
function i(x) {
    if (x <=0)
        return '';
    else
        return INDENT + i(x-1);
}

function comment(key, parent) {
    var text = parent[key + '_comment'];

    if (text)
        return text;
    else
        return null;
}

// copied from underscore
function isObject(obj) {
    return obj === Object(obj)
}

function isArray(obj) {
    return Array.isArray(obj)
}

function pbxWriter(contents) {
    this.contents = contents;
    this.sync = false;
    this.indentLevel = 0;
}

util.inherits(pbxWriter, EventEmitter);

pbxWriter.prototype.write = function (str) {
    var fmt = f.apply(null, arguments);

    if (this.sync) {
        this.buffer += f("%s%s", i(this.indentLevel), fmt);
    } else {
        // do stream write
    }
}

pbxWriter.prototype.writeFlush = function (str) {
    var oldIndent = this.indentLevel;

    this.indentLevel = 0;

    this.write.apply(this, arguments)

    this.indentLevel = oldIndent;
}

pbxWriter.prototype.writeSync = function () {
    this.sync = true;
    this.buffer = "";

    this.writeHeadComment();
    this.writeProject();

    return this.buffer;
}

pbxWriter.prototype.writeHeadComment = function () {
    if (this.contents.headComment) {
        this.write("// %s\n", this.contents.headComment)
    }
}

pbxWriter.prototype.writeProject = function () {
    var proj = this.contents.project,
        key, cmt, obj;

    this.write("{\n")

    if (proj) {
        this.indentLevel++;

        for (key in proj) {
            // skip comments
            if (COMMENT_KEY.test(key)) continue;

            cmt = comment(key, proj);
            obj = proj[key];

            if (isArray(obj)) {
                this.writeArray(obj, key)
            } else if (isObject(obj)) {
                this.write("%s = {\n", key);
                this.indentLevel++;

                if (key === 'objects') {
                    this.writeObjectsSections(obj)
                } else {
                    this.writeObject(obj)
                }

                this.indentLevel--;
                this.write("};\n");
            } else if (cmt) {
                this.write("%s = %s /* %s */;\n", key, obj, cmt)
            } else {
                this.write("%s = %s;\n", key, obj)
            }
        }

        this.indentLevel--;
    }

    this.write("}\n")
}

pbxWriter.prototype.writeObject = function (object) {
    var key, obj, cmt;

    for (key in object) {
        if (COMMENT_KEY.test(key)) continue;

        cmt = comment(key, object);
        obj = object[key];

        if (isArray(obj)) {
            this.writeArray(obj, key)
        } else if (isObject(obj)) {
            this.write("%s = {\n", key);
            this.indentLevel++;

            this.writeObject(obj)

            this.indentLevel--;
            this.write("};\n");
        } else {
            if (cmt) {
                this.write("%s = %s /* %s */;\n", key, obj, cmt)
            } else {
                this.write("%s = %s;\n", key, obj)
            }
        }
    }
}

pbxWriter.prototype.writeObjectsSections = function (objects) {
    var first = true,
        key, obj;

    for (key in objects) {
        if (!first) {
            this.writeFlush("\n")
        } else {
            first = false;
        }

        obj = objects[key];

        if (isObject(obj)) {
            this.writeSectionComment(key, true);

            this.writeSection(obj);

            this.writeSectionComment(key, false);
        }
    }
}

pbxWriter.prototype.writeArray = function (arr, name) {
    var i, entry;

    this.write("%s = (\n", name);
    this.indentLevel++;

    for (i=0; i < arr.length; i++) {
        entry = arr[i]

        if (entry.value && entry.comment) {
            this.write('%s /* %s */,\n', entry.value, entry.comment);
        } else if (isObject(entry)) {
            this.write('{\n');
            this.indentLevel++;
            
            this.writeObject(entry);

            this.indentLevel--;
            this.write('},\n');
        } else {
            this.write('%s,\n', entry);
        }
    }

    this.indentLevel--;
    this.write(");\n");
}

pbxWriter.prototype.writeSectionComment = function (name, begin) {
    if (begin) {
        this.writeFlush("/* Begin %s section */\n", name)
    } else { // end
        this.writeFlush("/* End %s section */\n", name)
    }
}

pbxWriter.prototype.writeSection = function (section) {
    var key, obj, cmt;

    // section should only contain objects
    for (key in section) {
        if (COMMENT_KEY.test(key)) continue;

        cmt = comment(key, section);
        obj = section[key]

        if (obj.isa == 'PBXBuildFile' || obj.isa == 'PBXFileReference') {
            this.writeInlineObject(key, cmt, obj);
        } else {
            if (cmt) {
                this.write("%s /* %s */ = {\n", key, cmt);
            } else {
                this.write("%s = {\n", key);
            }

            this.indentLevel++

            this.writeObject(obj)

            this.indentLevel--
            this.write("};\n");
        }
    }
}

pbxWriter.prototype.writeInlineObject = function (n, d, r) {
    var output = [];

    var inlineObjectHelper = function (name, desc, ref) {
        var key, cmt, obj;

        if (desc) {
            output.push(f("%s /* %s */ = {", name, desc));
        } else {
            output.push(f("%s = {", name));
        }

        for (key in ref) {
            if (COMMENT_KEY.test(key)) continue;

            cmt = comment(key, ref);
            obj = ref[key];

            if (isArray(obj)) {
                output.push(f("%s = (", key));
                
                for (var i=0; i < obj.length; i++) {
                    output.push(f("%s, ", obj[i]))
                }

                output.push("); ");
            } else if (isObject(obj)) {
                inlineObjectHelper(key, cmt, obj)
            } else if (cmt) {
                output.push(f("%s = %s /* %s */; ", key, obj, cmt))
            } else {
                output.push(f("%s = %s; ", key, obj))
            }
        }

        output.push("}; ");
    }

    inlineObjectHelper(n, d, r);

    this.write("%s\n", output.join('').trim());
}

module.exports = pbxWriter;
