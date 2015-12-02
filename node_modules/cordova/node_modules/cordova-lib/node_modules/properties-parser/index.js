var fs = require("fs");

function Iterator(text) {
	var pos = 0, length = text.length;

	this.peek = function(num) {
		num = num || 0;
		if(pos + num >= length) { return null; }

		return text.charAt(pos + num);
	};
	this.next = function(inc) {
		inc = inc || 1;

		if(pos >= length) { return null; }

		return text.charAt((pos += inc) - inc);
	};
	this.pos = function() {
		return pos;
	};
}

var rWhitespace = /\s/;
function isWhitespace(chr) {
	return rWhitespace.test(chr);
}
function consumeWhiteSpace(iter) {
	var start = iter.pos();

	while(isWhitespace(iter.peek())) { iter.next(); }

	return { type: "whitespace", start: start, end: iter.pos() };
}

function startsComment(chr) {
	return chr === "!" || chr === "#";
}
function isEOL(chr) {
	return chr == null || chr === "\n" || chr === "\r";
}
function consumeComment(iter) {
	var start = iter.pos();

	while(!isEOL(iter.peek())) { iter.next(); }

	return { type: "comment", start: start, end: iter.pos() };
}

function startsKeyVal(chr) {
	return !isWhitespace(chr) && !startsComment(chr);
}
function startsSeparator(chr) {
	return chr === "=" || chr === ":" || isWhitespace(chr);
}
function startsEscapedVal(chr) {
	return chr === "\\";
}
function consumeEscapedVal(iter) {
	var start = iter.pos();
	
	iter.next(); // move past "\"
	var curChar = iter.next();
	if(curChar === "u") { // encoded unicode char
		iter.next(4); // Read in the 4 hex values
	}

	return { type: "escaped-value", start: start, end: iter.pos() };
}
function consumeKey(iter) {
	var start = iter.pos(), children = [];

	var curChar;
	while((curChar = iter.peek()) !== null) {
		if(startsSeparator(curChar)) { break; }
		if(startsEscapedVal(curChar)) { children.push(consumeEscapedVal(iter)); continue; }

		iter.next();
	}

	return { type: "key", start: start, end: iter.pos(), children: children };
}
function consumeKeyValSeparator(iter) {
	var start = iter.pos();

	var seenHardSep = false, curChar;
	while((curChar = iter.peek()) !== null) {
		if(isEOL(curChar)) { break; }

		if(isWhitespace(curChar)) { iter.next(); continue; }

		if(seenHardSep) { break; }

		seenHardSep = (curChar === ":" || curChar === "=");
		if(seenHardSep) { iter.next(); continue; }

		break; // curChar is a non-separtor char
	}

	return { type: "key-value-separator", start: start, end: iter.pos() };
}
function startsLineBreak(iter) {
	return iter.peek() === "\\" && isEOL(iter.peek(1));
}
function consumeLineBreak(iter) {
	var start = iter.pos();

	iter.next(); // consume \
	if(iter.peek() === "\r") { iter.next(); }
	iter.next(); // consume \n

	var curChar;
	while((curChar = iter.peek()) !== null) {
		if(isEOL(curChar)) { break; }
		if(!isWhitespace(curChar)) { break; }

		iter.next();
	}

	return { type: "line-break", start: start, end: iter.pos() };
}
function consumeVal(iter) {
	var start = iter.pos(), children = [];

	var curChar;
	while((curChar = iter.peek()) !== null) {
		if(startsLineBreak(iter)) { children.push(consumeLineBreak(iter)); continue; }
		if(startsEscapedVal(curChar)) { children.push(consumeEscapedVal(iter)); continue; }
		if(isEOL(curChar)) { break; }

		iter.next();
	}

	return { type: "value", start: start, end: iter.pos(), children: children };
}
function consumeKeyVal(iter) {
	return {
		type: "key-value",
		start: iter.pos(),
		children: [
			consumeKey(iter),
			consumeKeyValSeparator(iter),
			consumeVal(iter)
		],
		end: iter.pos()
	};
}

var renderChild = {
	"escaped-value": function(child, text) {
		var type = text.charAt(child.start + 1);

		if(type === "t") { return "\t"; }
		if(type === "r") { return "\r"; }
		if(type === "n") { return "\n"; }
		if(type === "f") { return "\f"; }
		if(type !== "u") { return type; }

		return String.fromCharCode(parseInt(text.substr(child.start + 2, 4), 16));
	},
	"line-break": function (child, text) {
		return "";
	}
};
function rangeToBuffer(range, text) {
	var start = range.start, buffer = [];

	for(var i = 0; i < range.children.length; i++) {
		var child = range.children[i];

		buffer.push(text.substring(start, child.start));
		buffer.push(renderChild[child.type](child, text));
		start = child.end;
	}
	buffer.push(text.substring(start, range.end));

	return buffer;
}
function rangesToObject(ranges, text) {
	var obj = Object.create(null); // Creates to a true hash map

	for(var i = 0; i < ranges.length; i++) {
		var range = ranges[i];

		if(range.type !== "key-value") { continue; }

		var key = rangeToBuffer(range.children[0], text).join("");
		var val = rangeToBuffer(range.children[2], text).join("");
		obj[key] = val;
	}

	return obj;
}

function stringToRanges(text) {
	var iter = new Iterator(text), ranges = [];

	var curChar;
	while((curChar = iter.peek()) !== null) {
		if(isWhitespace(curChar)) { ranges.push(consumeWhiteSpace(iter)); continue; }
		if(startsComment(curChar)) { ranges.push(consumeComment(iter)); continue; }
		if(startsKeyVal(curChar)) { ranges.push(consumeKeyVal(iter)); continue; }

		throw Error("Something crazy happened. text: '" + text + "'; curChar: '" + curChar + "'");
	}

	return ranges;
}

function isNewLineRange(range) {
	if(!range) { return false; }

	if(range.type === "whitespace") { return true; }

	if(range.type === "literal") {
		return isWhitespace(range.text) && range.text.indexOf("\n") > -1;
	}

	return false;
}

function Editor(text, path) {
	text = text || "";

	var ranges = stringToRanges(text);
	var obj = rangesToObject(ranges, text);
	var keyRange = Object.create(null); // Creates to a true hash map

	for(var i = 0; i < ranges.length; i++) {
		var range = ranges[i];

		if(range.type !== "key-value") { continue; }

		var key = rangeToBuffer(range.children[0], text).join("");
		keyRange[key] = range;
	}

	this.addHeadComment = function(comment) {
		if(comment == null) { return; }

		ranges.unshift({ type: "literal", text: "# " + comment.replace(/\n/g, "\n# ") + "\n" });
	};

	this.get = function(key) { return obj[key]; };
	this.set = function(key, val, comment) {
		if(val == null) { this.unset(key); return; }

		obj[key] = val;

		var range = keyRange[key];
		if(!range) {
			keyRange[key] = range = { type: "literal", text: key + "=" + val };

			var prevRange = ranges[ranges.length - 1];
			if(prevRange != null && !isNewLineRange(prevRange)) {
				ranges.push({ type: "literal", text: "\n" });
			}
			ranges.push(range);
		}

		// comment === null deletes comment. if comment === undefined, it's left alone
		if(comment !== undefined) {
			range.comment = comment && "# " + comment.replace(/\n/g, "\n# ") + "\n";
		}

		if(range.type === "literal") {
			range.text = key + "=" + val;
			if(range.comment != null) { range.text = range.comment + range.text; }
		} else if(range.type === "key-value") {
			range.children[2] = { type: "literal", text: val };
		} else {
			throw "Unknown node type: " + range.type;
		}
	};
	this.unset = function(key) {
		if(!(key in obj)) { return; }

		var range = keyRange[key];
		var idx = ranges.indexOf(range);

		ranges.splice(idx, (isNewLineRange(ranges[idx + 1]) ? 2 : 1));

		delete keyRange[key];
		delete obj[key];
	};
	this.valueOf = this.toString = function() {
		var buffer = [], stack = [].concat(ranges);

		var node;
		while((node = stack.shift()) != null) {
			switch(node.type) {
				case "literal":
					buffer.push(node.text);
					break;
				case "key":
				case "value":
				case "comment":
				case "whitespace":
				case "key-value-separator":
				case "escaped-value":
				case "line-break":
					buffer.push(text.substring(node.start, node.end));
					break;
				case "key-value":
					Array.prototype.unshift.apply(stack, node.children);
					if(node.comment) { stack.unshift({ type: "literal", text: node.comment }); }
					break;
			}
		}

		return buffer.join("");
	};
	this.save = function(newPath, callback) {
		if(typeof newPath === 'function') {
			callback = newPath;
			newPath = path;
		}
		newPath = newPath || path;

		if(!newPath) { callback("Unknown path"); }

		fs.writeFile(newPath, this.toString(), callback || function() {});
	};
}
function createEditor(path, callback) {
	if(!path) { return new Editor(); }

	if(!callback) { return new Editor(fs.readFileSync(path).toString(), path); }

	return fs.readFile(path, function(err, text) {
		if(err) { return callback(err, null); }

		text = text.toString();
		return callback(null, new Editor(text, path));
	});
}

function parse(text) {
	text = text.toString();
	var ranges = stringToRanges(text);
	return rangesToObject(ranges, text);
}

function read(path, callback) {
	if(!callback) { return parse(fs.readFileSync(path)); }

	return fs.readFile(path, function(err, data) {
		if(err) { return callback(err, null); }

		return callback(null, parse(data));
	});
}

module.exports = { parse: parse, read: read, createEditor: createEditor };
