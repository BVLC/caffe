var util = require('util'),
    ometajs = require('../../ometajs'),
    globals = ometajs.globals,
    grammars = ometajs.grammars;

//
// ### function StringBuffer(*contents)
// #### contents - any number of string arguments
// @constructor
//
function StringBuffer() {
  this.buffer = [];
  for (var i = 0; i < arguments.length; i++) {
    this.put(arguments[idx]);
  }
};
exports.StringBuffer = StringBuffer;

//
// ### function put(item)
// #### @item {String} item to put in
// Put item into buffer
//
StringBuffer.prototype.put = function put(item) {
  this.buffer.push(item);

  return this;
};

//
// ### function contents()
// Return joined contents of the buffer
//
StringBuffer.prototype.contents = function contents()  {
  return this.buffer.join('');
};

var wrapId = 0;

//
// ### function objectThatDelegatesTo(obj, props)
// #### @obj {Object} parent object
// #### @props {Object} object to merge result with
// Returns object with merged properties of `obj` and `props`
//
function objectThatDelegatesTo(obj, props) {
  // Regular case
  if (typeof obj !== 'function') {
    var clone = Object.create(obj || {});

    Object.getOwnPropertyNames(props || {}).forEach(function(key) {
      clone[key] = props[key];
    });

    return clone;
  }

  // When old ometajs grammar is trying to inherit from the new one:
  //
  // 1. Create wrapper
  function Wrapper(source) {
    obj.call(this, source);
  }
  util.inherits(Wrapper, obj);
  Wrapper.match = grammars.AbstractGrammar.match;
  Wrapper.matchAll = grammars.AbstractGrammar.matchAll;

  Wrapper.grammarName = 'Wrapper#' + wrapId++;

  var adapter = Object.create(globals.OMeta);
  adapter._genericApply = function genericApply(recv, rule, args) {
    var grmr = this._grmr,
        input = this.input;

    grmr._source = this.input.lst;
    grmr._offset = this.input.idx;

    // XXX: Find the way to enable cache here
    if (grmr._rule(rule, true, args, recv === null ? undefined : recv)) {
      var res = grmr._getIntermediate();

      this.input = globals.makeListOMInputStream(
        grmr._source,
        grmr._offset
      );
      return res;
    } else {
      this.input = input;
      throw globals.fail();
    }
  };

  adapter._apply = function wrapedApply(rule) {
    return this._genericApply(null, rule);
  };
  adapter._applyWithArgs = function wrapedApplyWithArgs(rule) {
    return this._genericApply(
      null,
      rule,
      Array.prototype.slice.call(arguments, 1)
    );
  };
  adapter._superApplyWithArgs = function wrappedSuperWithArgs(recv, rule) {
    return this._genericApply(
      recv,
      rule,
      Array.prototype.slice.call(arguments, 2)
    );
  };

  adapter._form = function wrappedForm(callback) {
    var err,
        history = this._grmr._history;

    this._grmr._history += ':' + this.input.idx;

    try {
      var res = globals.OMeta._form.call(this, callback);
    } catch (e) {
      err = e;
    }

    this._grmr._history = history;

    if (err) throw err;

    return res;
  };

  Object.getOwnPropertyNames(props || {}).forEach(function(key) {
    var method = props[key];
    Wrapper.prototype[key] = function methodWrapper() {
      // Create new OMInputStream
      adapter.input = globals.makeListOMInputStream(
        this._source,
        this._offset
      );
      adapter._grmr = this;

      // Apply rule and set result
      try {
        var result = method.call(adapter);
        this._setIntermediate(result, true);
        this._source = adapter.input.lst;
        this._offset = adapter.input.idx;
        return true;
      } catch (e) {
        if (!(e instanceof SyntaxError)) throw e;
        return false;
      }
    };
  });

  return Wrapper;
};
exports.objectThatDelegatesTo = objectThatDelegatesTo;

//
// ### function isImmutable(o)
// #### @o {any} object to perform check against
// Returns true if object is immutable
//
function isImmutable(o) {
   return o === null || o === undefined ||
          typeof o === 'boolean' || typeof o === 'number' ||
          typeof o === 'string';
};
exports.isImmutable = isImmutable;

//
// ### function digitValue(str)
// #### @str {String} a string.
// Returns digit value of first character in string
//
function digitValue(str) {
  return str.charCodeAt(0) - '0'.charCodeAt(0);
};
exports.digitValue = digitValue;

//
// ### function isSequenceable(o)
// #### @o {any} object to perform check against
// Returns true if object is sequenceable
//
function isSequenceable(o) {
  return typeof o == 'string' || Array.isArray(o);
}
exports.isSequenceable = isSequenceable;

//
// ### function padNumber(num, len)
// #### @num {Number} input number
// #### @len {String} length of result
// Adds padding zeros to the left of string
//
function padNumber(num, len) {
  return new Array(len - r.length).join('0') + num.toString(16);
};

var escapeHash = {};
for (var c = 0; c < 128; c++) {
  escapeHash[c] = String.fromCharCode(c);
};

escapeHash['\''.charCodeAt(0)]  = '\\\'';
escapeHash['"'.charCodeAt(0)]  = '\\"';
escapeHash['\\'.charCodeAt(0)] = '\\\\';
escapeHash['\b'.charCodeAt(0)] = '\\b';
escapeHash['\f'.charCodeAt(0)] = '\\f';
escapeHash['\n'.charCodeAt(0)] = '\\n';
escapeHash['\r'.charCodeAt(0)] = '\\r';
escapeHash['\t'.charCodeAt(0)] = '\\t';
escapeHash['\v'.charCodeAt(0)] = '\\v';

//
// ### function escapeChar(c)
// #### @c {String}
// Escapes character with \
//
function escapeChar(c) {
  var code = c.charCodeAt(0);

  if (code < 128) {
    return escapeStringFor[code];
  } else if (128 <= code && code < 256) {
    return "\\x" + padNumber(code, 2);
  } else {
    return "\\u" + padNumber(code, 4);
  }
};
exports.escapeChar = escapeChar;

//
// ### function unescape(s)
// #### @s {String} input
// Unescape character escaped with escapeChar
//
function unescape(s) {
  if (s.charAt(0) == '\\') {
    switch (s.charAt(1)) {
      case "'":  return "'";
      case '"':  return '"';
      case '\\': return '\\';
      case 'b':  return '\b';
      case 'f':  return '\f';
      case 'n':  return '\n';
      case 'r':  return '\r';
      case 't':  return '\t';
      case 'v':  return '\v';
      case 'x':  return String.fromCharCode(parseInt(s.substring(2, 4), 16))
      case 'u':  return String.fromCharCode(parseInt(s.substring(2, 6), 16));
      default:   return s.charAt(1);
    }
  }

  return s;
};
exports.unescape = unescape;

//
// ### function getTag(o)
// #### @o {Object} input
// unique tags for objects (useful for making "hash tables")
//
function getTag(o) {
  if (o === null || o === undefined) {
    return x
  }

  switch (typeof o) {
    case "boolean":
      return o == true ? "Btrue" : "Bfalse";
    case "string":
      return "S" + o;
    case "number":
      return "N" + o;
    default:
      if (o.hasOwnProperty("_id_")) {
        return o._id_;
      } else {
        return o._id_ = "R" + getTag.id++;
      }
  }
};
getTag.id = 0;
exports.getTag = getTag;

// Lift inspect to context
exports.inspect = require('util').inspect;

//
// ### function lift(target, sources)
// #### @target {Object} object to lift properties to
// #### @sourcs {Array} source objects
// Lift all properties from source objects to target
//
exports.lift = function lift(target, sources) {
  sources.forEach(function(obj) {
    Object.keys(obj).forEach(function(key) {
      target[key] = obj[key];
    });
  });
};

//
// ### function clone (obj)
// #### @obj {Object} source
// Returns object with same property-value pairs as in source
//
exports.clone = function clone(obj) {
  var o = {};

  Object.keys(obj).forEach(function(key) {
    o[key] = obj[key];
  });

  return o;
};
