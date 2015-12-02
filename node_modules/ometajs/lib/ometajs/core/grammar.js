var ometajs = require('../../ometajs'),
    util = require('util');

//
// ### function AbstractGrammar (source, options)
// #### @source {Array|String} source code
// #### @options {Object} options
// Abstract Grammar constructor
//
function AbstractGrammar(source, options) {
  if (!options) options = {};

  ometajs.core.AbstractParser.call(this, source, options);
  this._name = 'AbstractGrammar';

  this._options = options;
  this._trackOffset = options.trackOffset || false;
  this._errorRule = '';
  this._errorHistory = '';
  this._errorOffset = -1;
  this._errorSource = source;
}
util.inherits(AbstractGrammar, ometajs.core.AbstractParser);
module.exports = AbstractGrammar;

//
// ### function match (source, rule, args)
// #### @source {Array|String} source code
// #### @rule {String} rule to start parsing with
// #### @args {Array} (optional) arguments
// #### @errback {Function} (optional) Error handler
// #### @options {Object} options
// Creates instance of the grammar, invokes rule and returns a result
//
AbstractGrammar.match = function match(source, rule, args, errback, options) {
  return this.matchAll([source], rule, args, errback, options);
};

//
// ### function matchAll (source, rule, args)
// #### @source {Array|String} source code
// #### @rule {String} rule to start parsing with
// #### @args {Array} (optional) arguments
// #### @errback {Function} (optional) Error handler
// #### @options {Object} options
// Creates instance of the grammar, invokes rule and returns a result
//
AbstractGrammar.matchAll = function matchAll(source,
                                             rule,
                                             args,
                                             errback,
                                             options) {
  if (!Array.isArray(args) && typeof args === 'object' &&
      !errback && !options) {
    options = args;
    args = null;
  }
  var grmr = new this(source, options);

  if (!grmr._rule(rule, false, args)) {
    return grmr._getError(errback);
  }

  return grmr._getIntermediate();
};

//
// ### function _getError (errback)
// #### @errback {Function} (optional) Error handler
// Throws or calls a callback with error
//
AbstractGrammar.prototype._getError = function getError(errback) {
  // Throw errors if no error handler was provided
  if (!errback) errback = function (err) { throw err; };

  if (this._errorOffset >= 0) {
    var line,
        lineNumber = 0,
        offset = 0,
        current = 0,
        error = this._errorStack || new Error();

    if (Array.isArray(this._errorSource)) {
      var strArray = util.inspect(this._errorSource, false, 4);
      error.message = this._errorRule + ' rule failed at: ' +
                      strArray + ':' +
                      this._errorOffset;
    } else {
      (this._errorSource || '').toString().split(/\n/g)
                               .some(function (source, i) {
        if (this._errorOffset > (current + source.length + 1)) {
          current += source.length + 1;
          return false;
        }

        offset = this._errorOffset - current;
        line = source;
        lineNumber = i;

        return true;
      }, this);


      if (line.length > 80) {
        var suboffset = Math.max(0, offset - 10);
        line = line.slice(suboffset, Math.min(suboffset + 70, line.length));
        offset -= suboffset;
      }
      line = line.replace(/\t/g, ' ');

      error.message = this._errorRule + ' rule failed at: ' +
                      lineNumber + ':' + offset + '\n' +
                      line + '\n' +
                      new Array(offset + 1).join(' ') + '^';
      error.line = lineNumber;
      error.offset = offset;
    }

    errback(error);
  } else {
    errback(new Error('Unexpected negative offset'));
  }
};

//
// ### function _lastError (rule)
// #### @rule {String} Rule where we've just failed
// Internal method for tracking errors
//
AbstractGrammar.prototype._lastError = function _lastError(rule) {
  if (this._errorHistory <= this._history &&
      this._errorOffset <= this._offset) {
    this._errorRule = rule;
    this._errorHistory = this._history;
    this._errorOffset = this._offset;
    this._errorSource = this._source;
  }
};

//
// ### function _invoke (grmr, rule, fn, nc, args)
// #### @grmr {String}   name of grammar
// #### @rule {String}   Rule's name
// #### @fn   {Function} Rule's body
// #### @nc   {Boolean}  If true - rule will be applied without left-recursion
// ####                  check.
// #### @args {Array}    Arguments to rule
//
AbstractGrammar.prototype._invoke = function _invoke(grmr, rule, fn, nc, args) {
  function invoke() {
    var body = fn || grmr.prototype[rule];

    // Nullify result
    this._result = undefined;

    // Invoke actual rule function
    if (body === undefined) {
      console.error('Rule: ' + rule + ' not found!');
      console.trace();
      return false;
    }
    return body.call(this);
  }

  if (args && args.length > 0) {
    return this._simulate(args, invoke);
  } else if (!nc) {
    return this._cache(grmr.grammarName || grmr.name, rule, invoke);
  } else {
    return invoke.call(this);
  }
};

//
// ### function findOffset (source, def)
// #### @source {String|Array} source
// #### @def {Number} default offset
// Finds current offset in source, either shadowed or normal
//
AbstractGrammar.prototype._findOffset = function findOffset(source, def) {
  if (source._info) return source._info.offset;

  var marker = {};

  // Process nested arrays
  if (Array.isArray(source)) {
    var offset = marker;
    source.some(function(subsource) {
      offset = this._findOffset(subsource, marker);

      return offset !== marker;
    }, this);
    if (offset !== marker) return offset;
  }

  return def;
};

//
// ### function rule (name, args)
// #### @name {String} rule name
// #### @args {Array} (optional) arguments
// #### @cons {AbstractGrammar}
// #### @body {Function} rule's body
//
AbstractGrammar.prototype._rule = function rule(name,
                                                nocache,
                                                args,
                                                cons,
                                                body) {
  var res = false,
      trackOffset;

  if (this._trackOffset) {
    trackOffset = this._findOffset(this._source, this._offset);
  }

  // `apply` is a meta rule that invokes rule provided in arguments
  if (name === 'apply') {
    res = this._rule(args[0], nocache, args.slice(1));
  // Token rule is quite magical :)
  // It'll remember a token at the current position
  // and then use _tokenCache for matching
  // that change will automatically add lexer for any grammar
  } else if (name === 'token') {
    var flat = this._history.length === 0 && this._type === 'string',
        offset = this._offset,
        cache;

    // This cache will work only on flat string source
    // If we hit cache - just do a simple comparison
    if (flat && (cache = this._tokenCache[offset]) !== undefined) {
      if (cache.token === args[0]) {
        this._skip(cache.skip);
        this._setIntermediate(cache.value, true);
        res = true;
      } else {
        res = false;
      }
    } else {
      res =  this._atomic(function() {
        // If not - invoke grammar code first
        if (!this._invoke(cons || this.constructor, name, body, nocache)) {
          return false;
        }

        // Get result
        var pair = this._getIntermediate();

        // And store it cache
        if (flat) {
          this._tokenCache[offset] = {
            token: pair[0],
            value: pair[1],
            skip: this._offset - offset
          };
        }

        // Anyway perform check
        if (pair[0] === args[0]) {
          this._setIntermediate(pair[1], true);
          return true;
        } else {
          return false;
        }
      });
    }
  } else {
    res = this._invoke(cons || this.constructor, name, body, nocache, args);
  }

  if (!res) {
    this._lastError(name);
    return false
  } else {
    if (this._trackOffset && this._result) {
      this._result._info = {
        offset: trackOffset
      };
    }
    return true;
  }
};

//
// ### function fromTo (from, to)
// #### @from {any}
// #### @to {any}
// Tries to match content between `from` and `to`
//
AbstractGrammar.prototype._fromTo = function fromTo(from, to) {
  var head = this._source.slice(this._offset, this._offset + from.length);
  if (head !== from) return false;

  var t = this._source.indexOf(to, this._offset);

  if (t === -1) return false;

  t += to.length;

  var value = this._source.slice(this._offset, t);

  this._skip(value.length);
  this._setIntermediate(value);

  return true;
};

//
// ### function seq (str, result)
// #### @str {String}
// #### @result {Boolean} set result or not
// Tries to match chars sequence
//
AbstractGrammar.prototype._seq = function seq(str, result) {
  if (str instanceof RegExp) {
    if (this._type !== 'string') return false;

    var match = this._source.slice(this._offset).match(str);
    if (match === null) return false;

    this._skip(match[0].length, result);
    this._setIntermediate(match[0], result);

    return true;
  }

  if (this._type === 'string' && str.length < 4) {
    var head = this._source.slice(this._offset, this._offset + str.length);
    if (head !== str) return false;

    this._skip(str.length, result);
  } else {
    for (var i = 0; i < str.length; i++) {
      if (!this._match(str[i])) return false;
    }
  }
  if (str.length > 1) this._setIntermediate(str, result);

  return true;
};

//
// ### function word ()
// Tries to match non-space chars sequence
//
AbstractGrammar.prototype._word = function word() {
  if (this._type === 'string') {
    var match = this._source.slice(this._offset).match(/^[^\s]+/);
    if (match === null) return false;
    this._skip(match[0].length);
    this._setIntermediate(match[0]);

    return true;
  } else {
    var chars = [],
        current;

    while (!this._isEnd() && !(/\s/.test(current = this._current()))) {
      this._skip();
      chars.push(current);
    }

    if (!this._isEnd()) return false;

    this._setIntermediate(chars.join(''));

    return true;
  }
};

//
// ### function any (fn)
// #### @fn {Function} function to iterate with
// Greedy matcher, count >= 0
//
AbstractGrammar.prototype._any = function any(fn) {
  var list = [];

  while (!this._isEnd() && fn.call(this)) {
    list.push(this._getIntermediate());
  }

  this._setIntermediate(list);

  return true;
};

//
// ### function many (fn)
// #### @fn {Function} function to iterate with
// Greedy matcher, count > 0
//
AbstractGrammar.prototype._many = function many(fn) {
  var list = [];

  if (!fn.call(this)) return false;
  list.push(this._getIntermediate());

  while (!this._isEnd() && fn.call(this)) {
    list.push(this._getIntermediate());
  }

  this._setIntermediate(list);

  return true;
};

//
// ### function optional (fn)
// #### @fn {Function} function to iterate with
// Match, or at least not fail
//
AbstractGrammar.prototype._optional = function optional(fn) {
  if (!fn.call(this)) {
    this._setIntermediate(undefined);
  }

  return true;
};

//
// ### function token ()
// Default token rule implementation
//
AbstractGrammar.prototype.token = function token() {
  if (!this._word()) return false;
  var token = this._getIntermediate();

  // Match whitespace after token and close 'chars'
  if (this._rule('spaces')) {
    this._setIntermediate([token, token], true);
    return true;
  } else {
    this._setIntermediate(undefined);
    return false;
  }
};

//
// ### function anything ()
// Default `anything` rule implementation
//
AbstractGrammar.prototype.anything = function anything() {
  return this._skip();
};

//
// ### function space ()
// Default `space` rule implementation
//
AbstractGrammar.prototype.space = function space() {
  return this._fnMatch(function(v) { return /^[\s\n\r]$/.test(v) });
};

//
// ### function spaces ()
// Default `spaces` rule implementation
//
AbstractGrammar.prototype.spaces = function spaces() {
  return this._any(function() {
    return this._rule('space');
  });
};

//
// ### function fromTo ()
// Default `fromTo` rule implementation
//
AbstractGrammar.prototype.fromTo = function fromTo() {
  this._skip();
  var from = this._getIntermediate();
  this._skip();
  var to = this._getIntermediate();

  return this._fromTo(from, to);
};

//
// ### function exactly ()
// Default `exactly` rule implementation
//
AbstractGrammar.prototype.exactly = function exactly() {
  this._skip();
  var target = this._getIntermediate();
  this._skip();
  var source = this._getIntermediate();

  return source === target;
};

//
// ### function firstAndRest ()
// Matches <first rest*>
// ** Compatibility-only method! **
//
AbstractGrammar.prototype.firstAndRest = function firstAndRest() {
  this._skip();
  var first = this._getIntermediate();
  this._skip();
  var rest = this._getIntermediate();

  var list = [];
  if (!this._rule(first)) return false;
  list.push(this._getIntermediate());

  while (!this._isEnd() && this._rule(rest)) {
    list.push(this._getIntermediate());
  }

  this._setIntermediate(list, true);

  return true;
};

//
// ### function char ()
// Default `char` rule implementation
//
AbstractGrammar.prototype.char = function char() {
  return this._fnMatch(function(curr) {
    return typeof curr === 'string' &&
           curr.length === 1;
  }, true);
};

//
// ### function letter ()
// Default `letter` rule implementation
//
AbstractGrammar.prototype.letter = function letter() {
  return this._fnMatch(function(curr) {
    return /^[a-zA-Z]$/.test(curr);
  }, true);
};

//
// ### function letter ()
// Default `digit` rule implementation
//
AbstractGrammar.prototype.digit = function digit() {
  return this._fnMatch(function(curr) {
    return /^\d$/.test(curr);
  }, true);
};

//
// ### function seq ()
// Default `seq` rule implementation
//
AbstractGrammar.prototype.seq = function seq() {
  this._skip();
  var seq = this._getIntermediate();

  return this._seq(seq);
};

//
// ### function listOf ()
// Default `listOf` rule implementation
//
AbstractGrammar.prototype.listOf = function listOf() {
  this._skip();
  var rule = this._getIntermediate();

  this._skip();
  var sep = this._getIntermediate();

  if (!this._rule(rule)) {
    this._setIntermediate([], true);
    return true;
  }
  var list = [this._getIntermediate()];

  function separator() {
    return this.spaces() && this._seq(sep) && this.spaces();
  }

  while (this._atomic(separator)) {
    if (!this._rule(rule)) return false;
    list.push(this._getIntermediate());
  }

  this._setIntermediate(list, true);

  return true;
};

//
// ### function empty ()
// Default `empty` rule implementation
//
AbstractGrammar.prototype.empty = function empty() {
  return true;
};

//
// ### function end ()
// Default `end` rule implementation
//
AbstractGrammar.prototype.end = function end() {
  return this._isEnd();
};
