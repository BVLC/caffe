//
// ### function AbstractParser (source, options)
// #### @source {Array|String} source code
// #### @options {Object} options
// Abstract Parser constructor
//
function AbstractParser(source, options) {
  // Allocate rule cache for left recursion and overall speedup
  this.__cache = {};
  // Magical token cache
  this._tokenCache = {};

  // Store source and current position
  this._source = source;
  this._offset = 0;

  // Remember type of stream (string, array or simulate)
  this._type = this._getType(source);

  // Allocate history for traversing through nested lists
  this._history = '';

  // Prepary property for future usage
  this._intermediate = undefined;
  this._intermediateLazy = undefined;
  this._result = undefined;

  // Some state info
  this._simulated = false;
};
module.exports = AbstractParser;

//
// ### function _type (source)
// #### @source {any} Input source
// Returns type of source
//
AbstractParser.prototype._getType = function _getType(source) {
  if (Array.isArray(source)) return 'array';
  if (typeof source === 'string') return 'string';
  if (source instanceof Simulate) return 'simulate';

  throw new Error('Non-sequencable source! (source: ' + source + ' )');
};

//
// ### function _save ()
// Saves parser's state
//
AbstractParser.prototype._save = function _save() {
  return {
    source: this._source,
    offset: this._offset,
    cache: this.__cache,
    type: this._type,
    history: this._history,
    intermediate: this._intermediate,
    intermediateLazy: this._intermediateLazy,
    result: this._result,
    simulated: this._simulated
  };
};

//
// ### function _load (state)
// #### @state {Object} state
// #### @values {Boolean} should load intermediate and result?
// Loads parser's state
//
AbstractParser.prototype._load = function _load(state, values) {
  this._source = state.source;
  this._offset = state.offset;
  this.__cache = state.cache;
  this._type = state.type;
  this._history = state.history;
  if (values) {
    this._intermediate = state.intermediate;
    this._intermediateLazy = state.intermediateLazy;
    this._result = state.result;
  }
  this._simulated = state.simulated;
};

//
// ### function cache (grammar, rule, body)
// #### @grammar {String} grammar name
// #### @rule {String} rule name
// #### @body {Function} rule's body
// Caches rule results and allows left recursion
//
AbstractParser.prototype._cache = function cache(grammar, rule, body) {
  // Simulates can't be cached
  // And left recursion isn't supported here too
  if (this._simulated || grammar === 'AbstractGrammar') return body.call(this);

  var key = this._history ? [this._history, grammar].join(':') : grammar,
      // Fast offset level
      cacheLevel = this.__cache[grammar] || (this.__cache[grammar] = {}),
      cache,
      res;

  // Slower history level
  cache = cacheLevel[rule];

  if (cache) {
    // Indicate that left recursion was met
    if (cache.lr) {
      cache.detected = true;
    }

    // If result is positive - move position to cached one
    if (res = cache.result) this._load(cache.state, true);
  } else {
    var state = this._save();

    cacheLevel[rule] = cache = {
      lr: true,
      detected: false,
      result: false,
      state: state
    };

    res = body.call(this);

    cache.lr = false;
    cache.result = res;
    cache.state = this._save();

    // Left recursion detected
    if (res && cache.detected) {
      var source = this._source,
          offset = this._offset;

      do {
        // Return to previous position and start seeding
        this._load(state);

        res = body.call(this);

        if (source === this._source && offset === this._offset) res = false;

        if (res) {
          cache.result = res;
          cache.state = this._save();
        }
      } while (res);

      res = true;
      this._load(cache.state, true);
    }

  }

  return res;
};

//
// ### function atomic (body, lookahead)
// #### @body {Function} rule's body
// #### @lookahead {Boolean} if true - don't move index even after success
// Starts atomic operation which can either fail or success
// (won't be commited partially)
//
AbstractParser.prototype._atomic = function atomic(body, lookahead) {
  // Inlined state save (should be kept on stack)
  var source = this._source,
      offset = this._offset,
      cache = this.__cache,
      type = this._type,
      history = this._history,
      intermediate = this._intermediate,
      intermediateLazy = this._intermediateLazy,
      result = this._result,
      simulated =  this._simulated,
      status = body.call(this);

  // Restore state on body fail or if we was doing lookahead
  if (!status || lookahead) {
    // Inlined state load
    this._source = source;
    this._offset = offset;
    this.__cache = cache;
    this._type = type;
    this._history = history;
    this._simulated = simulated;
  } else if (this._result !== undefined) {
    this._setIntermediate(this._result);
  } else {
    var start = offset,
        end = this._offset;

    this._setIntermediate(null, null, [source, start, end]);
  }

  return status;
};

//
// ### function list (body)
// #### @body {Function} rule's body
// #### @flat {Boolean} true if it should not go deeper
// Enters an array or string at the current position (if there are any)
// Will leave array automatically on body's fail or success
//
AbstractParser.prototype._list = function list(body, flat) {
  var current = this._current(),
      res;

  if (!Array.isArray(current) && typeof current !== 'string') return false;

  this._atomic(function() {
    if (!flat) {
      // Move into list
      this._history += ':' + this._offset;
      this._source = current;
      this._offset = 0;
      this.__cache = {};
      this._type = this._getType(current);
    }

    // And invoke body
    res = body.call(this) &&
    // If we successfully matched body - ensure that it was fully matched
          (flat || this._offset === current.length);

    this._result = undefined;

    // Fail to restore all state
    return flat && res;
  });

  if (!flat && res) {
    // Skip current item as we matched it
    this._skip();
  }

  return res;
};

//
// ### function simulate (source)
// #### @source {Array} data array
// Prepends source to the current one
//
AbstractParser.prototype._simulate = function simulate(source, body) {
  if (!Array.isArray(source)) {
    throw new Error('Only arrays can be prepended to the current source');
  }

  return this._atomic(function() {
    new Simulate(this, source);
    this._simulated = true;

    return body.call(this);
  });
};

//
// ### function getIntermediate ()
// Internal functions, should be called to get intermediate value
//
AbstractParser.prototype._getIntermediate = function getIntermediate() {
  if (this._intermediateLazy) {
    var lazy = this._intermediateLazy;
    this._intermediate = lazy[0].slice(lazy[1], lazy[2]);
  }
  return this._intermediate;
};

//
// ### function setIntermediate (value)
// #### @value {any}
// #### @result {Boolean} should we propagate value to the end of rule
// #### @lazy {Object|Undefined} lazy value
// Internal functions, should be called to set intermediate value
//
AbstractParser.prototype._setIntermediate = function setIntermediate(value,
                                                                     result,
                                                                     lazy) {
  if (result) this._result = value;
  this._intermediate = value;
  this._intermediateLazy = lazy;
};

//
// ### function match (str)
// #### @fn {String} value to match
//
AbstractParser.prototype._match = function match(value) {
  if (this._current() === value) {
    return this._skip();
  } else {
    return false;
  }
};

//
// ### function fnMatch (fn, result)
// #### @fn {Function} callback to test current value
// #### @result {Boolean} Set result
//
AbstractParser.prototype._fnMatch = function match(fn, result) {
  if (fn.call(this, this._current())) {
    return this._skip(null, result);
  } else {
    return false;
  }
};

//
// ### function exec (fn)
// #### @fn {Function} host code to execute
// Executes host code and sets intermediate value to it's result
//
AbstractParser.prototype._exec = function exec(result) {
  this._setIntermediate(result, true);

  return true;
};

//
// ### function current ()
// Returns value at the current index
//
AbstractParser.prototype._current = function current() {
  if (this._type === 'simulate') {
    return this._source.get();
  } else {
    return this._source[this._offset];
  }
};

//
// ### function isEnd ()
// Returns true if input source ended
//
AbstractParser.prototype._isEnd = function isEnd() {
  return this._source.length <= this._offset;
};

//
// ### function skip (num, result)
// #### @num {Number} number of entries to skip
// #### @result {Boolean} set result or not
// Skips element in current source
//
AbstractParser.prototype._skip = function skip(num, result) {
  if (this._type === 'simulate') {
    this._source.skip(num);
  } else {
    if (num) this._offset += num - 1;
    this._setIntermediate(this._current(), result);
    this._offset++;

    // If source doesn't have enough data to consume - fail!
    if (this._offset > this._source.length) return false;
  }

  this.__cache = {};

  return true;
};

//
// ### function Simulate (parser, source)
// #### @parser {Parser}
// #### @source {Array|String|Simulate} Source
// Simulates constructor
//
function Simulate(parser, source) {
  this.original = parser._save();
  this.parser = parser;

  this.source = source;
  this.ended = false;

  parser._source = this;
  parser.__cache = {};
  parser._type = parser._getType(this);
  parser._offset = 0;
}

//
// ### function get ()
// Gets current item in the simulate
//
Simulate.prototype.get = function get() {
  return this.source[this.parser._offset];
};

//
// ### function skip (num)
// #### @num {Number} number of entries to skip
// Skips element in simulate
//
Simulate.prototype.skip = function skip(num) {
  if (num) this.parser._offset += num - 1;
  this.parser._setIntermediate(this.get());
  this.parser._offset++;

  if (this.parser._offset >= this.source.length) {
    this.ended = true;
    this.parser._load(this.original);
  }
};

//
// ### function slice(from, to)
// #### @from {Number} from index
// #### @to {Number} to index
// Returns concatenated source
//
Simulate.prototype.slice = function slice(from, to) {
  var result = this.source.slice(from, to);

  if (to > this.source.length) {
    result.concat(this.original.source.slice(0, to - this.source.length));
  }

  return result;
};
