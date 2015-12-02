/*
  new syntax:
    #foo and `foo  match the string object 'foo' (it's also accepted in my JS)
    'abc'    match the string object 'abc'
    'c'      match the string object 'c'
    ``abc''    match the sequence of string objects 'a', 'b', 'c'
    "abc"    token('abc')
    [1 2 3]    match the array object [1, 2, 3]
    foo(bar)    apply rule foo with argument bar
    -> ...    semantic actions written in JS (see OMetaParser's atomicHostExpr rule)
*/

/*
ometa M {
  number = number:n digit:d -> { n * 10 + d.digitValue() }
         | digit:d          -> { d.digitValue() }
}

translates to...

M = objectThatDelegatesTo(OMeta, {
  number: function() {
            return this._or(function() {
                              var n = this._apply("number"),
                                  d = this._apply("digit")
                              return n * 10 + d.digitValue()
                            },
                            function() {
                              var d = this._apply("digit")
                              return d.digitValue()
                            }
                           )
          }
})
M.matchAll("123456789", "number")
*/

var utils = require('./utils'),
    objectThatDelegatesTo = utils.objectThatDelegatesTo,
    isSequenceable = utils.isSequenceable;

//
// failure exception
//
var fail = exports.fail = function fail() {
  return fail.error;
};
fail.error = new SyntaxError('match failed');

//
// ### function lookup (fn, success, fallback)
// #### @fn {Function} function that may throw
// #### @success {Function} call if function hasn't thrown
// #### @fallback {Function} call if function thrown fail()
//
function lookup(fn, success, fallback) {
  var value;
  try {
    value = fn();
  } catch (e) {
    if (!(e instanceof SyntaxError)) throw e;
    return fallback && fallback();
  }

  return success && success(value);
};

//
// ### function OMInputStream(hd, tl)
// #### @hd {any} Head
// #### @tl {Object} Tail
// Streams and memoization
//
function OMInputStream(hd, tl) {
  this.memo = { }
  this.lst  = tl.lst
  this.idx  = tl.idx
  this.hd   = hd
  this.tl   = tl
};

//
// ### function head ()
// Returns stream's `hd` property
//
OMInputStream.prototype.head = function() { return this.hd };

//
// ### function tail ()
// Returns stream's `tl` property
//
OMInputStream.prototype.tail = function() { return this.tl };

//
// ### function type ()
// Returns stream's `lst` property constructor
//
OMInputStream.prototype.type = function() { return this.lst.constructor };

//
// ### function upTo (that)
// #### @that {Object} target object
// Visit all tails and join all met heads and return string or array
// (depending on `.lst` type)
//
OMInputStream.prototype.upTo = function(that) {
  var r = [], curr = this
  while (curr != that) {
    r.push(curr.head())
    curr = curr.tail()
  }
  return this.type() == String ? r.join('') : r
};

//
// ### function OMInputStreamEnd (lst, idx)
// #### @lst {Array} list
// #### @idx {Number} index
// Internal class
//
function OMInputStreamEnd(lst, idx) {
  this.memo = { }
  this.lst = lst
  this.idx = idx
};
OMInputStreamEnd.prototype = objectThatDelegatesTo(OMInputStream.prototype);

//
// ### function head ()
// Not implemented
//
OMInputStreamEnd.prototype.head = function() { throw fail() };

//
// ### function tail ()
// Not implemented
//
OMInputStreamEnd.prototype.tail = function() { throw fail() };

//
// ### function ListOMInputStream (lst, idx)
// #### @lst {Array} list
// #### @idx {Number} index
// Returns self-expanding stream
//
function ListOMInputStream(lst, idx) {
  this.memo = { };
  this.lst  = lst;
  this.idx  = idx;
  this.hd   = lst[idx];
}
ListOMInputStream.prototype = objectThatDelegatesTo(OMInputStream.prototype);

//
// ### function head ()
// Returns stream's `hd` property's value
//
ListOMInputStream.prototype.head = function() { return this.hd };

//
// ### function tail ()
// Returns or creates stream's tail
//
ListOMInputStream.prototype.tail = function() {
  return this.tl || (this.tl = makeListOMInputStream(this.lst, this.idx + 1));
}

//
// ### function makeListOMInputStream (lst, idx)
// #### @lst {Array} List
// #### @idx {Number} index
// Returns either ListOMInputStream's or OMInputStreamEnd's instance
//
function makeListOMInputStream(lst, idx) {
  if (idx < lst.length) {
    return new ListOMInputStream(lst, idx);
  } else {
    return new OMInputStreamEnd(lst, idx);
  }
}
exports.makeListOMInputStream = makeListOMInputStream;

//
// ### function makeOMInputStreamProxy (target)
// #### @target {any} Delegate's constructor
// Returns object with stream's properties
// (has self-expanding tail)
//
function makeOMInputStreamProxy(target) {
  return objectThatDelegatesTo(target, {
    memo:   { },
    target: target,
    tl: undefined,
    tail: function() {
      return this.tl || (this.tl = makeOMInputStreamProxy(target.tail()));
    }
  })
}

//
// ### function Failer()
// (i.e., that which makes things fail)
// Used to detect (direct) left recursion and memoize failures
function Failer() {
  this.used = false;
};

//
// ### OMeta
// the OMeta "class" and basic functionality
//
var OMeta = exports.OMeta = {
  _apply: function(rule) {
    var self = this,
        memoRec = this.input.memo[rule];

    if (memoRec == undefined) {
      var origInput = this.input,
          failer    = new Failer();

      if (this[rule] === undefined) {
        throw 'tried to apply undefined rule "' + rule + '"'
      }

      this.input.memo[rule] = failer;
      this.input.memo[rule] = memoRec = {
        ans: this[rule].call(this),
        nextInput: this.input
      };

      if (failer.used) {
        var sentinel = this.input;
        while (true) {
          var result = lookup(function() {
            self.input = origInput;
            var ans = self[rule].call(self);

            if (self.input == sentinel) throw fail()

            memoRec.ans       = ans
            memoRec.nextInput = self.input
          }, function () {
            return false;
          }, function () {
            return true;
          });

          if (result) break;
        }
      }
    }
    else if (memoRec instanceof Failer) {
      memoRec.used = true
      throw fail()
    }
    this.input = memoRec.nextInput
    return memoRec.ans
  },

  // note: _applyWithArgs and _superApplyWithArgs are not memoized, so they can't be left-recursive
  _applyWithArgs: function(rule) {
    for (var idx = arguments.length - 1; idx > 0; idx--)
      this._prependInput(arguments[idx])
    return this[rule].call(this)
  },
  _superApplyWithArgs: function(recv, rule) {
    for (var idx = arguments.length - 1; idx > 1; idx--)
      recv._prependInput(arguments[idx])
    return this[rule].call(recv)
  },
  _prependInput: function(v) {
    this.input = new OMInputStream(v, this.input)
  },

  // if you want your grammar (and its subgrammars) to memoize parameterized rules, invoke this method on it:
  memoizeParameterizedRules: function() {
    this._prependInput = function(v) {
      var newInput
      if (isImmutable(v)) {
        newInput = this.input[getTag(v)]
        if (!newInput) {
          newInput = new OMInputStream(v, this.input)
          this.input[getTag(v)] = newInput
        }
      }
      else newInput = new OMInputStream(v, this.input)
      this.input = newInput
    }
    this._applyWithArgs = function(rule) {
      for (var idx = arguments.length - 1; idx > 0; idx--)
        this._prependInput(arguments[idx])
      return this._apply(rule)
    }
  },

  _pred: function(b) {
    if (b) return true;

    throw fail();
  },
  _not: function(x) {
    var self = this,
        origInput = this.input;

    return lookup(function() {
      x.call(self);
    }, function() {
      throw fail();
    }, function() {
      self.input = origInput
      return true
    });
  },
  _lookahead: function(x) {
    var origInput = this.input,
        r         = x.call(this)
    this.input = origInput
    return r
  },
  _or: function() {
    var self = this,
        origInput = this.input,
        ref = {},
        result = ref;

    for (var idx = 0; idx < arguments.length; idx++) {
      var arg = arguments[idx];

      lookup(function() {
        self.input = origInput;
        result = arg.call(self);
      });

      if (result !== ref) return result;
    }

    throw fail();
  },
  _xor: function(ruleName) {
    var self = this,
        origInput = this.input,
        idx = 1,
        newInput,
        ans;

    while (idx < arguments.length) {
      var arg = arguments[idx];

      lookup(function() {
        self.input = origInput;
        ans = arg.call(self);
        if (newInput) {
          throw 'more than one choice matched by "exclusive-OR" in ' + ruleName
        }
        newInput = self.input
      });
      idx++
    }

    if (newInput) {
      this.input = newInput
      return ans
    }
    else
      throw fail();
  },
  disableXORs: function() {
    this._xor = function(ruleName) {
      var self = this,
          origInput = this.input,
          ref = {},
          result = ref;

      for (var idx = 1; idx < arguments.length; idx++) {
        var arg = arguments[idx];

        lookup(function() {
          self.input = origInput;
          result = arg.call(self);
        });

        if (result !== ref) return result;
      }
      throw fail()
    }
  },
  _opt: function(x) {
    var self = this,
        origInput = this.input,
        ans;

    lookup(function() {
      ans = x.call(self);
    }, function() {
    }, function() {
      self.input = origInput;
    });

    return ans;
  },
  _many: function(x) {
    var self = this,
        ans = arguments[1] != undefined ? [arguments[1]] : [];

    while (true) {
      var origInput = this.input

      var result = lookup(function() {
        ans.push(x.call(self));
      }, function() {
        return false;
      }, function() {
        self.input = origInput;
        return true;
      });

      if (result) break;
    }
    return ans
  },
  _many1: function(x) { return this._many(x, x.call(this)) },
  _form: function(x) {
    var v = this._apply("anything")
    if (!isSequenceable(v))
      throw fail()
    var origInput = this.input
    this.input =  makeListOMInputStream(v, 0);
    var r = x.call(this)
    this._apply("end")
    this.input = origInput
    return v
  },
  _consumedBy: function(x) {
    var origInput = this.input
    x.call(this)
    return origInput.upTo(this.input)
  },
  _idxConsumedBy: function(x) {
    var origInput = this.input
    x.call(this)
    return {fromIdx: origInput.idx, toIdx: this.input.idx}
  },
  _interleave: function(mode1, part1, mode2, part2 /* ..., moden, partn */) {
    var currInput = this.input, ans = []
    for (var idx = 0; idx < arguments.length; idx += 2)
      ans[idx / 2] = (arguments[idx] == "*" || arguments[idx] == "+") ? [] : undefined
    while (true) {
      var idx = 0, allDone = true
      while (idx < arguments.length) {
        if (arguments[idx] != "0")
          try {
            this.input = currInput
            switch (arguments[idx]) {
              case "*": ans[idx / 2].push(arguments[idx + 1].call(this));                       break
              case "+": ans[idx / 2].push(arguments[idx + 1].call(this)); arguments[idx] = "*"; break
              case "?": ans[idx / 2] =    arguments[idx + 1].call(this);  arguments[idx] = "0"; break
              case "1": ans[idx / 2] =    arguments[idx + 1].call(this);  arguments[idx] = "0"; break
              default:  throw "invalid mode '" + arguments[idx] + "' in OMeta._interleave"
            }
            currInput = this.input
            break
          }
          catch (f) {
            if (!(f instanceof SyntaxError))
              throw f
            // if this (failed) part's mode is "1" or "+", we're not done yet
            allDone = allDone && (arguments[idx] == "*" || arguments[idx] == "?")
          }
        idx += 2
      }
      if (idx == arguments.length) {
        if (allDone)
          return ans
        else
          throw fail()
      }
    }
  },
  _currIdx: function() { return this.input.idx },

  // some basic rules
  anything: function() {
    var r = this.input.head()
    this.input = this.input.tail()
    return r
  },
  end: function() {
    return this._not(function() { return this._apply("anything") })
  },
  pos: function() {
    return this.input.idx
  },
  empty: function() { return true },
  apply: function() {
    var r = this._apply("anything")
    return this._apply(r)
  },
  foreign: function() {
    var g   = this._apply("anything"),
        r   = this._apply("anything"),
        gi  = objectThatDelegatesTo(g, {input: makeOMInputStreamProxy(this.input)})
    var ans = gi._apply(r)
    this.input = gi.input.target
    return ans
  },

  //  some useful "derived" rules
  exactly: function() {
    var wanted = this._apply("anything")
    if (wanted === this._apply("anything"))
      return wanted
    throw fail()
  },
  "true": function() {
    var r = this._apply("anything")
    this._pred(r === true)
    return r
  },
  "false": function() {
    var r = this._apply("anything")
    this._pred(r === false)
    return r
  },
  "undefined": function() {
    var r = this._apply("anything")
    this._pred(r === undefined)
    return r
  },
  number: function() {
    var r = this._apply("anything")
    this._pred(typeof r === "number")
    return r
  },
  string: function() {
    var r = this._apply("anything")
    this._pred(typeof r === "string")
    return r
  },
  "char": function() {
    var r = this._apply("anything")
    this._pred(typeof r === "string" && r.length == 1)
    return r
  },
  space: function() {
    var r = this._apply("char")
    this._pred(r.charCodeAt(0) <= 32)
    return r
  },
  spaces: function() {
    return this._many(function() { return this._apply("space") })
  },
  digit: function() {
    var r = this._apply("char")
    this._pred(r >= "0" && r <= "9")
    return r
  },
  lower: function() {
    var r = this._apply("char")
    this._pred(r >= "a" && r <= "z")
    return r
  },
  upper: function() {
    var r = this._apply("char")
    this._pred(r >= "A" && r <= "Z")
    return r
  },
  letter: function() {
    return this._or(function() { return this._apply("lower") },
                    function() { return this._apply("upper") })
  },
  letterOrDigit: function() {
    return this._or(function() { return this._apply("letter") },
                    function() { return this._apply("digit")  })
  },
  firstAndRest: function()  {
    var first = this._apply("anything"),
        rest  = this._apply("anything")
     return this._many(function() { return this._apply(rest) }, this._apply(first))
  },
  seq: function() {
    var xs = this._apply("anything")
    for (var idx = 0; idx < xs.length; idx++)
      this._applyWithArgs("exactly", xs[idx])
    return xs
  },
  notLast: function() {
    var rule = this._apply("anything"),
        r    = this._apply(rule)
    this._lookahead(function() { return this._apply(rule) })
    return r
  },
  listOf: function() {
    var rule  = this._apply("anything"),
        delim = this._apply("anything")
    return this._or(function() {
                      var r = this._apply(rule)
                      return this._many(function() {
                                          this._applyWithArgs("token", delim)
                                          return this._apply(rule)
                                        },
                                        r)
                    },
                    function() { return [] })
  },
  token: function() {
    var cs = this._apply("anything")
    this._apply("spaces")
    return this._applyWithArgs("seq", cs)
  },
  fromTo: function () {
    var x = this._apply("anything"),
        y = this._apply("anything")
    return this._consumedBy(function() {
                              this._applyWithArgs("seq", x)
                              this._many(function() {
                                this._not(function() { this._applyWithArgs("seq", y) })
                                this._apply("char")
                              })
                              this._applyWithArgs("seq", y)
                            })
  },

  initialize: function() { },
  // match and matchAll are a grammar's "public interface"
  _genericMatch: function(input, rule, args, matchFailed) {
    if (args == undefined)
      args = []
    var realArgs = [rule]
    for (var idx = 0; idx < args.length; idx++)
      realArgs.push(args[idx])
    var m = objectThatDelegatesTo(this, {input: input})
    m.initialize()

    return lookup(function() {
      return realArgs.length == 1 ?
          m._apply.call(m, realArgs[0])
          :
          m._applyWithArgs.apply(m, realArgs);
    }, function(value) {
      return value;
    }, function() {
      if (matchFailed != undefined) {
        var input = m.input
        if (input.idx != undefined) {
          while (input.tl != undefined && input.tl.idx != undefined)
            input = input.tl
          input.idx--
        }
        return matchFailed(m, input.idx)
      }
      throw f
    });
  },
  match: function(obj, rule, args, matchFailed) {
    return this._genericMatch(makeListOMInputStream([obj], 0), rule, args, matchFailed)
  },
  matchAll: function(listyObj, rule, args, matchFailed) {
    return this._genericMatch(makeListOMInputStream(listyObj, 0), rule, args, matchFailed)
  },
  createInstance: function() {
    var m = objectThatDelegatesTo(this)
    m.initialize()
    m.matchAll = function(listyObj, aRule) {
      m.input = makeListOMInputStream(listyObj, 0);
      return m._apply(aRule)
    }
    return m
  }
};
