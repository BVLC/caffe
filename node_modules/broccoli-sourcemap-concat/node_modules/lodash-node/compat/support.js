/**
 * Lo-Dash 2.4.1 (Custom Build) <http://lodash.com/>
 * Build: `lodash modularize exports="node" -o ./compat/`
 * Copyright 2012-2013 The Dojo Foundation <http://dojofoundation.org/>
 * Based on Underscore.js 1.5.2 <http://underscorejs.org/LICENSE>
 * Copyright 2009-2013 Jeremy Ashkenas, DocumentCloud and Investigative Reporters & Editors
 * Available under MIT license <http://lodash.com/license>
 */
var isNative = require('./internals/isNative');

/** Used to detect functions containing a `this` reference */
var reThis = /\bthis\b/;

/** `Object#toString` result shortcuts */
var argsClass = '[object Arguments]',
    objectClass = '[object Object]';

/**
 * Used for `Array` method references.
 *
 * Normally `Array.prototype` would suffice, however, using an array literal
 * avoids issues in Narwhal.
 */
var arrayRef = [];

/** Used for native method references */
var errorProto = Error.prototype,
    objectProto = Object.prototype;

/** Used to resolve the internal [[Class]] of values */
var toString = objectProto.toString;

/** Native method shortcuts */
var propertyIsEnumerable = objectProto.propertyIsEnumerable;

/**
 * An object used to flag environments features.
 *
 * @static
 * @memberOf _
 * @type Object
 */
var support = {};

(function() {
  var ctor = function() { this.x = 1; },
      object = { '0': 1, 'length': 1 },
      props = [];

  ctor.prototype = { 'valueOf': 1, 'y': 1 };
  for (var key in new ctor) { props.push(key); }
  for (key in arguments) { }

  /**
   * Detect if an `arguments` object's [[Class]] is resolvable (all but Firefox < 4, IE < 9).
   *
   * @memberOf _.support
   * @type boolean
   */
  support.argsClass = toString.call(arguments) == argsClass;

  /**
   * Detect if `arguments` objects are `Object` objects (all but Narwhal and Opera < 10.5).
   *
   * @memberOf _.support
   * @type boolean
   */
  support.argsObject = arguments.constructor == Object && !(arguments instanceof Array);

  /**
   * Detect if `name` or `message` properties of `Error.prototype` are
   * enumerable by default. (IE < 9, Safari < 5.1)
   *
   * @memberOf _.support
   * @type boolean
   */
  support.enumErrorProps = propertyIsEnumerable.call(errorProto, 'message') || propertyIsEnumerable.call(errorProto, 'name');

  /**
   * Detect if `prototype` properties are enumerable by default.
   *
   * Firefox < 3.6, Opera > 9.50 - Opera < 11.60, and Safari < 5.1
   * (if the prototype or a property on the prototype has been set)
   * incorrectly sets a function's `prototype` property [[Enumerable]]
   * value to `true`.
   *
   * @memberOf _.support
   * @type boolean
   */
  support.enumPrototypes = propertyIsEnumerable.call(ctor, 'prototype');

  /**
   * Detect if functions can be decompiled by `Function#toString`
   * (all but PS3 and older Opera mobile browsers & avoided in Windows 8 apps).
   *
   * @memberOf _.support
   * @type boolean
   */
  support.funcDecomp = !isNative(global.WinRTError) && reThis.test(function() { return this; });

  /**
   * Detect if `Function#name` is supported (all but IE).
   *
   * @memberOf _.support
   * @type boolean
   */
  support.funcNames = typeof Function.name == 'string';

  /**
   * Detect if `arguments` object indexes are non-enumerable
   * (Firefox < 4, IE < 9, PhantomJS, Safari < 5.1).
   *
   * @memberOf _.support
   * @type boolean
   */
  support.nonEnumArgs = key != 0;

  /**
   * Detect if properties shadowing those on `Object.prototype` are non-enumerable.
   *
   * In IE < 9 an objects own properties, shadowing non-enumerable ones, are
   * made non-enumerable as well (a.k.a the JScript [[DontEnum]] bug).
   *
   * @memberOf _.support
   * @type boolean
   */
  support.nonEnumShadows = !/valueOf/.test(props);

  /**
   * Detect if own properties are iterated after inherited properties (all but IE < 9).
   *
   * @memberOf _.support
   * @type boolean
   */
  support.ownLast = props[0] != 'x';

  /**
   * Detect if `Array#shift` and `Array#splice` augment array-like objects correctly.
   *
   * Firefox < 10, IE compatibility mode, and IE < 9 have buggy Array `shift()`
   * and `splice()` functions that fail to remove the last element, `value[0]`,
   * of array-like objects even though the `length` property is set to `0`.
   * The `shift()` method is buggy in IE 8 compatibility mode, while `splice()`
   * is buggy regardless of mode in IE < 9 and buggy in compatibility mode in IE 9.
   *
   * @memberOf _.support
   * @type boolean
   */
  support.spliceObjects = (arrayRef.splice.call(object, 0, 1), !object[0]);

  /**
   * Detect lack of support for accessing string characters by index.
   *
   * IE < 8 can't access characters by index and IE 8 can only access
   * characters by index on string literals.
   *
   * @memberOf _.support
   * @type boolean
   */
  support.unindexedChars = ('x'[0] + Object('x')[0]) != 'xx';

  /**
   * Detect if a DOM node's [[Class]] is resolvable (all but IE < 9)
   * and that the JS engine errors when attempting to coerce an object to
   * a string without a `toString` function.
   *
   * @memberOf _.support
   * @type boolean
   */
  try {
    support.nodeClass = !(toString.call(document) == objectClass && !({ 'toString': 0 } + ''));
  } catch(e) {
    support.nodeClass = true;
  }
}(1));

module.exports = support;
