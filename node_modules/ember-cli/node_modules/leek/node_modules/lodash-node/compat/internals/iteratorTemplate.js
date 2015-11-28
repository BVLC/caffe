/**
 * Lo-Dash 2.4.1 (Custom Build) <http://lodash.com/>
 * Build: `lodash modularize exports="node" -o ./compat/`
 * Copyright 2012-2013 The Dojo Foundation <http://dojofoundation.org/>
 * Based on Underscore.js 1.5.2 <http://underscorejs.org/LICENSE>
 * Copyright 2009-2013 Jeremy Ashkenas, DocumentCloud and Investigative Reporters & Editors
 * Available under MIT license <http://lodash.com/license>
 */
var support = require('../support');

/**
 * The template used to create iterator functions.
 *
 * @private
 * @param {Object} data The data object used to populate the text.
 * @returns {string} Returns the interpolated text.
 */
var iteratorTemplate = function(obj) {

  var __p = 'var index, iterable = ' +
  (obj.firstArg) +
  ', result = ' +
  (obj.init) +
  ';\nif (!iterable) return result;\n' +
  (obj.top) +
  ';';
   if (obj.array) {
  __p += '\nvar length = iterable.length; index = -1;\nif (' +
  (obj.array) +
  ') {  ';
   if (support.unindexedChars) {
  __p += '\n  if (isString(iterable)) {\n    iterable = iterable.split(\'\')\n  }  ';
   }
  __p += '\n  while (++index < length) {\n    ' +
  (obj.loop) +
  ';\n  }\n}\nelse {  ';
   } else if (support.nonEnumArgs) {
  __p += '\n  var length = iterable.length; index = -1;\n  if (length && isArguments(iterable)) {\n    while (++index < length) {\n      index += \'\';\n      ' +
  (obj.loop) +
  ';\n    }\n  } else {  ';
   }

   if (support.enumPrototypes) {
  __p += '\n  var skipProto = typeof iterable == \'function\';\n  ';
   }

   if (support.enumErrorProps) {
  __p += '\n  var skipErrorProps = iterable === errorProto || iterable instanceof Error;\n  ';
   }

      var conditions = [];    if (support.enumPrototypes) { conditions.push('!(skipProto && index == "prototype")'); }    if (support.enumErrorProps)  { conditions.push('!(skipErrorProps && (index == "message" || index == "name"))'); }

   if (obj.useHas && obj.keys) {
  __p += '\n  var ownIndex = -1,\n      ownProps = objectTypes[typeof iterable] && keys(iterable),\n      length = ownProps ? ownProps.length : 0;\n\n  while (++ownIndex < length) {\n    index = ownProps[ownIndex];\n';
      if (conditions.length) {
  __p += '    if (' +
  (conditions.join(' && ')) +
  ') {\n  ';
   }
  __p +=
  (obj.loop) +
  ';    ';
   if (conditions.length) {
  __p += '\n    }';
   }
  __p += '\n  }  ';
   } else {
  __p += '\n  for (index in iterable) {\n';
      if (obj.useHas) { conditions.push("hasOwnProperty.call(iterable, index)"); }    if (conditions.length) {
  __p += '    if (' +
  (conditions.join(' && ')) +
  ') {\n  ';
   }
  __p +=
  (obj.loop) +
  ';    ';
   if (conditions.length) {
  __p += '\n    }';
   }
  __p += '\n  }    ';
   if (support.nonEnumShadows) {
  __p += '\n\n  if (iterable !== objectProto) {\n    var ctor = iterable.constructor,\n        isProto = iterable === (ctor && ctor.prototype),\n        className = iterable === stringProto ? stringClass : iterable === errorProto ? errorClass : toString.call(iterable),\n        nonEnum = nonEnumProps[className];\n      ';
   for (k = 0; k < 7; k++) {
  __p += '\n    index = \'' +
  (obj.shadowedProps[k]) +
  '\';\n    if ((!(isProto && nonEnum[index]) && hasOwnProperty.call(iterable, index))';
          if (!obj.useHas) {
  __p += ' || (!nonEnum[index] && iterable[index] !== objectProto[index])';
   }
  __p += ') {\n      ' +
  (obj.loop) +
  ';\n    }      ';
   }
  __p += '\n  }    ';
   }

   }

   if (obj.array || support.nonEnumArgs) {
  __p += '\n}';
   }
  __p +=
  (obj.bottom) +
  ';\nreturn result';

  return __p
};

module.exports = iteratorTemplate;
