/*
 * cliff.js: CLI output formatting tools: "Your CLI Formatting Friend".
 *
 * (C) 2010, Nodejitsu Inc.
 *
 */
 
var colors = require('colors'),
    eyes = require('eyes'),
    winston = require('winston');

var cliff = exports,
    logger;

cliff.__defineGetter__('logger', function () {
  delete cliff.logger;
  return cliff.logger = logger;
});

cliff.__defineSetter__('logger', function (val) {
  logger = val;
  
  //
  // Setup winston to use the `cli` formats
  //
  if (logger.cli) {
    logger.cli();
  }
});

//
// Set the default logger for cliff.
//
cliff.logger = new winston.Logger({
  transports: [new winston.transports.Console()]
});

//
// Expose a default `eyes` inspector.
//
cliff.inspector = eyes.inspector;
cliff.inspect   = eyes.inspector({ stream: null,
  styles: {               // Styles applied to stdout
    all:     null,        // Overall style applied to everything
    label:   'underline', // Inspection labels, like 'array' in `array: [1, 2, 3]`
    other:   'inverted',  // Objects which don't have a literal representation, such as functions
    key:     'grey',      // The keys in object literals, like 'a' in `{a: 1}`
    special: 'grey',      // null, undefined...
    number:  'blue',      // 0, 1, 2...
    bool:    'magenta',   // true false
    regexp:  'green'      // /\d+/
  }
});

//
// ### function extractFrom (obj, properties)
// #### @obj {Object} Object to extract properties from. 
// #### @properties {Array} List of properties to output.
// Creates an array representing the values for `properties` in `obj`.
//
cliff.extractFrom = function (obj, properties) {
  return properties.map(function (p) {
    return obj[p];
  });
};

//
// ### function columnMajor (rows)
// #### @rows {ArrayxArray} Row-major Matrix to transpose
// Transposes the row-major Matrix, represented as an array of rows,
// into column major form (i.e. an array of columns).
//
cliff.columnMajor = function (rows) {
  var columns = [];
  
  rows.forEach(function (row) {
    for (var i = 0; i < row.length; i += 1) {
      if (!columns[i]) {
        columns[i] = [];
      }
      
      columns[i].push(row[i]);
    }
  });
  
  return columns;
};

//
// ### arrayLengths (arrs) 
// #### @arrs {ArrayxArray} Arrays to calculate lengths for
// Creates an array with values each representing the length
// of an array in the set provided.   
//
cliff.arrayLengths = function (arrs) {
  var i, lengths = [];
  for (i = 0; i < arrs.length; i += 1) {
    lengths.push(longestElement(arrs[i].map(cliff.stringifyLiteral)));
  }
  return lengths;
};

//
// ### function stringifyRows (rows, colors)
// #### @rows {ArrayxArray} Matrix of properties to output in row major form
// #### @colors {Array} Set of colors to use for the headers
// Outputs the specified `rows` as fixed-width columns, adding
// colorized headers if `colors` are supplied.
//
cliff.stringifyRows = function (rows, colors, options) {
  var lengths, columns, output = [], headers;

  options = options || {};
  options.columnSpacing = options.columnSpacing || 2;

  columns = cliff.columnMajor(rows);
  lengths = cliff.arrayLengths(columns);

  function stringifyRow(row, colorize) {
    var rowtext = '', padding, item, i, length;
    for (i = 0; i < row.length; i += 1) {
      item = cliff.stringifyLiteral(row[i]);

      if(colorize) {
        item = item[colors[i]] || item[colors[colors.length -1]] || item;
      }

      length = realLength(item);
      padding = length < lengths[i] ? lengths[i] - length + options.columnSpacing  : options.columnSpacing ;
      rowtext += item + new Array(padding).join(' ');
    }
    
    output.push(rowtext);
  }
  
  // If we were passed colors, then assume the first row
  // is the headers for the rows
  if (colors) {
    headers = rows.splice(0, 1)[0];
    stringifyRow(headers, true);
  }
  
  rows.forEach(function (row) {
    stringifyRow(row, false);
  });
  
  return output.join('\n');
};

//
// ### function rowifyObjects (objs, properties, colors)
// #### @objs {Array} List of objects to create output for
// #### @properties {Array} List of properties to output
// #### @colors {Array} Set of colors to use for the headers
// Extracts the lists of `properties` from the specified `objs`
// and formats them according to `cliff.stringifyRows`.
//
cliff.stringifyObjectRows = cliff.rowifyObjects = function (objs, properties, colors, options) {
  var rows = [properties].concat(objs.map(function (obj) {
    return cliff.extractFrom(obj, properties);
  }));
  
  return cliff.stringifyRows(rows, colors, options);
};

//
// ### function putRows (level, rows, colors) 
// #### @level {String} Log-level to use
// #### @rows {Array} Array of rows to log at the specified level
// #### @colors {Array} Set of colors to use for the specified row(s) headers.
// Logs the stringified table result from `rows` at the appropriate `level` using
// `cliff.logger`. If `colors` are supplied then use those when stringifying `rows`.
//
cliff.putRows = function (level, rows, colors) {
  cliff.stringifyRows(rows, colors).split('\n').forEach(function (str) {
    logger.log(level, str);
  });
};

//
// ### function putObjectRows (level, rows, colors) 
// #### @level {String} Log-level to use
// #### @objs {Array} List of objects to create output for
// #### @properties {Array} List of properties to output
// #### @colors {Array} Set of colors to use for the headers
// Logs the stringified table result from `objs` at the appropriate `level` using
// `cliff.logger`. If `colors` are supplied then use those when stringifying `objs`.
//
cliff.putObjectRows = function (level, objs, properties, colors) {
  cliff.rowifyObjects(objs, properties, colors).split('\n').forEach(function (str) {
    logger.log(level, str);
  });
};

//
// ### function putObject (obj, [rewriters, padding])
// #### @obj {Object} Object to log to the command line
// #### @rewriters {Object} **Optional** Set of methods to rewrite certain object keys
// #### @padding {Number} **Optional** Length of padding to put around the output.
// Inspects the object `obj` on the command line rewriting any properties which match
// keys in `rewriters` if any. Adds additional `padding` if supplied.  
//
cliff.putObject = function (/*obj, [rewriters, padding] */) {
  var args = Array.prototype.slice.call(arguments),
      obj = args.shift(),
      padding = typeof args[args.length - 1] === 'number' && args.pop(),
      rewriters = typeof args[args.length -1] === 'object' && args.pop(),
      keys = Object.keys(obj).sort(),
      sorted = {}, 
      matchers = {},
      inspected;
  
  padding = padding || 0;
  rewriters = rewriters || {};
  
  function pad () {
    for (var i = 0; i < padding / 2; i++) {
      logger.data('');
    }
  }
  
  keys.forEach(function (key) {
    sorted[key] = obj[key];
  });
  
  inspected = cliff.inspect(sorted);
  
  Object.keys(rewriters).forEach(function (key) {
    matchers[key] = new RegExp(key);
  });
  
  pad();
  inspected.split('\n').forEach(function (line) {
    Object.keys(rewriters).forEach(function (key) {
      if (matchers[key].test(line)) {
        line = rewriters[key](line);
      }
    });
    logger.data(line);
  });
  pad();
};

cliff.stringifyLiteral = function stringifyLiteral (literal) {
  switch (cliff.typeOf(literal)) {
    case 'number'   : return literal + '';
    case 'null'     : return 'null';
    case 'undefined': return 'undefined';
    case 'boolean'  : return literal + '';
    default         : return literal;
  }
};

cliff.typeOf = function typeOf(value) {
  var s = typeof(value),
      types = [Object, Array, String, RegExp, Number, Function, Boolean, Date];

  if (s === 'object' || s === 'function') {
    if (value) {
      types.forEach(function (t) {
        if (value instanceof t) { 
          s = t.name.toLowerCase();
        }
      });
    } else { 
      s = 'null';
    }
  }
  
  return s;
};

function realLength(str) {
  return ("" + str).replace(/\u001b\[\d+m/g,'').length;
}

function longestElement(a) {
  var l = 0;
  for (var i = 0; i < a.length; i++) {
    var new_l = realLength(a[i]);
    if (l < new_l) {
      l = new_l;
    }
  }

  return l;
}
