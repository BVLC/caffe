var api = exports;

var ometajs = require('../ometajs'),
    path = require('path'),
    fs = require('fs');

//
// ### function compile (code, options)
// #### @code {String} Ometajs source code
// #### @options {Object} Compiler options
// Compiles ometajs to javascript
//
api.compile = function compile(code, options) {
  options || (options = {});
  var ast = ometajs.parser.create(code).execute(),
      compiler = ometajs.compiler.create(ast, options);

  return compiler.execute();
};

//
// Allow require('filename.ometajs')
//
require.extensions['.ometajs'] = function loadExtension(module, filename) {
  var content = fs.readFileSync(filename).toString(),
      source = api.compile(content, {
        root: path.resolve(__dirname, '../../lib/ometajs')
      });

  module._compile(source, filename);
};
