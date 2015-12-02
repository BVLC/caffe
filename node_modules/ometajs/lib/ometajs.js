//
// OmetaJS
//

var ometajs = exports;

// Export utils
ometajs.utils = require('./ometajs/utils');

// Export lexer
ometajs.lexer = require('./ometajs/lexer');

// Export compiler
ometajs.compiler = {};
ometajs.compiler.ast = require('./ometajs/compiler/ast');
ometajs.compiler.ir = require('./ometajs/compiler/ir');
ometajs.compiler.create = require('./ometajs/compiler/core').create;

// Export parser
ometajs.parser = require('./ometajs/parser');

// Compiler routines
ometajs.core = {};
ometajs.core.AbstractParser = require('./ometajs/core/parser');
ometajs.core.AbstractGrammar = require('./ometajs/core/grammar');

// Export legacy methods
var firstTime = false;
Object.defineProperty(ometajs, 'globals', {
  get: function () {
    if (!firstTime) {
      firstTime = true;
      console.error('!!!\n' +
                    '!!! Warning: you\'re using grammar compiled with ' +
                    'previous version of ometajs. Please recompile it with ' +
                    'the newest one\n' +
                    '!!!\n');
      ometajs.utils.extend(ometajs, require('./ometajs/legacy'));
    }
    return require('./ometajs/legacy');
  }
});

// Export grammars
ometajs.grammars = require('./ometajs/grammars');

// Export API
ometajs.compile = require('./ometajs/api').compile;

// Export CLI
ometajs.cli = require('./ometajs/cli');
