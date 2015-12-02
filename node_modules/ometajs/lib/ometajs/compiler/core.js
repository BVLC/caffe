var util = require('util'),
    ometajs = require('../../ometajs'),
    utils = ometajs.utils;

//
// ### function Compiler (ast, options)
// #### @ast {Array} source code AST
// #### @options {Object} (optional) compiler options
// Compiler constructor
//
function Compiler(ast, options) {
  ometajs.compiler.ir.call(this);

  this.ast = ast;
  this.options = options || {};
  this.localId = 0;

  if (this.options.grammars !== false) {
    this.addGrammars(this.options.root || 'ometajs');
  }
};
util.inherits(Compiler, ometajs.compiler.ir);
exports.Compiler = Compiler;

//
// ### function addGrammars (root)
// #### @root {String} path to module
// Add globlas
//
Compiler.prototype.addGrammars = function addGrammars(root) {
  this.enter('code', function() {
    // Enter chunks
    this.push([]);

    this.insert('var ometajs_ = require(', JSON.stringify(root), ');');

    var keys = Object.keys(ometajs.grammars);
    for (var i = 0; i < keys.length; i++) {
      this.insert('var ', keys[i], ' = ometajs_.grammars.', keys[i], ';');
    }

    // Leave chunks
    this.pop();
  });
};

//
// ### function create (options, ast)
// #### @ast {Array} source code AST
// #### @options {Object} (optional) compiler options
// Compiler constructor wrapper
//
exports.create = function create(ast, options) {
  return new Compiler(ast, options);
};

//
// ### function execute ()
// Compiles code and returns function
//
Compiler.prototype.execute = function execute() {
  var self = this;

  this.ast.forEach(function(chunk) {
    switch (chunk[0]) {
      case 'code':
        self.push([ 'code', [ chunk[1] ] ]);
        self.pop();
        break;
      case 'grammar':
        self.compileGrammar(chunk.slice(1));
        break;
    }
  });

  return this.render();
};

//
// ### function compileGrammar (ast)
// #### @ast {Array} source code AST
// Compiles grammar and pushes it's contents into the internal buffer
//
Compiler.prototype.compileGrammar = function compileGrammar(ast) {
  var self = this,
      name = ast[0],
      parent = ast[1] || 'AbstractGrammar';

  // Prelude

  this.enter(['grammar', name, parent], function() {
    // Enter rules
    this.push([]);

    // Translate rules
    ast[2].forEach(function(rule) {
      if (rule[0] !== 'rule') throw new Error('unexpected: ' + rule[0]);

      self.enter(['rule', rule[1]], function() {
        var vars = [];
        self.insert(vars);

        // Enter rule expressions
        self.push([]);

        // Translate expressions
        self.compileExpressions(rule[2], vars);

        // Leave expressions
        this.pop();
      });
    });

    // Leave rules
    this.pop();
  });
};

//
// ### function compileExpressions (exps, args)
// #### @exps {Array} List of expressions
// #### @args {Array} Rule arguments
// Compiles expressions
//
Compiler.prototype.compileExpressions = function compileExpressions(exps,
                                                                    args) {
  var self = this;
  exps.every(function(exp, i, exps) {
    return self.compileExpression(
      exp,
      args,
      exps.slice(i + 1)
    );
  });
}

//
// ### function compileExpression (exp, args)
// #### @exp {Array} expression
// #### @args {Array} Rule arguments
// Compiles expression and pushes it's contents into the internal buffer
//
Compiler.prototype.compileExpression = function compileExpression(exp,
                                                                  args,
                                                                  rest) {
  var self = this;

  switch(exp[0]) {
    case 'null':
      this.insert(['match', 'null']);
      break;
    case 'bool':
    case 'number':
    case 'string':
      this.insert(['match', JSON.stringify(exp[1])]);
      break;
    case 'seq':
      this.insert(['seq', JSON.stringify(exp[1])]);
      break;
    case 're':
      this.insert(['re', exp[1]]);
      break;
    case 'match':
      if (exp[1] !== null) throw new Error('Not implemented yet');
      this.insert(['rule', exp[2]]);
      break;
    case 'super':
      var match = exp[1];
      if (match[1] !== null) throw new Error('Not implemented yet');

      this.insert(['super', match[2]]);
      break;
    case 'call':
      if (exp[1] !== null) throw new Error('Not implemented yet');
      this.insert(['rule', exp[2], exp[3]]);
      break;
    case 'arg':
      this.compileExpression(exp[1], args, []);
      this.insert(['store', exp[2]]);
      if (args.indexOf(exp[2]) === -1) {
        args.push(exp[2]);
      }
      break;
    case 'choice':
      this.enter('choice', function() {
        // Enter choices
        this.push([]);

        // Translate every body
        for (var i = 1; i < exp.length; i++) {
          var args = [];

          // Enter choice expressions
          this.push([]);
          this.push(['atomic', args]);
          this.push([]);

          self.compileExpressions(exp[i], args);

          // Leave choice expressions
          this.pop();
          this.pop();
          this.pop();
        }

        // Leave choices
        this.pop();
      });
      break;
    case 'list':
    case 'chars':
      this.enter(exp[0], function() {
        // Enter expressions
        this.push([]);

        for (var i = 1; i < exp.length; i++) {
          this.compileExpression(exp[i], args, []);
        }

        // Leave expressions
        this.pop();
      });
      break;
    case 'body':
    case 'result':
      this.insert(['exec', exp[1]]);
      break;
    case 'any':
    case 'many':
    case 'optional':
      this.enter(exp[0], function() {
        // Enter expressions
        this.push([]);

        this.compileExpression(exp[1], args, []);

        // Leave expressions
        this.pop();
      });
      break;
    case 'lookahead':
    case 'not':
      this.enter(exp[0], function() {
        // Enter expressions
        this.push([]);

        this.compileExpression(exp[1], args, []);

        // Leave expressions
        this.pop();
      });
      break;
    case 'predicate':
      this.insert(['predicate', exp[1]]);
      break;
    case 'local':
      var local = utils.localify(exp[1], this.localId);
      this.localId += local.vars.length;

      args.push.apply(args, local.vars);

      this.insert(['predicate', local.before]);

      // ?(set, true) (body ?(revert, true) || ?(revert, false))
      this.enter('choice', function() {
        this.push([]);

        this.push([]);
        this.compileExpressions(rest, args);
        this.insert(['predicate', local.afterSuccess]);
        this.pop();

        this.push([]);
        this.insert(['predicate', local.afterFail]);
        this.pop();

        this.pop();
      });

      // We'll process rest of operations manually
      return false;
    default:
      throw new Error('Unexpected node type:' + exp[0]);
  }

  return true;
};
