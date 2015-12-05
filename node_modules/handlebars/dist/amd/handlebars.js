define(['exports', 'module', './handlebars.runtime', './handlebars/compiler/ast', './handlebars/compiler/base', './handlebars/compiler/compiler', './handlebars/compiler/javascript-compiler', './handlebars/compiler/visitor', './handlebars/no-conflict'], function (exports, module, _handlebarsRuntime, _handlebarsCompilerAst, _handlebarsCompilerBase, _handlebarsCompilerCompiler, _handlebarsCompilerJavascriptCompiler, _handlebarsCompilerVisitor, _handlebarsNoConflict) {
  'use strict';

  var _interopRequire = function (obj) { return obj && obj.__esModule ? obj['default'] : obj; };

  var _runtime = _interopRequire(_handlebarsRuntime);

  // Compiler imports

  var _AST = _interopRequire(_handlebarsCompilerAst);

  var _JavaScriptCompiler = _interopRequire(_handlebarsCompilerJavascriptCompiler);

  var _Visitor = _interopRequire(_handlebarsCompilerVisitor);

  var _noConflict = _interopRequire(_handlebarsNoConflict);

  var _create = _runtime.create;
  function create() {
    var hb = _create();

    hb.compile = function (input, options) {
      return _handlebarsCompilerCompiler.compile(input, options, hb);
    };
    hb.precompile = function (input, options) {
      return _handlebarsCompilerCompiler.precompile(input, options, hb);
    };

    hb.AST = _AST;
    hb.Compiler = _handlebarsCompilerCompiler.Compiler;
    hb.JavaScriptCompiler = _JavaScriptCompiler;
    hb.Parser = _handlebarsCompilerBase.parser;
    hb.parse = _handlebarsCompilerBase.parse;

    return hb;
  }

  var inst = create();
  inst.create = create;

  _noConflict(inst);

  inst.Visitor = _Visitor;

  inst['default'] = inst;

  module.exports = inst;
});