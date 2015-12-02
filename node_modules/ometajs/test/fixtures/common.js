var ometajs = require('../../lib/ometajs'),
    fs = require('fs'),
    path = require('path'),
    assert = require('assert'),
    uglify = require('uglify-js');

ometajs.root = __dirname + '/../../lib/ometajs';

exports.ometajs = ometajs;

exports.loadFile = function loadFile(name) {
  return fs.readFileSync(
    path.resolve(
      __dirname,
      '../files',
      name + (path.extname(name) ? '' : '.ometajs')
    )
  ).toString()
};

exports.readFile = function readFile(name) {
  return fs.readFileSync(
    path.resolve(
      __dirname,
      '../files',
      name
    )
  ).toString();
};

exports.translate = function translate(name, options) {
  var code = exports.loadFile(name);
  return ometajs.translateCode(code, options);
};

exports.compile = function compile(code, options) {
  var ast = ometajs.parser.create(code).execute(),
      code = ometajs.compiler.create(ast, options).execute();

  // Just to check that it has correct syntax
  assert.doesNotThrow(function() {
    uglify(code);
  });

  return code;
};

exports.require = function compile(name) {
  return require(__dirname + '/../files/' + name);
};

exports.lexems = function(code) {
  var lexer = ometajs.lexer.create(code),
      lexems = [],
      lexem;

  while (lexem = lexer.token()) lexems.push(lexem);

  return lexems;
};

exports.parse = function(code) {
  var parser = ometajs.parser.create(code);

  return parser.execute();
};

exports.ap = function(code) {
  return new ometajs.core.AbstractParser(code);
}

exports.ag = function(code) {
  return new ometajs.core.AbstractGrammar(code);
}

exports.expressionify = ometajs.utils.expressionify;
exports.localify = ometajs.utils.localify;
