define(['exports', './parser', './ast', './whitespace-control', './helpers', '../utils'], function (exports, _parser, _ast, _whitespaceControl, _helpers, _utils) {
  'use strict';

  var _interopRequire = function (obj) { return obj && obj.__esModule ? obj['default'] : obj; };

  exports.__esModule = true;
  exports.parse = parse;

  var _parser2 = _interopRequire(_parser);

  var _AST = _interopRequire(_ast);

  var _WhitespaceControl = _interopRequire(_whitespaceControl);

  exports.parser = _parser2;

  var yy = {};
  _utils.extend(yy, _helpers, _AST);

  function parse(input, options) {
    // Just return if an already-compiled AST was passed in.
    if (input.type === 'Program') {
      return input;
    }

    _parser2.yy = yy;

    // Altering the shared object here, but this is ok as parser is a sync operation
    yy.locInfo = function (locInfo) {
      return new yy.SourceLocation(options && options.srcName, locInfo);
    };

    var strip = new _WhitespaceControl();
    return strip.accept(_parser2.parse(input));
  }
});