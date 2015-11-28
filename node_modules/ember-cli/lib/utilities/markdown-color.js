'use strict';

var fs          = require('fs');
var existsSync  = require('exists-sync');

var chalk       = require('chalk');
var SilentError = require('silent-error');
var isArray     = require('lodash/lang/isArray');
var merge       = require('lodash/object/merge');

var colors = ['red', 'green', 'blue', 'cyan', 'magenta', 'yellow', 'black', 'white', 'grey', 'gray'];
var backgroundColors = ['bgRed', 'bgGreen', 'bgBlue', 'bgCyan', 'bgMagenta', 'bgYellow', 'bgWhite', 'bgBlack'];

module.exports = MarkdownColor;

/*
  Parses markdown and applies coloring to output by matching defined tokens and
  applying styling. Default outputs chalk.js coloring for terminal output.
  Options:
    tokens: pass additional tokens in the following format:
    {
      name:{
        token: '(Reserved)', // example of token
        pattern: /(Reserved)/g, // regexp pattern
        render: function(string) { return string + '!'} // function that takes and returns a string
      }
    }

    markdownOptions: object containing options to pass (or override) to markdown-it-terminal upon
      instantiation. See https://github.com/trabus/markdown-it-terminal/blob/master/index.js#L8

    renderStyles: pass an object to render color styles, default is chalk.js

  @class MarkdownColor
  @constructor
  @param {Object} [options]
*/
function MarkdownColor(options) {
  var optionTokens = options && options.tokens || {};
  var renderStyles = options && options.renderStyles || chalk;
  var tokens = this.generateTokens(renderStyles);
  var markdownOptions = options && options.markdownOptions || {};
  var markdown = require('markdown-it');
  var terminal = require('markdown-it-terminal');
  this.options = options || {};
  this.markdown = markdown().use(terminal, markdownOptions);
  this.tokens = merge(tokens, optionTokens);

  return this;
}

/*
  Lookup markdown file and render contents
  @method renderFile
  @param {String} [filePath]
  @param {Object} [options]
*/
MarkdownColor.prototype.renderFile = function (filePath, options) {
  var file;
  if (existsSync(filePath)) {
    file = fs.readFileSync(filePath, 'utf-8');
  } else {
    throw new SilentError('The file \'' + filePath + '\' doesn\'t exist.' +
      ' Please check your filePath');
  }

  return this.render(file, options);
};

/*
  Parse markdown and output as string
  @method render
  @param {String} [string]
  @param {Object} [options]
  @return {String}
*/
MarkdownColor.prototype.render = function (string, options) {
  var indent = options && options.indent || '';
  var input  = this.markdown.render(string);
  var styles = Object.keys(this.tokens);
  input = input.replace(/^/mg, indent);
  styles.reverse().map(function(style) {
    input = input.replace(this.tokens[style].pattern, this.tokens[style].render);
  }.bind(this));
  input = input.replace(/\~\^(.*)\~\^/g, escapeToken);
  return input;
};

/*
  Generate default style tokens
  @method generateTokens
  @return {Object}
*/
MarkdownColor.prototype.generateTokens = function (renderer) {
  var defaultTokens = {
    // ember-cli styles
    option: {
      name: 'option',
      token: '--option',
      pattern: /((--\w*\b)|(<options>))/g,
      render: renderStylesFactory(renderer, 'cyan')
    },
    default: {
      name: 'default',
      token: '(Default: default)',
      pattern: /(\(Default:\s.*\))/g,
      render: renderStylesFactory(renderer, 'cyan')
    },
    required: {
      name: 'required',
      token: '(Required)',
      pattern: /(\(Required\))/g,
      render: renderStylesFactory(renderer, 'cyan')
    }
  };

  var colorTokens = unshiftValue(colors.concat(backgroundColors).map(getToken), {}).reduce(setToken);
  return merge(colorTokens, defaultTokens);
};


/*
  Looks up multiple styles to apply to the rendered output
  @method renderStylesFactory
  @param {Object} [renderer] should match chalk.js format
  @param {String, Array} [styleNames]
  @return {Function} Function that will wrap the input string with styling
*/
MarkdownColor.prototype.renderStylesFactory = renderStylesFactory;
function renderStylesFactory(renderer, styleNames){
  var styles;
  if (isArray(styleNames)) {
    styles = styleNames.map(checkStyleName.bind(null, renderer));
  } else {
    styles = [checkStyleName(renderer, styleNames)];
  }
  return function(match, pattern) {
    return styles.reverse().reduce(function (previous, current) {
      return renderer[current](previous);
    }, pattern);
  };
}

/*
  Check to see if style exists on the renderer
  @param {Object} [renderer] should match chalk.js format
  @param {String} [name]
*/
function checkStyleName(renderer, name) {
  if (Object.keys(renderer.styles).indexOf(name) > -1) {
    return name;
  } else {
    throw new SilentError('The style \'' + name + '\' is not supported by the markdown renderer.');
  }
}

/*
  @param {String} [name]
  @param {Object} [options]
  @return {RegExp} Returns lookup pattern
*/
function getColorTokenRegex(name, options) {
  options = options || {};
  var start = options.start || '(?:<';
  var end = options.end || '>)';
  var close = options.close || '(?:<\/';
  var middle = options.middle || '(.*?)';
  var tag = start + name + end;
  var closeTag = close + name + end;
  var pattern = tag + middle + closeTag;
  return new RegExp(pattern, 'g');
}

/*
  @param {String} [name]
  @param {Object} [options]
  @return {Object} Returns token object
*/
function getToken(name, options) {
  var renderer = options && options.renderer || chalk;
  return {
    name: name,
    token: '<' + name + '></' + name + '>',
    pattern: getColorTokenRegex(name),
    render: renderStylesFactory(renderer, name)
  };
}

function setToken(result, color) {
  result[color.name] = color;
  return result;
}

function escapeToken(match, pattern){
  var output = pattern.replace(/\~/g,'');
  return '<' + output + '>';
}

function unshiftValue(array, value) {
  array.unshift(value);
  return array;
}

/*
Formatting colors for ember-cli help

white: ember serve
yellow: <arg-option, >
cyan: --port, --important-option
cyan: (Default: something), (Default: 4200)
white: Description 1, Description 2
cyan: (Required)
*/
