var lexer = exports;

//
// ### function Lexer (code)
// #### @code {String} source code
// Lexer constructor
//
function Lexer(code) {
  this.code = code.trim();
  this.offset = 0;
  this.noTrim = false;
};

//
// ### function create (code)
// #### @code {String} source code
// Lexer's constructor wrapper
//
lexer.create = function create(code) {
  return new Lexer(code);
};

//
// ### function token ()
// Returns token or false (may throw)
//
Lexer.prototype.token = function token() {
  var result;

  return this.match('space', 0, /^(\/\/[^\r\n]*)([\r\n]|$)/) ||
         this.match('space', 0, /^\/\*(?:.|\r|\n)*\*\//) ||
         this.match('space', 0, /^[\s\r\n]+/) ||
         this.match('name', 0, /^(?:[$_a-z][$_a-z0-9]*|@[$_a-z0-9]+)/i) ||
         this.match('re', 1, /^@(\/(?:[^\/\n]|\\\/)+\/[gim]*)/) ||
         this.match('re', 0, /^\/(?:[^\/\\\n]|\\.)*\/[gim]*/i) ||
         this.match('punc', 1,
                    /^(->|<:|&&|\|\||[()\[\]{}<>,.~!+\-^=|:;*?&%\/])/) ||
         this.match('number', 0, /^-?\d+(?:\.\d+)?/) ||
         this.match('sequence', 0, /^``(?:[^'\\]|\\.)*''/, function(val) {
           return val.slice(2, -2);
         }) ||
         this.match('string', 1, /^([#`][$_\w\-]+)/, function(val) {
           return val.slice(1);
         }) ||
         this.match('string', 0, /^'(?:[^'\\]|\\.)*'/, function(val) {
           function swap(quote) {
             return quote === '"' ? '\'' : '"';
           }
           val = val.replace(/["']/g, swap);
           return JSON.parse(val).replace(/["']/g, swap);
         }) ||
         this.match('token', 0, /^"(?:[^"\\]|\\.)*"/, function(val) {
           return JSON.parse(val);
         }) ||
         this.unexpected();
};

//
// ### function match (type, index, re, sanitizer)
// #### @type {String} Token type
// #### @index {Number} Number of match in regexp to pick as value
// #### @re {RegExp} regexp itself
// #### @sanitizer {Function} (optional) preprocess value
// Tries to match current code against regexp and returns token on success
//
Lexer.prototype.match = function match(type, index, re, sanitizer) {
  var match = this.code.match(re);
  if (!match) return false;

  var offset = this.offset,
      value = match[index];

  this.skip(match[index].length);

  if (type === 'name' && value === 'ometa') type = 'keyword';
  if (sanitizer !== undefined) value = sanitizer(value);

  return { type: type, value: value, offset: offset };
};

//
// ### function trim ()
// Removes spaces at start and end of source code
//
Lexer.prototype.trim = function trim() {
  var code = this.code;

  this.code = this.code.replace(
    /^([\s\r\n]*|(\/\/[^\r\n]*)[\r\n]|\/\*(?:.|\r|\n)*\*\/)*/,
    ''
  );
  this.offset += code.length - this.code.length;
};

//
// ### function skip (chars)
// #### @char {Number} number chars to skip
// Skips number of chars
//
Lexer.prototype.skip = function skip(chars) {
  var code = this.code;

  this.code = this.code.slice(chars);

  this.offset += code.length - this.code.length;
};

Lexer.prototype.stringify = function stringify(token) {
  if (!token) return '';
  switch (token.type) {
    case 'string':
    case 'token':
      return JSON.stringify(token.value);
    case 'space':
      return token.value;
    default:
      return token.value;
  }
};

//
// ### function unexpected ()
// Returns false if reached end of code or throws exception
//
Lexer.prototype.unexpected = function unexpected() {
  if (this.code.length === 0) return false;
  throw new Error('Lexer failer at: "' + this.code.slice(0, 11) + '"');
};
