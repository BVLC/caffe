
/**
 * Module dependencies.
 */

var fmt = require('util').format;
var amp = require('amp');

/**
 * Proxy methods.
 */

var methods = [
  'push',
  'pop',
  'shift',
  'unshift'
];

/**
 * Expose `Message`.
 */

module.exports = Message;

/**
 * Initialize an AMP message with the
 * given `args` or message buffer.
 *
 * @param {Array|Buffer} args or blob
 * @api public
 */

function Message(args) {
  if (Buffer.isBuffer(args)) args = decode(args);
  this.args = args || [];
}

// proxy methods

methods.forEach(function(method){
  Message.prototype[method] = function(){
    return this.args[method].apply(this.args, arguments);
  };
});

/**
 * Inspect the message.
 *
 * @return {String}
 * @api public
 */

Message.prototype.inspect = function(){
  return fmt('<Message args=%d size=%d>',
    this.args.length,
    this.toBuffer().length);
};

/**
 * Return an encoded AMP message.
 *
 * @return {Buffer}
 * @api public
 */

Message.prototype.toBuffer = function(){
  return encode(this.args);
};

/**
 * Decode `msg` and unpack all args.
 *
 * @param {Buffer} msg
 * @return {Array}
 * @api private
 */

function decode(msg) {
  var args = amp.decode(msg);
  
  for (var i = 0; i < args.length; i++) {
    args[i] = unpack(args[i]);
  }

  return args;
}

/**
 * Encode and pack all `args`.
 *
 * @param {Array} args
 * @return {Buffer}
 * @api private
 */

function encode(args) {
  var tmp = new Array(args.length);

  for (var i = 0; i < args.length; i++) {
    tmp[i] = pack(args[i]);
  }

  return amp.encode(tmp);
}

/**
 * Pack `arg`.
 *
 * @param {Mixed} arg
 * @return {Buffer}
 * @api private
 */

function pack(arg) {
  // blob
  if (Buffer.isBuffer(arg)) return arg;

  // string
  if ('string' == typeof arg) return new Buffer('s:' + arg);

  // undefined
  if (arg === undefined) arg = null;

  // json
  return new Buffer('j:' + JSON.stringify(arg));
}

/**
 * Unpack `arg`.
 *
 * @param {Buffer} arg
 * @return {Mixed}
 * @api private
 */

function unpack(arg) {
  // json
  if (isJSON(arg)) return JSON.parse(arg.slice(2));

  // string
  if (isString(arg)) return arg.slice(2).toString();
 
  // blob
  return arg;
}

/**
 * String argument.
 */

function isString(arg) {
  return 115 == arg[0] && 58 == arg[1];
}

/**
 * JSON argument.
 */

function isJSON(arg) {
  return 106 == arg[0] && 58 == arg[1];
}

/**
 * ID argument.
 */

function isId(arg) {
  return 105 == arg[0] && 58 == arg[1];
}
