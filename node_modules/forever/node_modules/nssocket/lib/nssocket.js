/*
 * nssocket.js - Wraps a TLS/TCP socket to emit namespace events also auto-buffers.
 *
 * (C) 2011, Charlie Robbins, Paolo Fragomeni, & the Contributors.
 *
 */

var net = require('net'),
    tls = require('tls'),
    util = require('util'),
    events2 = require('eventemitter2'),
    Lazy = require('lazy'),
    common = require('./common');

//
// ### function NsSocket (socket, options)
// #### @socket {Object} TCP or TLS 'socket' either from a 'connect' 'new' or from a server
// #### @options {Object} Options for this NsSocket
// NameSpace Socket, NsSocket, is a thin wrapper above TLS/TCP.
// It provides automatic buffering and name space based data emits.
//
var NsSocket = exports.NsSocket = function (socket, options) {
  if (!(this instanceof NsSocket)) {
    return new NsSocket(socket, options);
  }

  //
  // If there is no Socket instnace to wrap,
  // create one.
  //
  if (!options) {
    options = socket;
    socket = common.createSocket(options);
  }

  //
  // Options should be
  //
  //    {
  //      type : 'tcp' or 'tls',
  //      delimiter : '::', delimiter that separates between segments
  //      msgLength : 3 //number of segments in a complete message
  //    }
  //
  options = options || {};

  var self = this,
      startName;

  //
  // Setup underlying socket state.
  //
  this.socket     = socket;
  this.connected  = options.connected || socket.writable && socket.readable || false;

  //
  // Setup reconnect options.
  //
  this._reconnect = options.reconnect  || false;
  this.retry      = {
    retries:  0,
    max:      options.maxRetries || 10,
    interval: options.retryInterval || 5000,
    wait:     options.retryInterval || 5000
  };

  //
  // Setup default instance variables.
  //
  this._options   = options;
  this._type      = options.type || 'tcp4',
  this._delimiter = options.delimiter || '::';

  //
  // Setup encode format.
  //
  this._encode   = options.encode || JSON.stringify;
  this._decode   = options.decode || JSON.parse;

  events2.EventEmitter2.call(this, {
    delimiter: this._delimiter,
    wildcard: true,
    maxListeners: options.maxListeners || 10
  });

  this._setup();
};

//
// Inherit from `events2.EventEmitter2`.
//
util.inherits(NsSocket, events2.EventEmitter2);

//
// ### function createServer (options, connectionListener)
// #### @options {Object} **Optional**
// Creates a new TCP/TLS server which wraps every incoming connection
// in an instance of `NsSocket`.
//
exports.createServer = function createServer(options, connectionListener) {
  if (!connectionListener && typeof options === 'function') {
    connectionListener = options;
    options = {};
  }

  options.type      = options.type || 'tcp4';
  options.delimiter = options.delimiter || '::';

  function onConnection (socket) {
    //
    // Incoming socket connections cannot reconnect
    // by definition.
    //
    options.reconnect = false;
    connectionListener(new NsSocket(socket, options));
  }

  return options.type === 'tls'
    ? tls.createServer(options, onConnection)
    : net.createServer(options, onConnection);
};

//
// ### function send (data, callback)
// #### @event {Array|string} The array (or string) that holds the event name
// #### @data {Literal|Object} The data to be sent with the event.
// #### @callback {Function} the callback function when send is done sending
// The send function follows write/send rules for TCP/TLS/UDP
// in that the callback is called when sending is complete, not when delivered
//
NsSocket.prototype.send = function send(event, data, callback) {
  var dataType = typeof data,
      message;

  // rebinds
  if (typeof event === 'string') {
    event = event.split(this._delimiter);
  }

  if (dataType === 'undefined' || dataType === 'function') {
    callback = data;
    data = null;
  }

  // if we aren't connected/socketed, then error
  if (!this.socket || !this.connected) {
    return this.emit('error', new Error('NsSocket: sending on a bad socket'));
  }

  message = Buffer(this._encode(event.concat(data)) + '\n');

  if (this.socket.cleartext) {
    this.socket.cleartext.write(message, callback);
  }
  else {
    // now actually write to the socket
    this.socket.write(message, callback);
  }
};

//
// ### function data (event, callback)
// #### @event {Array|string} Namespaced `data` event to listen to.
// #### @callback {function} Continuation to call when the event is raised.
// Shorthand function for listening to `['data', '*']` events.
//
NsSocket.prototype.data = function (event, callback) {
  if (typeof event === 'string') {
    event = event.split(this._delimiter);
  }

  this.on(['data'].concat(event), callback);
};

NsSocket.prototype.undata = function (event, listener) {
  this.off(['data'].concat(event), listener);
};

//
// ### function data (event, callback)
// #### @event {Array|string} Namespaced `data` event to listen to once.
// #### @callback {function} Continuation to call when the event is raised.
// Shorthand function for listening to `['data', '*']` events once.
//
NsSocket.prototype.dataOnce = function (event, callback) {
  if (typeof event === 'string') {
    event = event.split(this._delimiter);
  }

  this.once(['data'].concat(event), callback);
};

//
// ### function setIdle (time, callback)
// #### @time {Integer} how often to emit idle
// Set the idle/timeout timer
//
NsSocket.prototype.setIdle = function setIdle(time) {
  this.socket.setTimeout(time);
  this._timeout = time;
};

//
// ### function destroy (void)
// #### forcibly destroys this nsSocket, unregister socket, remove all callbacks
//
NsSocket.prototype.destroy = function destroy() {
  if (this.socket) {
    try {
      this.socket.end(); // send FIN
      this.socket.destroy(); // make sure fd's are gone
    }
    catch (ex) {
      // do nothing on errors
    }
  }

  // clear buffer
  this.data = '';
  this.emit('destroy');

  // this should forcibly remove EVERY listener
  this.removeAllListeners();
};

//
// ### function end (void)
// #### closes the underlying socket, recommend you call destroy after
//
NsSocket.prototype.end = function end() {
  var hadErr;
  this.connected = false;

  if (this.socket) {
    try {
      this.socket.end();
    }
    catch (ex) {
      this.emit('error', ex);
      hadErr = true;
      return;
    }

    this.socket = null;
  }

  return this.emit('close', hadErr || undefined);
};

//
// ### function connect (port[, host, callback])
// A passthrough to the underlying socket's connect function
//
NsSocket.prototype.connect = function connect(/*port, host, callback*/) {
  var args = Array.prototype.slice.call(arguments),
      self = this,
      callback,
      host,
      port;

  args.forEach(function handle(arg) {
    var type = typeof arg;
    switch (type) {
      case 'number':
        port = arg;
        break;
      case 'string':
        host = arg;
        break;
      case 'function':
        callback = arg;
        break;
      default:
        self.emit('error', new Error('bad argument to connect'));
        break;
    }
  });

  this.port = port || this.port;
  this.host = host || this.host;
  this.host = this.host || '127.0.0.1';
  args = this.port ? [this.port, this.host] : [this.host];

  if (callback) {
    args.push(callback);
  }

  if (['tcp4', 'tls'].indexOf(this._type) === -1) {
    return this.emit('error', new Error('Unknown Socket Type'));
  }

  var errHandlers = self.listeners('error');

  if (errHandlers.length > 0) {
    //
    // copy the last error from nssocker onto the error event.
    //
    self.socket._events.error = errHandlers[errHandlers.length-1];
  }

  this.connected = true;
  this.socket.connect.apply(this.socket, args);
};

//
// ### function reconnect ()
// Attempts to reconnect the current socket on `close` or `error`.
// This instance will attempt to reconnect until `this.retry.max` is reached,
// with an interval increasing by powers of 10.
//
NsSocket.prototype.reconnect = function reconnect() {
  var self = this;

  //
  // Helper function containing the core reconnect logic
  //
  function doReconnect() {
    //
    // Cleanup and recreate the socket associated
    // with this instance.
    //
    self.retry.waiting = true;
    self.socket.removeAllListeners();
    self.socket = common.createSocket(self._options);

    //
    // Cleanup reconnect logic once the socket connects
    //
    self.socket.once('connect', function () {
      self.retry.waiting = false;
      self.retry.retries = 0;
    });

    //
    // Attempt to reconnect the socket
    //
    self._setup();
    self.connect();
  }

  //
  // Helper function which attempts to retry if
  // it is less than the maximum
  //
  function tryReconnect() {
    self.retry.retries++;
    if (self.retry.retries >= self.retry.max) {
      return self.emit('error', new Error('Did not reconnect after maximum retries: ' + self.retry.max));
    }

    doReconnect();
  }

  this.retry.wait = this.retry.interval * Math.pow(10, this.retry.retries);
  setTimeout(tryReconnect, this.retry.wait);
};

//
// ### @private function _setup ()
// Sets up the underlying socket associate with this instance.
//
NsSocket.prototype._setup = function () {
  var self = this,
      startName;

  function bindData(sock) {
    Lazy(sock)
      .lines
      .map(String)
      .forEach(self._onData.bind(self));
  }

  //
  // Because of how the code node.js `tls` module works, we have
  // to separate some bindings. The main difference is on
  // connection, some socket activities.
  //
  if (this._type === 'tcp4') {
    startName = 'connect';

    bindData(this.socket);

    // create a stub for the setKeepAlive functionality
    this.setKeepAlive = function () {
      self.socket.setKeepAlive.apply(self.socket, arguments);
    };
  }
  else if (this._type === 'tls') {
    startName = 'secureConnection';

    if (this.connected) {
      bindData(self.socket);
    } else {
      this.socket.once('connect', function () {
        bindData(self.socket.cleartext);
      });
    }

    // create a stub for the setKeepAlive functionality
    this.setKeepAlive = function () {
      self.socket.socket.setKeepAlive.apply(self.socket.socket, arguments);
    };
  }
  else {
    // bad arguments, so throw an error
    this.emit('error', new Error('Bad Option Argument [type]'));
    return null;
  }

  // make sure we listen to the underlying socket
  this.socket.on(startName, this._onStart.bind(this));
  this.socket.on('close',   this._onClose.bind(this));

  if (this.socket.socket) {
    //
    // otherwise we get a error passed from net.js
    // they need to backport the fix from v5 to v4
    //
    this.socket.socket.on('error', this._onError.bind(this));
  }

  this.socket.on('error',   this._onError.bind(this));
  this.socket.on('timeout', this._onIdle.bind(this));
};

//
// ### @private function _onStart ()
// Emits a start event when the underlying socket finish connecting
// might be used to do other activities.
//
NsSocket.prototype._onStart = function _onStart() {
  this.emit('start');
};

//
// ### @private function _onData (message)
// #### @message {String} literal message from the data event of the socket
// Messages are assumed to be delimited properly (if using nssocket to send)
// otherwise the delimiter should exist at the end of every message
// We assume messages arrive in order.
//
NsSocket.prototype._onData = function _onData(message) {
  var parsed,
      data;

  try {
    parsed = this._decode(message);
    data = parsed.pop();
  }
  catch (err) {
    //
    // Don't do anything, assume that the message is only partially
    // received.
    //
  }
  this.emit(['data'].concat(parsed), data);
};

//
// ### @private function _onClose (hadError)
// #### @hadError {Boolean} true if there was an error, which then include the
// actual error included by the underlying socket
//
NsSocket.prototype._onClose = function _onClose(hadError) {
  this.connected = false;

  if (hadError) {
    this.emit('close', hadError, arguments[1]);
  }
  else {
    this.emit('close');
  }

  if (this._reconnect) {
    this.reconnect();
  }
};

//
// ### @private function _onError (error)
// #### @error {Error} emits and error event in place of the socket
// Error event is raise with an error if there was one
//
NsSocket.prototype._onError = function _onError(error) {
  this.connected = false;

  if (!this._reconnect) {
    return this.emit('error', error || new Error('An Unknown Error occured'));
  }

  this.reconnect();
};

//
// ### @private function _onIdle ()
// #### Emits the idle event (based on timeout)
//
NsSocket.prototype._onIdle = function _onIdle() {
  this.emit('idle');
  if (this._timeout) {
    this.socket.setTimeout(this._timeout);
  }
};
