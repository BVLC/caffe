'use strict';

var tls = require('tls');
var http = require('./http.js');
var util = require('util');
var inherits = util.inherits;

var debug;

if (util.debuglog) {
  debug = util.debuglog('https');
} else {
  debug = function (x) {
    if (process.env.NODE_DEBUG && /http/.test(process.env.NODE_DEBUG)) {
      console.error('HTTPS: %s', x);
    }
  };
}
function createConnection(port, host, options) {
  if (port !== null && typeof port === 'object') {
    options = port;
  } else if (host !== null && typeof host === 'object') {
    options = host;
  } else if (options === null || typeof options !== 'object') {
    options = {};
  }

  if (typeof port === 'number') {
    options.port = port;
  }

  if (typeof host === 'string') {
    options.host = host;
  }

  debug('createConnection', options);
  return tls.connect(options);
}


function Agent(options) {
  http.Agent.call(this, options);
  this.defaultPort = 443;
  this.protocol = 'https:';
}
inherits(Agent, http.Agent);
Agent.prototype.createConnection = createConnection;

Agent.prototype.getName = function(options) {
  var name = http.Agent.prototype.getName.call(this, options);

  name += ':';
  if (options.ca)
    name += options.ca;

  name += ':';
  if (options.cert)
    name += options.cert;

  name += ':';
  if (options.ciphers)
    name += options.ciphers;

  name += ':';
  if (options.key)
    name += options.key;

  name += ':';
  if (options.pfx)
    name += options.pfx;

  name += ':';
  if (options.rejectUnauthorized !== undefined)
    name += options.rejectUnauthorized;

  return name;
};

var globalAgent = new Agent();

exports.globalAgent = globalAgent;
exports.Agent = Agent;
