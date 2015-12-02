//
// HTTP Error objectst
//
var util = require('util');

exports.NotModified = function () {
  this.status = 304;
  this.options = {
    removeContentHeaders: true
  };
};

util.inherits(exports.NotModified, Error);

exports.BadRequest = function (msg) {
  msg = msg || 'Bad request';

  this.status = 400;
  this.headers = {};
  this.message = msg;
  this.body = { error: msg };
};

util.inherits(exports.BadRequest, Error);

exports.NotAuthorized = function (msg) {
  msg = msg || 'Not Authorized';

  this.status = 401;
  this.headers = {};
  this.message = msg;
  this.body = { error: msg };
};

util.inherits(exports.NotAuthorized, Error);

exports.Forbidden = function (msg) {
  msg = msg || 'Not Authorized';

  this.status = 403;
  this.headers = {};
  this.message = msg;
  this.body = { error: msg };
};

util.inherits(exports.Forbidden, Error);

exports.NotFound = function (msg) {
  msg = msg || 'Not Found';

  this.status = 404;
  this.headers = {};
  this.message = msg;
  this.body = { error: msg };
};

util.inherits(exports.NotFound, Error);

exports.MethodNotAllowed = function (allowed) {
  var msg = 'method not allowed.';

  this.status = 405;
  this.headers = { allow: allowed };
  this.message = msg;
  this.body = { error: msg };
};

util.inherits(exports.MethodNotAllowed, Error);

exports.NotAcceptable = function (accept) {
  var msg = 'cannot generate "' + accept + '" response';

  this.status = 406;
  this.headers = {};
  this.message = msg;
  this.body = {
    error: msg,
    only: 'application/json'
  };
};

util.inherits(exports.NotAcceptable, Error);

exports.NotImplemented = function (msg) {
  msg = msg || 'Not Implemented';

  this.status = 501;
  this.headers = {};
  this.message = msg;
  this.body = { error: msg };
};

util.inherits(exports.NotImplemented, Error);
