(function () {
  "use strict";

  function create(protowares, connect, http) {
    var createServer
      , lowerwares = [];

    function addMiddleware(wares) {
      if (!Array.isArray(wares)) {
        wares = Array.prototype.slice.call(arguments);
      }
      lowerwares = lowerwares.concat(wares);
    }

    function middlewareHandler(middlewares) {
      if (!Array.isArray(middlewares)) {
        middlewares = Array.prototype.slice.call(arguments);
      }
      return createServer.apply(connect, lowerwares.concat(middlewares));
    }

    if (!connect) {
      connect = require('connect');
    }
    createServer = connect.createServer;
    connect.createServer = middlewareHandler;

    if (!http) {
      http = require('http');
    }

    if (!Array.isArray(protowares)) {
      protowares = Array.prototype.slice.call(arguments);
    }

    protowares.forEach(function (protoware) {
      protoware(http.ClientRequest.prototype, http.ServerResponse.prototype, http);
    });
    
    connect.addMiddleware = addMiddleware;
    return connect;
  }

  module.exports = create;
  module.exports.create = create;
}());
