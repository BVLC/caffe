(function () {
  "use strict";

  var stack = require('./lib/stack')
    , gcf = require('express-chromeframe')
    , addSendJson = require('./lib/jason-res-json')
    , corsSession = require('./lib/connect-cors-session')
    , nowww = require('nowww')
    , queryparser = require('connect-queryparser')
    , xcors = require('connect-xcors')
    , cors
    , session
    , connect
    ;


  connect = stack.create(
      addSendJson()
  );

  cors = xcors();
  session = corsSession();
  connect.addMiddleware(
      nowww()
    , queryparser()
    , cors
    , session
    , gcf()
  );

  // TODO push into middleware layer
  cors.config.headers = cors.config.headers.concat(session.headers.slice());

  module.exports = connect;
}());
