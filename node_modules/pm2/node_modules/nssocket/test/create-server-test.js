/*
 * create-server-test.js : namespace socket unit test for TLS.
 *
 * (C) 2011, Charlie Robbins, Paolo Fragomeni, & the Contributors.
 *
 */

var assert = require('assert'),
    fs = require('fs'),
    net = require('net'),
    path = require('path'),
    tls = require('tls'),
    vows = require('vows'),
    nssocket = require('../lib/nssocket');


function getBatch() {
  var args = Array.prototype.slice.call(arguments),
      res = {};

  return {
    "the createServer() method": {
      topic: function () {
        var outbound = new nssocket.NsSocket(),
            server = nssocket.createServer(this.callback.bind(null, null, outbound));

        server.listen.apply(server, args.concat(function () {
          outbound.connect.apply(outbound, args);
        }));
      },
      "should create a full-duplex namespaced socket": {
        topic: function (outbound, inbound) {
          outbound.on(['data', 'here', 'is'], this.callback.bind(outbound, null));
          inbound.send(['here', 'is'], 'something.');
        },
        "should handle namespaced events": function (_, data) {
          assert.isArray(this.event);
          assert.lengthOf(this.event, 3);
          assert.isString(this.event[0]);
          assert.isString(this.event[1]);
          assert.isString(this.event[2]);
          assert.isString(data);
          assert.equal(this.event[0], 'data');
          assert.equal(this.event[1], 'here');
          assert.equal(this.event[2], 'is');
          assert.equal(data, 'something.');
        }
      }
    }
  };
}

var PORT = 9564,
    HOST = "127.0.0.1",
    PIPE = path.join(__dirname, "fixtures", "nssocket.sock"),
    HOSTNAME = "localhost";

vows.describe('nssocket/create-server').addBatch({
  "When using NsSocket": {
    "with `(PORT)` argument": getBatch(PORT),
    "with `(PORT, HOST)` arguments": getBatch(PORT + 1, HOST),
    "with `(PORT, HOSTNAME)` argument": getBatch(PORT + 2, HOSTNAME),
    "with `(PIPE)` argument": getBatch(PIPE)
  }
}).addBatch({
  "When tests are finished": {
    "`PIPE` should be removed": function () {
      fs.unlinkSync(PIPE);
    }
  }
}).export(module);

