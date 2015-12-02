/*
 * nssocket-test.js : namespace socket unit test for TLS.
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
    NsSocket = require('../lib/nssocket').NsSocket;

var TLS_PORT = 50305,
    fixturesDir = path.join(__dirname, 'fixtures');

var serverOpts = {
  key:  fs.readFileSync(path.join(fixturesDir, 'ryans-key.pem')),
  cert: fs.readFileSync(path.join(fixturesDir, 'ryans-cert.pem')),
  ca:   fs.readFileSync(path.join(fixturesDir, 'ryans-csr.pem'))
};

var tlsServer = tls.createServer(serverOpts),
    tlsOpt;

tlsOpt = {
  type:      'tls',
  delimiter: '::'
};

tlsServer.listen(TLS_PORT);

vows.describe('nssocket/tls').addBatch({
  "When using NsSocket with TLS": {
    topic: new NsSocket(tlsOpt),
    "should create a wrapped socket": function (outbound) {
      assert.instanceOf(outbound, NsSocket);
    },
    "should have the proper configuration settings": function (outbound) {
      assert.equal(outbound._type, tlsOpt.type);
      assert.equal(outbound._delimiter, tlsOpt.delimiter);
    },
    "the connect() method": {
      topic: function (outbound) {
        var that = this;
        tlsServer.on('secureConnection', this.callback.bind(null, null, outbound));
        outbound.connect(TLS_PORT);
      },
      "should actually connect": function (_, outbound, inbound) {
        assert.instanceOf(outbound, NsSocket);
        if (!inbound.authorized) {
          console.log('Certificate is not authorized: ' + inbound.authorizationError);
        }
      },
      "the on() method": {
        topic: function (outbound, inbound) {
          outbound.on(['data', 'here', 'is'], this.callback.bind(outbound, null));
          inbound.write(JSON.stringify(['here', 'is', 'something']) + '\n');
        },
        "should handle namespaced events": function (_, data) {
          assert.isString(data);
          assert.isArray(this.event);
          assert.lengthOf(this.event, 3);
          assert.equal(this.event[0], 'data');
          assert.equal(this.event[1], 'here');
          assert.equal(this.event[2], 'is');
          assert.equal(data, 'something');
        },
        "once idle": {
          topic: function (_, outbound, inbound) {
            outbound.once('idle', this.callback.bind(null, null, outbound, inbound));
            outbound.setIdle(100);
          },
          "it should emit `idle`": function (_, outbound, inbound) {
            assert.isNull(_);
          },
          "the send() method": {
            topic: function (outbound, inbound) {
              inbound.on('data', this.callback.bind(null, null, outbound, inbound));
              outbound.send(['hello','world'], { some: "json", data: 123 });
            },
            "we should see it on the other end": function (_, outbound, inbound, data) {
              assert.isObject(data);
              arr = JSON.parse(data.toString());
              assert.lengthOf(arr, 3);
              assert.equal(arr[0], 'hello');
              assert.equal(arr[1], 'world');
              assert.deepEqual(arr[2], { some: "json", data: 123 });
            },
            "the end() method": {
              topic: function (outbound, inbound) {
                outbound.on('close', this.callback.bind(null, null, outbound, inbound));
                inbound.end();
              },
              "should close without errors": function (_, _, _, err) {
                assert.isUndefined(err);
              }
            }
          }
        }
      }
    }
  }
}).export(module);
