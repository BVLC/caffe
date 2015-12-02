/*
 * nssocket-test.js : namespace socket unit test for TCP
 *
 * (C) 2011, Charlie Robbins, Paolo Fragomeni, & the Contributors.
 *
 */

var assert = require('assert'),
    fs = require('fs'),
    net = require('net'),
    path = require('path'),
    vows = require('vows'),
    NsSocket = require('../lib/nssocket').NsSocket;

var TCP_PORT = 30105;

var tcpServer = net.createServer(),
    tcpOpt;

tcpOpt = {
  type : 'tcp4',
  delimiter: '.}',
  reconnect: true,
  retryInterval: 1000
};

tcpServer.listen(TCP_PORT);

vows.describe('nssocket/tcp/reconnect').addBatch({
  "When using NsSocket with TCP": {
    topic: new NsSocket(tcpOpt),
    "the connect() method": {
      topic: function (outbound) {
        var that = this;
        tcpServer.on('connection', this.callback.bind(null, null, outbound));
        outbound.connect(TCP_PORT);
      },
      "should actually connect": function (_, outbound, inbound) {
        assert.instanceOf(outbound, NsSocket);
        assert.instanceOf(inbound, net.Socket);
      },
      "when the server closes": {
        topic: function (outbound, inbound) {
          outbound.once('close', this.callback.bind(this, null, outbound));
          tcpServer.close();
          inbound.destroy();
        },
        "and then restarts": {
          topic: function (outbound) {
            tcpServer = net.createServer();
            tcpServer.listen(TCP_PORT);
            tcpServer.on('connection', this.callback.bind(null, null, outbound));
          },
          "the socket should reconnect correctly": function (_, outbound, inbound) {
            assert.instanceOf(outbound, NsSocket);
            assert.instanceOf(inbound, net.Socket);
          },
          "the on() method": {
            topic: function (outbound, inbound) {
              outbound.on('data.}here.}is', this.callback.bind(outbound, null));
              inbound.write(JSON.stringify(['here', 'is', 'something.']) + '\n');
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
      }
    }
  }
}).export(module);
