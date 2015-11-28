'use strict';

var expect            = require('chai').expect;
var TestsServerAddon  = require('../../../../../lib/tasks/server/middleware/tests-server');
var Promise           = require('../../../../../lib/ext/promise');

describe('TestServerAddon', function () {
  describe('.serverMiddleware', function () {
    var addon = new TestsServerAddon();
    var nextWasCalled = false;
    var mockRequest = {
      method: 'GET',
      path: '',
      url: 'http://example.com',
      headers: {}
    };
    var app = {
      use: function (callback) {
        return callback(mockRequest, null, function () { nextWasCalled = true; });
      }
    };

    it('invokes next when the watcher succeeds', function(done) {
      addon.serverMiddleware({
        app: app,
        options: {
          watcher: Promise.resolve()
        },
        finally: function() {
          expect(nextWasCalled).to.true;
          done();
        }
      });
    });

    it('invokes next when the watcher fails', function (done) {
      var mockError = 'bad things are bad';

      addon.serverMiddleware({
        app: app,
        options: {
          watcher: Promise.reject(mockError)
        },
        finally: function() {
          expect(nextWasCalled).to.true;
          done();
        }
      });
    });
  });
});
