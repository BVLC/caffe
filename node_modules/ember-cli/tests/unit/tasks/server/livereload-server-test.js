'use strict';

var expect            = require('chai').expect;
var LiveReloadServer  = require('../../../../lib/tasks/server/livereload-server');
var MockUI            = require('../../../helpers/mock-ui');
var MockExpressServer = require('../../../helpers/mock-express-server');
var net               = require('net');
var EOL               = require('os').EOL;
var path              = require('path');
var MockWatcher       = require('../../../helpers/mock-watcher');

describe('livereload-server', function() {
  var subject;
  var ui;
  var watcher;
  var expressServer;

  beforeEach(function() {
    ui = new MockUI();
    watcher = new MockWatcher();
    expressServer = new MockExpressServer();

    subject = new LiveReloadServer({
      ui: ui,
      watcher: watcher,
      expressServer: expressServer,
      analytics: { trackError: function() { } },
      project: {
        liveReloadFilterPatterns: [],
        root: '/home/user/my-project'
      }
    });
  });

  afterEach(function() {
    try {
      if (subject._liveReloadServer) {
        subject._liveReloadServer.close();
      }
    } catch (err) { }
  });

  describe('start', function() {
    it('does not start the server if `liveReload` option is not true', function() {
      return subject.start({
        liveReloadPort: 1337,
        liveReload: false,
      }).then(function(output) {
        expect(output).to.equal('Livereload server manually disabled.');
        expect(!!subject._liveReloadServer).to.equal(false);
      });
    });

    it('correctly indicates which port livereload is present on', function() {
      return subject.start({
        liveReloadPort: 1337,
        liveReloadHost: 'localhost',
        liveReload: true
      }).then(function() {
        expect(ui.output).to.equal('Livereload server on http://localhost:1337' + EOL);
      });
    });

    it('informs of error during startup', function(done) {
      var preexistingServer = net.createServer();
      preexistingServer.listen(1337);

      return subject.start({
          liveReloadPort: 1337,
          liveReload: true
        })
        .catch(function(reason) {
          expect(reason).to.equal('Livereload failed on http://localhost:1337.  It is either in use or you do not have permission.' + EOL);
        })
        .finally(function() {
          preexistingServer.close(done);
        });
    });

    it('starts with custom host', function() {
      return subject.start({
        liveReloadHost: '127.0.0.1',
        liveReloadPort: 1337,
        liveReload: true
      }).then(function() {
        expect(ui.output).to.equal('Livereload server on http://127.0.0.1:1337' + EOL);
      });
    });
  });

  describe('start with https', function() {
    it('correctly indicates which port livereload is present on and running in https mode', function() {
      return subject.start({
        liveReloadPort: 1337,
        liveReloadHost: 'localhost',
        liveReload: true,
        ssl: true,
        sslKey: 'tests/fixtures/ssl/server.key',
        sslCert: 'tests/fixtures/ssl/server.crt'
      }).then(function() {
        expect(ui.output).to.equal('Livereload server on https://localhost:1337' + EOL);
      });
    });

    it('informs of error during startup', function(done) {
      var preexistingServer = net.createServer();
      preexistingServer.listen(1337);

      return subject.start({
          liveReloadPort: 1337,
          liveReload: true,
          ssl: true,
          sslKey: 'tests/fixtures/ssl/server.key',
          sslCert: 'tests/fixtures/ssl/server.crt'
        })
        .catch(function(reason) {
          expect(reason).to.equal('Livereload failed on https://localhost:1337.  It is either in use or you do not have permission.' + EOL);
        })
        .finally(function() {
          preexistingServer.close(done);
        });
    });
  });

  describe('express server restart', function() {
    it('triggers when the express server restarts', function() {
      var calls = 0;
      subject.didRestart = function () {
        calls++;
      };

      return subject.start({
          liveReloadPort: 1337,
          liveReload: true
        }).then(function () {
          expressServer.emit('restart');
          expect(calls).to.equal(1);
        });
    });
  });

  describe('livereload changes', function () {
    var liveReloadServer;
    var changedCount;
    var oldChanged;
    var stubbedChanged = function() {
      changedCount += 1;
    };
    var trackCount;
    var oldTrack;
    var stubbedTrack = function() {
      trackCount += 1;
    };

    beforeEach(function() {
      liveReloadServer = subject.liveReloadServer();
      changedCount = 0;
      oldChanged = liveReloadServer.changed;
      liveReloadServer.changed = stubbedChanged;

      trackCount = 0;
      oldTrack = subject.analytics.track;
      subject.analytics.track = stubbedTrack;
    });

    afterEach(function() {
      liveReloadServer.changed = oldChanged;
      subject.analytics.track = oldTrack;
      subject.project.liveReloadFilterPatterns = [];
    });

    describe('watcher events', function () {
      function watcherEventTest(eventName, expectedCount) {
        subject.project.liveReloadFilterPatterns = [];
        return subject.start({
          liveReloadPort: 1337,
          liveReload: true,
        }).then(function () {
            watcher.emit(eventName, {
              filePath: '/home/user/my-project/test/fixtures/proxy/file-a.js'
            });
          }).finally(function () {
            expect(changedCount).to.equal(expectedCount);
          });
      }

      it('triggers a livereload change on a watcher change event', function () {
        return watcherEventTest('change', 1);
      });

      it('triggers a livereload change on a watcher error event', function () {
        return watcherEventTest('error', 1);
      });

      it('does not trigger a livereload change on other watcher events', function () {
        return watcherEventTest('not-an-event', 0);
      });
    });

    describe('filter pattern', function() {
      it('triggers the livereload server of a change when no pattern matches', function() {
        subject.didChange({filePath: ''});
        expect(changedCount).to.equal(1);
        expect(trackCount).to.equal(1);
      });

      it('does not trigger livereload server of a change when there is a pattern match', function() {
        // normalize test regex for windows
        // path.normalize with change forward slashes to back slashes if test is running on windows
        // we then replace backslashes with double backslahes to escape the backslash in the regex
        var basePath = path.normalize('test/fixtures/proxy').replace(/\\/g, '\\\\');
        var filter = new RegExp('^' + basePath);
        subject.project.liveReloadFilterPatterns = [filter];

        subject.didChange({
          filePath: '/home/user/my-project/test/fixtures/proxy/file-a.js'
        });
        expect(changedCount).to.equal(0);
        expect(trackCount).to.equal(0);
      });
    });
  });
});
