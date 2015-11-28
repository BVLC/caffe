'use strict';

var expect         = require('chai').expect;
var TestServerTask = require('../../../lib/tasks/test-server');
var MockProject    = require('../../helpers/mock-project');
var MockUI         = require('../../helpers/mock-ui');
var MockWatcher    = require('../../helpers/mock-watcher');

describe('test server', function() {
  var subject;

  it('transforms the options and invokes testem properly', function(done) {
    var ui = new MockUI();
    var watcher = new MockWatcher();

    subject = new TestServerTask({
      project: new MockProject(),
      ui: ui,
      addonMiddlewares: function() {
        return ['middleware1', 'middleware2'];
      },
      testem: {
        startDev: function(options) {
          expect(options.host).to.equal('greatwebsite.com');
          expect(options.port).to.equal(123324);
          expect(options.cwd).to.equal('blerpy-derpy');
          expect(options.reporter).to.equal('xunit');
          expect(options.middleware).to.deep.equal(['middleware1', 'middleware2']);
          /* jshint ignore:start */
          expect(options.test_page).to.equal('http://my/test/page');
          expect(options.config_dir).to.be.an('string');
          /* jshint ignore:end*/
          done();
        }
      }
    });

    subject.run({
      host: 'greatwebsite.com',
      port: 123324,
      reporter: 'xunit',
      outputPath: 'blerpy-derpy',
      watcher: watcher,
      testPage: 'http://my/test/page'
    });
    watcher.emit('change');
  });
});
