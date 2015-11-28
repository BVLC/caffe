'use strict';

var expect      = require('chai').expect;
var TestTask    = require('../../../lib/tasks/test');
var MockProject = require('../../helpers/mock-project');

describe('test', function() {
  var subject;

  it('transforms the options and invokes testem properly', function() {
    subject = new TestTask({
      project: new MockProject(),
      addonMiddlewares: function() {
        return ['middleware1', 'middleware2'];
      },
      testem: {
        startCI: function(options, cb) {
          expect(options.host).to.equal('greatwebsite.com');
          expect(options.port).to.equal(123324);
          expect(options.cwd).to.equal('blerpy-derpy');
          expect(options.reporter).to.equal('xunit');
          expect(options.middleware).to.deep.equal(['middleware1', 'middleware2']);
          /* jshint ignore:start */
          expect(options.test_page).to.equal('http://my/test/page');
          expect(options.config_dir).to.be.an('string');
          /* jshint ignore:end*/
          cb(0);
        },
        app: { reporter: { total: 1 } }
      }
    });

    subject.run({
      host: 'greatwebsite.com',
      port: 123324,
      reporter: 'xunit',
      outputPath: 'blerpy-derpy',
      testPage: 'http://my/test/page'
    });
  });
});
