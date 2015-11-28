'use strict';
/*jshint expr: true*/

var expect  = require('chai').expect;
var path    = require('path');
var nodeModulesPath = require('./');

describe('cli/node-module-path.js', function() {

  afterEach(function() {
    delete process.env.EMBER_NODE_PATH;
  });

  it('nodeModulesPath() should return the local node_modules path by default.', function() {
    // Valid commands
    var expectedPath = path.join(process.cwd(),'node_modules');

    expect(
      nodeModulesPath(process.cwd())
    ).to.equal(expectedPath);
  });

  it('nodeModulesPath() should return subdirectories of EMBER_NODE_PATH when set to an absolute path.', function() {
    if (process.platform === 'win32') {
      process.env.EMBER_NODE_PATH = 'C:\\tmp\node_modules';
    } else {
      process.env.EMBER_NODE_PATH = '/tmp/node_modules';
    }

    expect(nodeModulesPath(process.cwd())).to.equal(path.resolve(process.env.EMBER_NODE_PATH));

    var addOnPath = path.resolve(process.env.EMBER_NODE_PATH, 'node_modules', 'my-add-on');
    var addOnModulesPath = path.resolve(process.env.EMBER_NODE_PATH, 'node_modules', 'my-add-on', 'node_modules');
    expect(nodeModulesPath(addOnPath)).to.equal(addOnModulesPath);
  });

  it('nodeModulesPath() should return subdirectories of EMBER_NODE_PATH when set to a relative path.', function() {
    process.env.EMBER_NODE_PATH = '../../tmp/node_modules';
    expect(nodeModulesPath(process.cwd())).to.equal(path.resolve('../../tmp','node_modules'));
    expect(nodeModulesPath('../../tmp/node_modules/my-add-on')).to.equal(path.resolve('../../tmp','node_modules','my-add-on','node_modules'));
  });

  it('should resolve node_modules if the directory is behind the context', function() {
    expect(nodeModulesPath(path.resolve(process.cwd(), 'fixtures/foo/bar/baz'))).to.equal(path.resolve(process.cwd(), 'fixtures/foo/node_modules'));
  });
});
