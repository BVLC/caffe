/*
 * hierarchy-test.js: Basic tests for hierarchical file stores.
 *
 * (C) 2011, Nodejitsu Inc.
 *
 */

var assert = require('assert'),
    fs = require('fs'),
    path = require('path'),
    spawn = require('child_process').spawn,
    vows = require('vows'),
    nconf = require('../lib/nconf');

var configDir = path.join(__dirname, 'fixtures', 'hierarchy'),
    globalConfig = path.join(configDir, 'global.json'),
    userConfig = path.join(configDir, 'user.json');

vows.describe('nconf/hierarchy').addBatch({
  "When using nconf": {
    "configured with two file stores": {
      topic: function () {
        nconf.add('user', { type: 'file', file: userConfig });
        nconf.add('global', { type: 'file', file: globalConfig });
        nconf.load();
        return nconf;
      },
      "should have the appropriate keys present": function () {
        assert.equal(nconf.get('title'), 'My specific title');
        assert.equal(nconf.get('color'), 'green');
        assert.equal(nconf.get('movie'), 'Kill Bill');
      }
    },
    "configured with two file stores using `file`": {
      topic: function () {
        nconf.file('user', userConfig);
        nconf.file('global', globalConfig);
        nconf.load();
        return nconf;
      },
      "should have the appropriate keys present": function () {
        assert.equal(nconf.get('title'), 'My specific title');
        assert.equal(nconf.get('color'), 'green');
        assert.equal(nconf.get('movie'), 'Kill Bill');
      }
    },
    "configured with .argv(), .env() and .file()": {
      topic: function () {
        var configFile = path.join(__dirname, 'fixtures', 'load-save.json'),
            script = path.join(__dirname, 'fixtures', 'scripts', 'nconf-hierarchical-load-save.js'),
            argv = ['--foo', 'foo', '--bar', 'bar'],
            that = this,
            data = '',
            child;
        
        try { fs.unlinkSync(configFile) }
        catch (ex) { }
        
        child = spawn('node', [script].concat(argv));
        
        child.stdout.on('data', function (d) {
          data += d;
        });
        
        child.on('exit', function () {
          fs.readFile(configFile, 'utf8', that.callback.bind(null, null, data));
        });
      },
      "should not persist information passed in to process.env and process.argv to disk ": function (_, data, _, ondisk){
        assert.equal(data, 'foo');
        assert.deepEqual(JSON.parse(ondisk), {
          database: {
            host: '127.0.0.1',
            port: 5984
          }
        });
      }
    },
    "configured with .argv(), .file() and invoked with nested command line options": {
      topic: function () {
        var script = path.join(__dirname, 'fixtures', 'scripts', 'nconf-hierarchical-load-merge.js'),
            argv = ['--candy:something', 'foo', '--candy:something5:second', 'bar'],
            that = this,
            data = '',
            child;

        child = spawn('node', [script].concat(argv));

        child.stdout.on('data', function (d) {
          data += d;
        });

        child.on('exit', function() {
          that.callback(null, data);
        });
      },
      "should merge nested objects ": function (err, data) {
        assert.deepEqual(JSON.parse(data), {
          apples: true,
          candy: {
            something: 'foo',
            something1: true,
            something2: true,
            something5: {
              first: 1,
              second: 'bar'
            }
          }
        });
      }
    }
  }
}).export(module);
