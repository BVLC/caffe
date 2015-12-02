/*
 * provider-test.js: Tests for the nconf Provider object.
 *
 * (C) 2011, Nodejitsu Inc.
 *
 */

var assert = require('assert'),
    fs = require('fs'),
    path = require('path'),
    spawn = require('child_process').spawn,
    vows = require('vows'),
    helpers = require('./helpers'),
    nconf = require('../lib/nconf');
    
var fixturesDir = path.join(__dirname, 'fixtures'),
    mergeFixtures = path.join(fixturesDir, 'merge'),
    files = [path.join(mergeFixtures, 'file1.json'), path.join(mergeFixtures, 'file2.json')],
    override = JSON.parse(fs.readFileSync(files[0]), 'utf8');

function assertProvider(test) {
  return {
    topic: new nconf.Provider(),
    "should use the correct File store": test
  };
}

vows.describe('nconf/provider').addBatch({
  "When using nconf": {
    "an instance of 'nconf.Provider'": {
      "calling the use() method with the same store type and different options": {
        topic: new nconf.Provider().use('file', { file: files[0] }),
        "should use a new instance of the store type": function (provider) {
          var old = provider.stores['file'];

          assert.equal(provider.stores.file.file, files[0]);
          provider.use('file', { file: files[1] });

          assert.notStrictEqual(old, provider.stores.file);
          assert.equal(provider.stores.file.file, files[1]);
        }
      },
      "when 'argv' is true": helpers.assertSystemConf({
        script: path.join(fixturesDir, 'scripts', 'provider-argv.js'),
        argv: ['--something', 'foobar']
      }),
      "when 'env' is true": helpers.assertSystemConf({
        script: path.join(fixturesDir, 'scripts', 'provider-env.js'),
        env: { SOMETHING: 'foobar' }
      })
    },
    "the default nconf provider": {
      "when 'argv' is set to true": helpers.assertSystemConf({
        script: path.join(fixturesDir, 'scripts', 'nconf-argv.js'),
        argv: ['--something', 'foobar'],
        env: { SOMETHING: true }
      }),
      "when 'env' is set to true": helpers.assertSystemConf({
        script: path.join(fixturesDir, 'scripts', 'nconf-env.js'),
        env: { SOMETHING: 'foobar' }
      }),
      "when 'argv' is set to true and process.argv is modified": helpers.assertSystemConf({
        script: path.join(fixturesDir, 'scripts', 'nconf-change-argv.js'),
        argv: ['--something', 'badValue', 'evenWorse', 'OHNOEZ', 'foobar']
      }),
      "when hierarchical 'argv' get": helpers.assertSystemConf({
        script: path.join(fixturesDir, 'scripts', 'nconf-hierarchical-file-argv.js'),
        argv: ['--something', 'foobar'],
        env: { SOMETHING: true }
      }),
      "when 'env' is set to true with a nested separator": helpers.assertSystemConf({
        script: path.join(fixturesDir, 'scripts', 'nconf-nested-env.js'),
        env: { SOME_THING: 'foobar' }
      })
    }
  }
}).addBatch({
  "When using nconf": {
    "an instance of 'nconf.Provider'": {
      "the merge() method": {
        topic: new nconf.Provider().use('file', { file: files[1] }),
        "should have the result merged in": function (provider) {
          provider.load();
          provider.merge(override);
          helpers.assertMerged(null, provider.stores.file.store);
          assert.equal(provider.stores.file.store.candy.something, 'file1');
        },
        "should merge Objects over null": function (provider) {
          provider.load();
          provider.merge(override);
          assert.equal(provider.stores.file.store.unicorn.exists, true);
        }
      }
    }
  }
}).addBatch({
  "When using nconf": {
    "an instance of 'nconf.Provider'": {
      "the load() method": {
        "when sources are passed in": {
          topic: new nconf.Provider({
            sources: {
              user: {
                type: 'file',
                file: files[0]
              },
              global: {
                type: 'file',
                file: files[1]
              }
            }
          }),
          "should respect the hierarchy ": function (provider) {
            var merged = provider.load();

            helpers.assertMerged(null, merged);
            assert.equal(merged.candy.something, 'file1');
          }
        },
        "when multiple stores are used": {
          topic: new nconf.Provider().overrides({foo: {bar: 'baz'}})
            .add('file1', {type: 'file', file: files[0]})
            .add('file2', {type: 'file', file: files[1]}),
          "should respect the hierarchy": function(provider) {
            var merged = provider.load();
            
            helpers.assertMerged(null, merged);
            assert.equal(merged.foo.bar, 'baz');
            assert.equal(merged.candy.something, 'file1');
          }
        }
      }
    }
  }
}).addBatch({
  "When using nconf": {
    "an instance of 'nconf.Provider'": {
      "the .file() method": {
        "with a single filepath": assertProvider(function (provider) {
          provider.file(helpers.fixture('store.json'));
          assert.isObject(provider.stores.file);
        }),
        "with a name and a filepath": assertProvider(function (provider) {
          provider.file('custom', helpers.fixture('store.json'));
          assert.isObject(provider.stores.custom);
        }),
        "with a single object": assertProvider(function (provider) {
          provider.file({
            dir: helpers.fixture(''),
            file: 'store.json',
            search: true
          });

          assert.isObject(provider.stores.file);
          assert.equal(provider.stores.file.file, helpers.fixture('store.json'));
        }),
        "with a name and an object": assertProvider(function (provider) {
          provider.file('custom', {
            dir: helpers.fixture(''),
            file: 'store.json',
            search: true
          });

          assert.isObject(provider.stores.custom);
          assert.equal(provider.stores.custom.file, helpers.fixture('store.json'));
        })
      }
    }
  }
}).export(module);
