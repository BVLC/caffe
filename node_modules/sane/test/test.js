var os = require('os');
var fs = require('fs');
var sane = require('../');
var rimraf = require('rimraf');
var path = require('path');
var assert = require('assert');
var tmp = require('tmp');

tmp.setGracefulCleanup();
var jo = path.join.bind(path);

describe('sane in polling mode', function() {
  harness.call(this, {poll: true});
});
describe('sane in normal mode', function() {
  harness.call(this, {});
});
describe('sane in watchman mode', function() {
  harness.call(this, {watchman: true})
});
describe('sane in watchman mode with offset project', function() {
  harness.call(this, {watchman: true, offset: true})
});

function getWatcherClass(mode) {
  if (mode.watchman) {
    return sane.WatchmanWatcher;
  } else if (mode.poll) {
    return sane.PollWatcher;
  } else {
    return sane.NodeWatcher;
  }
}

function harness(mode) {
  if (mode.poll) this.timeout(5000);
  var global_testdir = null;
  var testdir = null;
  after(function() {
    if (global_testdir) {
      try {
        rimraf.sync(global_testdir.name);
      } catch (e) {}
    }
  });
  before(function() {
    global_testdir = tmp.dirSync({
      prefix: 'sane-test',
      unsafeCleanup: true
    });
    testdir = fs.realpathSync(global_testdir.name);

    // Some Watchman deployments are restricted to watching
    // project roots.  Let's fake one
    fs.mkdirSync(jo(testdir, '.git'));

    // If testing watchman watch-project in offset mode, create an offset dir
    if (mode.offset) {
      testdir = jo(testdir, 'offset');
      fs.mkdirSync(testdir);
    }

    for (var i = 0; i < 10; i++) {
      fs.writeFileSync(jo(testdir, 'file_' + i), 'test_' + i);
      var subdir = jo(testdir, 'sub_' + i);
      try {
        fs.mkdirSync(subdir);
      } catch (e) {}
      for (var j = 0; j < 10; j++) {
        fs.writeFileSync(jo(subdir, 'file_' + j), 'test_' + j);
      }
    }
  });

  describe('sane plugin', function () {
    beforeEach(function () {
      this.watcher = sane(testdir, {
        glob: '**/file_1',
        watcher: './test/plugin_watcher',
      });
    });

    afterEach(function (done) {
      this.watcher.close(done);
    });

    it('uses the custom plugin watcher', function (done) {
      this.watcher.on('is-test-plugin', function () {
        done();
      });
    });

  });


  describe('sane(file)', function() {
    beforeEach(function () {
      var Watcher = getWatcherClass(mode);
      this.watcher = new Watcher(testdir);
    });

    afterEach(function(done) {
      this.watcher.close(done);
    });

    it('emits a ready event', function(done) {
      this.watcher.on('ready', done);
      this.watcher.on('error', function(error) {
        done(error);
      });
    });

    it('change emits event', function(done) {
      var testfile = jo(testdir, 'file_1');
      this.watcher.on('error', function(error) {
        done(error);
      });
      this.watcher.on('change', function(filepath, dir, stat) {
        assert(stat instanceof fs.Stats);
        assert.equal(filepath, path.relative(testdir, testfile));
        assert.equal(dir, testdir);
        done();
      });
      this.watcher.on('ready', function() {
        fs.writeFileSync(testfile, 'wow');
      });
    });

    it('emits change events for subdir files', function(done) {
      var subdir = 'sub_1';
      var testfile = jo(testdir, subdir, 'file_1');
      this.watcher.on('change', function(filepath, dir) {
        assert.equal(filepath, path.relative(testdir, testfile));
        assert.equal(dir, testdir);
        done();
      });
      this.watcher.on('ready', function() {
        fs.writeFileSync(testfile, 'wow');
      });
    });

    it('adding a file will trigger an add event', function(done) {
      var testfile = jo(testdir, 'file_x' + Math.floor(Math.random() * 10000));
      this.watcher.on('add', function(filepath, dir, stat) {
        assert(stat instanceof fs.Stats);
        assert.equal(filepath, path.relative(testdir, testfile));
        assert.equal(dir, testdir);
        done();
      });
      this.watcher.on('ready', function() {
        fs.writeFileSync(testfile, 'wow');
      });
    });

    it('removing a file will emit delete event', function(done) {
      var testfile = jo(testdir, 'file_9');
      this.watcher.on('delete', function(filepath, dir) {
        assert.equal(filepath, path.relative(testdir, testfile));
        assert.equal(dir, testdir);
        done();
      });
      this.watcher.on('ready', function() {
        fs.unlinkSync(testfile);
      });
    });

    it('chaging, removing, deleting should emit the "all" event', function(done) {
      var toChange = jo(testdir, 'file_4');
      var toDelete = jo(testdir, 'file_5');
      var toAdd = jo(testdir, 'file_x' + Math.floor(Math.random() * 10000));
      var i = 0;
      var added = false;

      this.watcher.on('all', function(type, filepath, dir, stat) {
        assert.equal(dir, testdir);
        if (type === 'change') {
          // Windows emits additional change events for newly created files.
          if (added && filepath === path.relative(dir, toAdd)) {
            return;
          }
          assert(stat instanceof fs.Stats);
          assert.equal(filepath, path.relative(dir, toChange));
        } else if (type === 'delete') {
          assert(!stat);
          assert.equal(filepath, path.relative(dir, toDelete));
        } else if (type === 'add') {
          assert(stat instanceof fs.Stats);
          assert.equal(filepath, path.relative(dir, toAdd));
          added = true;
        }
        if (++i === 3) {
          done();
        }
      });

      this.watcher.on('ready', function() {
        fs.writeFileSync(toChange, 'hai');
        fs.unlinkSync(toDelete);
        fs.writeFileSync(toAdd, 'hai wow');
      })
    });

    it('removing a dir will emit delete event', function(done) {
      var subdir = jo(testdir, 'sub_9');
      this.watcher.on('delete', function(filepath, dir) {
        // Ignore delete events for files in the dir.
        if (path.dirname(filepath) === path.relative(testdir, subdir)) {
          return;
        }
        assert.equal(filepath, path.relative(testdir, subdir));
        assert.equal(dir, testdir);
        done();
      });
      this.watcher.on('ready', function() {
        rimraf.sync(subdir);
      });
    });

    it('adding a dir will emit an add event', function(done) {
      var subdir = jo(testdir, 'sub_x' + Math.floor(Math.random() * 10000));
      this.watcher.on('add', function(filepath, dir, stat) {
        assert(stat instanceof fs.Stats);
        assert.equal(filepath, path.relative(testdir, subdir));
        assert.equal(dir, testdir);
        done();
      });
      this.watcher.on('ready', function() {
        fs.mkdirSync(subdir);
      });
    });

    it('adding in a subdir will trigger an add event', function(done) {
      var subdir = jo(testdir, 'sub_x' + Math.floor(Math.random() * 10000));
      var testfile = jo(subdir, 'file_x' + Math.floor(Math.random() * 10000));
      var i = 0;
      this.watcher.on('add', function(filepath, dir, stat) {
        assert(stat instanceof fs.Stats);
        if (++i === 1) {
          assert.equal(filepath, path.relative(testdir, subdir));
          assert.equal(dir, testdir);
        } else {
          assert.equal(filepath, path.relative(testdir, testfile));
          assert.equal(dir, testdir);
          done();
        }
      });
      this.watcher.on('ready', function() {
        fs.mkdirSync(subdir);
        defer(function() {
          fs.writeFileSync(testfile, 'wow');
        });
      });
    });

    it('closes watchers when dirs are deleted', function(done) {
      var subdir = jo(testdir, 'sub_1');
      var testfile = jo(subdir, 'file_1');
      var actualFiles = {};
      var expectedFiles = {};
      expectedFiles[path.relative(testdir, subdir)] = true;
      expectedFiles[path.relative(testdir, testfile)] = true;
      this.watcher.on('ready', function() {
        this.watcher.on('add', function(filepath) {
          // win32 order is not guaranteed and events may leak between tests
          if (expectedFiles[filepath]) {
            actualFiles[filepath] = true;
          }
          if (Object.keys(actualFiles).length === 2) {
            assert.deepEqual(
              expectedFiles,
              actualFiles
            );
            done();
          }
        });
        rimraf.sync(subdir);
        defer(function() {
          fs.mkdirSync(subdir);
          defer(function() {
            fs.writeFileSync(testfile, 'wow');
          });
        });
      }.bind(this));
    });

    it('should be ok to remove and then add the same file', function(done) {
      var testfile = jo(testdir, 'sub_8', 'file_1');
      var i = 0;
      this.watcher.on('add', function(filepath, dir) {
        assert.equal(filepath, path.relative(testdir, testfile));
        assert.equal(dir, testdir);
      });
      this.watcher.on('delete', function(filepath, dir) {
        assert.equal(filepath, path.relative(testdir, testfile));
        assert.equal(dir, testdir);
        done();
      });
      this.watcher.on('ready', function() {
        fs.unlink(testfile);
        defer(function() {
          fs.writeFileSync(testfile, 'wow');
        });
      });
    });
  });

  describe('sane(file, glob)', function() {
    beforeEach(function () {
      var Watcher = getWatcherClass(mode);
      this.watcher = new Watcher(
        testdir,
        { glob: ['**/file_1', '**/file_2'] }
      );
    });

    afterEach(function(done) {
      this.watcher.close(done);
    });

    it('ignore files according to glob', function (done) {
      var i = 0;
      this.watcher.on('change', function(filepath, dir) {
        assert.ok(filepath.match(/file_(1|2)/), 'only file_1 and file_2');
        assert.equal(dir, testdir);
        if (++i == 2) done();
      });
      this.watcher.on('ready', function() {
        fs.writeFileSync(jo(testdir, 'file_1'), 'wow');
        fs.writeFileSync(jo(testdir, 'file_9'), 'wow');
        fs.writeFileSync(jo(testdir, 'file_3'), 'wow');
        fs.writeFileSync(jo(testdir, 'file_2'), 'wow');
      });
    });
  });

  describe('sane(dir, {dot: false})', function() {
    beforeEach(function () {
      var Watcher = getWatcherClass(mode);
      this.watcher = new Watcher(
        testdir,
        { dot: false }
      );
    });

    afterEach(function(done) {
      this.watcher.close(done);
    });

    it('should ignore dot files', function (done) {
      var i = 0;
      this.watcher.on('change', function(filepath, dir) {
        assert.ok(filepath.match(/file_(1|2)/), 'only file_1 and file_2');
        assert.equal(dir, testdir);
        if (++i == 2) done();
      });
      this.watcher.on('ready', function() {
        fs.writeFileSync(jo(testdir, 'file_1'), 'wow');
        fs.writeFileSync(jo(testdir, '.file_9'), 'wow');
        fs.writeFileSync(jo(testdir, '.file_3'), 'wow');
        fs.writeFileSync(jo(testdir, 'file_2'), 'wow');
      });
    });

    it('should ignore dot dirs', function(done) {
      this.watcher.on('change', function(filepath, dir) {
        assert.ok(filepath.match(/file_1/), 'only file_1 got : ' + filepath);
        assert.equal(dir, testdir);
        done();
      });
      this.watcher.on('ready', function() {
        fs.mkdirSync(jo(testdir, '.lol'));
        fs.writeFileSync(jo(testdir, '.lol', 'file'), 'wow');
        fs.writeFileSync(jo(testdir, '.file_3'), 'wow');
        fs.writeFileSync(jo(testdir, 'file_1'), 'wow');
      });
    });
  });

  describe('sane shortcut alias', function () {
    beforeEach(function () {
      this.watcher = sane(testdir, {
        glob: '**/file_1',
        poll: mode.poll,
        watchman: mode.watchman
      });
    });

    afterEach(function(done) {
      this.watcher.close(done);
    });

    it('allows for shortcut mode using just a string as glob', function (done) {
      this.watcher.on('change', function (filepath, dir) {
        assert.ok(filepath.match(/file_1/));
        assert.equal(dir, testdir);
        done();
      });
      this.watcher.on('ready', function() {
        fs.writeFileSync(jo(testdir, 'file_1'), 'wow');
      });
    });
  });

  function defer(fn) {
    setTimeout(fn, mode.poll ? 1000 : 300);
  }
}
