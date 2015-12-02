/*
 * watch-test.js: Tests for restarting forever processes when a file changes.
 *
 * (C) 2010 Charlie Robbins & the Contributors
 * MIT LICENSE
 *
 */

var assert = require('assert'),
    path = require('path'),
    fs = require('fs'),
    vows = require('vows'),
    fmonitor = require('../../lib');

var watchDir = fs.realpathSync(path.join(__dirname, '..', 'fixtures', 'watch')),
    monitor;

vows.describe('forever-monitor/plugins/watch').addBatch({
  'When using forever with watch enabled': {
    'forever should': {
      topic: fmonitor.start('daemon.js', {
        silent: true,
        args: ['-p', '8090'],
        watch: true,
        sourceDir: path.join(__dirname, '..', 'fixtures', 'watch')
      }),
      'have correct options set': function (child) {
        monitor = child;
        assert.isTrue(child.watchIgnoreDotFiles);
        assert.equal(watchDir, fs.realpathSync(child.watchDirectory));
      },
      'read .foreverignore file and store ignore patterns': function (child) {
        setTimeout(function () {
          assert.deepEqual(
            child.watchIgnorePatterns,
            fs.readFileSync(
              path.join(watchDir, '.foreverignore'),
              'utf8'
            ).split("\n").filter(Boolean)
          );
        }, 100);
      }
    }
  }
}).addBatch({
  'When using forever with watch enabled': {
    'when a file matching an ignore pattern is added': {
      topic: function () {
        var self = this;
        this.filenames = [
          path.join(watchDir, 'ignore_newFile'),
          path.join(watchDir, 'ignoredDir', 'ignore_subfile')
        ];

        //
        // Setup a bad restart function
        //
        function badRestart() {
          this.callback(new Error('Monitor restarted at incorrect time.'));
        }

        monitor.once('restart', badRestart);
        this.filenames.forEach(function (filename) {
          fs.writeFileSync(filename, '');
        });

        //
        // `chokidar` does not emit anything when ignored
        // files have changed so we need a setTimeout here
        // to prove that nothing has happened.
        //
        setTimeout(function () {
          monitor.removeListener('restart', badRestart);
          self.callback();
        }, 5000);
      },
      'do nothing': function (err) {
        assert.isUndefined(err);
        this.filenames.forEach(function (filename) {
          fs.unlinkSync(filename);
        });
      }
    }
  }
}).addBatch({
  'When using forever with watch enabled': {
    'when file changes': {
      topic: function (child) {
        child.once('restart', this.callback);
        fs.writeFileSync(path.join(watchDir, 'file'), '// hello, I know nodejitsu.');
      },
      'restart the script': function (child, _) {
        fs.writeFileSync(path.join(watchDir, 'file'), '/* hello, I know nodejitsu. ');
      }
    }
  }
}).addBatch({
  'When using forever with watch enabled': {
    'when file is added': {
      topic: function () {
        monitor.once('restart', this.callback);
        fs.writeFileSync(path.join(watchDir, 'newFile'), '');
      },
      'restart the script': function (child, _) {
        fs.unlinkSync(path.join(watchDir, 'newFile'));
      }
    }
  }
}).addBatch({
  'When using forever with watch enabled': {
    'when file is removed': {
      topic: function () {
        monitor.once('restart', this.callback);
        try { fs.unlinkSync(path.join(watchDir, 'removeMe')) }
        catch (ex) { }
      },
      'restart the script': function (child, _) {
        fs.writeFileSync(path.join(watchDir, 'removeMe'), '');
        monitor.stop();
      }
    }
  }
}).export(module);
