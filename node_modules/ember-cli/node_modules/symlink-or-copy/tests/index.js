var assert = require('assert');
var symLinkOrCopy = require('..');

describe('node-symlink-or-copy', function() {
  beforeEach(function() {
    symLinkOrCopy.setOptions({}); // make sure we don't mix options between tests
  });

  it('windows falls back to copy', function() {
    var count = 0;
    symLinkOrCopy.setOptions({
      isWindows: true,
      copyDereferenceSync: function() {
        count++;
      },
      canSymLink: false
    });
    symLinkOrCopy.sync();
    assert.equal(count, 1);
  });

  it('windows symlinks when has permission', function() {
    var count = 0;
    symLinkOrCopy.setOptions({
      fs: {
        lstatSync: function() {
          return {
            isSymbolicLink: function() {
              count++;
              return true;
            },
            isDirectory: function() {
              return true;
            }
          }
        },
        realpathSync: function() {count++},
        symlinkSync: function() {count++;}
      },
      canSymlink: true
    });
    symLinkOrCopy.sync();
    assert.equal(count, 3);
  })
});

describe('testing mode', function() {
  it('allows fs to be mocked', function() {
    var count = 0;
    symLinkOrCopy.setOptions({
      canSymlink: true,
      fs: {
        lstatSync: function() {
          return {
            isSymbolicLink: function() {
              count++;
              return true;
            },
            isDirectory: function() {
              return true;
            }
          }
        },
        realpathSync: function() {count++},
        symlinkSync: function() {count++;}
      }
    });

    assert.equal(count, 0);
    symLinkOrCopy.sync();
    assert.equal(count, 3);
  });
});
