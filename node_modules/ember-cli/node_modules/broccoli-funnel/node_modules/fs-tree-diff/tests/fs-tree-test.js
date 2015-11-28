'use strict';

var expect = require('chai').expect;
var FSTree = require('../lib/index');
var Entry = require('../lib/entry');
var context = describe;
var fsTree;

require('chai').config.truncateThreshold = 0;
describe('FSTree', function() {
  function merge(x, y) {
    var result = {};

    Object.keys(x || {}).forEach(function(key) {
      result[key] = x[key];
    });

    Object.keys(y || {}).forEach(function(key) {
      result[key] = y[key];
    });

    return result;
  }

  function MockEntry(options) {
    this.relativePath = options.relativePath;
    this.mode = options.mode;
    this.size = options.size;
    this.mtime = options.mtime;
  }

  MockEntry.prototype.isDirectory = Entry.prototype.isDirectory;

  function file(relativePath, options) {
    return entry(merge({ relativePath: relativePath }, options));
  }

  function directory(relativePath, options) {
    return entry(merge({
      relativePath: relativePath,
      mode: 16877
    }, options));
  }

  function entry(options) {
    return new MockEntry({
      relativePath: options.relativePath,
      mode: options.mode || 0,
      size: options.size || 0,
      mtime: options.mtime || 0
    });
  }

  it('can be instantiated', function() {
    expect(new FSTree()).to.be.an.instanceOf(FSTree);
  });

  describe('.fromPaths', function() {
    it('creates empty trees', function() {
      fsTree = FSTree.fromPaths([ ]);
      expect(fsTree.size).to.eq(0);
    });

    it('creates trees from paths', function() {
      var result;

      fsTree = FSTree.fromPaths([
        'a.js',
        'foo/a.js',
      ]);

      result = fsTree.calculatePatch(
        FSTree.fromPaths([
          'a.js',
          'foo/b.js',
        ])
      );

      expect(result).to.deep.equal([
        ['unlink', 'foo/a.js', undefined],
        // This no-op is not fundamental: a future iteration could reasonably
        // optimize it away
        ['rmdir', 'foo',       undefined],
        ['mkdir', 'foo',       directory('foo')],
        ['create', 'foo/b.js', file('foo/b.js')]
      ]);
    });
  });

  describe('.fromEntries', function() {
    it('creates empty trees', function() {
      fsTree = FSTree.fromEntries([ ]);
      expect(fsTree.size).to.eq(0);
    });

    it('creates tree from entries', function() {
      var fsTree = FSTree.fromEntries([
        file('a/b.js', { size: 1, mtime: 1 }),
        file('c/d.js', { size: 1, mtime: 1 }),
        file('a/c.js', { size: 1, mtime: 1 })
      ]);

      expect(fsTree.size).to.eq(3);

      var result = fsTree.calculatePatch(FSTree.fromEntries([
        file('a/b.js', { size: 1, mtime: 2 }),
        file('c/d.js', { size: 1, mtime: 1 }),
        file('a/c.js', { size: 1, mtime: 1 })
       ]));

      expect(result).to.deep.equal([
        ['change', 'a/b.js', file('a/b.js', { mtime: 2, size: 1 })]
      ]);
    });
  });

  describe('#calculatePatch', function() {
    context('from an empty tree', function() {
      beforeEach( function() {
        fsTree = new FSTree();
      });

      context('to an empty tree', function() {
        it('returns 0 operations', function() {
          expect(fsTree.calculatePatch(FSTree.fromPaths([]))).to.deep.equal([]);
        });
      });

      context('to a non-empty tree', function() {
        it('returns n create operations', function() {
          expect(fsTree.calculatePatch(FSTree.fromPaths([
            'bar/baz.js',
            'foo.js',
          ]))).to.deep.equal([
            ['mkdir',  'bar',        directory('bar')],
            ['create', 'foo.js',     file('foo.js')],
            ['create', 'bar/baz.js', file('bar/baz.js')],
          ]);
        });
      });
    });

    context('from a simple non-empty tree', function() {
      beforeEach( function() {
        fsTree = FSTree.fromPaths([
          'bar/baz.js',
          'foo.js',
        ]);
      });

      context('to an empty tree', function() {
        it('returns n rm operations', function() {
          expect(fsTree.calculatePatch(FSTree.fromPaths([]))).to.deep.equal([
            ['unlink', 'bar/baz.js', undefined],
            ['rmdir',  'bar',        undefined],
            ['unlink', 'foo.js',     undefined]
          ]);
        });
      });
    });

    context('FSTree with entries', function() {
      beforeEach(function() {
        fsTree = new FSTree({
          entries: [
            file('a/b.js', { size: 1, mtime: 1, mode: '0o666' }),
            file('c/d.js', { size: 1, mtime: 1, mode: '0o666' }),
            file('a/c.js', { size: 1, mtime: 1, mode: '0o666' })
          ]
        });
      });

      it('should detect additions', function() {
        var result = fsTree.calculatePatch(new FSTree({
          entries: [
            file('a/b.js', { size: 1, mtime: 1, mode: '0o666' }),
            file('c/d.js', { size: 1, mtime: 1, mode: '0o666' }),
            file('a/c.js', { size: 1, mtime: 1, mode: '0o666' }),
            file('a/j.js', { size: 1, mtime: 1, mode: '0o666' })
          ]
        }));

        expect(result).to.deep.equal([
          ['create', 'a/j.js', file('a/j.js', { size: 1, mtime: 1, mode: '0o666'})]
        ]);
      });

      it('should detect removals', function() {
        var e = entry({
          relativePath: 'a/b.js',
          mode: '0o666',
          size: 1,
          mtime: 1
        });

        var result = fsTree.calculatePatch(new FSTree({
          entries: [e]
        }));

        expect(result).to.deep.equal([
          ['unlink', 'a/c.js', undefined],
          ['unlink', 'c/d.js', undefined],
          ['rmdir',  'c',      undefined]
        ]);
      });

      it('should detect updates', function() {
        var entries = [
          entry({ relativePath: 'a/b.js', mode: '0o666', size: 1, mtime: 1 }),
          entry({ relativePath: 'c/d.js', mode: '0o666', size: 1, mtime: 2 }),
          entry({ relativePath: 'a/c.js', mode: '0o666', size: 10, mtime: 1 })
        ];

        var result = fsTree.calculatePatch(new FSTree({
          entries: entries
        }));

        debugger;

        expect(result).to.deep.equal([
          ['change', 'c/d.js', entries[1]],
          ['change', 'a/c.js', entries[2]],
        ]);
      });
    });

    context('FSTree with updates at several different depths', function () {
      beforeEach( function() {
        fsTree = new FSTree({
          entries: [
            entry({ relativePath: 'a.js', mode: '0o666', size: 1, mtime: 1 }),
            entry({ relativePath: 'b.js', mode: '0o666', size: 1, mtime: 1 }),
            entry({ relativePath: 'one/a.js', mode: '0o666', size: 1, mtime: 1 }),
            entry({ relativePath: 'one/b.js', mode: '0o666', size: 1, mtime: 1 }),
            entry({ relativePath: 'one/two/a.js', mode: '0o666', size: 1, mtime: 1 }),
            entry({ relativePath: 'one/two/b.js', mode: '0o666', size: 1, mtime: 1 }),
          ]
        });
      });

      it('catches each update', function() {
        var result = fsTree.calculatePatch(new FSTree({
          entries: [
            entry({ relativePath: 'a.js', mode: '0o666', size: 1, mtime: 2 }),
            entry({ relativePath: 'b.js', mode: '0o666', size: 1, mtime: 1 }),
            entry({ relativePath: 'one/a.js', mode: '0o666', size: 10, mtime: 1 }),
            entry({ relativePath: 'one/b.js', mode: '0o666', size: 1, mtime: 1 }),
            entry({ relativePath: 'one/two/a.js', mode: '0o667', size: 1, mtime: 1 }),
            entry({ relativePath: 'one/two/b.js', mode: '0o666', size: 1, mtime: 1 }),
          ]
        }));

        expect(result).to.deep.equal([
          ['change', 'a.js', entry({ relativePath: 'a.js', size: 1, mtime: 2, mode: '0o666' })],
          ['change', 'one/a.js', entry({ relativePath: 'one/a.js', size: 10, mtime: 1, mode: '0o666'})],
          ['change', 'one/two/a.js', entry({ relativePath: 'one/two/a.js', mode: '0o667', size: 1, mtime: 1})],
        ]);
      });
    });

    context('with unchanged paths', function() {
      beforeEach( function() {
        fsTree = FSTree.fromPaths([
          'bar/baz.js',
          'foo.js',
        ]);
      });

      it('returns an empty changeset', function() {
        expect(fsTree.calculatePatch(FSTree.fromPaths([
          'bar/baz.js',
          'foo.js'
        ]))).to.deep.equal([
          // when we work with entries, will potentially return updates
        ]);
      });
    });


    context('from a non-empty tree', function() {
      beforeEach( function() {
        fsTree = FSTree.fromPaths([
          'foo/one.js',
          'foo/two.js',
          'bar/one.js',
          'bar/two.js',
        ]);
      });

      context('with removals', function() {
        it('reduces the rm operations', function() {
          expect(fsTree.calculatePatch(FSTree.fromPaths([
            'bar/two.js'
          ]))).to.deep.equal([
            ['unlink', 'foo/one.js', undefined],
            ['unlink', 'foo/two.js', undefined],
            ['unlink', 'bar/one.js', undefined],
            ['rmdir',  'foo',        undefined],
          ]);
        });
      });

      context('with removals and additions', function() {
        it('reduces the rm operations', function() {
          expect(fsTree.calculatePatch(FSTree.fromPaths([
            'bar/three.js'
          ]))).to.deep.equal([
            ['unlink', 'foo/one.js', undefined],
            ['unlink', 'foo/two.js', undefined],
            ['unlink', 'bar/one.js', undefined],
            ['unlink', 'bar/two.js', undefined],
            ['rmdir',  'foo',        undefined],

            // TODO: we could detect this NOOP [[rmdir bar] => [mkdir bar]] , but leaving it made File ->
            // Folder & Folder -> File transitions easiest. Maybe some future
            // work can explore, but the overhead today appears to be
            // neglibable

            ['rmdir', 'bar', undefined],
            ['mkdir', 'bar', directory('bar')],

            ['create', 'bar/three.js', file('bar/three.js')]
          ]);
        });
      });
    });

    context('from a deep non-empty tree', function() {
      beforeEach( function() {
        fsTree = FSTree.fromPaths([
          'bar/quz/baz.js',
          'foo.js',
        ]);
      });

      context('to an empty tree', function() {
        it('returns n rm operations', function() {
          expect(fsTree.calculatePatch(FSTree.fromPaths([]))).to.deep.equal([
            ['unlink', 'bar/quz/baz.js', undefined],
            ['rmdir', 'bar/quz',         undefined],
            ['rmdir', 'bar',             undefined],
            ['unlink', 'foo.js',         undefined]
          ]);
        });
      });
    });

    context('from a deep non-empty tree \w intermediate entry', function() {
      beforeEach( function() {
        fsTree = FSTree.fromPaths([
          'bar/quz/baz.js',
          'bar/foo.js',
        ]);
      });

      context('to an empty tree', function() {
        it('returns one unlink operation', function() {
          expect(fsTree.calculatePatch(FSTree.fromPaths([
            'bar/quz/baz.js'
          ]))).to.deep.equal([
            ['unlink', 'bar/foo.js', undefined]
          ]);
        });
      });
    });

    context('another nested scenario', function() {
      beforeEach( function() {
        fsTree = FSTree.fromPaths([
          'subdir1/subsubdir1/foo.png',
          'subdir2/bar.css'
        ]);
      });

      context('to an empty tree', function() {
        it('returns one unlink operation', function() {
          expect(fsTree.calculatePatch(FSTree.fromPaths([
            'subdir1/subsubdir1/foo.png'
          ]))).to.deep.equal([
            ['unlink', 'subdir2/bar.css', undefined],
            ['rmdir',  'subdir2',         undefined]
          ]);
        });
      });
    });

    context('folder => file', function() {
      beforeEach( function() {
        fsTree = FSTree.fromPaths([
          'subdir1/foo'
        ]);
      });

      it('it unlinks the file, and rmdir the folder and then creates the file', function() {
        expect(fsTree.calculatePatch(FSTree.fromPaths([
          'subdir1'
        ]))).to.deep.equal([
          ['unlink', 'subdir1/foo', undefined],
          ['rmdir',  'subdir1',     undefined],
          ['create', 'subdir1',     file('subdir1')]
        ]);
      });
    });

    context('file => folder', function() {
      beforeEach( function() {
        fsTree = FSTree.fromPaths([
          'subdir1'
        ]);
      });

      it('it unlinks the file, and makes the folder and then creates the file', function() {
        expect(fsTree.calculatePatch(FSTree.fromPaths([
          'subdir1/foo'
        ]))).to.deep.equal([
          ['unlink', 'subdir1',     undefined],
          ['mkdir',  'subdir1',     directory('subdir1')],
          ['create', 'subdir1/foo', file('subdir1/foo')]
        ]);
      });
    });

    context('folders', function() {
      beforeEach( function() {
        fsTree = FSTree.fromPaths([
          'dir/',
          'dir2/subdir1/',
          'dir3/subdir1/'
        ]);
      });

      it('it unlinks the file, and makes the folder and then creates the file', function() {
        var result = fsTree.calculatePatch(FSTree.fromPaths([
          'dir2/subdir1/',
          'dir3/',
          'dir4/',
        ]));

        expect(result).to.deep.equal([
          ['rmdir', 'dir3/subdir1', undefined],
          ['rmdir', 'dir',          undefined],
          // This no-op (rmdir dir3; mkdir dir3) is not fundamental: a future
          // iteration could reasonably optimize it away
          ['rmdir', 'dir3', undefined],
          ['mkdir', 'dir3', directory('dir3')],
          ['mkdir', 'dir4', directory('dir4')]
        ]);
      });
    });
  });
});
