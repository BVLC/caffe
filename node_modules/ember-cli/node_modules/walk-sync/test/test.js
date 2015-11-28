var fs = require('fs')
var test = require('tap').test
var walkSync = require('../')

test('walkSync', function (t) {
  t.deepEqual(walkSync('fixtures'), [
    'dir/',
    'dir/bar.txt',
    'dir/subdir/',
    'dir/subdir/baz.txt',
    'dir/zzz.txt',
    'foo.txt',
    'some-other-dir/',
    'some-other-dir/qux.txt',
    'symlink1/',
    'symlink1/qux.txt',
    'symlink2',
  ])

  t.throws(function () {
    walkSync('doesnotexist')
  }, {
    name: 'Error',
    message: "ENOENT, no such file or directory 'doesnotexist/'"
  })

  t.throws(function () {
    walkSync('fixtures/foo.txt')
  }, {
    name: 'Error',
    message: "ENOTDIR, not a directory 'fixtures/foo.txt/'"
  })

  t.end()
})
