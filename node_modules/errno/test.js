#!/usr/bin/env node

var assert = require('assert')
  , errno = require('./')

assert(errno.all, 'errno.all not found')
assert(errno.errno, 'errno.errno not found')
assert(errno.code, 'errno.code not found')

assert(errno.all.length === 59, 'found ' + errno.all.length + ', expected 59')

assert(errno.errno['-1'] === errno.all[0], 'errno -1 not first element')

assert(errno.code['UNKNOWN'] === errno.all[0], 'code UNKNOWN not first element')

assert(errno.errno[1] === errno.all[2], 'errno 1 not third element')

assert(errno.code['EOF'] === errno.all[2], 'code EOF not third element')
