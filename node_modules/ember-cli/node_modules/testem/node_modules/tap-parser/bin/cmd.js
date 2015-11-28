#!/usr/bin/env node

var Parser = require('../')
var etoa = require('events-to-array')
var util = require('util')

var args = process.argv.slice(2)
var json = null

args.forEach(function (arg, i) {
  if (arg === '-j') {
    json = args[i + 1] || 2
  } else {
    var m = arg.match(/^--json(?:=([0-9]+))$/)
    if (m)
      json = +m[1] || args[i + 1] || 2
  }

  if (arg === '-h' || arg === '--help')
    usage()
})

function usage () {
  console.log(function () {/*
Usage:
  tap-parser [-j [<indent>] | --json[=indent]]

Parses TAP data from stdin, and outputs an object representing
the data found in the TAP stream to stdout.

If there are any failures in the TAP stream, then exits with a
non-zero status code.

Data is output by default using node's `util.format()` method, but
JSON can be specified using the `-j` or `--json` flag with a number
of spaces to use as the indent (default=2).
*/}.toString().split('\n').slice(1, -1).join('\n'))

  if (!process.stdin.isTTY)
    process.stdin.resume()

  process.exit()
}

function format (msg) {
  if (json !== null)
    return JSON.stringify(msg, null, +json)
  else
    return util.inspect(events, null, Infinity)
}

var parser = new Parser()
var events = etoa(parser, [ 'pipe', 'unpipe', 'prefinish', 'finish', 'line' ])

process.stdin.pipe(parser)
process.on('exit', function () {
  console.log(format(events))
  if (!parser.ok)
    process.exit(1)
})
