fs            = require 'fs'
{print}       = require 'util'
{spawn, exec} = require 'child_process'
watchit       = require 'watchit'

build = (watch, callback) ->
  if typeof watch is 'function'
    callback = watch
    watch = false
  options = ['-c', '-o', 'lib', 'src']
  options.unshift '-w' if watch

  coffee = spawn 'coffee', options
  coffee.stdout.on 'data', (data) -> print data.toString()
  coffee.stderr.on 'data', (data) -> print data.toString()
  coffee.on 'exit', (status) -> callback?() if status is 0

task 'docs', 'Generate annotated source code with Docco', ->
  fs.readdir 'src', (err, contents) ->
    files = ("src/#{file}" for file in contents when /\.coffee$/.test file)
    docco = spawn 'docco', files
    docco.stdout.on 'data', (data) -> print data.toString()
    docco.stderr.on 'data', (data) -> print data.toString()
    docco.on 'exit', (status) -> callback?() if status is 0

task 'build', 'Compile CoffeeScript source files', ->
  build()

task 'watch', 'Recompile CoffeeScript source files when modified', ->
  build true

task 'test', 'Run the test suite (and re-run if anything changes)', ->
  suite = null
  build ->
    do runTests = ->
      suite?.kill()
      suiteNames = ['test']
      suiteIndex = 0
      do runNextTestSuite = ->
        return unless suiteName = suiteNames[suiteIndex]
        suite = spawn "coffee", ["-e", "{reporters} = require 'nodeunit'; reporters.default.run ['#{suiteName}.coffee']"], cwd: 'test'
        suite.stdout.on 'data', (data) -> print data.toString()
        suite.stderr.on 'data', (data) -> print data.toString()
        suite.on 'exit', -> suiteIndex++; runNextTestSuite()
      invoke 'docs'  # lest I forget
    watchTargets = (targets..., callback) ->
      for target in targets
        watchit target, include: true, (event) ->
          callback() unless event is 'success'
    watchTargets 'src', -> build runTests
    watchTargets 'test', 'Cakefile', runTests