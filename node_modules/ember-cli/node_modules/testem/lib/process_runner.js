/*

process_runner.js
================

Model objects (via Backbone) for a browser client (a connection to a browser + the test run session)
and the returned test results for a run of tests.

*/

var Backbone = require('backbone')
var TestResults = require('./test_results')
var TapConsumer = require('./runner_tap_consumer')
var log = require('npmlog')

var ProcessRunner = Backbone.Model.extend({
  defaults: {
    type: 'process'
  },
  initialize: function(attrs){
    this.launcher = attrs.launcher
    // Assume launcher has already launched
    this.set({
      name: this.launcher.name,
      messages: new Backbone.Collection,
      results: this.isTap() ? new TestResults : null
    })

    this.startTests()
  },
  isTap: function(){
    return this.launcher.settings.protocol === 'tap'
  },
  hasResults: function(){
    return this.isTap()
  },
  hasMessages: function(){
    return this.get('messages').length > 0
  },
  registerProcess: function(process){
    var settings = this.launcher.settings
    var stdout = process.stdout
    var stderr = process.stderr
    var self = this
    if (!settings.hide_stdout){
      stdout.on('data', function(data){
        self.get('messages').push({
          type: 'log',
          text: '' + data
        })
      })
    }
    if (!settings.hide_stderr){
      stderr.on('data', function(data){
        self.get('messages').push({
          type: 'error',
          text: '' + data
        })
      })
    }
    process.on('exit', function(code){
      self.set('allPassed', code === 0)
      self.trigger('tests-end')
    })
    if (this.isTap()){
      this.setupTapConsumer(stdout)
    }
  },
  setupTapConsumer: function(stdout){
    this.message = null
    var tapConsumer = new TapConsumer(this)
    stdout.pipe(tapConsumer.stream)
  },
  startTests: function(){
    var self = this
    if (this.get('results')){
      this.get('results').reset()
    }else{
      this.set('results', this.isTap() ? new TestResults : null)
    }
    this.get('messages').reset([])
    this.set('allPassed', undefined)

    this.launcher.launch(function(process){
      self.registerProcess(process)
      setTimeout(function(){
        self.trigger('tests-start')
      }, 1)
    })
  }
})

module.exports = ProcessRunner
