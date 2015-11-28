var log = require('npmlog')
var Backbone = require('backbone')
var TestResults = require('./test_results')
var RunnerTapConsumer = require('./runner_tap_consumer')
var BrowserTapConsumer = require('./browser_tap_consumer')

function TestMessagesCollector(runner){
  var socket = runner.get('socket')
  var results = runner.get('results')
  var messages = runner.get('messages')

  socket.on('top-level-error', function(msg, url, line){
    var message = msg + ' at ' + url + ', line ' + line + '\n'
    messages.add({
      type: 'error',
      text: message
    }, {at: 0})
  })
  socket.on('error', function(message){
    messages.push({
      type: 'error',
      text: message + '\n'
    })
  })
  socket.on('info', function(message) {
    messages.push({
      type: 'info',
      text: message + '\n',
      color: 'green'
    })
  })
  socket.on('warn', function(message){
    messages.push({
      type: 'warn',
      text: message + '\n',
      color: 'cyan'
    })
  })
  socket.on('log', function(message){
    messages.push({
      type: 'log',
      text: message + '\n'
    })
  })
  socket.on('tests-start', function(){
    runner.trigger('tests-start')
  })
  socket.on('test-result', function(result){
    results.addResult(result)
    runner.trigger('test-result', result)
  })
  socket.on('all-test-results', function(){
    results.set('all', true)
    runner.trigger('tests-end')
    runner.trigger('all-test-results', results)
  })
}

var BrowserRunner = Backbone.Model.extend({
  defaults: {
    type: 'browser'
  },
  initialize: function(){
    this.set({
      messages: new Backbone.Collection,
      results: new TestResults
    }, {silent: true})
    this.registerSocketEvents()
    this.on('change:socket', function(){
      this.previous('socket').removeAllListeners()
      this.registerSocketEvents()
    }, this)
  },
  registerSocketEvents: function(){
    var self = this
    var results = this.get('results')
    var messages = this.get('messages')
    var socket = this.get('socket')
    new TestMessagesCollector(this)
    BrowserTapConsumer(socket, RunnerTapConsumer(this))

    socket.on('disconnect', function(){
      log.info('Client disconnected ' + self.get('name'))
      self.get('results').reset()
      self.get('messages').reset()
      self.trigger('disconnect')
    })
  },
  startTests: function(){
    this.get('results').reset()
    this.get('socket').emit('start-tests')
  },
  hasResults: function(){
    var results = this.get('results')
    var total = results.get('total')
    return total > 0
  },
  hasMessages: function(){
    return this.get('messages').length > 0
  }
})

module.exports = BrowserRunner
