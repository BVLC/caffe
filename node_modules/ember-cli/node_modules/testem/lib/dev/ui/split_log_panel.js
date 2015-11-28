var View = require('./view')
var ScrollableTextPanel = require('./scrollable_text_panel')
var log = require('npmlog')
var tabs = require('./constants')
var StyledString = require('styled_string')
var Chars = require('../../chars')
var indent = require('../../strutils').indent
var Screen = require('./screen')

function failureDisplay(item){
  var extra = []
  var stacktrace = item.stack
  if (stacktrace){
    var stacklines = stacktrace.split('\n')
    if (stacklines[0] === item.message)
      stacktrace = stacklines.slice(1).map(function(line){
        return line.trim()
      }).join('\n')
    extra.push(stacktrace)
  }else{
    if (item.file)
      extra.push(item.file)
    if (item.line)
      extra.push(' ' + item.line)
  }

  if (item.expected) {
    extra.push(' expected ' + item.expected)
  }

  if (item.actual) {
    extra.push(' actual ' + item.actual)
  }

  if (item.at){
    extra.push(' at ' + item.at)
  }

  return Chars.cross + ' ' + (item.message || 'failed') +
    (extra ? '\n' + indent(extra.join('\n')) : '')
}


function failedTestDisplay(test){
  var failedItems = test.get('items').filter(function(item){
    return !item.passed
  })
  return test.get('name') + '\n' + 
    indent(failedItems.map(failureDisplay).join('\n'))
}

var SplitLogPanel = module.exports = View.extend({
  defaults: {
    visible: false,
    focus: 'top'
  },
  initialize: function(attrs){
    if (!this.get('screen')){
      this.set('screen', Screen())
    }
    var runner = this.get('runner')
    var results = runner.get('results')
    var messages = runner.get('messages')
    var appview = this.get('appview')
    var visible = this.get('visible')
    var self = this
    var screen = this.get('screen')
    var topPanel = this.topPanel = new ScrollableTextPanel({
      line: tabs.TabStartLine + tabs.TabHeight - 1,
      col: 0,
      visible: visible,
      screen: screen
    })
    var bottomPanel = this.bottomPanel = new ScrollableTextPanel({
      col: 0,
      visible: visible,
      screen: screen
    })
    this.observe(appview, 'change:cols change:lines', function(){
      self.syncDimensions()
      self.render()
    })
    if (results){
      this.observe(results, 'change', function(){
        self.syncDimensions()
        self.syncResultsDisplay()
      })
    }
    this.observe(messages, 'reset add remove', function(){
      self.syncDimensions()
      self.syncMessages()
    })
    this.observe(this, 'change:visible', function(){
      var visible = self.get('visible')
      topPanel.set('visible', visible, {silent: true})
      bottomPanel.set('visible', visible, {silent: true})
      self.syncDimensions({silent: true})
      self.render()
    })
    this.syncDimensions({silent: true})
    this.syncResultsDisplay({silent: true})
    this.syncMessages({silent: true})
    this.render()
  },
  toggleFocus: function(){
    var focus = this.get('focus')
    this.set('focus', focus === 'top' ? 'bottom' : 'top')
  },
  resetScrollPositions: function(){
    this.topPanel.set('scrollOffset', 0)
    this.bottomPanel.set('scrollOffset', 0)
  },
  targetPanel: function(){
    var runner = this.get('runner')
    var bottomPanel = this.bottomPanel
    var topPanel = this.topPanel
    if (runner.hasMessages() && runner.hasResults()){
      return (this.get('focus') === 'top') ? topPanel : bottomPanel
    }else if (runner.hasMessages()){
      return bottomPanel
    }else if (runner.hasResults()){
      return topPanel
    }else{
      return topPanel
    }
  },
  scrollUp: function(){
    this.targetPanel().scrollUp()
  },
  scrollDown: function(){
    this.targetPanel().scrollDown()
  },
  pageUp: function(){
    this.targetPanel().pageUp()
  },
  pageDown: function(){
    this.targetPanel().pageDown()
  },
  halfPageUp: function(){
    this.targetPanel().halfPageUp()
  },
  halfPageDown: function(){
    this.targetPanel().halfPageDown()
  },
  syncMessages: function(options){
    this.bottomPanel.set('text', this.getMessagesText(), options)
  },
  syncResultsDisplay: function(options){
    this.topPanel.set('text', this.getResultsDisplayText(), options)
  },
  syncDimensions: function(options){
    var appview = this.get('appview')
    var termCols = appview.get('cols')
    var termLines = appview.get('lines')
    var runner = this.get('runner')
    if (runner.hasMessages() && runner.hasResults()){
      var midLine = Math.floor((termLines - tabs.LogPanelUnusedLines) / 2)
            
      this.topPanel.set({
        height: midLine,
        width: termCols
      }, options)
      var line = midLine + tabs.TabStartLine + tabs.TabHeight - 1
      var bottomPanelAttrs = {
        line: line,
        height: termLines - line - 1,
        width: termCols
      }
      this.bottomPanel.set(bottomPanelAttrs, options)
    }else if (runner.hasMessages()){ // only has messages
      this.topPanel.set({
        height: 0,
        width: termCols
      }, options)
      var height = termLines - tabs.LogPanelUnusedLines
      this.bottomPanel.set({
        line: tabs.TabStartLine + tabs.TabHeight - 1,
        height: height,
        width: termCols
      }, options)
    }else{ // only has results

      // Hide the bottom panel if there are no messages 
      // to be displayed
      var topPanelHeight = termLines - tabs.LogPanelUnusedLines
      this.topPanel.set({
        height: topPanelHeight,
        width: termCols
      }, options)
      this.bottomPanel.set({
        line: tabs.TabStartLine + tabs.TabHeight + topPanelHeight,
        height: 0,
        width: termCols
      }, options)
    }
  },
  render: function(){
    this.topPanel.render()
    this.bottomPanel.render()
  },
  getResultsDisplayText: function(){
    var appview = this.get('appview')
    var runner = this.get('runner')
    var idx = appview.get('currentTab')
    var results = runner.get('results')
    var topLevelError = results ? results.get('topLevelError') : null
    var tests = null
    var out = ''
    var pendingOut = ''

    if (topLevelError){
      out += "Top Level:\n" + indent(topLevelError) + '\n\n'
    }

    if (results && (tests = results.get('tests'))){
      var total = results.get('total')
      var pending = results.get('pending')
      var allDone = results.get('all')
      if (!total){
        out = allDone ? 'No tests were run :(' : 'Please be patient :)'
      }else{
        var failedTests = tests.filter(function(test){
          return test.get('failed') > 0
        })
        if (failedTests.length > 0){
           out += failedTests.map(failedTestDisplay).join('\n')
        }else{ // All passed
          if (allDone){
            var pendingTests = tests.filter(function(test){
              return test.get('pending') > 0
            })
            out += Chars.success + ' ' + total + ' tests complete'
            if (pending > 0){
              out += ' (' + pending + ' pending)'
            }
            out += '.'
            if (pending){
              pendingOut += '\n\n'
              pendingOut += pendingTests.map(function(test) {
                return '[PENDING] ' + test.get('name')
              }).join('\n')
            }
          }else{
            out += 'Looking good...'
          }
        }
      }
    }

    return StyledString(out, {foreground: 'cyan'})
      .append(StyledString(pendingOut, {foreground: 'yellow'}))
  },
  getMessagesText: function(){
    var messages = this.get('runner').get('messages')
    var retval = StyledString('')
    
    messages.forEach(function(message, i){
      var type = message.get('type')
      var text = message.get('text')
      var color = message.get('color') || ( type === 'error' ? 'red' : 'yellow' )
      retval = retval.concat(StyledString(text, {foreground: color}))
    })
    return retval
  }
})
