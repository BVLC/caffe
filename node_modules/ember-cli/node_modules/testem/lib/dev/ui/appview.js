/*

appview.js
==========

The actual AppView. This encapsulates the entire UI.

*/

var View = require('./view')
var tabs = require('./runner_tabs')
var constants = require('./constants')
var RunnerTab = tabs.RunnerTab
var RunnerTabs = tabs.RunnerTabs
var Screen = require('./screen')
var pad = require('../../strutils').pad
var log = require('npmlog')
var ErrorMessagesPanel = require('./error_messages_panel')


var AppView = module.exports = View.extend({
  defaults: {
    currentTab: 0,
    atLeastOneRunner: false
  },
  initialize: function(attrs){
    var app = attrs.app
    this.name = 'Testem'
    this.app = app
    this.config = app.config
    if (!this.get('screen')){
      this.set('screen', Screen())
    }

    this.initCharm()

    var screen = this.get('screen')

    var runnerTabs = this.runnerTabs = new RunnerTabs([], {
      appview: this,
      screen: screen
    })
    this.set({
      runnerTabs: runnerTabs
    })
    var self = this
    var runners = this.runners()
    runners.on('add', function(runner, options){
      var idx = options.index || runners.length - 1
      var tab = new RunnerTab({
        runner: runner,
        index: idx,
        appview: self,
        screen: screen
      })
      runnerTabs.push(tab)
    })
    runners.on('add remove', function(){
      self.set('atLeastOneRunner', runners.length > 0)
    })
    runnerTabs.on('add', function(){
      runnerTabs.render()
    })
    this.on('change:atLeastOneRunner', function(atLeastOne){
      if (atLeastOne && self.get('currentTab') < 0){
        self.set('currentTab', 0)
      }
      self.renderMiddle()
      self.renderBottom()
    })
    this.on('change:lines change:cols', function(){
      self.render()
    })

    this.errorMessagesPanel = new ErrorMessagesPanel({
      appview: this,
      text: '',
      screen: screen
    })
    this.errorMessagesPanel.on('change:text', function(m, text){
      self.set('isPopupVisible', !!text)
    })
    this.startMonitorTermSize()
  },
  initCharm: function(){
    var screen = this.get('screen')
    screen.reset()
    screen.erase('screen')
    screen.cursor(false)
    screen.on('data', this.onInputChar.bind(this))
    screen.removeAllListeners('^C')
    screen.on('^C', function(buf){
      this.trigger('ctrl-c')
    }.bind(this))
  },
  startMonitorTermSize: function(){
    var self = this
    var screen = self.get('screen')
    this.updateScreenDimensions()
    process.stdout.on('resize', function(){
      var cols = process.stdout.columns
      var lines = process.stdout.rows
      if (cols !== self.get('cols') || lines !== self.get('lines')){
        self.updateScreenDimensions()
      }
    })
  },
  updateScreenDimensions: function(){
    var screen = this.get('screen')
    var cols = process.stdout.columns
    var lines = process.stdout.rows
    screen.enableScroll(constants.LogPanelUnusedLines, lines - 1)
    this.set({
      cols: cols,
      lines: lines
    })
    this.updateErrorMessagesPanelSize()
  },
  updateErrorMessagesPanelSize: function(){
    this.errorMessagesPanel.set({
      line: 2,
      col: 4,
      width: this.get('cols') - 8,
      height: this.get('lines') - 4
    })
  },
  render: function(){
    this.renderTop()
    if (!this.get('atLeastOneRunner')){
      this.renderMiddle()
    }
    this.renderBottom()
  },
  renderTop: function(){
    if (this.isPopupVisible()) return

    var screen = this.get('screen')
    var url = this.config.get('url')
    var cols = this.get('cols')
    screen
      .position(0, 1)
      .write(pad('TEST\u0027EM \u0027SCRIPTS!', cols, ' ', 1))
      .position(0, 2)
      .write(pad('Open the URL below in a browser to connect.', cols, ' ', 1))
      .position(0, 3)
      .display('underscore')
      .write(url, cols, ' ', 1)
      .display('reset')
      .position(url.length + 1, 3)
      .write(pad('', cols - url.length, ' ', 1))

  },
  renderMiddle: function(){
    if (this.isPopupVisible()) return
    if (this.runners.length > 0) return

    var screen = this.get('screen')
    var lines = this.get('lines')
    var cols = this.get('cols')
    var textLineIdx = Math.floor(lines / 2 + 2)
    for (var i = constants.LogPanelUnusedLines; i < lines; i++){
      var text = (i === textLineIdx ? 'Waiting for runners...' : '')
      screen
        .position(0, i)
        .write(pad(text, cols, ' ', 2))
    }
  },
  renderBottom: function(){
    if (this.isPopupVisible()) return

    var screen = this.get('screen')
    var cols = this.get('cols')
    var pauseStatus = this.app.paused ? '; p to unpause (PAUSED)' : '; p to pause'

    var msg = (
      !this.get('atLeastOneRunner') ?
      'q to quit' :
      'Press ENTER to run tests; q to quit'
      )
    msg = '[' + msg + pauseStatus + ']'
    screen
      .position(0, this.get('lines'))
      .write(pad(msg, cols - 1, ' ', 1))
  },
  runners: function(){
    return this.app.runners
  },
  currentRunnerTab: function(){
    var idx = this.get('currentTab')
    return this.runnerTabs.at(idx)
  },
  onInputChar: function(buf){
    try{
      var chr = String(buf).charAt(0)
      var i = chr.charCodeAt(0)
      var key = (buf[0] === 27 && buf[1] === 91) ? buf[2] : null
      var currentRunnerTab = this.currentRunnerTab()
      var splitPanel = currentRunnerTab && currentRunnerTab.splitPanel

      //log.info([buf[0], buf[1], buf[2]].join(','))
      if (key === 67){ // right arrow
        this.nextTab()
      }else if (key === 68){ // left arrow
        this.prevTab()
      }else if (key === 66){ // down arrow
        splitPanel.scrollDown()
      }else if (key === 65){ // up arrow
        splitPanel.scrollUp()
      }else if (chr === '\t'){
        splitPanel.toggleFocus()
      }else if (chr === ' ' && splitPanel){
        splitPanel.pageDown()
      }else if (chr === 'b'){
        splitPanel.pageUp()
      }else if (chr === 'u'){
        splitPanel.halfPageUp()
      }else if (chr === 'd'){
        splitPanel.halfPageDown()
      }
      this.trigger('inputChar', chr, i)
    }catch(e){
      log.error('In onInputChar: ' + e + '\n' + e.stack)
    }
  },
  nextTab: function(){
    var currentTab = this.get('currentTab')
    currentTab++
    if (currentTab >= this.runners().length)
      currentTab = 0

    var runner = this.runners().at(currentTab)
    this.set('currentTab', currentTab)
  },
  prevTab: function(){
    var currentTab = this.get('currentTab')
    currentTab--
    if (currentTab < 0)
      currentTab = this.runners().length - 1

    var runner = this.runners().at(currentTab)
    this.set('currentTab', currentTab)
  },
  setErrorPopupMessage: function(msg){
    this.errorMessagesPanel.set('text', msg)
  },
  clearErrorPopupMessage: function(){
    this.errorMessagesPanel.set('text', '')
    this.render()
  },
  isPopupVisible: function(){
    return !! this.get('isPopupVisible')
  },
  setRawMode: function() {
    if (process.stdin.isTTY) {
      process.stdin.setRawMode(false)
    }
  },
  cleanup: function(cb){
    var screen = this.get('screen')
    screen.display('reset')
    screen.erase('screen')
    screen.position(0, 0)
    screen.enableScroll()
    screen.cursor(true)
    this.setRawMode(false)
    screen.destroy()
    if (cb) cb()
  }
})
