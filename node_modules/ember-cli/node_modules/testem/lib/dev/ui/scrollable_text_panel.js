var View = require('./view')
var log = require('npmlog')
var splitLines = require('../../strutils').splitLines
var Screen = require('./screen')

// This is a generic scrollable text viewer widget. Should be refactored
// out to another file or npm module at some point.
var ScrollableTextPanel = module.exports = View.extend({
  defaults: {
    visible: true,
    text: '',
    textLines: [],
    scrollOffset: 0
  },
  // expect the attributes to have
  // -----------------------------
  //
  // * line and col (top left coordinates)
  // * height and width
  initialize: function(){
    var self = this
    if (!this.get('screen')){
      this.set('screen', Screen())
    }
    this.updateTextLines()
    this.observe(this, 'change:text change:width', function(model, text){
      self.updateTextLines()
    })
    this.observe(this, 'change:visible change:textLines change:height', function(){
      self.render()
    })
    this.render()
  },
  updateTextLines: function(){
    var text = this.get('text')
    var lines = splitLines(text, this.get('width'))
    this.set('textLines', lines)
  },
  scrollUp: function(){
    var line = this.get('line')
    var height = this.get('height')
    var scrollOffset = this.get('scrollOffset')
    if (scrollOffset > 0){
      scrollOffset--
      this.set('scrollOffset', scrollOffset, {silent: true})
      this.render()
    }
  },
  scrollDown: function(){
    var line = this.get('line')
    var height = this.get('height')
    var scrollOffset = this.get('scrollOffset')
    var textLines = this.get('textLines')
    if (textLines.length > height + scrollOffset){
      scrollOffset++
      this.set('scrollOffset', scrollOffset, {silent: true})
      this.render()
    }
  },
  pageUp: function(){
    var height = this.get('height')
    var scrollOffset = this.get('scrollOffset')
    this.scrollTo(Math.max(0, scrollOffset - height))
  },
  pageDown: function(){
    var height = this.get('height')
    var scrollOffset = this.get('scrollOffset')
    var textLines = this.get('textLines')
    this.scrollTo(Math.min(scrollOffset + height, textLines.length - height))
  },
  halfPageUp: function(){
    var height = this.get('height')
    var scrollOffset = this.get('scrollOffset')
    this.scrollTo(Math.max(0, scrollOffset - Math.ceil(height / 2)))
  },
  halfPageDown: function(){
    var height = this.get('height')
    var scrollOffset = this.get('scrollOffset')
    var textLines = this.get('textLines')
    this.scrollTo(Math.min(scrollOffset + Math.ceil(height / 2), textLines.length - height))
  },
  scrollTo: function(newOffset){
    var scrollOffset = this.get('scrollOffset')
    if (scrollOffset !== newOffset){
      this.set('scrollOffset', newOffset, {slient: true})
      this.render()
    }
  },
  render: function(firstOrLast){
    if (!this.get('visible')) return

    var screen = this.get('screen')        
    var startLine = this.get('line')
    var col = this.get('col')
    var width = this.get('width')
    var height = this.get('height')
    var textLines = this.get('textLines')
    var scrollOffset = this.get('scrollOffset')

    function renderLine(i){
      var idx = i + scrollOffset
      var textLine = textLines[idx] || ''
      var output = textLine.toString()
      screen
        .position(col, startLine + i + 1)
        .write(output)
      if (textLine.length < width)
        screen.erase('end')
    }
        
    if (!firstOrLast){
      for (var i = 0; i < height; i++){
        renderLine(i)
      }
    }else if (firstOrLast === 'first'){
      renderLine(0)
    }else if (firstOrLast === 'last'){
      renderLine(height - 1)
    }

    screen.display('reset')
  }
})
