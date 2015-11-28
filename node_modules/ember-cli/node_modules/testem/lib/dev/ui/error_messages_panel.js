var ScrollableTextPanel = require('./scrollable_text_panel')
var log = require('npmlog')
var Chars = require('../../chars')

var ErrorMessagesPanel = module.exports = ScrollableTextPanel.extend({
  initialize: function(attrs){
    this.updateVisibility()
    this.on('change:text', function(self, text){
      self.updateVisibility()
    })
    ScrollableTextPanel.prototype.initialize.call(this, attrs)
  },
  updateVisibility: function(){
    var text = this.get('text')
    this.set('visible', !!text)
  },
  render: function(){
    var visible = this.get('visible')
    if (!visible) return

    ScrollableTextPanel.prototype.render.call(this)

    // draw a box
    var line = this.get('line')
    var col = this.get('col') - 1
    var width = this.get('width') + 1
    var height = this.get('height') + 1
        
    var screen = this.get('screen')
    screen.foreground('red')
    screen.position(col, line)
    screen.write(Chars.topLeft + Array(width).join(Chars.horizontal) + Chars.topRight)

    for (var l = line + 1; l < line + height; l++){
      screen.position(col, l)
      screen.write(Chars.vertical)
      screen.position(col + width, l)
      screen.write(Chars.vertical)
    }

    screen.position(col, line + height)
    screen.write(Chars.bottomLeft + Array(width).join(Chars.horizontal) + Chars.bottomRight)
    screen.display('reset')
  }
})
