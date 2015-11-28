/*

view.js
==========

Base-class for our view-models.
*/

var tty = require('tty')
var log = require('npmlog')
var Backbone = require('backbone')


// ============== Backbone-based View Models ============================


// View is the base class for our view models. That's right, view-models.
// All of our views carry state of some sort.
var View = module.exports = Backbone.Model.extend({
  observe: function(model, thing){
    var eventMap
    if (typeof thing === 'string' && arguments.length === 3){
      eventMap = {}
      eventMap[thing] = arguments[2]
    }else{
      eventMap = thing
    }
    for (var event in eventMap){
      model.on(event, eventMap[event])
    }
    if (!this.observers)
      this.observers = []
    this.observers.push([model, eventMap])
  },
  destroy: function(){
    this.removeObservers()
  },
  removeObservers: function(){
    if (!this.observers) return
    this.observers.forEach(function(observer){
      var model = observer[0]
      var eventMap = observer[1]
      for (var event in eventMap){
        model.off(event, eventMap[event])
      }
    })
  }
})
