var Backbone = require('backbone')

var TestResults = Backbone.Model.extend({
  initialize: function(){
    this.reset()
  },
  reset: function(){
    this.set({
      topLevelError: null,
      failed: 0,
      passed: 0,
      pending: 0,
      total: 0,
      tests: new Backbone.Collection,
      all: false
    })
  },
  addResult: function(result){
    var total = this.get('total')
    var pending = this.get('pending')
    var passed = this.get('passed')
    var failed = this.get('failed')
    total++
    if (result.pending)
      pending++
    else if (result.failed == 0)
      passed++
    else
      failed++
    this.get('tests').push(result)
    this.set({
      total: total,
      pending: pending,
      passed: passed,
      failed: failed,
      items: result.items
    })
  }
})

module.exports = TestResults
