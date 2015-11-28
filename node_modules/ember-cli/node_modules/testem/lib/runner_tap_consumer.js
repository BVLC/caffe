var TapConsumer = require('./tap_consumer')

function RunnerTapConsumer(runner){
  this.runner = runner
  var tapConsumer = new TapConsumer
  tapConsumer.on('test-result', function(test){
    runner.get('results').addResult(test)
  })
  tapConsumer.on('error', function(){
    runner.set('results', null)
  })
  tapConsumer.on('all-test-results', function(){
    runner.get('results').set('all', true)
    tapConsumer.removeAllListeners()
    runner.trigger('all-test-results', this.results)
    runner.trigger('tests-end')
  })
  return tapConsumer
}

module.exports = RunnerTapConsumer
