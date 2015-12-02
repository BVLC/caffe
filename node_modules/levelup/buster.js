var config = module.exports

config['unit'] = {
    environment: 'node'
  , tests: [ 'test/*-test.js' ]
  , libs: [ 'test/common.js' ]
}