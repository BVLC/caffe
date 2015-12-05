var Promise = require('rsvp').Promise;

module.exports = function promiseMapSeries(array, iterator, thisArg) {
  var length = array.length
  var current = Promise.resolve()
  var results = new Array(length)
  var cb = arguments.length > 2 ? iterator.bind(thisArg) : iterator

  for (var i = 0; i < length; ++i) {
    current = results[i] = current.then(function(i) {
      return cb(array[i], i, array)
    }.bind(undefined, i))
  }

  return Promise.all(results)
}
