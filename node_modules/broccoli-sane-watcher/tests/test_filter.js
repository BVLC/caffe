var Promise = require('rsvp').Promise;

module.exports = TestFilter;
function TestFilter(inputs, output) {
  this.inputs = inputs;
  this.output = output;
}

TestFilter.prototype.read = function (readTree) {
  var inputs = this.inputs;
  var output = this.output;
  var sequence = Promise.resolve();

  this.inputs.forEach(function (input) {
    sequence = sequence.then(function () {
      return readTree(input);
    });
  });

  return sequence.then(function () {
    return output();
  });
};
