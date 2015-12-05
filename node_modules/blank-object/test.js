var BlankObject = require('./index');
var assert = require('assert');

var blank = new BlankObject();

assert.deepEqual(Object.keys(blank), [], 'object is blank');

var propertyNamesOfObjectPrototype = Object.getOwnPropertyNames(Object.prototype);

propertyNamesOfObjectPrototype.forEach(function (propertyName) {
  assert.strictEqual(blank[propertyName], undefined, propertyName + ' is undefined');

  blank[propertyName] = 1
  assert.strictEqual(blank[propertyName], 1, propertyName + ' can be written and read');
});

assert.deepEqual(Object.keys(blank), propertyNamesOfObjectPrototype, ' keys returns properties that are written');
