var cliff = require('../lib/cliff');

var rows = [
  ['Name',  'Flavor',    'Dessert'],
  ['Alice', 'cherry',    'yogurt'],
  ['Bob',   'carmel',    'apples'],
  ['Joe',   'chocolate', 'cake'],
  ['Nick',  'vanilla',   'ice cream']
];

console.log(cliff.stringifyRows(rows, ['red', 'blue', 'green']));
