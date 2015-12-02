var cliff = require('../lib/cliff');

var rows = [
  ['Name',        'Flavor',              'Dessert'],
  ['Alice'.grey,  'cherry'.cyan,         'yogurt'.yellow],
  ['Bob'.magenta, 'carmel'.rainbow,      'apples'.white],
  ['Joe'.italic,  'chocolate'.underline, 'cake'.inverse],
  ['Nick'.bold,   'vanilla',             'ice cream']
];

cliff.putRows('data', rows, ['red', 'blue', 'green']);

