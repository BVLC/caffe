var is = require('core-util-is');

Object.keys(is).forEach(function(m) {
  var name = m.slice(2);
  name = name[0].toLowerCase() + name.slice(1);
  exports[name] = is[m];
});
