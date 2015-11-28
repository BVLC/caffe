'use strict';

module.exports = function(name) {
  name = name.toLowerCase();

  if (['test', 'ember', 'ember-cli', 'vendor', 'app'].indexOf(name) > -1) { return false; }
  if (name.indexOf('.') > -1) { return false; }
  if (!isNaN(parseInt(name.charAt(0), 10))) { return false; }

  return true;
};
