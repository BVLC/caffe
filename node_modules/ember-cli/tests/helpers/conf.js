'use strict';

var Configstore = require('configstore');

var conf = new Configstore('insight-ember-cli');
var optOut = conf.get('optOut');

module.exports.setup = function() {
  conf.set('optOut', true);
};

module.exports.restore = function() {
  conf.set('optOut', optOut);
};
