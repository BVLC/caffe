'use strict';

module.exports = function(output) {
  return JSON.parse(output.substr(output.indexOf('{')));
};
