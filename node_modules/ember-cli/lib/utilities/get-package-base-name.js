'use strict';

module.exports = function (name) {
  var packageParts;

  if (!name) {
    return null;
  }

  packageParts = name.split('/');
  return packageParts[(packageParts.length - 1)];
};
