var grammars = exports,
    ometajs = require('../../ometajs');

grammars.AbstractGrammar = ometajs.core.AbstractGrammar;

// Lazy getters for BSJS stuff
var bsjs = undefined;

function lazyDescriptor(property) {
  return {
    enumerable: true,
    get: function get() {
      if (bsjs === undefined) bsjs = require('./bsjs');

      return bsjs[property];
    }
  };
}

Object.defineProperties(grammars, {
  BSJSParser: lazyDescriptor('BSJSParser'),
  BSJSIdentity: lazyDescriptor('BSJSIdentity'),
  BSJSTranslator: lazyDescriptor('BSJSTranslator')
});
