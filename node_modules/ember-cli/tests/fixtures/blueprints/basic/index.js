'use strict';

var Blueprint = require('../../../../lib/models/blueprint');
var Promise = require('../../../../lib/ext/promise');

module.exports = Blueprint.extend({
  description: 'A basic blueprint',
  beforeInstall: function(options, locals){
      return Promise.resolve().then(function(){
          locals.replacementTest = 'TESTY';
      });
  }
});
