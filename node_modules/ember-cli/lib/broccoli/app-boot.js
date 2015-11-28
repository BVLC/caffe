/* jshint ignore:start */

define('{{MODULE_PREFIX}}/config/environment', ['ember'], function(Ember) {
  {{content-for 'config-module'}}
});

{{content-for 'app-boot'}}

/* jshint ignore:end */
