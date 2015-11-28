/*jshint strict:false */
/* globals QUnit */

import config from '../../config/environment';

QUnit.test('the correct config is used', function(assert) {
  assert.equal(config.fileUsed, 'config/something-else.js');
});
