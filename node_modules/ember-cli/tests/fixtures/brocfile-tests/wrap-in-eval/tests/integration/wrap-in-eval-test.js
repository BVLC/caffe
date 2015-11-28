/*jshint strict:false */
/* globals visit, andThen */

import Ember from 'ember';
import startApp from '../helpers/start-app';
import destroyApp from '../helpers/destroy-app';
import { module, test } from 'qunit';

module('wrapInEval in-app test', {
  beforeEach: function() {
    this.application = startApp();
  },
  afterEach: function() {
    destroyApp(this.application);
  }
});

test('the application boots properly with wrapInEval', function(assert) {
  assert.expect(1);

  visit('/');

  andThen(function() {
    assert.equal(Ember.$('#title').text(), 'Welcome to Ember');
  });
});
