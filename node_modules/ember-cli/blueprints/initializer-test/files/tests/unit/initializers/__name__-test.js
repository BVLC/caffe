import Ember from 'ember';
import <%= classifiedModuleName %>Initializer from '<%= dependencyDepth %>/initializers/<%= dasherizedModuleName %>';
import { module, test } from 'qunit';

let application;

module('<%= friendlyTestName %>', {
  beforeEach() {
    Ember.run(function() {
      application = Ember.Application.create();
      application.deferReadiness();
    });
  }
});

// Replace this with your real tests.
test('it works', function(assert) {
  <%= classifiedModuleName %>Initializer.initialize(application);

  // you would normally confirm the results of the initializer here
  assert.ok(true);
});
