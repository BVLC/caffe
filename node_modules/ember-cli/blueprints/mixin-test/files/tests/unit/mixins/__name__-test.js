import Ember from 'ember';
import <%= classifiedModuleName %>Mixin from '../../../mixins/<%= dasherizedModuleName %>';
import { module, test } from 'qunit';

module('<%= friendlyTestName %>');

// Replace this with your real tests.
test('it works', function(assert) {
  let <%= classifiedModuleName %>Object = Ember.Object.extend(<%= classifiedModuleName %>Mixin);
  let subject = <%= classifiedModuleName %>Object.create();
  assert.ok(subject);
});
