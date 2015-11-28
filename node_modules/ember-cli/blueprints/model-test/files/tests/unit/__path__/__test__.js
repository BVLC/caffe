import { moduleForModel, test } from 'ember-qunit';

moduleForModel('<%= dasherizedModuleName %>', '<%= friendlyDescription %>', {
  // Specify the other units that are required for this test.
<%= typeof needs !== 'undefined' ? needs : '' %>
});

test('it exists', function(assert) {
  let model = this.subject();
  // let store = this.store();
  assert.ok(!!model);
});
