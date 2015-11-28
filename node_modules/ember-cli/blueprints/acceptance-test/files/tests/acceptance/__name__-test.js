import { test } from 'qunit';
import moduleForAcceptance from '<%= testFolderRoot %>/tests/helpers/module-for-acceptance';

moduleForAcceptance('<%= friendlyTestName %>');

test('visiting /<%= dasherizedModuleName %>', function(assert) {
  visit('/<%= dasherizedModuleName %>');

  andThen(function() {
    assert.equal(currentURL(), '/<%= dasherizedModuleName %>');
  });
});
