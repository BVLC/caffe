import { test } from 'qunit';
import moduleForAcceptance from 'my-app/tests/helpers/module-for-acceptance';

moduleForAcceptance('Acceptance | foo');

test('visiting /foo', function(assert) {
  visit('/foo');

  andThen(function() {
    assert.equal(currentURL(), '/foo');
  });
});
