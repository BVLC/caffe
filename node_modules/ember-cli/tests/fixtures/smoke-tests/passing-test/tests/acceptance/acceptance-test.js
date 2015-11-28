import moduleForAcceptance from '../helpers/module-for-acceptance';
import QUnit from 'qunit';

let application, firstArgument;

moduleForAcceptance('Module', {
  beforeEach(assert) {
    application = this.application;
    firstArgument = assert;
  },

  afterEach() {
    console.log('afterEach called');
  }
});

QUnit.test('it works', function(assert) {
  assert.ok(application, 'beforeEach binds to the setup context');
  assert.ok(
    Object.getPrototypeOf(firstArgument) === QUnit.assert,
    'first argument is QUnit assert'
  );
});
