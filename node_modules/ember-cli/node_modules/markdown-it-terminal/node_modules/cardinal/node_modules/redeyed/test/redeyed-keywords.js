'use strict';
/*jshint asi: true*/

var test = require('tap').test
  , util = require('util')
  , redeyed = require('..')

function inspect (obj) {
  return util.inspect(obj, false, 5, true)
}

test('adding custom asserts ... ', function (t) {
  t.constructor.prototype.assertSurrounds = function (code, opts, expected) {
    var optsi = inspect(opts);
    var result = redeyed(code, opts).code

    this.equals(  result
                , expected
                , util.format('%s: %s => %s', optsi, inspect(code), inspect(expected))
               )
    return this;
  }

  t.end()
})

test('types', function (t) {
  t.test('\n# Keyword', function (t) {
    var keyconfig = { 'Keyword': { _default: '$:%' } };
    t.assertSurrounds('import foo from \'foo\';', keyconfig, '$import% foo from \'foo\';')
    t.assertSurrounds('export default foo;', keyconfig, '$export% $default% foo;')
    t.assertSurrounds('if(foo) { let bar = 1;}', keyconfig, '$if%(foo) { $let% bar = 1;}')
    t.assertSurrounds('const x = "foo";', keyconfig, '$const% x = "foo";')
    t.assertSurrounds('"use strict";(function* () { yield *v })', keyconfig, '"use strict";($function%* () { $yield% *v })')
    t.assertSurrounds('"use strict"; (class A { static constructor() { super() }})', keyconfig
                      ,'"use strict"; ($class% A { $static% constructor() { $super%() }})')
    t.assertSurrounds('class Foo { constructor(name){this.name = name;}}', keyconfig
                      , '$class% Foo { constructor(name){$this%.name = name;}}')
    t.assertSurrounds('class Foo extends Bar { constructor(name,value){super(value);this.name = name;}}', keyconfig
                      , '$class% Foo $extends% Bar { constructor(name,value){$super%(value);$this%.name = name;}}')
    t.end()
  })
})