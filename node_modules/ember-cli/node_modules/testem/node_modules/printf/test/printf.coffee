
semver = require 'semver'
should = require 'should'
printf = if process.env.PRINTF_COV then require '../lib-cov/printf' else require '../lib/printf'

describe 'sprintf', ->
  it 'Specifier: b', ->
    printf('%b', 123).should.eql '1111011'

  it 'Flag: (space)', ->
    printf('% d', 42).should.eql    ' 42'
    printf('% d', -42).should.eql   '-42'
    printf('% 5d', 42).should.eql   '   42'
    printf('% 5d', -42).should.eql  '  -42'
    printf('% 15d', 42).should.eql  '             42'
    printf('% 15d', -42).should.eql '            -42'

  it 'Flag: +', ->
    printf('%+d', 42).should.eql    '+42'
    printf('%+d', -42).should.eql   '-42'
    printf('%+5d', 42).should.eql   '  +42'
    printf('%+5d', -42).should.eql  '  -42'
    printf('%+15d', 42).should.eql  '            +42'
    printf('%+15d', -42).should.eql '            -42'

  it 'Flag: 0', ->
    printf('%0d', 42).should.eql    '42'
    printf('%0d', -42).should.eql   '-42'
    printf('%05d', 42).should.eql   '00042'
    printf('%05d', -42).should.eql  '-00042'
    printf('%015d', 42).should.eql  '000000000000042'
    printf('%015d', -42).should.eql '-000000000000042'

  it 'Flag: -', ->
    printf('%-d', 42).should.eql     '42'
    printf('%-d', -42).should.eql    '-42'
    printf('%-5d', 42).should.eql    '42   '
    printf('%-5d', -42).should.eql   '-42  '
    printf('%-15d', 42).should.eql   '42             '
    printf('%-15d', -42).should.eql  '-42            '
    printf('%-0d', 42).should.eql    '42'
    printf('%-0d', -42).should.eql   '-42'
    printf('%-05d', 42).should.eql   '42   '
    printf('%-05d', -42).should.eql  '-42  '
    printf('%-015d', 42).should.eql  '42             '
    printf('%-015d', -42).should.eql '-42            '
    printf('%0-d', 42).should.eql    '42'
    printf('%0-d', -42).should.eql   '-42'
    printf('%0-5d', 42).should.eql   '42   '
    printf('%0-5d', -42).should.eql  '-42  '
    printf('%0-15d', 42).should.eql  '42             '
    printf('%0-15d', -42).should.eql '-42            '

  it 'Precision', ->
    printf('%d', 42.8952).should.eql     '42'
    printf('%.2d', 42.8952).should.eql   '42' # Note: the %d format is an int
    printf('%.2i', 42.8952).should.eql   '42'
    printf('%.2f', 42.8952).should.eql   '42.90'
    printf('%.2F', 42.8952).should.eql   '42.90'
    printf('%.10f', 42.8952).should.eql  '42.8952000000'
    printf('%1.2f', 42.8952).should.eql  '42.90'
    printf('%6.2f', 42.8952).should.eql  ' 42.90'
    printf('%06.2f', 42.8952).should.eql '042.90'
    printf('%+6.2f', 42.8952).should.eql '+42.90'
    printf('%5.10f', 42.8952).should.eql '42.8952000000'
    printf('%1.4g', 1.06800e-10).should.eql '1.068e-10'

  it 'Bases', ->
    printf('%c', 0x7f).should.eql ''
    error = false
    try
      printf '%c', -100
    catch e
      e.message.should.eql 'invalid character code passed to %c in printf'
      error = true
    error.should.be.true
    error = false
    try
      printf '%c', 0x200000
    catch e
      e.message.should.eql 'invalid character code passed to %c in printf'
      error = true
    error.should.be.true

  it 'Mapping', ->
    # %1$s format
    printf('%1$').should.eql '%1$'
    printf('%0$s').should.eql '%0$s'
    printf('%1$s %2$s', 'Hot', 'Pocket').should.eql 'Hot Pocket'
    printf('%1$.1f %2$s %3$ss', 12, 'Hot', 'Pocket').should.eql '12.0 Hot Pockets'
    printf('%1$*.f', '42', 3).should.eql ' 42'
    error = false
    try
      printf '%2$*s', 'Hot Pocket'
    catch e
      e.message.should.eql "got 1 printf arguments, insufficient for '%2$*s'"
      error = true
    error.should.be.true
    # %(map)s format
    printf('%(foo', {}).should.eql '%(foo'
    printf('%(temperature)s %(crevace)s',
      temperature: 'Hot'
      crevace: 'Pocket'
    ).should.eql 'Hot Pocket'
    printf('%(quantity).1f %(temperature)s %(crevace)ss',
      quantity: 12
      temperature: 'Hot'
      crevace: 'Pocket'
    ).should.eql '12.0 Hot Pockets'
    error = false
    try
      printf '%(foo)s', 42
    catch e
      e.message.should.eql 'format requires a mapping'
      error = true
    error.should.be.true
    error = false
    try
      printf '%(foo)s %(bar)s', 'foo', 42
    catch e
      e.message.should.eql 'format requires a mapping'
      error = true
    error.should.be.true
    error = false
    try
      printf '%(foo)*s',
        foo: 'Hot Pocket'
    catch e
      e.message.should.eql '* width not supported in mapped formats'
      error = true
    error.should.be.true

  it 'Positionals', ->
    printf('%*s', 'foo', 4).should.eql ' foo'
    printf('%*.*f', 3.14159265, 10, 2).should.eql '      3.14'
    printf('%0*.*f', 3.14159265, 10, 2).should.eql '0000003.14'
    printf('%-*.*f', 3.14159265, 10, 2).should.eql '3.14      '
    error = false
    try
      printf '%*s', 'foo', 'bar'
    catch e
      e.message.should.eql 'the argument for * width at position 2 is not a number in %*s'
      error = true
    error.should.be.true
    error = false
    try
      printf '%10.*f', 'foo', 42
    catch e
      e.message.should.eql "format argument 'foo' not a float; parseFloat returned NaN"
      error = true
    error.should.be.true

  it 'vs. Formatter', ->
    i = 0
    while i < 1000
      printf '%d %s Pockets', i, 'Hot'
      i++

  it 'Formatter', ->
    str = new printf.Formatter('%d %s Pockets')
    i = 0
    while i < 1000
      str.format i, 'Hot'
      i++

  it 'Miscellaneous', ->
    printf('+%s+', 'hello').should.eql '+hello+'
    printf('+%d+', 10).should.eql '+10+'
    printf('%c', 'a').should.eql 'a'
    printf('%c', 34).should.eql '\"'
    printf('%c', 36).should.eql '$'
    printf('%d', 10).should.eql '10'
    error = false
    try
      printf '%s%s', 42
    catch e
      e.message.should.eql "got 1 printf arguments, insufficient for '%s%s'"
      error = true
    error.should.be.true
    error = false
    try
      printf '%c'
    catch e
      e.message.should.eql "got 0 printf arguments, insufficient for '%c'"
      error = true
    error.should.be.true
    printf('%10', 42).should.eql '%10'

  it 'Escape', ->
    printf('%d %', 10).should.eql '10 %'

  it 'Object inspection', ->
    test =
      foo:
        is:
          bar: true
          baz: false
        isnot:
          array: [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
        maybe: undefined
    printf('%.0O', test).replace(/\s+/g, ' ').should.eql '{ foo: [Object] }'
    printf('%#O', test.foo.isnot.array).should.eql '[ 1, 2, 3, 4, 5, 6, 7, 8, 9, 0 ]'
    # Object inspect serialize object in different order when showHidden is true
    return if semver.lt process.version, 'v0.9.0'
    printf('%O', test).replace(/\s+/g, ' ').should.eql '{ foo: { is: { bar: true, baz: false }, isnot: { array: [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, [length]: 10 ] }, maybe: undefined } }'
    printf('%.2O', test).replace(/\s+/g, ' ').should.eql '{ foo: { is: { bar: true, baz: false }, isnot: { array: [Object] }, maybe: undefined } }'

