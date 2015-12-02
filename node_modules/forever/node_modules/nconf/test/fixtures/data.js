/*
 * data.js: Simple data fixture for configuration test.
 *
 * (C) 2011, Nodejitsu Inc.
 *
 */
 
exports.data = {
  isNull: null,
  literal: 'bazz', 
  arr: ['one', 2, true, { value: 'foo' }],
  obj: {
    host: 'localhost',
    port: 5984,
    array: ['one', 2, true, { foo: 'bar' }],
    auth: {
      username: 'admin',
      password: 'password'
    }
  }
};

exports.merge = {
  prop1: 1,
  prop2: [1, 2, 3],
  prop3: {
    foo: 'bar',
    bar: 'foo'
  }
};