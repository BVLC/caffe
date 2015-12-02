/*
 * put-object.js: Example usage for `cliff.putObject`.
 *
 * (C) 2010, Nodejitsu Inc.
 *
 */
 
var cliff = require('../lib/cliff');

console.log(cliff.inspect({
  literal: "bazz",
  arr: [
    "one",
    2,
  ],
  obj: {
    host: "localhost",
    port: 5984,
    auth: {
      username: "admin",
      password: "password"
    }
  }
}));
