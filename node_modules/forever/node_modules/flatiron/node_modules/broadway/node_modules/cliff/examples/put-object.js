/*
 * put-object.js: Example usage for `cliff.putObject`.
 *
 * (C) 2010, Nodejitsu Inc.
 *
 */
 
var cliff = require('../lib/cliff');

cliff.putObject({
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
});