/*
 * put-object-rows.js: Example usage for `cliff.putObjectRows`.
 *
 * (C) 2010, Nodejitsu Inc.
 *
 */
 
var cliff = require('../lib/cliff');

var objs = [], obj = {
  name: "bazz",
  address: "1234 Nowhere Dr.",
};

for (var i = 0; i < 10; i++) {
  objs.push({
    name: obj.name,
    address: obj.address,
    id: Math.random().toString()
  });
}

cliff.putObjectRows('data', objs, ['id', 'name', 'address']);