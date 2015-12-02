var test = require('tape');
var oldToNew = require('../index').oldToNew;
var newToOld = require('../index').newToOld;

test('plugin mappings exist', function(t) {
    t.plan(2);

    t.equal('cordova-plugin-device', oldToNew['org.apache.cordova.device']);

    t.equal('org.apache.cordova.device', newToOld['cordova-plugin-device']);
})
