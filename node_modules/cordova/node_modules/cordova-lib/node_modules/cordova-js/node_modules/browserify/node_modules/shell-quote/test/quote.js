var test = require('tap').test;
var quote = require('../').quote;

test('quote', function (t) {
    t.equal(quote([ 'a', 'b', 'c d' ]), 'a b \'c d\'');
    t.equal(
        quote([ 'a', 'b', "it's a \"neat thing\"" ]),
        'a b "it\'s a \\"neat thing\\""'
    );
    t.equal(
        quote([ '$', '`', '\'' ]),
        '\\$ \\` "\'"'
    );
    t.end();
});
