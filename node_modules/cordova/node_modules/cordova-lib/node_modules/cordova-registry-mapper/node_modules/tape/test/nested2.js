var test = require('../');

test(function(t) {
    var i = 0
    t.test('setup', function(t) {
        process.nextTick(function() {
            t.equal(i, 0, 'called once')
            i++
            t.end()
        })
    })


    t.test('teardown', function(t) {
        t.end()
    })

    t.end()
})
