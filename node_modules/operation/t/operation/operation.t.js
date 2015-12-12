require('proof')(5, prove)

function prove (assert) {
    var Operation = require('../..'), operation
    operation = new Operation(function f (one) {
        assert(one, 1, 'as function')
    })
    operation.apply([ 1 ])

    var object = {
        direct: function (one) {
            assert(this === object, 'direct object method this')
            assert(one, 1, 'direct object method parameters')
        },
        named: function (one) {
            assert(this === object, 'named object method this')
            assert(one, 1, 'named object method parameters')
        }
    }
    operation = new Operation({ object: object, method: 'named' })
    operation.apply([ 1 ])
    operation = new Operation({ object: object, method: object.direct })
    operation.apply([ 1 ])
}
