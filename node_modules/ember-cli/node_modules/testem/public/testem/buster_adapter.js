/*

buster_adapter.js
=================

Testem's adapter for Buster.js. It works by attaching event listeners to the test runner.


*/

function busterAdapter(socket){

    var id = 1
    var started = false

    var results = {
        failed: 0
        , passed: 0
        , total: 0
        , pending: 0
        , tests: []
    }

    var runner = buster.testRunner
    var currContext = null

    runner.on('context:start', function(context){
        if (!started) emit('tests-start')
        currContext = context
    })
    runner.on('context:end', function(){
        currContext = null
    })

    function onSuccess(test){
        var test = {
            passed: 1
            , failed: 0
            , total: 1
            , pending: 0
            , id: id++
            , name: currContext ? (currContext.name + ' ' + test.name) : test.name
        }
        emit('test-result', test)
        results.passed++
        results.total++
    }    

    function onFailure(test){
        var test = {
            passed: 0
            , failed: 1
            , total: 1
            , pending: 0
            , id: id++
            , name: currContext ? (currContext.name + ' ' + test.name) : test.name
            , items: [{    
                passed: false,
                message: test.error.message,
                stack: test.error.stack ? test.error.stack : undefined
            }]
        }
        emit('test-result', test)
        results.failed++
        results.total++
    }

    function onDeferred(test){
        var test = {
            passed: 0
            , failed: 0
            , total: 1
            , pending: 1
            , id: id++
            , name: currContext ? (currContext.name + ' ' + test.name) : test.name
        }
        emit('test-result', test)
        results.total++
    }

    runner.on('test:success', onSuccess)
    runner.on('test:failure', onFailure)
    runner.on('test:error', onFailure)
    runner.on('test:timeout', onFailure)
    runner.on('test:deferred', onDeferred)

    runner.on('suite:end', function(){
        emit('all-test-results', results)
    })

}
