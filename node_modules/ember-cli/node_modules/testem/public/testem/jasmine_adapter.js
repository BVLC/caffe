/*

jasmine_adapter.js
==================

Testem's adapter for Jasmine. It works by adding a custom reporter.

*/

function jasmineAdapter(socket){

    var results = 
        { failed: 0
        , passed: 0
        , total: 0
        , tests: []
        }

    function JasmineAdapterReporter(){}
    JasmineAdapterReporter.prototype.reportRunnerStarting = function(runner){
        emit('tests-start')
    }
    JasmineAdapterReporter.prototype.reportSpecResults = function(spec){
        if (spec.results().skipped) return
        var test = {
            passed: 0,
            failed: 0,
            total: 0,
            id: spec.id + 1,
            name: spec.getFullName(),
            items: []
        }
        
        var items = spec.results().getItems()
        
        for (var i = 0, len = items.length; i < len; i++){
            var item = items[i]
            if (item.type === 'log') continue
            var passed = item.passed()
            test.total++
            if (passed)
                test.passed++
            else
                test.failed++
            test.items.push({    
                passed: passed,
                message: item.message,
                stack: item.trace.stack ? item.trace.stack : undefined
            })
        }
        
        results.total++
        if (test.failed > 0)
            results.failed++
        else
            results.passed++

        emit('test-result', test)
    }
    JasmineAdapterReporter.prototype.reportRunnerResults = function(runner){
        emit('all-test-results', results)
    }
    jasmine.getEnv().addReporter(new JasmineAdapterReporter)

}