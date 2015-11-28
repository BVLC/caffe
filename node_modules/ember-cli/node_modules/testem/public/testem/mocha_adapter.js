/*

mocha_adapter.js
================

Testem`s adapter for Mocha. It works by monkey-patching `Runner.prototype.emit`.

*/

function mochaAdapter(socket){
	var results = 
		{ failed: 0
	    , passed: 0
	    , total: 0
	    , pending: 0
	    , tests: []
		}
	var id = 1
	var Runner
	var ended = false
	var waiting = 0
	
	try{
		Runner = mocha.Runner || Mocha.Runner
	}catch(e){
		console.error('Testem: failed to register adapter for mocha.')
	}

	function getFullName(test){
		var name = ''
		while (test){
			name = test.title + ' ' + name
			test = test.parent
		}
		return name.replace(/^ /, '')
	}

	/* Store a reference to the global setTimeout function, in case it's
	 * manipulated by test helpers */
	var _setTimeout = setTimeout

	var oEmit = Runner.prototype.emit
	Runner.prototype.emit = function(evt, test, err){
		if (evt === 'start'){
			emit('tests-start')
		}else if (evt === 'end'){
			if (waiting === 0) {
				emit('all-test-results', results)
			}
			ended = true
		}else if (evt === 'test end'){
			var name = getFullName(test)
			waiting++
			_setTimeout(function(){
				waiting--
				if (test.state === 'passed'){
					testPass(test)
				}else if (test.state === 'failed'){
					testFail(test, err)
				}else if (test.pending){
					testPending(test)
				}
				if (ended && waiting === 0){
					emit('all-test-results', results)
				}
			}, 0)
		}else if(evt === 'fail'){
			test.err = test.err || err
		}

		oEmit.apply(this, arguments)

		function testPass(test){
			var tst = 
				{ passed: 1
				, failed: 0
				, total: 1
				, pending: 0
				, id: id++
				, name: name
				, items: []
				}
			results.passed++
			results.total++
			results.tests.push(tst)
			emit('test-result', tst)
		}

		function makeFailingTest(test, err){
			err = err || test.err
			var items = [
				{ passed: false
				, message: err.message
				, stack: (err && err.stack) ? err.stack : undefined
				}
			]
			var tst = 
				{ passed: 0
				, failed: 1
				, total: 1
				, pending: 0
				, id: id++
				, name: name
				, items: items
				}
			return tst
		}

		function testFail(test, err){
			var tst = makeFailingTest(test, err)
			results.failed++
			results.total++
			results.tests.push(tst)
			emit('test-result', tst)

		}

		function testPending(test){
			var tst =
			{ passed: 0
			, failed: 0
			, total: 1
			, pending: 1
			, id: id++
			, name: name
			, items: []
			}
			results.total++
			results.tests.push(tst)
			emit('test-result', tst)
		}
	}

}

