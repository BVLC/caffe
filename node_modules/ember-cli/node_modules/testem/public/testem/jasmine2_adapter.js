/*

 jasmine2_adapter.js
 ==================

 Testem's adapter for Jasmine. It works by adding a custom reporter.

 */

function jasmine2Adapter() {

	var results = {
		failed: 0,
		passed: 0,
		total: 0,
		pending: 0,
		tests: []
	};

	function Jasmine2AdapterReporter() {

		this.jasmineStarted = function() {
			emit('tests-start')
		};

		this.specDone = function(spec) {

			var test = {
				passed: 0,
				failed: 0,
				total: 0,
				pending: 0,
				id: spec.id + 1,
				name: spec.fullName,
				items: []
			};

			var i, l, failedExpectations, item;

			if(spec.status === 'passed') {
				test.passed++;
				test.total++;
				results.passed++;
			} else if(spec.status === 'pending') {
				test.pending++;
				test.total++;
				results.pending++;
			} else {
				failedExpectations = spec.failedExpectations;
				for(i = 0, l = failedExpectations.length; i < l; i++) {
					item = failedExpectations[i];
					test.items.push({
						passed: item.passed,
						message: item.message,
						stack: item.stack || undefined
					});
				}
				test.failed++;
				results.failed++;
				test.total++;
			}

			results.total++;

			emit('test-result', test)
		};

		this.jasmineDone = function() {
			emit('all-test-results', results)
		};

	}

	jasmine.getEnv().addReporter(new Jasmine2AdapterReporter);
}
