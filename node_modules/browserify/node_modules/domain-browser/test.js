// Import
var events = require('events')
var expect = require('chai').expect
var joe = require('joe')
var domain = require('./')

// =====================================
// Tests

joe.describe('domain-browser', function(describe,it){
	it('should work on throws', function(done){
		var d = domain.create()
		d.on('error', function(err){
			expect(err && err.message).to.eql('a thrown error')
			done()
		})
		d.run(function(){
			throw new Error('a thrown error')
		})
	})

	it('should be able to add emitters', function(done){
		var d = domain.create()
		var emitter = new events.EventEmitter()

		d.add(emitter)
		d.on('error', function (err) {
			expect(err && err.message).to.eql('an emitted error')
			done()
		})

		emitter.emit('error', new Error('an emitted error'))
	})

	it('should be able to remove emitters', function (done){
		var emitter = new events.EventEmitter()
		var d = domain.create()

		d.add(emitter)
		var domainGotError = false
		d.on('error', function (err) {
			domainGotError = true
		})

		emitter.on('error', function (err) {
			expect(err && err.message).to.eql('This error should not go to the domain')

			// Make sure nothing race condition-y is happening
			setTimeout(function () {
				expect(domainGotError).to.eql(false)
				done()
			}, 0)
		})

		d.remove(emitter)
		emitter.emit('error', new Error('This error should not go to the domain'))
	})

	it('bind should work', function(done){
		var d = domain.create()
		d.on('error', function(err){
			expect(err && err.message).to.eql('a thrown error')
			done()
		})
		d.bind(function(err, a, b){
			expect(err && err.message).to.equal('a passed error')
			expect(a).to.equal(2)
			expect(b).to.equal(3)
			throw new Error('a thrown error')
		})(new Error('a passed error'), 2, 3)
	})

	it('intercept should work', function(done){
		var d = domain.create()
		var count = 0
		d.on('error', function(err){
			if ( count === 0 ) {
				expect(err && err.message).to.eql('a thrown error')
			} else if ( count === 1 ) {
				expect(err && err.message).to.eql('a passed error')
				done()
			}
			count++
		})

		d.intercept(function(a, b){
			expect(a).to.equal(2)
			expect(b).to.equal(3)
			throw new Error('a thrown error')
		})(null, 2, 3)

		d.intercept(function(a, b){
			throw new Error('should never reach here')
		})(new Error('a passed error'), 2, 3)
	})

})