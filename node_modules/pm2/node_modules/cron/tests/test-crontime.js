var testCase = require('nodeunit').testCase,
    cron = require('../lib/cron');

module.exports = testCase({
        'test stars (* * * * * *)': function(assert) {
            assert.expect(1);
            assert.doesNotThrow(function() {
                new cron.CronTime('* * * * * *');
            });
            assert.done();
        },
        'test digit (0 * * * * *)': function(assert) {
            assert.expect(1);
            assert.doesNotThrow(function() {
                new cron.CronTime('0 * * * * *');
            });
            assert.done();
        },
        'test multi digits (08 * * * * *)': function(assert) {
            assert.expect(1);
            assert.doesNotThrow(function() {
                new cron.CronTime('08 * * * * *');
            });
            assert.done();
        },
        'test all digits (08 8 8 8 8 5)': function(assert) {
            assert.expect(1);
            assert.doesNotThrow(function() {
                new cron.CronTime('08 * * * * *');
            });
            assert.done();
        },
        'test too many digits (08 8 8 8 8 5)': function(assert) {
            assert.expect(1);
            assert.doesNotThrow(function() {
                new cron.CronTime('08 * * * * *');
            });
            assert.done();
        },
        'test no second digit doesnt throw, i.e. standard cron format (* * * * *)': function(assert) {
            assert.expect(1);
            assert.doesNotThrow(function() {
                new cron.CronTime('* * * * *');
            });
            assert.done();
        },
        'test no second digit defaults to 0, i.e. standard cron format (8 8 8 8 5)': function(assert) {
            assert.expect(6);
            var now = Date.now();
            var standard = new cron.CronTime('8 8 8 8 5');
            var extended = new cron.CronTime('0 8 8 8 8 5');

						assert.deepEqual(standard.dayOfWeek, extended.dayOfWeek);
						assert.deepEqual(standard.month, extended.month);
						assert.deepEqual(standard.dayOfMonth, extended.dayOfMonth);
						assert.deepEqual(standard.hour, extended.hour);
						assert.deepEqual(standard.minute, extended.minute);
						assert.deepEqual(standard.second, extended.second);

            assert.done();
        },
        'test hyphen (0-10 * * * * *)': function(assert) {
            assert.expect(1);
            assert.doesNotThrow(function() {
                new cron.CronTime('0-10 * * * * *');
            });
            assert.done();
        },
        'test multi hyphens (0-10 0-10 * * * *)': function(assert) {
            assert.expect(1);
            assert.doesNotThrow(function() {
                new cron.CronTime('0-10 0-10 * * * *');
            });
            assert.done();
        },
        'test all hyphens (0-10 0-10 0-10 0-10 0-10 0-1)': function(assert) {
            assert.expect(1);
            assert.doesNotThrow(function() {
                new cron.CronTime('0-10 0-10 0-10 0-10 0-10 0-1');
            });
            assert.done();
        },
        'test comma (0,10 * * * * *)': function(assert) {
            assert.expect(1);
            assert.doesNotThrow(function() {
                new cron.CronTime('0,10 * * * * *');
            });
            assert.done();
        },
        'test multi commas (0,10 0,10 * * * *)': function(assert) {
            assert.expect(1);
            assert.doesNotThrow(function() {
                new cron.CronTime('0,10 0,10 * * * *');
            });
            assert.done();
        },
        'test all commas (0,10 0,10 0,10 0,10 0,10 0,1)': function(assert) {
            assert.expect(1);
            assert.doesNotThrow(function() {
                new cron.CronTime('0,10 0,10 0,10 0,10 0,10 0,1');
            });
            assert.done();
        },
        'test alias (* * * * jan *)': function(assert) {
            assert.expect(1);
            assert.doesNotThrow(function() {
                new cron.CronTime('* * * * jan *');
            });
            assert.done();
        },
        'test multi aliases (* * * * jan,feb *)': function(assert) {
            assert.expect(1);
            assert.doesNotThrow(function() {
                new cron.CronTime('* * * * jan,feb *');
            });
            assert.done();
        },
        'test all aliases (* * * * jan,feb mon,tue)': function(assert) {
            assert.expect(1);
            assert.doesNotThrow(function() {
                new cron.CronTime('* * * * jan,feb mon,tue');
            });
            assert.done();
        },
        'test unknown alias (* * * * jar *)': function(assert) {
            assert.expect(1);
            assert.throws(function() {
                new cron.CronTime('* * * * jar *');
            });
            assert.done();
        },
        'test unknown alias - short (* * * * j *)': function(assert) {
            assert.expect(1);
            assert.throws(function() {
                new cron.CronTime('* * * * j *');
            });
            assert.done();
        },
        'test Date': function(assert) {
          assert.expect(1);
          var d = new Date();
          var ct = new cron.CronTime(d);
          assert.ok(ct.source.isSame(d.getTime()));
          assert.done();
        },
        'test day roll-over': function(assert) {
          var numHours = 24;
          assert.expect(numHours * 2);
          var ct = new cron.CronTime('0 0 17 * * *');
          
          for (var hr = 0; hr < numHours; hr++) {
            var start = new Date(2012, 3, 16, hr, 30, 30);
            var next = ct._getNextDateFrom(start);
            assert.ok(next - start < 24*60*60*1000);
            assert.ok(next > start);
          }
          assert.done();
        },
        'test illegal repetition syntax': function(assert){
          assert.throws(function(){
            new cron.CronTime('* * /4 * * *');
          });
          assert.done();
				},
				'test next date': function(assert) {
          assert.expect(2);
          var ct = new cron.CronTime('0 0 */4 * * *');

					var nextDate = new Date();
					nextDate.setHours(23);
					var nextdt = ct._getNextDateFrom(nextDate);
					assert.ok(nextdt > nextDate);
					assert.ok(nextdt.hours() % 4 === 0);
					assert.done();
				},
				'test next date from invalid date': function(assert) {
					assert.expect(1);
					var ct = new cron.CronTime('0 0 * * * *');
					var nextDate = new Date('My invalid date string');
					var nextdt = ct._getNextDateFrom(nextDate);
					assert.equal(nextdt.toString(), 'Invalid date');
					assert.done();
				},
				'test next real date': function(assert) {
          assert.expect(2);
          var ct = new cron.CronTime(new Date());

					var nextDate = new Date();
					nextDate.setMonth(nextDate.getMonth()+1);
					assert.ok(nextDate > ct.source);
					var nextdt = ct._getNextDateFrom(nextDate);
					assert.ok(nextdt.isSame(nextDate));
					assert.done();
				},
				'test < constraints day of month': function(assert) {
					assert.expect(5);

					var ltm = [1, 3, 5, 8, 10];
					for (var i = 0; i < ltm.length; i++) {
						(function(m) {
							assert.throws(function() {
								var ct = new cron.CronTime('0 0 0 33 ' + m + ' *');
							});
						})(ltm[i]);
					}

					assert.done();
				},
				'test next month selection': function(assert) {
					assert.expect(1);
					var date = new Date();
					var dom = date.getDate() + 1;
					var ct = new cron.CronTime('0 0 0 ' + dom + ' * *');

					var saDate = ct.sendAt();

					if (dom < date.getDate())
						date.setMonth(date.getMonth()+1);

					assert.equal(date.getMonth(), saDate.month());

					assert.done()
				}
});
