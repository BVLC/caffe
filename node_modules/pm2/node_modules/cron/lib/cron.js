var exports,
		timeUnits = ['second', 'minute', 'hour', 'dayOfMonth', 'month', 'dayOfWeek'],
		spawn = require('child_process').spawn,
		moment = require('moment-timezone');

function CronTime(source, zone) {
	this.source = source;
	this.zone = zone;

	var that = this;
	timeUnits.map(function(timeUnit){
		that[timeUnit] = {};
	});

	if (this.source instanceof Date) {
		this.source = moment(this.source);
		this.realDate = true;
	} else {
		this._parse();
		if (!this._verifyParse())
			throw Error("Could not verify a valid date. Please verify your parameters.");
	}
}

CronTime.constraints = [
		[0, 59],
		[0, 59],
		[0, 23],
		[1, 31],
		[0, 11],
		[0, 6]
	];
	CronTime.monthConstraints = [
		31,
		29, //support leap year...not perfect
		31,
		30,
		31,
		30,
		31,
		31,
		30,
		31,
		30,
		31
	];
	CronTime.parseDefaults = ['0', '*', '*', '*', '*', '*'];
	CronTime.aliases = {
		jan: 0,
		feb: 1,
		mar: 2,
		apr: 3,
		may: 4,
		jun: 5,
		jul: 6,
		aug: 7,
		sep: 8,
		oct: 9,
		nov: 10,
		dec: 11,
		sun: 0,
		mon: 1,
		tue: 2,
		wed: 3,
		thu: 4,
		fri: 5,
		sat: 6
	};


CronTime.prototype = {
	_verifyParse: function() {
		var months = Object.keys(this.month);
		for (var i = 0; i < months.length; i++) {
			var m = months[i];
			var con = CronTime.monthConstraints[parseInt(m, 10)];
			var ok = false;
			var dsom = Object.keys(this.dayOfMonth);
			for (var j = 0; j < dsom.length; j++) {
				var dom = dsom[j];
				if (dom <= con)
					ok = true;
			}

			if (!ok) {
				console.error("Month '" + m + "' is limited to '" + con + "' days.");
				return false;
			}
		}

		return true;
	},

	/**
	 * calculates the next send time
	 */
	sendAt: function() {
		var date = this.realDate ? this.source : moment();
		// Set the timezone if given (http://momentjs.com/timezone/docs/#/using-timezones/parsing-in-zone/)
		if (this.zone)
			date = date.tz(this.zone);

		if (this.realDate)
			return date;

		//add 1 second so next time isn't now (can cause timeout to be 0 or negative number)
		var now = new Date();
		var targetSecond = date.seconds();
		var diff = Math.abs(targetSecond - now.getSeconds())
		// there was a problem when `date` is 1424777177999 and `now` is 1424777178000
		// 1 ms diff but this is another second...
		if ( diff == 0 || (diff == 1 && now.getMilliseconds() <= date.milliseconds() ) ) {
			//console.log('add 1 second?');
			date = date.add(1, 's');
		}

		date = this._getNextDateFrom(date);

		return date;
	},

	/**
	 * Get the number of milliseconds in the future at which to fire our callbacks.
	 */
	getTimeout: function() {
		return Math.max(-1, this.sendAt() - moment());
	},

	/**
	 * writes out a cron string
	 */
	toString: function() {
		return this.toJSON().join(' ');
	},

	/**
	 * Json representation of the parsed cron syntax.
	 */
	toJSON: function() {
		var self = this;
		return timeUnits.map(function(timeName){
			return self._wcOrAll(timeName);
		});
	},

	/**
	 * get next date that matches parsed cron time
	 */
	_getNextDateFrom: function(start) {
		var date = moment(start);
		if (date.toString() == 'Invalid date') {
			console.log("ERROR: You specified an invalid date.");
			return date;
		}
		if (this.realDate && start < new Date())
			console.log("WARNING: Date in past. Will never be fired.");
		if (this.realDate) return date;

		//sanity check
		while (1) {
			var diff = date - start,
				origDate = new Date(date);

			if (!(date.month() in this.month)) {
				date.add(1, 'M');
				date.date(1);
				date.hours(0);
				date.minutes(0);
				date.seconds(0);
				continue;
			}

			if (!(date.date() in this.dayOfMonth)) {
				date.add(1, 'd');
				date.hours(0);
				date.minutes(0);
				date.seconds(0);
				continue;
			}

			if (!(date.day() in this.dayOfWeek)) {
				date.add(1, 'd');
				date.hours(0);
				date.minutes(0);
				date.seconds(0);
				if (date <= origDate) {
					date = this._findDST(origDate);
				}
				continue;
			}

			if (!(date.hours() in this.hour)) {
				origDate = moment(date);
				date.hours(date.hours() == 23 && diff > 86400000 ? 0 : date.hours() + 1);
				date.minutes(0);
				date.seconds(0);
				if (date <= origDate) {
					date = this._findDST(origDate);
				}
				continue;
			}

			if (!(date.minutes() in this.minute)) {
				origDate = moment(date);
				date.minutes(date.minutes() == 59 && diff > 60 * 60 * 1000 ? 0 : date.minutes() + 1);
				date.seconds(0);
				if (date <= origDate) {
					date = this._findDST(origDate);
				}
				continue;
			}

			if (!(date.seconds() in this.second)) {
				origDate = moment(date);
				date.seconds(date.seconds() == 59 && diff > 60 * 1000 ? 0 : date.seconds() + 1);
				if (date <= origDate) {
					date = this._findDST(origDate);
				}
				continue;
			}

			break;
		}

		return date;
	},

	/**
	 * get next date that is a valid DST date
	 */
	_findDST: function(date) {
		var newDate = moment(date),
		addSeconds = 1;
		while (newDate <= date)
			newDate.add(1, 's');

		return newDate;
	},

	/**
	 * wildcard, or all params in array (for to string)
	 */
	_wcOrAll: function(type) {
		if (this._hasAll(type)) return '*';

		var all = [];
		for (var time in this[type]) {
			all.push(time);
		}

		return all.join(',');
	},

	_hasAll: function(type) {
		var constrain = CronTime.constraints[timeUnits.indexOf(type)];

		for (var i = constrain[0], n = constrain[1]; i < n; i++) {
			if (!(i in this[type])) return false;
		}

		return true;
	},


	_parse: function() {
		var aliases = CronTime.aliases,
		source = this.source.replace(/[a-z]{1,3}/ig, function(alias) {
			alias = alias.toLowerCase();

			if (alias in aliases) {
				return aliases[alias];
			}

			throw new Error('Unknown alias: ' + alias);
		}),
		split = source.replace(/^\s\s*|\s\s*$/g, '').split(/\s+/),
		cur, i = 0,
		len = timeUnits.length;

		for (; i < timeUnits.length; i++) {
			// If the split source string doesn't contain all digits,
			// assume defaults for first n missing digits.
			// This adds support for 5-digit standard cron syntax
			cur = split[i - (len - split.length)] || CronTime.parseDefaults[i];
			this._parseField(cur, timeUnits[i], CronTime.constraints[i]);
		}
	},

	_parseField: function(field, type, constraints) {
		 var rangePattern = /^(\d+)(?:-(\d+))?(?:\/(\d+))?$/g,
		 typeObj = this[type],
		 diff, pointer,
		 low = constraints[0],
		 high = constraints[1];

		 // * is a shortcut to [lower-upper] range
		 field = field.replace(/\*/g, low + '-' + high);

		 //commas separate information, so split based on those
		 var allRanges = field.split(',');

		 for (var i = 0; i < allRanges.length; i++) {
			 if (allRanges[i].match(rangePattern)) {
				 allRanges[i].replace(rangePattern, function($0, lower, upper, step) {
					 step = parseInt(step) || 1;
					 // Positive integer higher than constraints[0]
					 lower = Math.min(Math.max(low, ~~Math.abs(lower)), high);

					 // Positive integer lower than constraints[1]
					 upper = upper ? Math.min(high, ~~Math.abs(upper)) : lower;

					 // Count from the lower barrier to the upper
					 pointer = lower;

					 do {
						 typeObj[pointer] = true
					 pointer += step;
					 } while (pointer <= upper);
				 });
			 } else {
				 throw new Error('Field (' + field + ') cannot be parsed');
			 }
		 }
	 }
};

function command2function(cmd) {
	switch (typeof cmd) {
		case 'string':
			var args = cmd.split(' ');
			var command = args.shift();

			cmd = spawn.bind(undefined, command, args);
			break;
		case 'object':
			var command = cmd && cmd.command;
			if (command) {
				var args = cmd.args;
				var options = cmd.options;

				cmd = spawn.bind(undefined, command, args, options);
			}
			break;
	}

	return cmd
}

function CronJob(cronTime, onTick, onComplete, startNow, timeZone, context) {
	if (typeof cronTime != "string" && arguments.length == 1) {
		//crontime is an object...
		onTick = cronTime.onTick;
		onComplete = cronTime.onComplete;
		context = cronTime.context;
		startNow = cronTime.start || cronTime.startNow;
		timeZone = cronTime.timeZone;
		cronTime = cronTime.cronTime;
	}

	this.context = (context || this);
	this._callbacks = [];
	this.onComplete = command2function(onComplete);
	this.cronTime = new CronTime(cronTime, timeZone);

	addCallback.call(this, command2function(onTick));

	if (startNow) start.call(this);

	return this;
}

var addCallback = function(callback) {
	if (typeof callback == 'function') this._callbacks.push(callback);
}
CronJob.prototype.addCallback = addCallback;

CronJob.prototype.setTime = function(time) {
	if (!(time instanceof CronTime)) throw '\'time\' must be an instance of CronTime.';
	this.stop();
	this.cronTime = time;
}

CronJob.prototype.nextDate = function() {
	return this.cronTime.sendAt();
}

var start = function() {
	if (this.running) return;

	var MAXDELAY = 2147483647; // The maximum number of milliseconds setTimeout will wait.
	var self = this;
	var timeout = this.cronTime.getTimeout();
	var remaining = 0;

	if (this.cronTime.realDate) this.runOnce = true;

	// The callback wrapper checks if it needs to sleep another period or not
	// and does the real callback logic when it's time.

	function callbackWrapper() {

		// If there is sleep time remaining, calculate how long and go to sleep
		// again. This processing might make us miss the deadline by a few ms
		// times the number of sleep sessions. Given a MAXDELAY of almost a
		// month, this should be no issue.

		if (remaining) {
			if (remaining > MAXDELAY) {
				remaining -= MAXDELAY;
				timeout = MAXDELAY;
			} else {
				timeout = remaining;
				remaining = 0;
			}

			self._timeout = setTimeout(callbackWrapper, timeout);
		} else {

			// We have arrived at the correct point in time.

			self.running = false;

			//start before calling back so the callbacks have the ability to stop the cron job
			if (!(self.runOnce)) self.start();

			for (var i = (self._callbacks.length - 1); i >= 0; i--)
				self._callbacks[i].call(self.context, self.onComplete);
		}
	}

	if (timeout >= 0) {
		this.running = true;

		// Don't try to sleep more than MAXDELAY ms at a time.

		if (timeout > MAXDELAY) {
			remaining = timeout - MAXDELAY;
			timeout = MAXDELAY;
		}

		this._timeout = setTimeout(callbackWrapper, timeout);
	} else {
		this.stop();
	}
}

CronJob.prototype.start = start;

/**
 * Stop the cronjob.
 */
CronJob.prototype.stop = function() {
	if (this._timeout)
		clearTimeout(this._timeout);
	this.running = false;
	if (typeof this.onComplete == 'function') this.onComplete();
}

if (exports) {
	exports.job = function(cronTime, onTick, onComplete) {
		return new CronJob(cronTime, onTick, onComplete);
	}

	exports.time = function(cronTime, timeZone) {
		return new CronTime(cronTime, timeZone);
	}

	exports.sendAt = function(cronTime) {
		return exports.time(cronTime).sendAt();
	}

	exports.timeout = function(cronTime) {
		return exports.time(cronTime).getTimeout();
	}

	exports.CronJob = CronJob;
	exports.CronTime = CronTime;
}
