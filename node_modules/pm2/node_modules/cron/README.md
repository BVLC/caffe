node-cron
=========

[![Build Status](https://secure.travis-ci.org/ncb000gt/node-cron.png)](http://travis-ci.org/#!/ncb000gt/node-cron)
[![wercker status](https://app.wercker.com/status/0cadfe5d45ad7bc819efb636026cf230/s "wercker status")](https://app.wercker.com/project/bykey/0cadfe5d45ad7bc819efb636026cf230)

Originally this project was a NodeJS fork of [James Padolsey's][jamespadolsey] [cron.js](http://github.com/padolsey/cron.js).

After [Craig Condon][crcn] made some updates and changes to the code base this has evolved to something that has a bit of both. The cron syntax parsing is mostly James' while using timeout instead of interval is Craig's.

Additionally, this library goes beyond the basic cron syntax and allows you to supply a Date object. This will be used as the trigger for your callback. Cron syntax is still an acceptable CronTime format. Although the Cron patterns suported here extend on the standard Unix format to support seconds digits, leaving it off will default to 0 and match the Unix behavior.

If You Are Submitting Bugs/Issues
=============

Because we can't magically know what you are doing to expose an issue, it is best if you provide a snippet of code. This snippet need not include your secret sauce, but it must replicate the issue you are describing. The issues that get closed without resolution tend to be the ones without code examples. Thanks.


Versions and Backwards compatability breaks:
==========

As goes with semver, breaking backwards compatibility should be explicit in the versioning of your library. As such, we'll upgrade the version of this module in accordance with breaking changes (I'm not always great about doing it this way so if you notice that there are breaking changes that haven't been bumped appropriately please let me know). This table lists out the issues which were the reason for the break in backward compatibility.

<table>
<tr>
<td>Node Cron Ver</td><td>Issue #</td>
</tr>
<tr>
<td>1.0.0</td><td><ul><li><a href="https://github.com/ncb000gt/node-cron/pull/41">GH-41</a></li><li><a href="https://github.com/ncb000gt/node-cron/pull/36">GH-36</a></li></ul></td>
</tr>
</table>


Usage (basic cron usage):
==========

    var CronJob = require('cron').CronJob;
    new CronJob('* * * * * *', function(){
        console.log('You will see this message every second');
    }, null, true, "America/Los_Angeles");


Available Cron patterns:
==========

    Asterisk. E.g. *
    Ranges. E.g. 1-3,5
    Steps. E.g. */2

[Read up on cron patterns here](http://crontab.org).

Cron Ranges
==========

When specifying your cron values you'll need to make sure that your values fall within the ranges. For instance, some cron's use a 0-7 range for the day of week where both 0 and 7 represent Sunday. We do not.

 * Seconds: 0-59
 * Minutes: 0-59
 * Hours: 0-23
 * Day of Month: 1-31
 * Months: 0-11
 * Day of Week: 0-6

Another cron example
==========

    var CronJob = require('cron').CronJob;
    var job = new CronJob('00 30 11 * * 1-5', function(){
        // Runs every weekday (Monday through Friday)
        // at 11:30:00 AM. It does not run on Saturday
        // or Sunday.
      }, function () {
        // This function is executed when the job stops
      },
      true /* Start the job right now */,
      timeZone /* Time zone of this job. */
    );

Another example with Date
==========

    var CronJob = require('cron').CronJob;
    var job = new CronJob(new Date(), function(){
        //runs once at the specified date.
      }, function () {
        // This function is executed when the job stops
      },
      true /* Start the job right now */,
      timeZone /* Time zone of this job. */
    );

For good measure
==========

    var CronJob = require('cron').CronJob;
    var job = new CronJob({
      cronTime: '00 30 11 * * 1-5',
      onTick: function() {
        // Runs every weekday (Monday through Friday)
        // at 11:30:00 AM. It does not run on Saturday
        // or Sunday.
      },
      start: false,
      timeZone: "America/Los_Angeles"
    });
    job.start();


How to check if a cron pattern is valid:
==========

		try {
			new CronJob('invalid cron pattern', function() {
				console.log('this should not be printed');
			})
		} catch(ex) {
			console.log("cron pattern not valid");
		}


Install
==========

    From source: `npm install`
    From npm: `npm install cron`


API
==========

Parameter Based 

`CronJob`

  * `constructor(cronTime, onTick, onComplete, start, timezone, context)` - Of note, the first parameter here can be a JSON object that has the below names and associated types (see examples above).
    * `cronTime` - [REQUIRED] - The time to fire off your job. This can be in the form of cron syntax or a JS [Date](https://developer.mozilla.org/en/JavaScript/Reference/Global_Objects/Date) object.
    * `onTick` - [REQUIRED] - The function to fire at the specified time.
    * `onComplete` - [OPTIONAL] - A function that will fire when the job is complete, when it is stopped.
    * `start` - [OPTIONAL] - Specifies whether to start the job just before exiting the constructor. By default this is set to false. If left at default you will need to call `job.start()` in order to start the job (assuming `job` is the variable you set the cronjob to).
    * `timeZone` - [OPTIONAL] - Specify the timezone for the execution. This will modify the actual time relative to your timezone.
    * `context` - [OPTIONAL] - The context within which to execute the onTick method. This defaults to the cronjob itself allowing you to call `this.stop()`. However, if you change this you'll have access to the functions and values within your context object.
  * `start` - Runs your job.
  * `stop` - Stops your job.

`CronTime`

  * `constructor(time)`
    * `time` - [REQUIRED] - The time to fire off your job. This can be in the form of cron syntax or a JS [Date](https://developer.mozilla.org/en/JavaScript/Reference/Global_Objects/Date) object.

Contributors
===========

* [Romain Beauxis][toots]
* [James Padolsey][jamespadolsey]
* [Craig Condon][crcn]
* [Finn Herpich][errorprone]
* [cliftonc][cliftonc]
* [neyric][neyric]
* [humanchimp][humanchimp]
* [danhbear][danhbear]

License
==========

MIT


[toots]:http://github.com/toots
[jamespadolsey]:http://github.com/padolsey
[crcn]:http://github.com/crcn
[cliftonc]:http://github.com/cliftonc
[neyric]:http://github.com/neyric
[humanchimp]:http://github.com/humanchimp
[errorprone]:http://github.com/ErrorProne
[danhbear]:http://github.com/danhbear
