var CronJob = require('./lib/cron').CronJob;
var sinon = require('sinon');

//var clock = sinon.useFakeTimers(new Date().getTime());

//var previous = 0;
//var counter = 0;
//var cronSyncSchedule = '*/1 * * * * *';

//var scheduleJob = new CronJob(cronSyncSchedule, function() {
	//counter++;
	//checkCounter();
//}, function() {
	//console.log('scheduled job is completed successfully');
//}, true, null);

//function checkCounter() {
	//console.log(counter + ' - ' + previous);
	//if (counter != (previous + 1))
		//console.log('SKIPPED: ' + (counter - 1) + ", COUNTER: " + counter);
	//previous = counter;
//}

//clock.tick(31*24*60*60*1000);

//new CronJob('0 0 0 1 * *',function(){
	//console.log("triggered");
//},null,true);

//clock.tick(24 * 60 * 60 * 1000);


//clock.restore();

//console.log(CronJob.prototype);
//
new CronJob('* * * * * *', function() {
	console.log('tick: ' + Date.now());
}, null, true);
