'use strict';

// Load modules

const Sntp = require('../lib');


// Declare internals

const internals = {};


// All options are optional

const options = {
    host: 'nist1-sj.ustiming.org',  // Defaults to pool.ntp.org
    port: 123,                      // Defaults to 123 (NTP)
    resolveReference: true,         // Default to false (not resolving)
    timeout: 1000                   // Defaults to zero (no timeout)
};

// Request server time

Sntp.time(options, (err, time) => {

    if (err) {
        console.log('Failed: ' + err.message);
        process.exit(1);
    }

    console.log(time);
    console.log('Local clock is off by: ' + time.t + ' milliseconds');
    process.exit(0);
});
