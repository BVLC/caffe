'use strict';

// Load modules

const Sntp = require('../lib');


// Declare internals

const internals = {};


// Request offset once

Sntp.offset((err, offset1) => {

    console.log(offset1);                    // New (served fresh)

    // Request offset again

    Sntp.offset((err, offset2) => {

        console.log(offset2);                // Identical (served from cache)
    });
});
