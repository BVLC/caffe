/*
 * file-open-test.js: Tests for File transport "open" event
 *
 * (C) 2014 William Wong
 * MIT LICENSE
 *
 */

!function (assert, fs, os, path, vows, winston) {
    'use strict';

    vows.describe('winston/transports/file').addBatch({
        'An instance of the File Transport': {
            topic: function () {
                var callback = this.callback.bind(this),
                    logPath = path.resolve(__dirname, '../fixtures/logs/file-open-test.log');

                try {
                    fs.unlinkSync(logPath);
                } catch (ex) {
                    if (ex && ex.code !== 'ENOENT') { return callback(ex); }
                }

                var fileTransport = new (winston.transports.File)({
                        filename: logPath
                    }),
                    logger = new (winston.Logger)({
                        transports: [fileTransport]
                    }),
                    timeline = {};

                fileTransport.open(function () {
                    timeline.open = Date.now();

                    setTimeout(function () {
                        logger.info('Hello, World!', function () {
                            timeline.logged = Date.now();
                        });
                    }, 100);

                    setTimeout(function () {
                        callback(null, timeline);
                    }, 1000);
                });
            },
            'should fire "open" event': function (results) {
                assert.isTrue(!!results.open);
            },
            'should fire "logged" event': function (results) {
                assert.isTrue(!!results.logged);
            }
        }
    }).export(module);
}(
    require('assert'),
    require('fs'),
    require('os'),
    require('path'),
    require('vows'),
    require('../../lib/winston')
);