/**
 Licensed to the Apache Software Foundation (ASF) under one
 or more contributor license agreements.  See the NOTICE file
 distributed with this work for additional information
 regarding copyright ownership.  The ASF licenses this file
 to you under the Apache License, Version 2.0 (the
 "License"); you may not use this file except in compliance
 with the License.  You may obtain a copy of the License at

 http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing,
 software distributed under the License is distributed on an
 "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 KIND, either express or implied.  See the License for the
 specific language governing permissions and limitations
 under the License.
 */

var chalk   = require('chalk'),
    express = require('express'),
    Q       = require('q');

/**
 * @desc Launches a server with the specified options and optional custom handlers.
 * @param {{root: ?string, port: ?number, noLogOutput: ?bool, noServerInfo: ?bool, router: ?express.Router, events: EventEmitter}} opts
 * @returns {*|promise}
 */
module.exports = function (opts) {
    var deferred = Q.defer();

    opts = opts || {};
    var port = opts.port || 8000;

    var log = module.exports.log = function (msg) {
        if (!opts.noLogOutput) {
            if (opts.events) {
                opts.events.emit('log', msg);
            } else {
                console.log(msg);
            }
        }
    };

    var app = this.app;
    var server = require('http').Server(app);
    this.server = server;

    if (opts.router) {
        app.use(opts.router);
    }

    if (opts.root) {
        this.root = opts.root;
        app.use(express.static(opts.root));
    }

    var that = this;
    server.listen(port).on('listening', function () {
        that.port = port;
        if (!opts.noServerInfo) {
            log('Static file server running on: ' + chalk.green('http://localhost:' + port) + ' (CTRL + C to shut down)');
        }
        deferred.resolve();
    }).on('error', function (e) {
        if (e && e.toString().indexOf('EADDRINUSE') !== -1) {
            port++;
            server.listen(port);
        } else {
            deferred.reject(e);
        }
    });

    return deferred.promise;
};
