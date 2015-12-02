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

var cordova_util = require('./util'),
    crypto       = require('crypto'),
    path         = require('path'),
    shell        = require('shelljs'),
    url          = require('url'),
    platforms    = require('../platforms/platforms'),
    ConfigParser = require('cordova-common').ConfigParser,
    HooksRunner  = require('../hooks/HooksRunner'),
    Q            = require('q'),
    fs           = require('fs'),
    events       = require('cordova-common').events,
    serve        = require('cordova-serve');

var projectRoot;
var installedPlatforms;

function handleRoot(request, response, next) {
    if (url.parse(request.url).pathname !== '/') {
        response.sendStatus(404);
        return;
    }

    response.writeHead(200, {'Content-Type': 'text/html'});
    var config = new ConfigParser(cordova_util.projectConfig(projectRoot));
    response.write('<html><head><title>' + config.name() + '</title></head><body>');
    response.write('<table border cellspacing=0><thead><caption><h3>Package Metadata</h3></caption></thead><tbody>');
    ['name', 'packageName', 'version'].forEach(function (c) {
        response.write('<tr><th>' + c + '</th><td>' + config[c]() + '</td></tr>');
    });
    response.write('</tbody></table>');
    response.write('<h3>Platforms</h3><ul>');
    Object.keys(platforms).forEach(function (platform) {
        if (installedPlatforms.indexOf(platform) >= 0) {
            response.write('<li><a href="' + platform + '/www/">' + platform + '</a></li>\n');
        } else {
            response.write('<li><em>' + platform + '</em></li>\n');
        }
    });
    response.write('</ul>');
    response.write('<h3>Plugins</h3><ul>');
    var pluginPath = path.join(projectRoot, 'plugins');
    var plugins = cordova_util.findPlugins(pluginPath);
    Object.keys(plugins).forEach(function (plugin) {
        response.write('<li>' + plugins[plugin] + '</li>\n');
    });
    response.write('</ul>');
    response.write('</body></html>');
    response.end();
}

function getPlatformHandler(platform, wwwDir, configXml) {
    return function (request, response, next) {
        switch (url.parse(request.url).pathname) {
            case '/' + platform + '/config.xml':
                response.sendFile(configXml);
                break;

            case '/' + platform + '/project.json':
                response.send({
                    'configPath': '/' + platform + '/config.xml',
                    'wwwPath': '/' + platform + '/www',
                    'wwwFileList': shell.find(wwwDir)
                        .filter(function (a) { return !fs.statSync(a).isDirectory() && !/(^\.)|(\/\.)/.test(a); })
                        .map(function (a) { return {'path': a.slice(wwwDir.length), 'etag': '' + calculateMd5(a)}; })
                });
                break;

            default:
                next();
        }
    };
}

function calculateMd5(fileName) {
    var md5sum,
        BUF_LENGTH = 64*1024,
        buf = new Buffer(BUF_LENGTH),
        bytesRead = BUF_LENGTH,
        pos = 0,
        fdr = fs.openSync(fileName, 'r');

    try {
        md5sum = crypto.createHash('md5');
        while (bytesRead === BUF_LENGTH) {
            bytesRead = fs.readSync(fdr, buf, 0, BUF_LENGTH, pos);
            pos += bytesRead;
            md5sum.update(buf.slice(0, bytesRead));
        }
    } finally {
        fs.closeSync(fdr);
    }
    return md5sum.digest('hex');
}

module.exports = function server(port) {
    var d = Q.defer();
    projectRoot = cordova_util.cdProjectRoot();
    port = +port || 8000;

    var hooksRunner = new HooksRunner(projectRoot);
    hooksRunner.fire('before_serve').then(function () {
        // Run a prepare first!
        return require('./cordova').raw.prepare([]);
    }).then(function () {
        var server = serve();

        installedPlatforms = cordova_util.listPlatforms(projectRoot);
        installedPlatforms.forEach(function (platform) {
            var locations = platforms.getPlatformApi(platform).getPlatformInfo().locations;
            server.app.use('/' + platform + '/www', serve.static(locations.www));
            server.app.get('/' + platform + '/*', getPlatformHandler(platform, locations.www, locations.configXml));
        });
        server.app.get('*', handleRoot);

        server.launchServer({port: port, events: events});
        hooksRunner.fire('after_serve').then(function () {
            d.resolve(server.server);
        });
    });
    return d.promise;
};
