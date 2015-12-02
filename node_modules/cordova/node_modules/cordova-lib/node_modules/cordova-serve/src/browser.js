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

var exec = require('./exec'),
    Q = require('q');

/**
 * Launches the specified browser with the given URL.
 * Based on https://github.com/domenic/opener
 * @param {{target: ?string, url: ?string, dataDir: ?string}} opts - parameters:
 *   target - the target browser - ie, chrome, safari, opera, firefox or chromium
 *   url - the url to open in the browser
 *   dataDir - a data dir to provide to Chrome (can be used to force it to open in a new window)
 * @return {Q} Promise to launch the specified browser
 */
module.exports = function (opts) {
    var target = opts.target || 'chrome';
    var url = opts.url || '';

    return getBrowser(target, opts.dataDir).then(function (browser) {
        var args;

        var urlAdded = false;
        switch (process.platform) {
            case 'darwin':
                args = ['open'];
                if (target == 'chrome') {
                    // Chrome needs to be launched in a new window. Other browsers, particularly, opera does not work with this.        
                    args.push('-n');
                }
                args.push('-a', browser);
                break;
            case 'win32':
                // On Windows, we really want to use the "start" command. But, the rules regarding arguments with spaces, and 
                // escaping them with quotes, can get really arcane. So the easiest way to deal with this is to pass off the 
                // responsibility to "cmd /c", which has that logic built in. 
                // 
                // Furthermore, if "cmd /c" double-quoted the first parameter, then "start" will interpret it as a window title, 
                // so we need to add a dummy empty-string window title: http://stackoverflow.com/a/154090/3191

                if (target === 'edge') {
                    browser += ':' + url;
                    urlAdded = true;
                }

                args = ['cmd /c start ""', browser];
                break;
            case 'linux':
                // if a browser is specified, launch it with the url as argument
                // otherwise, use xdg-open.
                args = [browser];
                break;
        }

        if (!urlAdded) {
            args.push(url);
        }
        var command = args.join(' ');
        return exec(command);
    });
};

function getBrowser(target, dataDir) {
    dataDir = dataDir || 'temp_chrome_user_data_dir_for_cordova';

    var chromeArgs = ' --user-data-dir=/tmp/' + dataDir;
    var browsers = {
        'win32': {
            'ie': 'iexplore',
            'chrome': 'chrome --user-data-dir=%TEMP%\\' + dataDir,
            'safari': 'safari',
            'opera': 'opera',
            'firefox': 'firefox',
            'edge': 'microsoft-edge'
        },
        'darwin': {
            'chrome': '"Google Chrome" --args' + chromeArgs,
            'safari': 'safari',
            'firefox': 'firefox',
            'opera': 'opera'
        },
        'linux' : {
            'chrome': 'google-chrome' + chromeArgs ,
            'chromium': 'chromium-browser' + chromeArgs,
            'firefox': 'firefox',
            'opera': 'opera'
        }
    };
    target = target.toLowerCase();
    if (target in browsers[process.platform]) {
        return Q(browsers[process.platform][target]);
    }
    return Q.reject('Browser target not supported: ' + target);
}
