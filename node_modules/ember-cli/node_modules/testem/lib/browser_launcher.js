/*

browser_launcher.js
===================

This file more or less figures out how to launch any browser on any platform.

*/

var path = require('path')
var rimraf = require('rimraf')
var async = require('async')
var fs = require('fs')
var fileutils = require('./fileutils')
var browserExeExists = fileutils.browserExeExists
var findableByWhich = fileutils.findableByWhich
var findableByWhichOrModule = fileutils.findableByWhichOrModule
var findableByWhereOrModule = fileutils.findableByWhereOrModule
var os = require('os')

// Find the temporary directory for the system
var tempDir = function(){
  return os.tmpdir()
}()

var userHomeDir = process.env.HOME || process.env.USERPROFILE

function setupFirefoxProfile(profileDir, done){
  rimraf(profileDir, function(){
    // using prefs.js to suppress the check default browser popup
    // and the welcome start page
    var prefs = [
      'user_pref("browser.shell.checkDefaultBrowser", false);',
      'user_pref("browser.cache.disk.smart_size.first_run", false);'
    ]
    fs.mkdir(profileDir, function(){
      fs.writeFile(profileDir + '/prefs.js', prefs.join('\n'), function(){
        done()
      })
    })
  })
}

function buildPhantomJsArgs(config) {
  var options = [path.join(path.dirname( __dirname ), '/assets/phantom.js'), this.getUrl()]
  var debug_port = config.get( "phantomjs_debug_port" )
  if (debug_port) {
    options.unshift( "--remote-debugger-autorun=true" )
    options.unshift( "--remote-debugger-port=" + debug_port )
  }
  var phantom_args = config.get("phantomjs_args")
  if (phantom_args) {
    options = phantom_args.concat(options)
  }
  return options
}


// Return the catalogue of the browsers that Testem supports for the platform. Each "browser object"
// will contain these fields:
//
// * `name` - the display name of the browser
// * `exe` - path to the executable to use to launch the browser
// * `setup(app, done)` - any initial setup needed before launching the executable(this is async,
//        the second parameter `done()` must be invoked when done).
// * `supported(cb)` - an async function which tells us whether the browser is supported by the current machine.
function browsersForPlatform(){
  var platform = process.platform

  if (platform === 'win32'){
    return  [
      {
        name: "IE",
        exe: "C:\\Program Files\\Internet Explorer\\iexplore.exe",
        supported: browserExeExists
      },
      {
        name: "Firefox",
        exe: [
          "C:\\Program Files\\Mozilla Firefox\\firefox.exe",
          "C:\\Program Files (x86)\\Mozilla Firefox\\firefox.exe"
        ],
        args: ["-profile", tempDir + "\\testem.firefox"],
        setup: function(config, done){
          setupFirefoxProfile(tempDir + '/testem.firefox', done)
        },
        supported: browserExeExists
      },
      {
        name: "Chrome",
        exe: [
          userHomeDir + "\\Local Settings\\Application Data\\Google\\Chrome\\Application\\chrome.exe",
          userHomeDir + "\\AppData\\Local\\Google\\Chrome\\Application\\chrome.exe",
          "C:\\Program Files\\Google\\Chrome\\Application\\Chrome.exe",
          "C:\\Program Files (x86)\\Google\\Chrome\\Application\\Chrome.exe"
        ],
        args: ["--user-data-dir=" + tempDir + "\\testem.chrome", "--no-default-browser-check", "--no-first-run", "--ignore-certificate-errors", "--test-type"],
        setup: function(config, done){
          rimraf(tempDir + '\\testem.chrome', done)
        },
        supported: browserExeExists
      },
      {
        name: "Safari",
        exe: [
          "C:\\Program Files\\Safari\\safari.exe",
          "C:\\Program Files (x86)\\Safari\\safari.exe"
        ],
        supported: browserExeExists
      },
      {
        name: "Opera",
        exe: [
          "C:\\Program Files\\Opera\\opera.exe",
          "C:\\Program Files (x86)\\Opera\\opera.exe"
        ],
        args: ["--user-data-dir=" + tempDir + "\\testem.opera", "-pd", tempDir + "\\testem.opera"],
        setup: function(config, done){
          rimraf(tempDir + '\\testem.opera', done)
        },
        supported: browserExeExists
      },
      {
        name: 'PhantomJS',
        exe: 'phantomjs',
        useCrossSpawn: true,
        args: buildPhantomJsArgs,
        supported: findableByWhereOrModule
      }
    ]
  }else if (platform === 'darwin'){
    return [
      {
        name: "Chrome",
        exe: [
          process.env.HOME + "/Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome",
          "/Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome"
        ],
        args: ["--user-data-dir=" + tempDir + "/testem.chrome", "--no-default-browser-check", "--no-first-run", "--ignore-certificate-errors", "--test-type"],
        setup: function(config, done){
          rimraf(tempDir + '/testem.chrome', done)
        },
        supported: browserExeExists
      },
      {
        name: "Chrome Canary",
        exe: [
          process.env.HOME + "/Applications/Google\ Chrome\ Canary.app/Contents/MacOS/Google\ Chrome\ Canary",
          "/Applications/Google\ Chrome\ Canary.app/Contents/MacOS/Google\ Chrome\ Canary"
        ],
        args: ["--user-data-dir=" + tempDir + "/testem.chrome-canary", "--no-default-browser-check", "--no-first-run", "--ignore-certificate-errors", "--test-type"],
        setup: function(config, done){
          rimraf(tempDir + '/testem.chrome-canary', done)
        },
        supported: browserExeExists
      },
      {
        name: "Firefox",
        exe: [
          process.env.HOME + "/Applications/Firefox.app/Contents/MacOS/firefox",
          "/Applications/Firefox.app/Contents/MacOS/firefox"
        ],
        args: ["-profile", tempDir + "/testem.firefox"],
        setup: function(config, done){
          setupFirefoxProfile(tempDir + '/testem.firefox', done)
        },
        supported: browserExeExists
      },
      {
        name: "Safari",
        exe: [
          process.env.HOME + "/Applications/Safari.app/Contents/MacOS/Safari",
          "/Applications/Safari.app/Contents/MacOS/Safari"
        ],
        setup: function(config, done){
          var url = this.getUrl()
          fs.writeFile(tempDir + '/testem.safari.html', "<script>window.location = '" + url + "'</script>", done)
        },
        args: function(){
          return [tempDir + '/testem.safari.html']
        },
        supported: browserExeExists
      },
      {
        name: "Opera",
        exe: [
          process.env.HOME + "/Applications/Opera.app/Contents/MacOS/Opera",
          "/Applications/Opera.app/Contents/MacOS/Opera"
        ],
        args: ["--user-data-dir=" + tempDir + "/testem.opera", "-pd", tempDir + "/testem.opera"],
        setup: function(config, done){
          rimraf(tempDir + '/testem.opera', done)
        },
        supported: browserExeExists
      },
      {
        name: 'PhantomJS',
        exe: 'phantomjs',
        args: buildPhantomJsArgs,
        supported: findableByWhichOrModule
      }
    ]
  }else if (platform === 'linux'){
    return [
      {
        name: 'Firefox',
        exe: 'firefox',
        args: ["-no-remote", "-profile", tempDir + "/testem.firefox"],
        setup: function(config, done){
          rimraf(tempDir + '/testem.firefox', function(err){
            if (!err){
              fs.mkdir(tempDir + '/testem.firefox', done)
            }else{
              done()
            }
          })
        },
        supported: findableByWhich
      },
      {
        name: 'Chrome',
        exe: 'google-chrome',
        args: ["--user-data-dir=" + tempDir + "/testem.chrome",
          "--no-default-browser-check", "--no-first-run", "--ignore-certificate-errors"],
        setup: function(config, done){
          rimraf(tempDir + '/testem.chrome', done)
        },
        supported: findableByWhich
      },
      {
        name: 'Chromium',
        exe: ['chromium', 'chromium-browser'],
        args: ["--user-data-dir=" + tempDir + "/testem.chromium",
          "--no-default-browser-check", "--no-first-run", "--ignore-certificate-errors"],
        setup: function(config, done){
          rimraf(tempDir + '/testem.chromium', done)
        },
        supported: findableByWhich
      },
      {
        name: 'PhantomJS',
        exe: 'phantomjs',
        args: buildPhantomJsArgs,
        supported: findableByWhichOrModule
      }
    ]
  }else if (platform === 'sunos') {
  	return [
        {
          name: 'PhantomJS',
          exe: 'phantomjs',
          args: buildPhantomJsArgs,
          supported: findableByWhichOrModule
        }
  	]
  }else if (platform === 'freebsd') {
  	return [
        {
          name: 'PhantomJS',
          exe: 'phantomjs',
          args: buildPhantomJsArgs,
          supported: findableByWhichOrModule
        }
  	]
  }else{
    return []
  }
}

// Returns the avaliable browsers on the current machine.
function getAvailableBrowsers(cb){
  var browsers = browsersForPlatform()
  browsers.forEach(function(b){
    b.protocol = 'browser'
  })
  async.filter(browsers, function(browser, cb){
    browser.supported(cb)
  }, function(available){
    cb(available)
  })
}

exports.getAvailableBrowsers = getAvailableBrowsers
