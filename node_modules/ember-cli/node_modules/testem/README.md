Got Scripts? Test&rsquo;em!
=================

[![Build Status](https://travis-ci.org/airportyh/testem.svg?branch=master)](https://travis-ci.org/airportyh/testem) [![Dependency Status](https://david-dm.org/airportyh/testem.svg)](https://david-dm.org/airportyh/testem) [![npm version](https://badge.fury.io/js/testem.svg)](http://badge.fury.io/js/testem) [![Windows build status](https://ci.appveyor.com/api/projects/status/l948rc4rv391ayge/branch/master?svg=true)](https://ci.appveyor.com/project/johanneswuerbach/testem/branch/master)

Unit testing in Javascript can be tedious and painful, but Testem makes it so easy that you will actually *want* to write tests.

Features
--------

* Test-framework agnostic. Support for
    - [Jasmine](http://jasmine.github.io/)
    - [QUnit](http://qunitjs.com/)
    - [Mocha](http://mochajs.org/)
    - [Buster.js](http://docs.busterjs.org/)
    - Others, through custom test framework adapters.
* Run tests in all major browsers as well as [Node](http://nodejs.org) and [PhantomJS](http://phantomjs.org/)
* Two distinct use-cases:
    - Test-Driven-Development(TDD) &mdash; designed to streamline the TDD workflow
    - Continuous Integration(CI) &mdash; designed to work well with popular CI servers like Jenkins or Teamcity
* Cross-platform support
    - OS X
    - Windows
    - Linux
* Preprocessor support
    - CoffeeScript
    - Browserify
    - JSHint/JSLint
    - everything else

Screencasts
-----------

* Watch this **[introductory screencast (11:39)](http://www.youtube.com/watch?v=-1mjv4yk5JM)** to see it in action! This one demonstrates the TDD workflow.
* [Launchers (12:10)](http://www.youtube.com/watch?v=Up0lVjWk9Rk) &mdash; more detail about launchers: how to specify what to auto-launch and how to configure one yourself to run tests in **Node**.
* [Continuous Integration (CI) Mode (4:24)](http://www.youtube.com/watch?v=Js16Cj80HKY) &mdash; details about how CI mode works.
* [Making JavaScript Testing Fun With Testem (22:53)](http://net.tutsplus.com/tutorials/javascript-ajax/make-javascript-testing-fun-with-testem/) &mdash; a thorough screencast by NetTuts+'s Jeffery Way covering the basics, Jasmine, Mocha/Chai, CoffeeScript and more!

Installation
------------
You need [Node](http://nodejs.org/) version 0.10+ or iojs installed on your system. Node is extremely easy to install and has a small footprint, and is really awesome otherwise too, so [just do it](http://nodejs.org/).

Once you have Node installed:

    npm install testem -g

This will install the `testem` executable globally on your system.

Usage
-----

As stated before, Testem supports two use cases: test-driven-development and continuous integration. Let's go over each one.

Development Mode
----------------

The simplest way to use Testem, in the TDD spirit, is to start in an empty directory and run the command

    testem

You will see a terminal-based interface which looks like this

![Initial interface](https://github.com/airportyh/testem/raw/master/images/initial.png)

Now open your browser and go to the specified URL. You should now see

![Zero of zero](https://github.com/airportyh/testem/raw/master/images/zeros.png)

We see 0/0 for tests because at this point we haven't written any code. As we write them, Testem will pick up any `.js` files
that were added, include them, and if there are tests, run them automatically. So let's first write `hello_spec.js` in the spirit of "test first"(written in Jasmine)

```javascript
describe('hello', function(){
  it('should say hello', function(){
    expect(hello()).toBe('hello world');
  });
});
```
Save that file and now you should see

![Red](https://github.com/airportyh/testem/raw/master/images/red.png)

Testem should automatically pick up the new files you've added and also any changes that you make to them and rerun the tests. The test fails as we'd expect. Now we implement the spec like so in `hello.js`

```javascript
function hello(){
  return "hello world";
}
```

So you should now see

![Green](https://github.com/airportyh/testem/raw/master/images/green.png)

### Using the Text User Interface

In development mode, Testem has a text-based graphical user interface which uses keyboard-based controls. Here is a list of the control keys

* ENTER : Run the tests
* q : Quit
* ← LEFT ARROW  : Move to the next browser tab on the left
* → RIGHT ARROW : Move to the next browser tab on the right
* TAB : switch the target text panel between the top and bottom halves of the split panel (if a split is present)
* ↑ UP ARROW : scroll up in the target text panel
* ↓ DOWN ARROW : scroll down in the target text panel
* SPACE : page down in the target text panel
* b : page up in the target text panel
* d : half a page down target text panel
* u : half a page up target text panel

### Command line options

To see all command line options

    testem --help

Continuous Integration Mode
---------------------------

To use Testem for continuous integration

    testem ci

In CI mode, Testem runs your tests on all the browsers that are available on the system one after another.

You can run multiple browsers in parallel in CI mode by specifying the `--parallel` (or `-P`) option to be the number of concurrent running browsers.

    testem ci -P 5 # run 5 browser in parallel

To find out what browsers are currently available - those that Testem knows about and can make use of

    testem launchers

Will print them out. The output might look like

    $ testem launchers
    Browsers available on this system:
    IE7
    IE8
    IE9
    Chrome
    Firefox
    Safari
    Opera
    PhantomJS

Did you notice that this system has IE versions 7-9? Yes, actually it has only IE9 installed, but Testem uses IE's compatibility mode feature to emulate IE 7 and 8.

When you run `testem ci` to run tests, it outputs the results in the [TAP](http://testanything.org/) format by default, which looks like

    ok 1 Chrome 16.0 - hello should say hello.

    1..1
    # tests 1
    # pass  1

    # ok

TAP is a human-readable and language-agnostic test result format. TAP plugins exist for popular CI servers

* [Jenkins TAP plugin](https://wiki.jenkins-ci.org/display/JENKINS/TAP+Plugin) - I've added [detailed instructions](https://github.com/airportyh/testem/blob/master/docs/use_with_jenkins.md) for setup with Jenkins.
* [TeamCity TAP plugin](https://github.com/pavelsher/teamcity-tap-parser)

## Other Test Reporters

Testem has other test reporters than TAP: `dot`, `xunit` and `teamcity`. You can use the `-R` to specify them

    testem ci -R dot

You can also [add your own reporter](docs/custom_reporter.md).

### Example xunit reporter output

Note that the real output is not pretty printed.
```xml
<testsuite name="Testem Tests" tests="4" failures="1" timestamp="Wed Apr 01 2015 11:56:20 GMT+0100 (GMT Daylight Time)" time="9">
  <testcase classname="PhantomJS 1.9" name="myFunc returns true when input is valid" time="0"/>
  <testcase classname="PhantomJS 1.9" name="myFunc returns false when user tickles it" time="0"/>
  <testcase classname="Chrome" name="myFunc returns true when input is valid" time="0"/>
  <testcase classname="Chrome" name="myFunc returns false when user tickles it" time="0">
    <failure name="myFunc returns false when user tickles it" message="function is not ticklish">
      <![CDATA[
      Callstack...
      ]]>
    </failure>
  </testcase>
</testsuite>
```

### Example teamcity reporter output

    ##teamcity[testStarted name='PhantomJS 1.9 - hello should say hello']
    ##teamcity[testFinished name='PhantomJS 1.9 - hello should say hello']
    ##teamcity[testStarted name='PhantomJS 1.9 - hello should say hello to person']
    ##teamcity[testFinished name='PhantomJS 1.9 - hello should say hello to person']
    ##teamcity[testStarted name='PhantomJS 1.9 - goodbye should say goodbye']
    ##teamcity[testFailed name='PhantomJS 1.9 - goodbye should say goodbye' message='expected |'hello world|' to equal |'goodbye world|'' details='AssertionError: expected |'hello world|' to equal |'goodbye world|'|n    at http://localhost:7357/testem/chai.js:873|n    at assertEqual (http://localhost:7357/testem/chai.js:1386)|n    at http://localhost:7357/testem/chai.js:3627|n    at http://localhost:7357/hello_spec.js:14|n    at callFn (http://localhost:7357/testem/mocha.js:4338)|n    at http://localhost:7357/testem/mocha.js:4331|n    at http://localhost:7357/testem/mocha.js:4728|n    at http://localhost:7357/testem/mocha.js:4819|n    at next (http://localhost:7357/testem/mocha.js:4653)|n    at http://localhost:7357/testem/mocha.js:4663|n    at next (http://localhost:7357/testem/mocha.js:4601)|n    at http://localhost:7357/testem/mocha.js:4630|n    at timeslice (http://localhost:7357/testem/mocha.js:5761)']
    ##teamcity[testFinished name='PhantomJS 1.9 - goodbye should say goodbye']

    ##teamcity[testSuiteFinished name='mocha.suite' duration='11091']

### Command line options

To see all command line options for CI

    testem ci --help

Configuration File
------------------

For the simplest JavaScript projects, the TDD workflow described above will work fine. There are times when you want
to structure your source files into separate directories, or want to have finer control over what files to include.
This calls for the `testem.json` configuration file (you can also alternatively use the YAML format with a `testem.yml` file). It looks like

```json
{
  "framework": "jasmine",
  "src_files": [
    "hello.js",
    "hello_spec.js"
  ]
}
```

The `src_files` can also be unix glob patterns.

```json
{
  "src_files": [
    "js/**/*.js",
    "spec/**/*.js"
  ]
}
```

You can also ignore certain files using `src_files_ignore`.
***Update: I've removed the ability to use a space-separated list of globs as a string in the src_files property because it disallowed matching files or directories with spaces in them.***

```json
{
  "src_files": [
    "js/**/*.js",
    "spec/**/*.js"
  ],
  "src_files_ignore": "js/toxic/*.js"
}
```

Read [more details](docs/config_file.md) about the config options.

Custom Test Pages
-----------------

You can also use a custom page for testing. To do this, first you need to specify `test_page` to point to your test page in the config file (`framework` and `src_files` are irrelevant in this case)

```json
{
  "test_page": "tests.html"
}
```

Next, the test page you use needs to have the adapter code installed on them, as specified in the next section.

### Include Snippet

Include this snippet directly after your `jasmine.js`, `qunit.js` or `mocha.js` or `buster.js` include to enable *Testem* with your test page.

```html
<script src="/testem.js"></script>
```

Or if you are using require.js or another loader, just make sure you load `/testem.js` as the next script after the test framework.

### Dynamic Substitution

To enable dynamically substituting in the Javascript files in your custom test page, you must

1. name your test page using `.mustache` as the extension
2. use `{{#serve_files}}` to loop over the set of Javascript files to be served, and then reference its `src` property to access their path (or `{{#css_files}}` for stylesheets)

Example:

    {{#serve_files}}
    <script src="{{src}}"></script>
    {{/serve_files}}

    {{#css_files}}
    <link rel="stylesheet" href="{{src}}">
    {{/css_files}}

Launchers
---------

Testem has the ability to automatically launch browsers or processes for you. To see the list of launchers Testem knows about, you can use the command

    testem launchers

This will display something like the following

    Have 5 launchers available; auto-launch info displayed on the right.

    Launcher      Type          CI  Dev
    ------------  ------------  --  ---
    Chrome        browser       ✔
    Firefox       browser       ✔
    Safari        browser       ✔
    Opera         browser       ✔
    Mocha         process(TAP)  ✔

This displays the current list of launchers that are available. Launchers can launch either a browser or a custom process &mdash; as shown in the "Type" column. Custom launchers can be defined to launch custom processes. The "CI" column indicates the launchers which will be automatically launch in CI-mode. Similarly, the "Dev" column those that will automatically launch in dev-mode.

Running Tests in Node and Custom Process Launchers
--------------------------------------------------

To run tests in Node you need to create a custom launcher which launches a process which will run your tests. This is nice because it means you can use any test framework - or lack thereof. For example, to make a launcher that runs mocha tests, you would write the following in the config file `testem.json`

```javascript
"launchers": {
  "Mocha": {
    "command": "mocha tests/*_tests.js"
  }
}
```

When you run `testem`, it will auto-launch the mocha process based on the specified command every time the tests are run. It will display the stdout and well as the stderr of the process inside of the "Mocha" tab in the UI. It will base the pass/fail status on the exit code of the process. In fact, because Testem can launch any arbitrary process for you, you could very well be using it to run programs in other languages.

Processes with TAP Output
-------------------------

If your process outputs test results in [TAP](http://en.wikipedia.org/wiki/Test_Anything_Protocol) format, you can tell that to testem via the `protocol` property. For example

```javascript
"launchers": {
  "Mocha": {
    "command": "mocha tests/*_tests.js -R tap",
    "protocol": "tap"
  }
}
```

When this is done, Testem will read in the process's stdout and parse it as TAP, and then display the test results in Testem's normal format. It will also hide the process's stdout output from the console log panel, although it will still display the stderr.

PhantomJS
---------

PhantomJS is a Webkit-based headless browser. It's fast and it's awesome! Testem will pick it up if you have [PhantomJS](http://www.phantomjs.org/) installed in your system and the `phantomjs` executable is in your path. Run

    testem launchers

And verify that it's in the list.

If you want to debug tests in PhantomJS, include the `phantomjs_debug_port` option in your testem configuration, referencing an available port number.  Once testem has started PhantomJS, navigate (with a traditional browser) to http://localhost:<port> and attach to one of PhantomJS's browser tabs (probably the second one in the list).  `debugger` statements will now break in the debugging console.

Preprocessors (CoffeeScript, LESS, Sass, Browserify, etc)
---------------------------------------------------------

If you need to run a preprocessor (or indeed any shell command before the start of the tests) use the `before_tests` option, such as

    "before_tests": "coffee -c *.coffee"

And Testem will run it before each test run. For file watching, you may still use the `src_files` option

```javascript
"src_files": [
  "*.coffee"
]
```

Since you want to be serving the `.js` files that are generated and not the `.coffee` files, you want to specify the `serve_files` option to tell it that

```javascript
"serve_files": [
  "*.js"
]
```

Testem will throw up a big ol' error dialog if the preprocessor command exits with an error code, so code checkers like jshint can used here as well.

If you need to run a command after your tests have completed (such as removing compiled `.js` files), use the `after_tests` option.

```javascript
"after_tests": "rm *.js"
```

If you would prefer simply to clean up when Testem exits, you can use the `on_exit` option.

Custom Routes
-------------

Sometimes you may want to re-map a URL to a different directory on the file system. Maybe you have the following file structure:

    + src
      + hello.js
      + tests.js
    + css
      + styles.css
    + public
      + tests.html

Let's say you want to serve `tests.html` at the top level url `/tests.html`, all the Javascripts under `/js` and all the css under `/css` you can use the "routes" option to do that

```javascript
"routes": {
  "/tests.html": "public/tests.html",
  "/js": "src",
  "/css": "css"
}
```

DIY: Use Any Test Framework
---------------------------

If you want to use Testem with a test framework that's not supported out of the box, you can write your own custom test framework adapter. See [customAdapter.js](https://github.com/airportyh/testem/blob/master/examples/custom_adapter/customAdapter.js) for an example of how to write a custom adapter.

Then, to use it, in your config file simply set

```javascript
"framework": "custom"
```

And then make sure you include the adapter code in your test suite and you are ready to go. Here for the [full example](https://github.com/airportyh/testem/tree/master/examples/custom_adapter).

Growl or Growl-ish Notifications
--------------------------------

If you'd prefer not to be looking at the terminal while developing, you can us growl notification (or simply desktop notifications on some platforms) using the `-g` option.

But, to use this option, you may first need to install some additional software, see the [node-growl page](https://github.com/visionmedia/node-growl#install) for more details.

API Proxy
--------------------------------

The proxy option allows you to transparently forward http requests to an external endpoint.

Simply add a `proxies` section to the `testem.json` configuration file.

```json
{
  "proxies": {
    "/api": {
      "target": "http://localhost:4200",
      "onlyContentTypes": ["xml", "json"]
    },
    "/xmlapi": {
      "target": "https://localhost:8000",
      "secure": false
    }
  }
}
```

This functionality is implemented as a *transparent proxy* hence a request to
`http://localhost:7357/api/posts.json` will be proxied to `http://localhost:4200/api/posts.json` without removing the `/api` prefix. Setting the `secure` option to false as in the above `/xmlapi` configuration block will ignore TLS certificate validation and allow tests to successfully reach that URL even if testem was launched over http. Other available options can be found here: https://github.com/nodejitsu/node-http-proxy#options

To limit the functionality to only certain content types, use "onlyContentTypes".

Example Projects
----------------

I've created [examples](https://github.com/airportyh/testem/tree/master/examples/) for various setups

* [Simple QUnit project](https://github.com/airportyh/testem/tree/master/examples/qunit_simple)
* [Simple Jasmine project](https://github.com/airportyh/testem/tree/master/examples/jasmine_simple)
* [Jasmine 2](https://github.com/airportyh/testem/tree/master/examples/jasmine2)
* [Custom Jasmine project](https://github.com/airportyh/testem/tree/master/examples/jasmine_custom)
* [Custom Jasmine project using Require.js](https://github.com/airportyh/testem/tree/master/examples/jasmine_requirejs)
* [Simple Mocha Project](https://github.com/airportyh/testem/tree/master/examples/mocha_simple)
* [Mocha + Chai](https://github.com/airportyh/testem/tree/master/examples/mocha_chai_simple)
* [Hybrid Project](https://github.com/airportyh/testem/tree/master/examples/hybrid_simple) - Mocha tests running in both the browser and Node.
* [Buster.js Project](https://github.com/airportyh/testem/tree/master/examples/buster)
* [Coffeescript Project](https://github.com/airportyh/testem/tree/master/examples/coffeescript)
* [Browserify Project](https://github.com/airportyh/testem/tree/master/examples/browserify)
* [JSHint Example](https://github.com/airportyh/testem/tree/master/examples/jshint)
* [Custom Test Framework](https://github.com/airportyh/testem/tree/master/examples/custom_adapter)
* [Tape Example](https://github.com/airportyh/testem/tree/master/examples/tape_example)
* [BrowserStack Integration](https://github.com/airportyh/testem/tree/master/examples/browserstack) **bleeding edge**
* [SauceLabs Integration](https://github.com/airportyh/testem/tree/master/examples/saucelabs) **bleeding edge**
* [Code Coverage with Istanbul](https://github.com/airportyh/testem/tree/master/examples/coverage_istanbul) **bleeding edge**

Known Issues
------------

1. On Windows, Mocha fails to run under Testem due to an [issue](https://github.com/joyent/node/issues/3871) in Node core. Until that gets resolved, I've made a [workaround](https://github.com/airportyh/mocha/tree/windowsfix) for mocha. To install this fork of Mocha, do

        npm install https://github.com/airportyh/mocha/tarball/windowsfix -g

2. If you are using prototype.js version 1.6.3 or below, you will [encounter issues](https://github.com/airportyh/testem/issues/130).

Contributing
------------

If you want to [contribute to the project](https://github.com/airportyh/testem/blob/master/CONTRIBUTING.md), I am going to do my best to stay out of your way.

Roadmap
-------

1. [BrowserStack](http://www.browserstack.com/user/dashboard) integration - following [Bunyip](http://www.thecssninja.com/javascript/bunyip)'s example
2. Figure out a happy path for testing on mobile browsers (maybe BrowserStack).

Contributors
------------

* [Toby Ho](https://github.com/airportyh)
* [Johannes Würbach](https://github.com/johanneswuerbach)
* [Raynos](https://github.com/Raynos)
* [Derek Brans](https://github.com/dbrans)

Community
---------

* **Mailing list**: <https://groups.google.com/forum/?fromgroups#!forum/testem-users>

Credits
-------

Testem depends on these great software

* [Jasmine](http://jasmine.github.io/)
* [QUnit](http://code.google.com/p/jqunit/)
* [Mocha](http://mochajs.org/)
* [Node](http://nodejs.org/)
* [Socket.IO](http://socket.io/)
* [PhantomJS](http://www.phantomjs.org/)
* [Node-Tap](https://github.com/isaacs/node-tap)
* [Node-Charm](https://github.com/substack/node-charm)
* [Node Commander](http://tjholowaychuk.com/post/9103188408/commander-js-nodejs-command-line-interfaces-made-easy)
* [JS-Yaml](https://github.com/nodeca/js-yaml)
* [Express](http://expressjs.com/)
* [jQuery](http://jquery.com/)
* [Backbone](http://backbonejs.org/)

License
-------

(The MIT License)

Copyright (c) 2012 Toby Ho &lt;airportyh@gmail.com&gt;

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the 'Software'), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED 'AS IS', WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
