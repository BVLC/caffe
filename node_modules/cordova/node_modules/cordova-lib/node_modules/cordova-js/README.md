<!--
#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
#  KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#
-->

[![Build Status](https://travis-ci.org/apache/cordova-js.svg?branch=master)](https://travis-ci.org/apache/cordova-js)

[![Build status](https://ci.appveyor.com/api/projects/status/github/apache/cordova-js?branch=master&svg=true)](https://ci.appveyor.com/project/Humbedooh/cordova-js/branch/master)


A unified JavaScript layer for [Apache Cordova](http://cordova.apache.org/) projects.

# Project Structure

    ./
     |-src/
     |  |-cordova.js ........ common Cordova stuff
     |  |-common/ ........... base modules shared across platfoms
     |  |  |-builder.js ..... injects in our classes onto window and navigator
     |  |  |-channel.js ..... pub/sub impl for custom framework events 
     |  |  |-init.js ........ common locations to add Cordova objects to browser globals
     |  |  |-exec.js ........ interace stub for each platform specific version of exec.js
     |  |  |-platform.js .... stub for platform's specific version of platform.js
     |  |  '-utils.js ....... closures, uuids, object, cloning, extending prototypes
     |  |
     |  |-scripts/ .......... non-module JS that gets concated to cordova.<platform>.js
     |  |  |-bootstrap.js ... bootstrap the Cordova platform, inject APIs and fire events
     |  |  '-require.js ..... module definition and require() impl
     |  |
     |  |-legacy-exec/ ...... contains old platform specific modules
     |  |  |-<platform>/ .... contains the platform-specific base modules
     |
     |-tasks/ ............... custom grunt tasks
     |-tests/ ............... unit tests
     |
     '-pkg/ ................. generated platform cordova.js files

# Building

Make sure you have [node.js](http://nodejs.org) installed. It should come pre-installed with [npm](http://npmjs.org) - but if you install node and can't run `npm` then head over to the website and install it yourself. Make sure you have all of the node dependencies installed by running the following command from the repository root:

    npm install

All of the build tasks can be run via the `grunt` node module. Install it globally first by running:

    sudo npm install -g grunt-cli

`grunt compile` task assumes that users have cordova-platforms as siblings to this cordova.js directory. When generating cordova.js, `grunt compile` will grab platform specific files from these directories if they exist. The default locations are defined in `package.json`.

Then from the repository root run:

    grunt 

To compile the js for just one platform, run:

    grunt compile:android --platformVersion=4.0.0

To comiple the js for all platforms but pass in a custom path for your cordova-android and cordova-ios platforms, run:

    grunt compile --android='../custompath/cordova-android' --ios='../custompath/cordova-ios'

To create the browserify version of the js, run:

    grunt compile-browserify --platformVersion=4.0.0

To compile the browserify version of the js for just one platform, run:

    grunt compile-browserify:android --platformVersion=4.0.0

NOTE: browserify method does not support custom paths for platform repos.

For integration, see the 'Integration' section below.

## Known Issues

- On Windows, when you run `npm install`, you may get errors regarding
  contextify. This is necessary for running the tests. Make sure you
  are running `node` **0.10.1** at the least (and `npm` **1.2.15** which should
  come bundled with `node` **0.10.1**). Also, install [Python 2.7.x](http://python.org/download/releases/2.7.3) and [Visual C++ 2010 Express](http://www.microsoft.com/visualstudio/en-us/products/2010-editions/visual-cpp-express). When that is done, run `npm install` again and it should build
  contextify natively on Windows.

# How It Works

The `tasks/lib/packager.js` tool is a node.js script that concatenates all of the core Cordova plugins in this repository into a `cordova.<platform>.js` file under the `pkg/` folder. It also wraps the plugins with a RequireJS-compatible module syntax that works in both browser and node environments. We end up with a `cordova.js` file that wraps each **Cordova** *plugin* into its own module.

**Cordova** defines a `channel` module under `src/common/channel.js`, which is a *publish/subscribe* implementation that the project uses for event management.

The **Cordova** *native-to-webview* bridge is initialized in `src/scripts/bootstrap.js`. This file attaches the `boot` function to the `channel.onNativeReady` event - fired by native with a call to:

    cordova.require('cordova/channel).onNativeReady.fire()

The `boot` method does all the work.  First, it grabs the common platform definition (under `src/common/common.js`) and injects all of the objects defined there onto `window` and other global namespaces. Next, it grabs all of the platform-specific object definitions (as defined under `src/<platform>/platform.js`) and overrides those onto `window`. Finally, it calls the platform-specific `initialize` function (located in the platform definition). At this point, Cordova is fully initialized and ready to roll. Last thing we do is wait for the `DOMContentLoaded` event to fire to make sure the page has loaded properly. Once that is done, Cordova fires the `deviceready` event where you can safely attach functions that consume the Cordova APIs.

# Testing

Tests run in node or the browser. To run the tests in node:
    
    grunt test --platformVersion=3.6.0

To run them in the browser:

    grunt btest

Final testing should always be done with the [Mobile Spec test application](https://github.com/apache/cordova-mobile-spec).

To get current tests coverage:

    grunt cover --platformVersion=3.6.0

# Integration

## Cordova

Build the js files by running **grunt** as described above. Update each platform independently. For a given platform:

Replace the `cordova.js` file in the cordova <platform>platform_www/cordova.js directory with the newly generated `cordova.js` file. If necessary, change the name of the new file to match that of the overwritten one.

Once the new js file has been added, any new projects created will use the updated js. To update an already existing project, directly replace the cordova.js file within the project's `www/` folder with the generated `cordova.PLATFORM.js`. Make sure to change the file name to match the original.

# Adding a New Platform

1. Add your platform as a directory under the `legacy-exec` folder.
2. Write a module that encapsulates your platform's `exec` method and
   call it `exec.js`. The `exec` method is a JavaScript function
   that enables communication from the platform's JavaScript environment
   into the platform's native environment. Each platform uses a different
   mechanism to enable this bridge. We recommend you check out the other
   platform `exec` definitions for inspiration. Drop this into the
   `src/legacy-exec/<platform>` folder you created in step 1. The `exec` method has the following method
   signature: `function(success, fail, service, action, args)`, with the
   following parameters:
  - `success`: a success function callback
  - `fail`: a failure function callback
  - `service`: a string identifier that the platform can resolve to a
    native class
  - `action`: a string identifier that the platform can resolve to a
    specific method inside the class pointed to by `service`
  - `args`: an array of parameters to pass to the native method invoked
    by the `exec` call
   It is required that new platform additions be as consistent as
   possible with the existing `service` and `action` labels.
2. Define your platform definition object and name it `platform.js`. Drop this into the `src/legacy-exec/<platform>` folder. This file should contain a **JSON** object with the following properties:
    - `id`: a string representing the platform. This should be the same
      name the .js file has
    - `objects`: the property names defined as children of this property
      are injected into `window`, and also *overrides any existing
      properties*. Each property can have the following
      child properties:
      - `path`: a string representing the module ID that will define
        this object. For example, the file `lib/plugin/accelerometer.js`
        can be accessed as `"cordova/plugin/accelerometer"`. More details on how
        the module IDs are defined are above under the "How It Works" section.
      - `children`: in a recursive fashion, can have `path` and
        `children` properties of its own that are defined as children of
        the parent property object
    - `merges`: similar to the above `objects` property, this one will
      not clobber existing objects, instead it will recursively merge
      this object into the specific target
    - `initialize`: a function that fires immediately after the `objects` (see above) are defined in the global scope
   
   The following is a simple example of a platform definition:

    <pre>
    {
      id:"atari",
      initialize:function(){
        console.log('firing up Cordova in my Atari, yo.');
      },
      objects:{
        cordova:{
          path:"cordova",
          children:{
            joystick:{
              path:"cordova/plugin/atari/joystick"
            }
          }
        }
      }
    }
    </pre>

3. You should probably add a `<platform>:{}` entry to the `Gruntfile` compile arrays.
4. Make sure your native implementation executes the following JavaScript once all of the native side is initialized and ready: `require('cordova/channel').onNativeReady.fire()`.
5. The `deviceready` event is important. To make sure that the stock
   common JavaScript fires this event off, the device and network
   connection plugins must successfully be instantiated and return
   information about the connectivity and device information. The
   success callbacks for these plugins should include calls to
   `require('cordova/channel').onCordovaInfoReady.fire()` (for device
   information) and
   `require('cordova/channel').OnCordovaConnectionReady.fire()` (for
   network information).
6. Last but certainly not least: add yourself to the contributors list!
   It's in the `package.json` file in the root of this repository. You
   deserve it!
