[![Build Status](https://travis-ci.org/stevengill/cordova-registry-mapper.svg?branch=master)](https://travis-ci.org/stevengill/cordova-registry-mapper)

#Cordova Registry Mapper

This module is used to map Cordova plugin ids to package names and vice versa.

When Cordova users add plugins to their projects using ids
(e.g. `cordova plugin add org.apache.cordova.device`),
this module will map that id to the corresponding package name so `cordova-lib` knows what to fetch from **npm**.

This module was created so the Apache Cordova project could migrate its plugins from
the [Cordova Registry](http://registry.cordova.io/)
to [npm](https://registry.npmjs.com/)
instead of having to maintain a registry.
