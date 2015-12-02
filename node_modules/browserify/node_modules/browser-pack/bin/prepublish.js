#!/usr/bin/env node

var uglify = require('uglify-js');
var fs = require('fs');
var path = require('path');

var src = fs.readFileSync(path.join(__dirname, '..', 'prelude.js'), 'utf8');
fs.writeFileSync(path.join(__dirname, '..', '_prelude.js'), uglify(src));
