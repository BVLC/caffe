/*
  © 2014 by Philipp Dunkel <pip@pipobscure.com>
  Licensed under MIT License.
 */

/* jshint node:true */
'use strict';

var fs = require('fs');
var path = require('path');
var test = require('tap').test;

test('functionality testing', function(t) {
  try {
    fs.mkdirSync(__dirname + '/temp');
  } catch (ex) {}
  var evt = require('../')(__dirname + '/temp').start();
  t.on('end', function() {
    evt.stop();
  });
  t.plan(16);

  evt.on('fsevent', function(name, flags, id) {
    console.error("id:\t" + id);
    console.error("flags:\t" + JSON.stringify(flags));
    if (name === __dirname + '/temp') return;
    if (path.basename(name) === 'created-fsevent') {
      t.ok('number' === typeof flags, 'created file was caught with flags:' + flags);
      t.ok('number' === typeof id, 'id is a number ' + id);
    }
    if (path.basename(name) === 'moved-fsevent') {
      t.ok('number' === typeof flags, 'renamed file was caught with flags:' + flags);
      t.ok('number' === typeof id, 'id is a number ' + id);
    }
  });

  evt.on('change', function(name, info) {
    console.error("name:\t" + name);
    console.error("base:\t" + path.basename(name));
    console.error("event:\t" + info.event);
    console.error("info:\t" + JSON.stringify(info));
    if (name === __dirname + '/temp') return;
    t.ok(name === info.path, 'matched path');
    switch (info.event) {
      case 'created':
      case 'modified':
        // NOTE(bajtos) The recent versions apparently report `modified` event
        // instead of `created`.
        t.ok(path.basename(name) === 'created-fsevent', 'file created: ' + path.basename(name));
        break;
      case 'moved-out':
        t.ok(path.basename(name) === 'created-fsevent', 'file moved out: ' + path.basename(name));
        break;
      case 'moved-in':
        t.ok(path.basename(name) === 'moved-fsevent', 'file moved in: ' + path.basename(name));
        break;
      case 'deleted':
        t.ok(path.basename(name) === 'moved-fsevent', 'file deleted: ' + path.basename(name));
        break;
      default:
        t.ok(false, 'Uknown event type: ' + info.event);
        break;
    }
  });

  setTimeout(function() {
    console.error("===========================================================================");
    console.error("\twriteFileSync(__dirname + '/temp/created-fsevent', 'created-fsevent');");
    fs.writeFileSync(__dirname + '/temp/created-fsevent', 'created-fsevent');

    console.error("===========================================================================");
  }, 500);
  setTimeout(function() {
    console.error("===========================================================================");
    console.error("\trenameSync(__dirname + '/temp/created-fsevent', __dirname + '/temp/moved-fsevent');");
    fs.renameSync(__dirname + '/temp/created-fsevent', __dirname + '/temp/moved-fsevent');

    console.error("===========================================================================");

  }, 1000);
  setTimeout(function() {
    console.error("===========================================================================");
    console.error("\tunlinkSync(__dirname + '/temp/moved-fsevent');");
    fs.unlinkSync(__dirname + '/temp/moved-fsevent');
    console.error("===========================================================================");
  }, 1500);
  setTimeout(function() {
    console.error("===========================================================================");
    console.error("\trmdirSync(__dirname + '/temp');");
    fs.rmdirSync(__dirname + '/temp');
    console.error("===========================================================================");
  }, 2000);
});
