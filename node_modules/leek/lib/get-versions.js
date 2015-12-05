'use strict';

var os = require('os');
var debug = require('debug')('leek:get-version');

var map = {
  darwin: {
    '14.0.0': 'OSX Yosemite',
    '14.0':   'OSX Yosemite',
    '13.4.0': 'OSX Mavericks',
    '13.3.0': 'OSX Mavericks',
    '13.2.0': 'OSX Mavericks',
    '13.1.0': 'OSX Mavericks',
    '12.5.0': 'OSX Mountain Lion',
    '12.0.0': 'OSX Mountain Lion',
    '11.4.2': 'Mac OSX Lion',
    '11.0.0': 'Mac OSX Lion',
    '10.8':   'Mac OSX Snow Leopard',
    '10.0':   'Mac OSX Snow Leopard',
    '9.8':    'Leopard',
    '9.0':    'Leopard'
  },
  win32: {
    '6.3.9600': 'Windows 8.1',
    '6.2.9200': 'Windows 8',
    '6.1.7601': 'Windows 7 SP1',
    '6.1.7600': 'Windows 7',
    '6.0.6002': 'Windows Vista SP2',
    '6.0.6000': 'Windows Vista',
    '5.1.2600': 'Windows XP'
  },
  linux: { }
};

function getVersion(platform, release) {
  var result;

  if (!map[platform]) {
    result = platform + ' ' + release;
  } else {
    var p = os.platform();
    var r = os.release();

    result = map[p][r] || ' ' + p + ' ' + r;
  }

  debug('getVersion platform:%s release:%s output:', platform, release, result);

  return result;
}

module.exports = function getVersions() {
  return {
    platform: getVersion(os.platform(), os.release()),
    version:  process.version
  };
};
