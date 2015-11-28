'use strict';
var rx = require('rx-lite');

function normalizeKeypressEvents(value, key) {
  return { value: value, key: key };
}

module.exports = function (rl) {
  var keypress = rx.Observable.fromEvent(rl, 'keypress', normalizeKeypressEvents);

  return {
    line: rx.Observable.fromEvent(rl, 'line'),
    keypress: keypress,

    normalizedUpKey: keypress.filter(function (e) {
      return e.key && (e.key.name === 'up' || e.key.name === 'k');
    }).share(),

    normalizedDownKey: keypress.filter(function (e) {
      return e.key && (e.key.name === 'down' || e.key.name === 'j');
    }).share(),

    numberKey: keypress.filter(function (e) {
      return e.value && '123456789'.indexOf(e.value) >= 0;
    }).map(function (e) {
      return Number(e.value);
    }).share(),

    spaceKey: keypress.filter(function (e) {
      return e.key && e.key.name === 'space';
    }).share(),

  };
};
