'use strict';
var _ = require('lodash');
var readlineFacade = require('readline2');


/**
 * Base interface class other can inherits from
 */

var UI = module.exports = function (opt) {
  // Instantiate the Readline interface
  // @Note: Don't reassign if already present (allow test to override the Stream)
  if (!this.rl) {
    this.rl = readlineFacade.createInterface(opt);
  }
  this.rl.resume();

  this.onForceClose = this.onForceClose.bind(this);
  this.onKeypress = this.onKeypress.bind(this);

  // Make sure new prompt start on a newline when closing
  this.rl.on('SIGINT', this.onForceClose);
  process.on('exit', this.onForceClose);

  // Propagate keypress events directly on the readline
  this.rl.input.addListener('keypress', this.onKeypress);
};


/**
 * Handle the ^C exit
 * @return {null}
 */

UI.prototype.onForceClose = function () {
  this.close();
  console.log('\n'); // Line return
};


/**
 * Close the interface and cleanup listeners
 */

UI.prototype.close = function () {
  // Remove events listeners
  this.rl.removeListener('SIGINT', this.onForceClose);
  this.rl.input.removeListener('keypress', this.onKeypress);
  process.removeListener('exit', this.onForceClose);

  // Restore prompt functionnalities
  this.rl.output.unmute();
  this.rl.output.write('\x1B[?25h'); // show cursor

  // Close the readline
  this.rl.output.end();
  this.rl.pause();
  this.rl.close();
  this.rl = null;
};


/**
 * Handle and propagate keypress events
 */

UI.prototype.onKeypress = function (s, key) {
  // Ignore `enter` key (readline `line` event is the only one we care for)
  if (key && (key.name === 'enter' || key.name === 'return')) {
    return;
  }

  if (this.rl) {
    this.rl.emit( 'keypress', s, key );
  }
};
