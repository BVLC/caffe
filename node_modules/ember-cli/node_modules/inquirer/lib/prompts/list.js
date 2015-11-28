/**
 * `list` type prompt
 */

var _ = require("lodash");
var util = require("util");
var chalk = require("chalk");
var Base = require("./base");
var utils = require("../utils/utils");


/**
 * Module exports
 */

module.exports = Prompt;


/**
 * Constructor
 */

function Prompt() {
  Base.apply( this, arguments );

  if (!this.opt.choices) {
    this.throwParamError("choices");
  }

  this.firstRender = true;
  this.selected = 0;

  var def = this.opt.default;

  // Default being a Number
  if ( _.isNumber(def) && def >= 0 && def < this.opt.choices.realLength ) {
    this.selected = def;
  }

  // Default being a String
  if ( _.isString(def) ) {
    this.selected = this.opt.choices.pluck("value").indexOf( def );
  }

  this.opt.choices.setRender( listRender );

  // Make sure no default is set (so it won't be printed)
  this.opt.default = null;

  return this;
}
util.inherits( Prompt, Base );


/**
 * Start the Inquiry session
 * @param  {Function} cb      Callback when prompt is done
 * @return {this}
 */

Prompt.prototype._run = function( cb ) {
  this.done = cb;

  // Move the selected marker on keypress
  this.rl.on( "keypress", this.onKeypress.bind(this) );

  // Once user confirm (enter key)
  this.rl.once( "line", this.onSubmit.bind(this) );

  // Init the prompt
  this.render();
  this.hideCursor();

  // Prevent user from writing
  this.rl.output.mute();

  return this;
};


/**
 * Render the prompt to screen
 * @return {Prompt} self
 */

Prompt.prototype.render = function() {

  // Render question
  var message    = this.getQuestion();
  var choicesStr = "\n" + this.opt.choices.render( this.selected );

  if ( this.firstRender ) {
    message += "(Use arrow keys)";
  }

  // Render choices or answer depending on the state
  if ( this.status === "answered" ) {
    message += chalk.cyan( this.opt.choices.getChoice(this.selected).name ) + "\n";
  } else {
    message += choicesStr;
  }

  this.firstRender = false;

  var msgLines = message.split(/\n/);
  this.height = msgLines.length;

  // Write message to screen and setPrompt to control backspace
  this.rl.setPrompt( _.last(msgLines) );
  this.write( message );

  return this;
};


/**
 * When user press `enter` key
 */

Prompt.prototype.onSubmit = function() {
  var choice = this.opt.choices.getChoice( this.selected );
  this.status = "answered";

  // Rerender prompt
  this.rl.output.unmute();
  this.clean().render();

  this.showCursor();

  this.rl.removeAllListeners("keypress");
  this.done( choice.value );
};


/**
 * When user press a key
 */

Prompt.prototype.onKeypress = function( s, key ) {
  // Only process up, down, j, k and 1-9 keys
  var keyWhitelist = [ "up", "down", "j", "k" ];
  if ( key && !_.contains(keyWhitelist, key.name) ) return;
  if ( key && (key.name === "j" || key.name === "k") ) s = undefined;
  if ( s && "123456789".indexOf(s) < 0 ) return;

  this.rl.output.unmute();

  var len = this.opt.choices.realLength;
  if ( key && (key.name === "up" || key.name === "k") ) {
    (this.selected > 0) ? this.selected-- : (this.selected = len - 1);
  } else if ( key && (key.name === "down" || key.name === "j") ) {
    (this.selected < len - 1) ? this.selected++ : (this.selected = 0);
  } else if ( Number(s) <= len ) {
    this.selected = Number(s) - 1;
  }

  // Rerender
  this.clean().render();

  this.rl.output.mute();
};


/**
 * Function for rendering list choices
 * @param  {Number} pointer Position of the pointer
 * @return {String}         Rendered content
 */

function listRender( pointer ) {
  var output = "";
    var separatorOffset = 0;

    this.choices.forEach(function( choice, i ) {
      if ( choice.type === "separator" ) {
        separatorOffset++;
        output += "  " + choice + "\n";
        return;
      }

      var isSelected = (i - separatorOffset === pointer);
      var line = (isSelected ? utils.getPointer() + " " : "  ") + choice.name;
      if ( isSelected ) {
        line = chalk.cyan( line );
      }
      output += line + " \n";
    }.bind(this));

    return output.replace(/\n$/, "");
}
