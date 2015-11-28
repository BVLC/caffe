/**
 * `rawlist` type prompt
 */

var _ = require("lodash");
var util = require("util");
var clc = require("cli-color");
var chalk = require("chalk");
var Base = require("./base");
var Separator = require("../objects/separator");


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

  this.opt.validChoices = this.opt.choices.filter(Separator.exclude);

  this.selected = 0;
  this.rawDefault = 0;

  this.opt.choices.setRender( renderChoices );

  var def = this.opt.default;
  if ( _.isNumber(def) && def >= 0 && def < this.opt.choices.realLength ) {
    this.selected = this.rawDefault = def;
  }

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

  // Save user answer and update prompt to show selected option.
  this.rl.on( "line", this.onSubmit.bind(this) );
  this.rl.on( "keypress", this.onKeypress.bind(this) );

  // Init the prompt
  this.render();

  return this;
};


/**
 * Render the prompt to screen
 * @return {Prompt} self
 */

Prompt.prototype.render = function() {
  // Render question
  var message    = this.getQuestion();
  var choicesStr = this.opt.choices.render( this.selected );

  if ( this.status === "answered" ) {
    message += chalk.cyan(this.opt.choices.getChoice(this.selected).name) + "\n";
  } else {
    message += choicesStr;
    message += "\n  Answer: ";
  }

  var msgLines = message.split(/\n/);
  this.height  = msgLines.length;

  this.rl.setPrompt( _.last(msgLines) );
  this.write( message );

  return this;
};


/**
 * When user press `enter` key
 */

Prompt.prototype.onSubmit = function( input ) {
  if ( input == null || input === "" ) {
    input = this.rawDefault;
  } else {
    input -= 1;
  }

  var selectedChoice = this.opt.choices.getChoice(input);

  // Input is valid
  if ( selectedChoice != null ) {
    this.status = "answered";
    this.selected = input;

    // Re-render prompt
    this.down().clean(2).render();

    this.rl.removeAllListeners("line");
    this.rl.removeAllListeners("keypress");
    this.done( selectedChoice.value );
    return;
  }

  // Input is invalid
  this
    .error("Please enter a valid index")
    .write( clc.bol(0, true) )
    .clean()
    .render();
};


/**
 * When user press a key
 */

Prompt.prototype.onKeypress = function( s, key ) {
  var index = this.rl.line.length ? Number(this.rl.line) - 1 : 0;

  if ( this.opt.choices.getChoice(index) ) {
    this.selected = index;
  } else {
    this.selected = undefined;
  }

  this.cacheCursorPos().down().clean(1).render().write( this.rl.line ).restoreCursorPos();
};


/**
 * Function for rendering list choices
 * @param  {Number} pointer Position of the pointer
 * @return {String}         Rendered content
 */

function renderChoices( pointer ) {
  var output = "";
  var separatorOffset = 0;

  this.choices.forEach(function( choice, i ) {
    output += "\n  ";

    if ( choice.type === "separator" ) {
      separatorOffset++;
      output += " " + choice;
      return;
    }

    var index = i - separatorOffset;
    var display = (index + 1) + ") " + choice.name;
    if ( index === pointer ) {
      display = chalk.cyan( display );
    }
    output += display;
  }.bind(this));

  return output;
}
