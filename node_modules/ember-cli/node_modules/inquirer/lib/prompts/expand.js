/**
 * `rawlist` type prompt
 */

var _ = require("lodash");
var util = require("util");
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

  if ( !this.opt.choices ) {
    this.throwParamError("choices");
  }

  this.validate();

  // Add the default `help` (/expand) option
  this.opt.choices.push({
    key   : "h",
    name  : "Help, list all options",
    value : "help"
  });

  this.opt.choices.setRender( renderChoice );

  // Setup the default string (capitalize the default key)
  var defIndex = 0;
  if ( _.isNumber(this.opt.default) && this.opt.choices.getChoice(this.opt.default) ) {
    defIndex = this.opt.default;
  }
  var defStr = this.opt.choices.pluck("key");
  this.rawDefault = defStr[ defIndex ];
  defStr[ defIndex ] = String( defStr[defIndex] ).toUpperCase();
  this.opt.default = defStr.join("");

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
  var message = this.getQuestion();

  if ( this.status === "answered" ) {
    message += chalk.cyan( this.selected.name ) + "\n";
  } else if ( this.status === "expanded" ) {
    message += this.opt.choices.render( this.selectedKey );
    message += "\n  Answer: ";
  }

  var msgLines = message.split(/\n/);
  this.height  = msgLines.length;

  this.rl.setPrompt( _.last(msgLines) );
  this.write( message );

  return this;
};


/**
 * Generate the prompt choices string
 * @return {String}  Choices string
 */

Prompt.prototype.getChoices = function() {
  var output = "";

  this.opt.choices.forEach(function( choice, i ) {
    output += "\n  ";

    if ( choice.type === "separator" ) {
      output += " " + choice;
      return;
    }

    var choiceStr = choice.key + ") " + choice.name;
    if ( this.selectedKey === choice.key ) {
      choiceStr = chalk.cyan( choiceStr );
    }
    output += choiceStr;
  }.bind(this));

  return output;
};


/**
 * When user press `enter` key
 */

Prompt.prototype.onSubmit = function( input ) {
  if ( input == null || input === "" ) {
    input = this.rawDefault;
  }

  var selected = this.opt.choices.where({ key : input.toLowerCase() })[0];

  if ( selected != null && selected.key === "h" ) {
    this.selectedKey = "";
    this.status = "expanded";
    this.down().clean(2).render();
    return;
  }

  if ( selected != null ) {
    this.status = "answered";
    this.selected = selected;

    // Re-render prompt
    this.down().clean(2).render();

    this.rl.removeAllListeners("line");
    this.rl.removeAllListeners("keypress");
    this.done( this.selected.value );
    return;
  }

  // Input is invalid
  this
    .error("Please enter a valid command")
    .clean()
    .render();
};


/**
 * When user press a key
 */

Prompt.prototype.onKeypress = function( s, key ) {
  this.selectedKey = this.rl.line.toLowerCase();
  var selected = this.opt.choices.where({ key : this.selectedKey })[0];
  this.cacheCursorPos();
  if ( this.status === "expanded" )  {
    this.clean().render();
  } else {
    this
      .down()
      .hint( selected ? selected.name : "" )
      .clean()
      .render();
  }

  this.write( this.rl.line ).restoreCursorPos();
};


/**
 * Validate the choices
 */

Prompt.prototype.validate = function() {
  var formatError;
  var errors = [];
  var keymap = {};
  this.opt.choices.filter(Separator.exclude).map(function( choice ) {
    if ( !choice.key || choice.key.length !== 1 ) {
      formatError = true;
    }
    if ( keymap[choice.key] ) {
      errors.push(choice.key);
    }
    keymap[ choice.key ] = true;
    choice.key = String( choice.key ).toLowerCase();
  });

  if ( formatError ) {
    throw new Error("Format error: `key` param must be a single letter and is required.");
  }
  if ( keymap.h ) {
    throw new Error("Reserved key error: `key` param cannot be `h` - this value is reserved.");
  }
  if ( errors.length ) {
    throw new Error( "Duplicate key error: `key` param must be unique. Duplicates: " +
        _.uniq(errors).join(", ") );
  }
};


/**
 * Function for rendering checkbox choices
 * @param  {String} pointer Selected key
 * @return {String}         Rendered content
 */

function renderChoice( pointer ) {
  var output = "";

  this.choices.forEach(function( choice, i ) {
    output += "\n  ";

    if ( choice.type === "separator" ) {
      output += " " + choice;
      return;
    }

    var choiceStr = choice.key + ") " + choice.name;
    if ( pointer === choice.key ) {
      choiceStr = chalk.cyan( choiceStr );
    }
    output += choiceStr;
  }.bind(this));

  return output;
}
