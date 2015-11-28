/**
 * `confirm` type prompt
 */

var _ = require("lodash");
var util = require("util");
var chalk = require("chalk");
var Base = require("./base");


/**
 * Module exports
 */

module.exports = Prompt;


/**
 * Constructor
 */

function Prompt() {
  Base.apply( this, arguments );

  var rawDefault = true;

  _.extend( this.opt, {
    filter: function( input ) {
      var value = rawDefault;
      if ( input != null && input !== "" ) {
        value = /^y(es)?/i.test(input);
      }
      return value;
    }.bind(this)
  });

  if ( _.isBoolean(this.opt.default) ) {
    rawDefault = this.opt.default;
  }

  this.opt.default = rawDefault ? "Y/n" : "y/N";

  return this;
}
util.inherits( Prompt, Base );


/**
 * Start the Inquiry session
 * @param  {Function} cb   Callback when prompt is done
 * @return {this}
 */

Prompt.prototype._run = function( cb ) {
  this.done = cb;

  // Once user confirm (enter key)
  this.rl.once( "line", this.onSubmit.bind(this) );

  // Init
  this.render();

  return this;
};


/**
 * Render the prompt to screen
 * @return {Prompt} self
 */

Prompt.prototype.render = function() {
  var message = this.getQuestion();

  this.write( message );

  var msgLines = message.split(/\n/);
  this.height = msgLines.length;
  this.rl.setPrompt( _.last(msgLines) );

  return this;
};


/**
 * When user press "enter" key
 */

Prompt.prototype.onSubmit = function( input ) {
  this.status = "answered";

  // Filter value to write final answer to screen
  this.filter( input, function( output ) {
    this.clean(1).render();
    this.write( chalk.cyan(output ? "Yes" : "No") + "\n" );

    this.done( input ); // send "input" because the master class will refilter
  }.bind(this));
};
