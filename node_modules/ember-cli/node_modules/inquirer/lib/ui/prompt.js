/**
 * Base interface class other can inherits from
 */

var _ = require("lodash");
var async = require("async");
var util = require("util");
var utils = require("../utils/utils");
var Base = require("./baseUI");

var inquirer = require("../inquirer");


/**
 * Module exports
 */

module.exports = PromptUI;


/**
 * Constructor
 */

function PromptUI( questions, allDone ) {
  Base.call(this);

  // Keep global reference to the answers
  this.answers = {};
  this.completed = allDone;

  // Make sure questions is an array.
  if ( !_.isArray(questions) ) {
    questions = [questions];
  }

  // Start running the questions
  async.mapSeries( questions, this.onEachPrompt.bind(this), this.onCompletion.bind(this) );
}
util.inherits( PromptUI, Base );


/**
 * Once all prompt are over
 */

PromptUI.prototype.onCompletion = function() {
  this.close();

  if ( _.isFunction(this.completed) ) {
    this.completed( this.answers );
  }
};


/**
 * Each prompt rendering
 * @param  {Object}   question Question object
 * @param  {Function} done     Callback
 */

PromptUI.prototype.onEachPrompt = function( question, done ) {
  // Callback to save answer and continu to next question
  var after = function( answer ) {
    this.answers[question.name] = answer;
    done( null );
  }.bind(this);

  // Default type to input
  if ( !inquirer.prompts[question.type] ) {
    question.type = "input";
  }

  if ( _.isFunction(question.default) ) {
    question.default = question.default( this.answers );
  }

  if ( _.isFunction(question.choices) ) {
    question.choices = question.choices( this.answers );
  }

  var prompt = new inquirer.prompts[question.type]( question, this.rl, this.answers );

  // Check if prompt should be runned (if `when` return true)
  utils.runAsync( prompt.opt.when, function( continu ) {
    if ( continu ) {
      prompt.run( after );
    } else {
      done( null );
    }
  }, this.answers );
};
