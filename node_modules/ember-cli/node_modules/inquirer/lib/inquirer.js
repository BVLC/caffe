/**
 * Inquirer.js
 * A collection of common interactive command line user interfaces.
 */

var inquirer = module.exports;


/**
 * Client interfaces
 */

inquirer.prompts = {
  list     : require("./prompts/list"),
  input    : require("./prompts/input"),
  confirm  : require("./prompts/confirm"),
  rawlist  : require("./prompts/rawlist"),
  expand   : require("./prompts/expand"),
  checkbox : require("./prompts/checkbox"),
  password : require("./prompts/password")
};

inquirer.Separator = require("./objects/separator");

inquirer.ui = {
  BottomBar: require("./ui/bottom-bar"),
  Prompt: require("./ui/prompt")
};


/**
 * Public CLI helper interface
 * @param  {Array}   questions  Questions settings array
 * @param  {Function} cb        Callback being passed the user answers
 * @return {null}
 */

inquirer.prompt = function( questions, allDone ) {
  return new inquirer.ui.Prompt( questions, allDone );
};
