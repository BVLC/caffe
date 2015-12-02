var moment  = require('moment');
var chalk   = require('chalk');
var path    = require('path');

// Prototypes
String.prototype.repeat = function(count) {
    return new Array(count+1).join(this);
}

var logger  = {

  log: function(text) {
    console.log(text);
  },

  // Second level logging
  ok: function (text) {
    logger.log(chalk.green('   ✔') + chalk.dim(text));
  },

  skip: function (text){
    logger.log(chalk.cyan('   ★') + chalk.dim(text));
  },

  fail: function (text){
    logger.log(chalk.red('   ✗') + chalk.dim(text));
  },

  // First level logging
  done: function (text) {
    logger.log(chalk.dim('  Done ') + chalk.green('✔') + "\n");
  },

  info: function (text) {
    logger.log(chalk.bold('➡ ') + chalk.bold(text));
  },

  // Top level logging
  footer: function (text) {
    logger.header(text);
  },

  header: function (text) {
    var char      = "—";
    var space     = " ";
    var max       = 80;
    var spacing   = 2;
    var buffer    = (text.length % 2);
    var padding   = (max - spacing * 2 - text.length + buffer) / 2;
    var line      = char.repeat(max);

    logger.log("");
    // logger.log(chalk.bold(line));
    logger.log(chalk.bold(char.repeat(padding) + space.repeat(spacing) +
                text.toUpperCase() +
                space.repeat(spacing) + char.repeat(padding - buffer)));
    // logger.log(chalk.bold(line));
    logger.log("");
  },

  error: function (text) {
    logger.log(chalk.bgRed.bold(text));
  },

  asset: function(category, file) {
    var attributes = chalk.cyan(category);
    attributes    += chalk.dim(" ❯ ");
    attributes    += chalk.bold(file.path ? path.basename(file.path) : file);
    logger.log("   " + attributes);
  },

  cache: function (fromCache, entry) {
   var attributes = chalk.dim(entry.locale);
   attributes += chalk.dim(" ❯ ");

   if (entry.type === 'component') {
     attributes += chalk.dim(entry.page);
     attributes += chalk.dim(" ❯ ");
     attributes += chalk.cyan(entry.component);
     attributes += chalk.dim (" ❯ ");
     attributes += chalk.bold(entry.componentFileType);
   } else if (entry.type === 'page') {
     attributes += chalk.cyan(entry.page);
     attributes += chalk.dim (" ❯ ");
     attributes += chalk.bold(entry.pageFileType);
   } else {
     attributes += chalk.dim(entry.source + ":") + chalk.bold(entry.file);
   }
   logger.log("   " + (fromCache ? chalk.green("[FROM CACHE]") : chalk.green.bold("[NOW CACHED]")) + " " + attributes);
  },

  fromCache: function (entry) {
    logger.cache(true, entry);
  },

  toCache: function (entry) {
    logger.cache(false, entry);
  },
}

var utils = {

  logger: logger,

  contentType: function (filename) {
    var lowercase = filename.toLowerCase();

    if (lowercase.indexOf('.html') >= 0) return 'text/html';
    else if (lowercase.indexOf('.css') >= 0) return 'text/css';
    else if (lowercase.indexOf('.json') >= 0) return'application/json';
    else if (lowercase.indexOf('.js') >= 0) return 'application/x-javascript';
    else if (lowercase.indexOf('.png') >= 0) return 'image/png';
    else if (lowercase.indexOf('.jpg') >= 0) return 'image/jpg';

    return 'application/octet-stream';
  },

  uuid: function () {
      var d = new Date().getTime();
      var uuid = 'xxxxxxxx-xxxx-0xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
          var r = (d + Math.random()*16)%16 | 0;
          d = Math.floor(d/16);
          return (c=='x' ? r : (r&0x3|0x8)).toString(16);
      });
      return uuid;
  },

  merge: function (data) {
      var result = {};
      if (data) {
        data.forEach(function(dataObject) {
          if (dataObject) {
            for (var object in dataObject)  { if (dataObject[object]) { result[object] = dataObject[object]; } }
          }
        });
      }

      return result;
  },

  datestamp: function () {
    var now = new Date();
    year = "" + now.getFullYear();
    month = "" + (now.getMonth() + 1); if (month.length == 1) { month = "0" + month; }
    day = "" + now.getDate(); if (day.length == 1) { day = "0" + day; }
    hour = "" + now.getHours(); if (hour.length == 1) { hour = "0" + hour; }
    minute = "" + now.getMinutes(); if (minute.length == 1) { minute = "0" + minute; }
    second = "" + now.getSeconds(); if (second.length == 1) { second = "0" + second; }
    return year + month + day + hour + minute + second;
  },

  niceDate: function(string, locale) {
    moment.locale(locale);
    return moment(string).format('MMMM Do, YYYY');
  }
};

module.exports = utils;
