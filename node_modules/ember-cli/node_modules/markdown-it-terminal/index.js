// var md            = require('markdown-it');
var terminal      = require('./lib/markdown-it-terminal');
var merge         = require('lodash-node/modern/object/merge');
var styles        = require('ansi-styles');

module.exports = function terminal_plugin(md,options) {
  var defaultOptions = {
    styleOptions: {
      code: styles.yellow,
      blockquote: terminal.compoundStyle(['gray','italic']),
      html: styles.gray,
      heading: terminal.compoundStyle(['green','bold']),
      firstHeading: terminal.compoundStyle(['magenta','underline','bold']),
      hr: styles.reset,
      listitem: styles.reset,
      table: styles.reset,
      paragraph: styles.reset,
      strong: styles.bold,
      em: styles.italic,
      codespan: styles.yellow,
      del: terminal.compoundStyle(['dim','gray','strikethrough']),
      link: styles.blue,
      href: terminal.compoundStyle(['blue','underline'])
    },
    unescape: true
  };

  var opts = merge(defaultOptions, options);
  terminal(md,opts);
  // console.log(styles)
};