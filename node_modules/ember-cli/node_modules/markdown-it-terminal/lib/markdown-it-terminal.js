'use strict';

var cardinal        = require('cardinal');
var styles          = require('ansi-styles');
var Table           = require('cli-table');
var color_block     = require('./color_block');
var color_inline    = require('./color_inline');
var assign          = require('./utils').assign;
var escapeHtml      = require('./utils').escapeHtml;
var compoundStyle   = require('./utils').compoundStyle;

var TABLE_CELL_SPLIT = '^*||*^';
var TABLE_ROW_WRAP = '*|*|*|*';
var TABLE_ROW_WRAP_REGEXP = new RegExp(escapeRegExp(TABLE_ROW_WRAP), 'g');
var colors = ['red', 'green', 'blue', 'cyan', 'magenta', 'yellow', 'black', 'white', 'grey', 'gray'];
var backgroundColors = ['bgRed', 'bgGreen', 'bgBlue', 'bgCyan', 'bgMagenta', 'bgYellow', 'bgWhite', 'bgBlack'];

module.exports = function(md, options) {
  var styleOptions = options.styleOptions;
  var unescape     = options.unescape;
  var highlight    = options.highlight || cardinal.highlight;
  
  md.renderer.rules.blockquote_open  = function () { 
    return '\n' + styleOptions.blockquote.open;
  };
  md.renderer.rules.blockquote_close = function () {
    return styleOptions.blockquote.close + '\n\n'; 
  };


  md.renderer.rules.code_block = function (tokens, idx /*, options, env */) {
    var e = unescape ? unescapeEntities : escapeHtml;
    return '\n' + styleOptions.code.open + e(tokens[idx].content) + styleOptions.code.close + '\n\n';
  };
  md.renderer.rules.code_inline = function (tokens, idx /*, options, env */) {
    var e = unescape ? unescapeEntities : escapeHtml;
    return styleOptions.codespan.open + e(tokens[idx].content) + styleOptions.codespan.close;
  };


  md.renderer.rules.fence = function (tokens, idx, options, env, self) {
    var token = tokens[idx];
    var langName = '';
    var highlighted;
    var e = unescape ? unescapeEntities : escapeHtml;
    if (token.info) {
      langName = e(token.info.trim().split(/\s+/g)[0]);
    }
    if (langName === 'javascript' || langName === 'js') {
      highlighted = highlight(token.content, langName) || e(token.content);
    } else {
      highlighted = e(token.content);
    }
    return '\n' + styleOptions.code.open + highlighted + styleOptions.code.close + '\n\n';
  };

  md.renderer.rules.heading_open = function (tokens, idx, options, env) {
    if (tokens[idx].tag === 'h1') {
      return styleOptions.firstHeading.open + '\n';
    }
    return styleOptions.heading.open + '\n';
  };
  md.renderer.rules.heading_close = function (tokens, idx, options, env) {
    if (tokens[idx].tag === 'h1') {
      return styleOptions.firstHeading.close + '\n';
    }
    return styleOptions.heading.close + '\n';
  };


  md.renderer.rules.hr = function (tokens, idx, options /*, env */) {
    return styleOptions.hr.open + hr('-') + styleOptions.hr.close + '\n';
  };


  md.renderer.rules.bullet_list_open   = function () { return ''; };
  md.renderer.rules.bullet_list_close  = function () { return '\n'; };
  md.renderer.rules.list_item_open     = function (tokens, idx /*, options, env */) {
    var next = tokens[idx + 1];
    var bullet = '* ';
    if ((next.type === 'list_item_close') ||
        (next.type === 'paragraph_open' && next.hidden)) {
      if(tokens[idx].order){
        bullet = tokens[idx].order + ' ';
      }
      return tab() + styleOptions.listitem.open + bullet;
    }
  };
  md.renderer.rules.list_item_close    = function (tokens, idx) { 
    return styleOptions.listitem.close + '\n'; 
  };
  
  md.renderer.rules.ordered_list_open  = function (tokens, idx /*, options, env */) {
    var count = 0;
    var item  = 1;
    if (tokens[idx].order > 1) {
      return styleOptions.listItem.open + tokens[idx].order + '\n';
    } else {
      while(tokens[idx + count].type !== 'ordered_list_close') {
        if(tokens[idx + count].type === 'list_item_open') {
          tokens[idx + count].order = item;
          item++;
        }
        count++;
      }
    }
    return '';
  };
  md.renderer.rules.ordered_list_close = function () { 
    return styleOptions.listitem.close + '\n'; 
  };

  md.renderer.rules.paragraph_open = function (tokens, idx, options, env ) {
    return tokens[idx].hidden ? '' : styleOptions.paragraph.open;
  };
  md.renderer.rules.paragraph_close = function (tokens, idx, options, env ) {
    if (tokens[idx].hidden === true) {
      return tokens[idx + 1].type.slice(-5) === 'close' ? '' : '\n';
    }
    return styleOptions.paragraph.close + '\n\n';
  };


  md.renderer.rules.link_open = function (tokens, idx /*, options, env */) {
    return styleOptions.link.open;
  };
  md.renderer.rules.link_close = function (/* tokens, idx, options, env */) {
    return styleOptions.link.close;
  };


  md.renderer.rules.image = function (tokens, idx, options, env, self) {
    var token = tokens[idx];
    var e = unescape ? unescapeEntities : escapeHtml;
    var src = e(token.attrs[token.attrIndex('src')][1]);
    var title = token.title ? ' – ' + e(token.title) : '';
    return styleOptions.link.open + '![' + src + title + '](' + src + ') '+ styleOptions.link.close + '\n';
  };

  md.renderer.rules.strong_open  = function () { return styleOptions.strong.open; };
  md.renderer.rules.strong_close = function () { return styleOptions.strong.close; };


  md.renderer.rules.em_open  = function () { return styleOptions.em.open; };
  md.renderer.rules.em_close = function () { return styleOptions.em.close; };


  md.renderer.rules.s_open  = function () { return styleOptions.del.open; };
  md.renderer.rules.s_close = function () { return styleOptions.del.close; };


  md.renderer.rules.hardbreak = function (tokens, idx, options /*, env */) {
    return '\n';
  };
  md.renderer.rules.softbreak = function (tokens, idx, options /*, env */) {
    return '\n';
  };


  md.renderer.rules.text = function (tokens, idx /*, options, env */) {
    var e = unescape ? unescapeEntities : escapeHtml;
    return e(tokens[idx].content);
  };


  md.renderer.rules.html_block = function (tokens, idx /*, options, env */) {
    return styleOptions.html.open + tokens[idx].content + styleOptions.html.close;
  };
  md.renderer.rules.html_inline = function (tokens, idx /*, options, env */) {
    return styleOptions.html.open + tokens[idx].content + styleOptions.html.close;
  };
  
  //TODO: replace with cli-table 
  
  md.renderer.rules.table_open  = function () { return '<table>\n'; };
  md.renderer.rules.table_close = function () { return '</table>\n'; };
  md.renderer.rules.thead_open  = function () { return '<thead>\n'; };
  md.renderer.rules.thead_close = function () { return '</thead>\n'; };
  md.renderer.rules.tbody_open  = function () { return '<tbody>\n'; };
  md.renderer.rules.tbody_close = function () { return '</tbody>\n'; };
  md.renderer.rules.tr_open     = function () { return '<tr>'; };
  md.renderer.rules.tr_close    = function () { return '</tr>\n'; };
  md.renderer.rules.th_open     = function (tokens, idx /*, options, env */) {
    if (tokens[idx].align) {
      return '<th style="text-align:' + tokens[idx].align + '">';
    }
    return '<th>';
  };
  md.renderer.rules.th_close    = function () { return '</th>'; };
  md.renderer.rules.td_open     = function (tokens, idx /*, options, env */) {
    if (tokens[idx].align) {
      return '<td style="text-align:' + tokens[idx].align + '">';
    }
    return '<td>';
  };
  md.renderer.rules.td_close    = function () { return '</td>'; };

  md.renderer.rules.table       = function () {
    
  };
    
  //TODO: Finish implementation of color blocks and inline colors. 
  // add color block rules
  // colors.concat(backgroundColors).forEach(function(color) {
    /*
    var rule = color_block(color);
    var ruleName = color + '_block';
    md.block.ruler.before('html_block', ruleName, rule);
    md.renderer.rules[ruleName] = function(tokens, idx) {
      return styles[color].open + tokens[idx].content + styles[color].close;
    };
    */
    /*
    var rule = color_inline(color);
    var ruleName = color;
    md.inline.ruler.before('html_inline', ruleName, rule);
    md.renderer.rules[color + '_open'] = function(tokens, idx) {
      console.log(tokens)
      return styles[color].open;
    };
    md.renderer.rules[color + '_close'] = function(tokens, idx) {
      console.log(tokens)
      return styles[color].close;
    };
    */
//  });
  
  // console.log(styles.red.open + 'red text ' + styles.blue.open + 'blue text' + styles.blue.close + 'more red' + styles.red.close)
};

module.exports.compoundStyle = compoundStyle;

function changeToOrdered(text) {
  var i = 1;
  return text.split('\n').reduce(function (acc, line) {
    if (!line) return acc;
    return acc + tab() + (i++) + '.' + line.substring(tab().length + 1) + '\n';
  }, '');
}
function hr(inputHrStr) {
  return (new Array(process.stdout.columns)).join(inputHrStr);
}
function tab(size) {
  size = size || 4;
  return (new Array(size)).join(' ');
}
function unescapeEntities(html) {
  return html
      .replace(/&amp;/g, '&')
      .replace(/&lt;/g, '<')
      .replace(/&gt;/g, '>')
      .replace(/&quot;/g, '"')
      .replace(/&#39;/g, "'");
}
function identity (str) {
  return str;
}
function generateTableRow(text) {
  if (!text) return [];
  var lines = text.split('\n');

  var data = [];
  lines.forEach(function (line) {
    if (!line) return;
    var parsed = line.replace(TABLE_ROW_WRAP_REGEXP, '').split(TABLE_CELL_SPLIT);

    data.push(parsed.splice(0, parsed.length - 1));
  });
  return data;
}

function escapeRegExp(str) {
  return str.replace(/[\-\[\]\/\{\}\(\)\*\+\?\.\\\^\$\|]/g, "\\$&");
}