'use strict';

function isLetter(ch) {
  /*eslint no-bitwise:0*/
  var lc = ch | 0x20; // to lower case
  return (lc >= 0x61/* a */) && (lc <= 0x7a/* z */);
}

function openTagRegex(name) {
  return new RegExp('(?:<' + name + '>)');
}

function closeTagRegex(name) {
  return new RegExp('(?:<\/' + name + '>)');
}

module.exports = function color_inline(name) {
  var token;
  var open_tag = '<' + name + '>';
  var close_tag = '<\/' + name + '>';
  var HTML_TAG_OPEN_RE = openTagRegex(name);
  var HTML_TAG_CLOSE_RE = closeTagRegex(name);
  var HTML_TAG_RE = new RegExp('^(?:' + open_tag + '|' + close_tag + ')');
  return function color_inline(state, silent) {
    var ch, matchopen, matchclose, 
    pos = state.pos,
    max = state.posMax,
    start = state.pos;

    // if (!state.md.options.html) { return false; }
    // console.log('start ' + state.src.charAt(pos), 'end '+state.src.charAt(state.posMax-1))

    if (state.src.charCodeAt(pos) !== 0x3C/* < */ ||
        pos + 2 >= max) {
      return false;
    }

    // Quick fail on second char
    ch = state.src.charCodeAt(pos + 1);
    if (ch !== 0x21/* ! */ &&
        ch !== 0x3F/* ? */ &&
        ch !== 0x2F/* / */ &&
        !isLetter(ch)) {
      return false;
    }
    // console.log(state.src.slice(pos))
    matchopen = state.src.slice(pos).match(HTML_TAG_OPEN_RE);
    matchclose = state.src.slice(pos).match(HTML_TAG_CLOSE_RE);
    console.log('matching '+open_tag,matchopen)
    console.log('----------')
    console.log('matching '+close_tag,matchclose)
    
    if (!matchopen && !matchclose) { return false; }

    if (!silent) {
    /*  state.push({
        type: name + '_inline',
        content: state.src.slice(pos, pos + match[0].length),
        level: state.level
      });
    */
    state.posMax = state.pos;
    state.pos = start;
    state.push({ type: name + '_open', level: state.level++ });
    state.md.inline.tokenize(state);
    state.push({ type: name + '_close', level: state.level--});
    state.pos = state.posMax + 1;
    state.posMax = max;
    }

    return true;
  };
};