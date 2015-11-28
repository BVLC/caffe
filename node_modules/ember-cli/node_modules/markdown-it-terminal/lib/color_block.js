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

module.exports = function color_block(name) {
  var HTML_TAG_OPEN_RE = openTagRegex(name);
  var HTML_TAG_CLOSE_RE = closeTagRegex(name);

  return function(state, startLine, endLine, silent) {
    var ch, match, nextLine, content,
        pos = state.bMarks[startLine],
        max = state.eMarks[startLine],
        shift = state.tShift[startLine];

    pos += shift;

    if (shift > 3 || pos + 2 >= max) { return false; }

    if (state.src.charCodeAt(pos) !== 0x3C/* < */) { return false; }

    ch = state.src.charCodeAt(pos + 1);

    if (ch === 0x21/* ! */ || ch === 0x3F/* ? */) {
      // Directive start / comment start / processing instruction start
      if (silent) { return true; }

    } else if (ch === 0x2F/* / */ || isLetter(ch)) {

      // Probably start or end of tag
      if (ch === 0x2F/* \ */) {
        // closing tag
        match = state.src.slice(pos, max).match(HTML_TAG_CLOSE_RE);
        if (!match) { return false; }
      } else {
        // opening tag
        match = state.src.slice(pos, max).match(HTML_TAG_OPEN_RE);
        if (!match) { return false; }
      }
      
      if (silent) { return true; }

    } else {
      return false;
    }

    // If we are here - we detected HTML block.
    // Let's roll down till empty line (block end).
    nextLine = startLine + 1;
    while (nextLine < state.lineMax && !state.isEmpty(nextLine)) {
      nextLine++;
    }

    content =  state.getLines(startLine, nextLine, 0, true)
      .replace(HTML_TAG_OPEN_RE, '')
      .replace(HTML_TAG_CLOSE_RE, '');

    state.line = nextLine;
    state.tokens.push({
      type: name + '_block',
      level: state.level,
      lines: [ startLine, state.line ],
      content: content
    });

    return true;
  };
};