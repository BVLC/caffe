/*

Javascript doctest runner
Copyright 2006-2010 Ian Bicking

This program is free software; you can redistribute it and/or modify it under
the terms of the MIT License.

*/


function doctest(verbosity/*default=0*/, elements/*optional*/,
                 outputId/*optional*/) {
  var output = document.getElementById(outputId || 'doctestOutput');
  var reporter = new doctest.Reporter(output, verbosity || 0);
  if (elements) {
      if (typeof elements == 'string') {
        // Treat it as an id
        elements = [document.getElementById(elementId)];
      }
      if (! elements.length) {
          throw('No elements');
      }
      var suite = new doctest.TestSuite(elements, reporter);
  } else {
      var els = doctest.getElementsByTagAndClassName('pre', 'doctest');
      var suite = new doctest.TestSuite(els, reporter);
  }
  suite.run();
}

doctest.runDoctest = function (el, reporter) {
  logDebug('Testing element', el);
  reporter.startElement(el);
  if (el === null) {
    throw('runDoctest() with a null element');
  }
  var parsed = new doctest.Parser(el);
  var runner = new doctest.JSRunner(reporter);
  runner.runParsed(parsed);
};

doctest.TestSuite = function (els, reporter) {
  if (this === window) {
    throw('you forgot new!');
  }
  this.els = els;
  this.parsers = [];
  for (var i=0; i<els.length; i++) {
    this.parsers.push(new doctest.Parser(els[i]));
  }
  this.reporter = reporter;
};

doctest.TestSuite.prototype.run = function (ctx) {
  if (! ctx) {
    ctx = new doctest.Context(this);
  }
  if (! ctx.runner ) {
    ctx.runner = new doctest.JSRunner(this.reporter);
  }
  return ctx.run();
};

// FIXME: should this just be part of TestSuite?
doctest.Context = function (testSuite) {
  if (this === window) {
    throw('You forgot new!');
  }
  this.testSuite = testSuite;
  this.runner = null;
};

doctest.Context.prototype.run = function (parserIndex) {
  var self = this;
  parserIndex = parserIndex || 0;
  if (parserIndex >= this.testSuite.parsers.length) {
    logInfo('All examples from all sections tested');
    this.runner.reporter.finish();
    return;
  }
  logInfo('Testing example ' + (parserIndex+1) + ' of '
           + this.testSuite.parsers.length);
  var runNext = function () {
    self.run(parserIndex+1);
  };
  this.runner.runParsed(this.testSuite.parsers[parserIndex], 0, runNext);
};

doctest.Parser = function (el) {
  if (this === window) {
    throw('you forgot new!');
  }
  if (! el) {
    throw('Bad call to doctest.Parser');
  }
  if (el.getAttribute('parsed-id')) {
    var examplesID = el.getAttribute('parsed-id');
    if (doctest._allExamples[examplesID]) {
      this.examples = doctest._allExamples[examplesID];
      return;
    }
  }
  var newHTML = document.createElement('span');
  newHTML.className = 'doctest-example-set';
  var examplesID = doctest.genID('example-set');
  newHTML.setAttribute('id', examplesID);
  el.setAttribute('parsed-id', examplesID);
  var text = doctest.getText(el);
  var lines = text.split(/(?:\r\n|\r|\n)/);
  this.examples = [];
  var example_lines = [];
  var output_lines = [];
  for (var i=0; i<lines.length; i++) {
    var line = lines[i];
    if (/^[$]/.test(line)) {
      if (example_lines.length) {
        var ex = new doctest.Example(example_lines, output_lines);
        this.examples.push(ex);
        newHTML.appendChild(ex.createSpan());
      }
      example_lines = [];
      output_lines = [];
      line = line.substr(1).replace(/ *$/, '').replace(/^ /, '');
      example_lines.push(line);
    } else if (/^>/.test(line)) {
      if (! example_lines.length) {
        throw('Bad example: '+doctest.repr(line)+'\n'
              +'> line not preceded by $');
      }
      line = line.substr(1).replace(/ *$/, '').replace(/^ /, '');
      example_lines.push(line);
    } else {
      output_lines.push(line);
    }
  }
  if (example_lines.length) {
    var ex = new doctest.Example(example_lines, output_lines);
    this.examples.push(ex);
    newHTML.appendChild(ex.createSpan());
  }
  el.innerHTML = '';
  el.appendChild(newHTML);
  doctest._allExamples[examplesID] = this.examples;
};

doctest._allExamples = {};

doctest.Example = function (example, output) {
  if (this === window) {
    throw('you forgot new!');
  }
  this.example = example.join('\n');
  this.output = output.join('\n');
  this.htmlID = null;
  this.detailID = null;
};

doctest.Example.prototype.createSpan = function () {
  var id = doctest.genID('example');
  var span = document.createElement('span');
  span.className = 'doctest-example';
  span.setAttribute('id', id);
  this.htmlID = id;
  var exampleSpan = document.createElement('span');
  exampleSpan.className = 'doctest-example-code';
  var exampleLines = this.example.split(/\n/);
  for (var i=0; i<exampleLines.length; i++) {
    var promptSpan = document.createElement('span');
    promptSpan.className = 'doctest-example-prompt';
    promptSpan.innerHTML = i == 0 ? '$ ' : '&gt; ';
    exampleSpan.appendChild(promptSpan);
    var lineSpan = document.createElement('span');
    lineSpan.className = 'doctest-example-code-line';
    lineSpan.appendChild(document.createTextNode(doctest.rstrip(exampleLines[i])));
    exampleSpan.appendChild(lineSpan);
    exampleSpan.appendChild(document.createTextNode('\n'));
  }
  span.appendChild(exampleSpan);
  var outputSpan = document.createElement('span');
  outputSpan.className = 'doctest-example-output';
  outputSpan.appendChild(document.createTextNode(this.output));
  span.appendChild(outputSpan);
  span.appendChild(document.createTextNode('\n'));
  return span;
};

doctest.Example.prototype.markExample = function (name, detail) {
  if (! this.htmlID) {
    return;
  }
  if (this.detailID) {
    var el = document.getElementById(this.detailID);
    el.parentNode.removeChild(el);
    this.detailID = null;
  }
  var span = document.getElementById(this.htmlID);
  span.className = span.className.replace(/ doctest-failure/, '')
                   .replace(/ doctest-success/, '')
                   + ' ' + name;
  if (detail) {
    this.detailID = doctest.genID('doctest-example-detail');
    var detailSpan = document.createElement('span');
    detailSpan.className = 'doctest-example-detail';
    detailSpan.setAttribute('id', this.detailID);
    detailSpan.appendChild(document.createTextNode(detail));
    span.appendChild(detailSpan);
  }
};

doctest.Reporter = function (container, verbosity) {
  if (this === window) {
    throw('you forgot new!');
  }
  if (! container) {
    throw('No container passed to doctest.Reporter');
  }
  this.container = container;
  this.verbosity = verbosity;
  this.success = 0;
  this.failure = 0;
  this.elements = 0;
};

doctest.Reporter.prototype.startElement = function (el) {
  this.elements += 1;
  logDebug('Adding element', el);
};

doctest.Reporter.prototype.reportSuccess = function (example, output) {
  if (this.verbosity > 0) {
    if (this.verbosity > 1) {
      this.write('Trying:\n');
      this.write(this.formatOutput(example.example));
      this.write('Expecting:\n');
      this.write(this.formatOutput(example.output));
      this.write('ok\n');
    } else {
      this.writeln(example.example + ' ... passed!');
    }
  }
  this.success += 1;
  if ((example.output.indexOf('...') >= 0
       || example.output.indexOf('?') >= 0)
      && output) {
    example.markExample('doctest-success', 'Output:\n' + output);
  } else {
    example.markExample('doctest-success');
  }
};

doctest.Reporter.prototype.reportFailure = function (example, output) {
  this.write('Failed example:\n');
  this.write('<span style="color: #00f"><a href="#'
             + example.htmlID
             + '" class="doctest-failure-link" title="Go to example">'
             + this.formatOutput(example.example)
             +'</a></span>');
  this.write('Expected:\n');
  this.write(this.formatOutput(example.output));
  this.write('Got:\n');
  this.write(this.formatOutput(output));
  this.failure += 1;
  example.markExample('doctest-failure', 'Actual output:\n' + output);
};

doctest.Reporter.prototype.finish = function () {
  this.writeln((this.success+this.failure)
               + ' tests in ' + this.elements + ' items.');
  if (this.failure) {
    var color = '#f00';
  } else {
    var color = '#0f0';
  }
  this.writeln('<span class="passed">' + this.success + '</span> tests of '
               + '<span class="total">' + (this.success+this.failure) + '</span> passed, '
               + '<span class="failed" style="color: '+color+'">'
               + this.failure + '</span> failed.');
};

doctest.Reporter.prototype.writeln = function (text) {
  this.write(text + '\n');
};

doctest.Reporter.prototype.write = function (text) {
  var leading = /^[ ]*/.exec(text)[0];
  text = text.substr(leading.length);
  for (var i=0; i<leading.length; i++) {
    text = String.fromCharCode(160)+text;
  }
  text = text.replace(/\n/g, '<br>');
  this.container.innerHTML += text;
};

doctest.Reporter.prototype.formatOutput = function (text) {
  if (! text) {
    return '    <span style="color: #999">(nothing)</span>\n';
  }
  var lines = text.split(/\n/);
  var output = '';
  for (var i=0; i<lines.length; i++) {
    output += '    '+doctest.escapeSpaces(doctest.escapeHTML(lines[i]))+'\n';
  }
  return output;
};

doctest.JSRunner = function (reporter) {
  if (this === window) {
    throw('you forgot new!');
  }
  this.reporter = reporter;
};

doctest.JSRunner.prototype.runParsed = function (parsed, index, finishedCallback) {
  var self = this;
  index = index || 0;
  if (index >= parsed.examples.length) {
    if (finishedCallback) {
      finishedCallback();
    }
    return;
  }
  var example = parsed.examples[index];
  if (typeof example == 'undefined') {
    throw('Undefined example (' + (index+1) + ' of ' + parsed.examples.length + ')');
  }
  doctest._waitCond = null;
  this.run(example);
  var finishThisRun = function () {
    self.finishRun(example);
    if (doctest._AbortCalled) {
      // FIXME: I need to find a way to make this more visible:
      logWarn('Abort() called');
      return;
    }
    self.runParsed(parsed, index+1, finishedCallback);
  };
  if (doctest._waitCond !== null) {
    if (typeof doctest._waitCond == 'number') {
      var condition = null;
      var time = doctest._waitCond;
      var maxTime = null;
    } else {
      var condition = doctest._waitCond;
      // FIXME: shouldn't be hard-coded
      var time = 100;
      var maxTime = doctest._waitTimeout || doctest.defaultTimeout;
    }
    var start = (new Date()).getTime();
    var timeoutFunc = function () {
      if (condition === null
          || condition()) {
        finishThisRun();
      } else {
        // Condition not met, try again soon...
        if ((new Date()).getTime() - start > maxTime) {
          // Time has run out
          var msg = 'Error: wait(' + repr(condition) + ') has timed out';
          writeln(msg);
          logDebug(msg);
          logDebug('Timeout after ' + ((new Date()).getTime() - start)
                   + ' milliseconds');
          finishThisRun();
          return;
        }
        setTimeout(timeoutFunc, time);
      }
    };
    setTimeout(timeoutFunc, time);
  } else {
    finishThisRun();
  }
};

doctest.formatTraceback = function (e, skipFrames) {
  skipFrames = skipFrames || 0;
  var lines = [];
  if (typeof e == 'undefined' || !e) {
    var caughtErr = null;
    try {
      (null).foo;
    } catch (caughtErr) {
      e = caughtErr;
    }
    skipFrames++;
  }
  if (e.stack) {
    var stack = e.stack.split('\n');
    for (var i=skipFrames; i<stack.length; i++) {
      if (stack[i] == '@:0' || ! stack[i]) {
        continue;
      }
      if (stack[i].indexOf('@') == -1) {
        lines.push(stack[i]);
        continue;
      }
      var parts = stack[i].split('@');
      var context = parts[0];
      parts = parts[1].split(':');
      var filename = parts[parts.length-2].split('/');
      filename = filename[filename.length-1];
      var lineno = parts[parts.length-1];
      context = context.replace('\\n', '\n');
      if (context != '' && filename != 'doctest.js') {
        lines.push('  ' + context + ' -> ' + filename + ':' + lineno);
      }
    }
  }
  if (lines.length) {
    return lines;
  } else {
    return null;
  }
};

doctest.logTraceback = function (e, skipFrames) {
  var tracebackLines = doctest.formatTraceback(e, skipFrames);
  if (! tracebackLines) {
    return;
  }
  for (var i=0; i<tracebackLines.length; i++) {
    logDebug(tracebackLines[i]);
  }
};

doctest.JSRunner.prototype.run = function (example) {
  this.capturer = new doctest.OutputCapturer();
  this.capturer.capture();
  try {
    var result = doctest.eval(example.example);
  } catch (e) {
    var tracebackLines = doctest.formatTraceback(e);
    writeln('Error: ' + (e.message || e));
    var result = null;
    logWarn('Error in expression: ' + example.example);
    logDebug('Traceback for error', e);
    if (tracebackLines) {
      for (var i=0; i<tracebackLines.length; i++) {
        logDebug(tracebackLines[i]);
      }
    }
    if (e instanceof Abort) {
      throw e;
    }
  }
  if (typeof result != 'undefined'
      && result !== null
      && example.output) {
    writeln(doctest.repr(result));
  }
};

doctest._AbortCalled = false;

doctest.Abort = function (message) {
  if (this === window) {
    return new Abort(message);
  }
  this.message = message;
  // We register this so Abort can be raised in an async call:
  doctest._AbortCalled = true;
};

doctest.Abort.prototype.toString = function () {
  return this.message;
};

if (typeof Abort == 'undefined') {
  Abort = doctest.Abort;
}

doctest.JSRunner.prototype.finishRun = function(example) {
  this.capturer.stopCapture();
  var success = this.checkResult(this.capturer.output, example.output);
  if (success) {
    this.reporter.reportSuccess(example, this.capturer.output);
  } else {
    this.reporter.reportFailure(example, this.capturer.output);
    logDebug('Failure: '+doctest.repr(example.output)
             +' != '+doctest.repr(this.capturer.output));
    if (location.href.search(/abort/) != -1) {
      doctest.Abort('abort on first failure');
    }
  }
};

doctest.JSRunner.prototype.checkResult = function (got, expected) {
  // Make sure trailing whitespace doesn't matter:
  got = got.replace(/ +\n/, '\n');
  expected = expected.replace(/ +\n/, '\n');
  got = got.replace(/[ \n\r]*$/, '') + '\n';
  expected = expected.replace(/[ \n\r]*$/, '') + '\n';
  if (expected == '...\n') {
    return true;
  }
  expected = RegExp.escape(expected);
  // Note: .* doesn't match newlines, [^] doesn't work on IE
  expected = '^' + expected.replace(/\\\.\\\.\\\./g, "[\\S\\s\\r\\n]*") + '$';
  expected = expected.replace(/\\\?/g, "[a-zA-Z0-9_.]+");
  expected = expected.replace(/[ \t]+/g, " +");
  expected = expected.replace(/\n/g, '\\n');
  var re = new RegExp(expected);
  var result = got.search(re) != -1;
  if (! result) {
    if (doctest.strip(got).split('\n').length > 1) {
      // If it's only one line it's not worth showing this
      var check = this.showCheckDifference(got, expected);
      logWarn('Mismatch of output (line-by-line comparison follows)');
      for (var i=0; i<check.length; i++) {
        logDebug(check[i]);
      }
    }
  }
  return result;
};

doctest.JSRunner.prototype.showCheckDifference = function (got, expectedRegex) {
  if (expectedRegex.charAt(0) != '^') {
    throw 'Unexpected regex, no leading ^';
  }
  if (expectedRegex.charAt(expectedRegex.length-1) != '$') {
    throw 'Unexpected regex, no trailing $';
  }
  expectedRegex = expectedRegex.substr(1, expectedRegex.length-2);
  // Technically this might not be right, but this is all a heuristic:
  var expectedRegex = expectedRegex.replace(/\(\?:\.\|\[\\r\\n\]\)\*/g, '...');
  var expectedLines = expectedRegex.split('\\n');
  for (var i=0; i<expectedLines.length; i++) {
    expectedLines[i] = expectedLines[i].replace(/\.\.\./g, '(?:.|[\r\n])*');
  }
  var gotLines = got.split('\n');
  var result = [];
  var totalLines = expectedLines.length > gotLines.length ?
    expectedLines.length : gotLines.length;
  function displayExpectedLine(line) {
    return line;
    line = line.replace(/\[a-zA-Z0-9_.\]\+/g, '?');
    line = line.replace(/ \+/g, ' ');
    line = line.replace(/\(\?:\.\|\[\\r\\n\]\)\*/g, '...');
    // FIXME: also unescape values? e.g., * became \*
    return line;
  }
  for (var i=0; i<totalLines; i++) {
    if (i >= expectedLines.length) {
      result.push('got extra line: ' + repr(gotLines[i]));
      continue;
    } else if (i >= gotLines.length) {
      result.push('expected extra line: ' + displayExpectedLine(expectedLines[i]));
      continue;
    }
    var gotLine = gotLines[i];
    try {
      var expectRE = new RegExp('^' + expectedLines[i] + '$');
    } catch (e) {
      result.push('regex match failed: ' + repr(gotLine) + ' ('
            + expectedLines[i] + ')');
      continue;
    }
    if (gotLine.search(expectRE) != -1) {
      result.push('match: ' + repr(gotLine));
    } else {
      result.push('no match: ' + repr(gotLine) + ' ('
            + displayExpectedLine(expectedLines[i]) + ')');
    }
  }
  return result;
};

// Should I really be setting this on RegExp?
RegExp.escape = function (text) {
  if (!arguments.callee.sRE) {
    var specials = [
      '/', '.', '*', '+', '?', '|',
      '(', ')', '[', ']', '{', '}', '\\'
    ];
    arguments.callee.sRE = new RegExp(
      '(\\' + specials.join('|\\') + ')', 'g'
    );
  }
  return text.replace(arguments.callee.sRE, '\\$1');
};

doctest.OutputCapturer = function () {
  if (this === window) {
    throw('you forgot new!');
  }
  this.output = '';
};

doctest._output = null;

doctest.OutputCapturer.prototype.capture = function () {
  doctest._output = this;
};

doctest.OutputCapturer.prototype.stopCapture = function () {
  doctest._output = null;
};

doctest.OutputCapturer.prototype.write = function (text) {
  if (typeof text == 'string') {
    this.output += text;
  } else {
    this.output += repr(text);
  }
};

// Used to create unique IDs:
doctest._idGen = 0;

doctest.genID = function (prefix) {
  prefix = prefix || 'generic-doctest';
  var id = doctest._idGen++;
  return prefix + '-' + doctest._idGen;
};

doctest.writeln = function () {
  for (var i=0; i<arguments.length; i++) {
    write(arguments[i]);
    if (i) {
      write(' ');
    }
  }
  write('\n');
};

if (typeof writeln == 'undefined') {
  writeln = doctest.writeln;
}

doctest.write = function (text) {
  if (doctest._output !== null) {
    doctest._output.write(text);
  } else {
    log(text);
  }
};

if (typeof write == 'undefined') {
  write = doctest.write;
}

doctest._waitCond = null;

function wait(conditionOrTime, hardTimeout) {
  // FIXME: should support a timeout even with a condition
  if (typeof conditionOrTime == 'undefined'
      || conditionOrTime === null) {
    // same as wait-some-small-amount-of-time
    conditionOrTime = 0;
  }
  doctest._waitCond = conditionOrTime;
  doctest._waitTimeout = hardTimeout;
};

doctest.wait = wait;

doctest.assert = function (expr, statement) {
  if (typeof expr == 'string') {
    if (! statement) {
      statement = expr;
    }
    expr = doctest.eval(expr);
  }
  if (! expr) {
    throw('AssertionError: '+statement);
  }
};

if (typeof assert == 'undefined') {
  assert = doctest.assert;
}

doctest.getText = function (el) {
  if (! el) {
    throw('You must pass in an element');
  }
  var text = '';
  for (var i=0; i<el.childNodes.length; i++) {
    var sub = el.childNodes[i];
    if (sub.nodeType == 3) {
      // TEXT_NODE
      text += sub.nodeValue;
    } else if (sub.childNodes) {
      text += doctest.getText(sub);
    }
  }
  return text;
};

doctest.reload = function (button/*optional*/) {
  if (button) {
    button.innerHTML = 'reloading...';
    button.disabled = true;
  }
  location.reload();
};

/* Taken from MochiKit, with an addition to print objects */
doctest.repr = function (o, indentString, maxLen) {
    indentString = indentString || '';
    if (doctest._reprTracker === null) {
      var iAmTheTop = true;
      doctest._reprTracker = [];
    } else {
      var iAmTheTop = false;
    }
    try {
      if (doctest._reprTrackObj(o)) {
        return '..recursive..';
      }
      if (maxLen === undefined) {
        maxLen = 120;
      }
      if (typeof o == 'undefined') {
          return 'undefined';
      } else if (o === null) {
          return "null";
      }
      try {
          if (typeof(o.__repr__) == 'function') {
              return o.__repr__(indentString, maxLen);
          } else if (typeof(o.repr) == 'function' && o.repr != arguments.callee) {
              return o.repr(indentString, maxLen);
          }
          for (var i=0; i<doctest.repr.registry.length; i++) {
              var item = doctest.repr.registry[i];
              if (item[0](o)) {
                  return item[1](o, indentString, maxLen);
              }
          }
      } catch (e) {
          if (typeof(o.NAME) == 'string' && (
                  o.toString == Function.prototype.toString ||
                      o.toString == Object.prototype.toString)) {
              return o.NAME;
          }
      }
      try {
          var ostring = (o + "");
          if (ostring == '[object Object]' || ostring == '[object]') {
            ostring = doctest.objRepr(o, indentString, maxLen);
          }
      } catch (e) {
          return "[" + typeof(o) + "]";
      }
      if (typeof(o) == "function") {
          var ostring = ostring.replace(/^\s+/, "").replace(/\s+/g, " ");
          var idx = ostring.indexOf("{");
          if (idx != -1) {
              ostring = ostring.substr(o, idx) + "{...}";
          }
      }
      return ostring;
    } finally {
      if (iAmTheTop) {
        doctest._reprTracker = null;
      }
    }
};

doctest._reprTracker = null;

doctest._reprTrackObj = function (obj) {
  if (typeof obj != 'object') {
    return false;
  }
  for (var i=0; i<doctest._reprTracker.length; i++) {
    if (doctest._reprTracker[i] === obj) {
      return true;
    }
  }
  doctest._reprTracker.push(obj);
  return false;
};

doctest._reprTrackSave = function () {
  return doctest._reprTracker.length-1;
};

doctest._reprTrackRestore = function (point) {
  doctest._reprTracker.splice(point, doctest._reprTracker.length - point);
};

doctest._sortedKeys = function (obj) {
  var keys = [];
  for (var i in obj) {
    // FIXME: should I use hasOwnProperty?
    if (typeof obj.prototype == 'undefined'
        || obj[i] !== obj.prototype[i]) {
      keys.push(i);
    }
  }
  keys.sort();
  return keys;
};

doctest.objRepr = function (obj, indentString, maxLen) {
  var restorer = doctest._reprTrackSave();
  var ostring = '{';
  var keys = doctest._sortedKeys(obj);
  for (var i=0; i<keys.length; i++) {
    if (ostring != '{') {
      ostring += ', ';
    }
    ostring += keys[i] + ': ' + doctest.repr(obj[keys[i]], indentString, maxLen);
  }
  ostring += '}';
  if (ostring.length > (maxLen - indentString.length)) {
    doctest._reprTrackRestore(restorer);
    return doctest.multilineObjRepr(obj, indentString, maxLen);
  }
  return ostring;
};

doctest.multilineObjRepr = function (obj, indentString, maxLen) {
  var keys = doctest._sortedKeys(obj);
  var ostring = '{\n';
  for (var i=0; i<keys.length; i++) {
    ostring += indentString + '  ' + keys[i] + ': ';
    ostring += doctest.repr(obj[keys[i]], indentString+'  ', maxLen);
    if (i != keys.length - 1) {
      ostring += ',';
    }
    ostring += '\n';
  }
  ostring += indentString + '}';
  return ostring;
};

doctest.arrayRepr = function (obj, indentString, maxLen) {
  var restorer = doctest._reprTrackSave();
  var s = "[";
  for (var i=0; i<obj.length; i++) {
    s += doctest.repr(obj[i], indentString, maxLen);
    if (i != obj.length-1) {
      s += ", ";
    }
  }
  s += "]";
  if (s.length > (maxLen + indentString.length)) {
    doctest._reprTrackRestore(restorer);
    return doctest.multilineArrayRepr(obj, indentString, maxLen);
  }
  return s;
};

doctest.multilineArrayRepr = function (obj, indentString, maxLen) {
  var s = "[\n";
  for (var i=0; i<obj.length; i++) {
    s += indentString + '  ' + doctest.repr(obj[i], indentString+'  ', maxLen);
    if (i != obj.length - 1) {
      s += ',';
    }
    s += '\n';
  }
  s += indentString + ']';
  return s;
};

doctest.xmlRepr = function (doc, indentString) {
  var i;
  if (doc.nodeType == doc.DOCUMENT_NODE) {
    return doctest.xmlRepr(doc.childNodes[0], indentString);
  }
  indentString = indentString || '';
  var s = indentString + '<' + doc.tagName;
  var attrs = [];
  if (doc.attributes && doc.attributes.length) {
    for (i=0; i<doc.attributes.length; i++) {
      attrs.push(doc.attributes[i].nodeName);
    }
    attrs.sort();
    for (i=0; i<attrs.length; i++) {
      s += ' ' + attrs[i] + '="';
      var value = doc.getAttribute(attrs[i]);
      value = value.replace('&', '&amp;');
      value = value.replace('"', '&quot;');
      s += value;
      s += '"';
    }
  }
  if (! doc.childNodes.length) {
    s += ' />';
    return s;
  } else {
    s += '>';
  }
  var hasNewline = false;
  for (i=0; i<doc.childNodes.length; i++) {
    var el = doc.childNodes[i];
    if (el.nodeType == doc.TEXT_NODE) {
      s += doctest.strip(el.textContent);
    } else {
      if (! hasNewline) {
        s += '\n';
        hasNewline = true;
      }
      s += doctest.xmlRepr(el, indentString + '  ');
      s += '\n';
    }
  }
  if (hasNewline) {
    s += indentString;
  }
  s += '</' + doc.tagName + '>';
  return s;
};

doctest.repr.registry = [
    [function (o) {
         return typeof o == 'string';},
     function (o) {
         o = '"' + o.replace(/([\"\\])/g, '\\$1') + '"';
         o = o.replace(/[\f]/g, "\\f")
         .replace(/[\b]/g, "\\b")
         .replace(/[\n]/g, "\\n")
         .replace(/[\t]/g, "\\t")
         .replace(/[\r]/g, "\\r");
         return o;
     }],
    [function (o) {
         return typeof o == 'number';},
     function (o) {
         return o + "";
     }],
    [function (o) {
          return (typeof o == 'object' && o.xmlVersion);
     },
     doctest.xmlRepr],
    [function (o) {
         var typ = typeof o;
         if ((typ != 'object' && ! (type == 'function' && typeof o.item == 'function')) ||
             o === null ||
             typeof o.length != 'number' ||
             o.nodeType === 3) {
             return false;
         }
         return true;
     },
     doctest.arrayRepr
     ]];

doctest.objDiff = function (orig, current) {
  var result = {
    added: {},
    removed: {},
    changed: {},
    same: {}
  };
  for (var i in orig) {
    if (! (i in current)) {
      result.removed[i] = orig[i];
    } else if (orig[i] !== current[i]) {
      result.changed[i] = [orig[i], current[i]];
    } else {
      result.same[i] = orig[i];
    }
  }
  for (i in current) {
    if (! (i in orig)) {
      result.added[i] = current[i];
    }
  }
  return result;
};

doctest.writeDiff = function (orig, current, indentString) {
  if (typeof orig != 'object' || typeof current != 'object') {
    writeln(indentString + repr(orig, indentString) + ' -> ' + repr(current, indentString));
    return;
  }
  indentString = indentString || '';
  var diff = doctest.objDiff(orig, current);
  var i, keys;
  var any = false;
  keys = doctest._sortedKeys(diff.added);
  for (i=0; i<keys.length; i++) {
    any = true;
    writeln(indentString + '+' + keys[i] + ': '
            + repr(diff.added[keys[i]], indentString));
  }
  keys = doctest._sortedKeys(diff.removed);
  for (i=0; i<keys.length; i++) {
    any = true;
    writeln(indentString + '-' + keys[i] + ': '
            + repr(diff.removed[keys[i]], indentString));
  }
  keys = doctest._sortedKeys(diff.changed);
  for (i=0; i<keys.length; i++) {
    any = true;
    writeln(indentString + keys[i] + ': '
            + repr(diff.changed[keys[i]][0], indentString)
            + ' -> '
            + repr(diff.changed[keys[i]][1], indentString));
  }
  if (! any) {
    writeln(indentString + '(no changes)');
  }
};

doctest.objectsEqual = function (ob1, ob2) {
  var i;
  if (typeof ob1 != 'object' || typeof ob2 != 'object') {
    return ob1 === ob2;
  }
  for (i in ob1) {
    if (ob1[i] !== ob2[i]) {
      return false;
    }
  }
  for (i in ob2) {
    if (! (i in ob1)) {
      return false;
    }
  }
  return true;
};

doctest.getElementsByTagAndClassName = function (tagName, className, parent/*optional*/) {
    parent = parent || document;
    var els = parent.getElementsByTagName(tagName);
    var result = [];
    var regexes = [];
    if (typeof className == 'string') {
      className = [className];
    }
    for (var i=0; i<className.length; i++) {
      regexes.push(new RegExp("\\b" + className[i] + "\\b"));
    }
    for (i=0; i<els.length; i++) {
      var el = els[i];
      if (el.className) {
        var passed = true;
        for (var j=0; j<regexes.length; j++) {
          if (el.className.search(regexes[j]) == -1) {
            passed = false;
            break;
          }
        }
        if (passed) {
          result.push(el);
        }
      }
    }
    return result;
};

doctest.strip = function (str) {
    str = str + "";
    return str.replace(/\s+$/, "").replace(/^\s+/, "");
};

doctest.rstrip = function (str) {
  str = str + "";
  return str.replace(/\s+$/, "");
};

doctest.escapeHTML = function (s) {
    return s.replace(/&/g, '&amp;')
    .replace(/\"/g, "&quot;")
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
};

doctest.escapeSpaces = function (s) {
  return s.replace(/  /g, '&nbsp; ');
};

doctest.extend = function (obj, extendWith) {
    for (i in extendWith) {
        obj[i] = extendWith[i];
    }
    return obj;
};

doctest.extendDefault = function (obj, extendWith) {
    for (i in extendWith) {
        if (typeof obj[i] == 'undefined') {
            obj[i] = extendWith[i];
        }
    }
    return obj;
};

if (typeof repr == 'undefined') {
    repr = doctest.repr;
}

doctest._consoleFunc = function (attr) {
  if (typeof window.console != 'undefined'
      && typeof window.console[attr] != 'undefined') {
    if (typeof console[attr].apply === 'function') {
      result = function() {
        console[attr].apply(console, arguments);
      };
    } else {
      result = console[attr];
    }
  } else {
    result = function () {
      // FIXME: do something
    };
  }
  return result;
};

if (typeof log == 'undefined') {
  log = doctest._consoleFunc('log');
}

if (typeof logDebug == 'undefined') {
  logDebug = doctest._consoleFunc('log');
}

if (typeof logInfo == 'undefined') {
  logInfo = doctest._consoleFunc('info');
}

if (typeof logWarn == 'undefined') {
  logWarn = doctest._consoleFunc('warn');
}

doctest.eval = function () {
  return window.eval.apply(window, arguments);
};

doctest.useCoffeeScript = function (options) {
  options = options || {};
  options.bare = true;
  options.globals = true;
  if (! options.fileName) {
    options.fileName = 'repl';
  }
  if (typeof CoffeeScript == 'undefined') {
    doctest.logWarn('coffee-script.js is not included');
    throw 'coffee-script.js is not included';
  }
  doctest.eval = function (code) {
    var src = CoffeeScript.compile(code, options);
    logDebug('Compiled code to:', src);
    return window.eval(src);
  };
};

doctest.autoSetup = function (parent) {
  var tags = doctest.getElementsByTagAndClassName('div', 'test', parent);
  // First we'll make sure everything has an ID
  var tagsById = {};
  for (var i=0; i<tags.length; i++) {
    var tagId = tags[i].getAttribute('id');
    if (! tagId) {
      tagId = 'test-' + (++doctest.autoSetup._idCount);
      tags[i].setAttribute('id', tagId);
    }
    // FIXME: test uniqueness here, warn
    tagsById[tagId] = tags[i];
  }
  // Then fill in the labels
  for (i=0; i<tags.length; i++) {
    var el = document.createElement('span');
    el.className = 'test-id';
    var anchor = document.createElement('a');
    anchor.setAttribute('href', '#' + tags[i].getAttribute('id'));
    anchor.appendChild(document.createTextNode(tags[i].getAttribute('id')));
    var button = document.createElement('button');
    button.innerHTML = 'test';
    button.setAttribute('type', 'button');
    button.setAttribute('test-id', tags[i].getAttribute('id'));
    button.onclick = function () {
      location.hash = '#' + this.getAttribute('test-id');
      location.reload();
    };
    el.appendChild(anchor);
    el.appendChild(button);
    tags[i].insertBefore(el, tags[i].childNodes[0]);
  }
  // Lastly, create output areas in each section
  for (i=0; i<tags.length; i++) {
    var outEl = doctest.getElementsByTagAndClassName('pre', 'output', tags[i]);
    if (! outEl.length) {
      outEl = document.createElement('pre');
      outEl.className = 'output';
      outEl.setAttribute('id', tags[i].getAttribute('id') + '-output');
    }
  }
  if (location.hash.length > 1) {
    // This makes the :target CSS work, since if the hash points to an
    // element whose id has just been added, it won't be noticed
    location.hash = location.hash;
  }
  var output = document.getElementById('doctestOutput');
  if (! tags.length) {
    tags = document.getElementsByTagName('body');
  }
  if (! output) {
    output = document.createElement('pre');
    output.setAttribute('id', 'doctestOutput');
    output.className = 'output';
    tags[0].parentNode.insertBefore(output, tags[0]);
  }
  var reloader = document.getElementById('doctestReload');
  if (! reloader) {
    reloader = document.createElement('button');
    reloader.setAttribute('type', 'button');
    reloader.setAttribute('id', 'doctest-testall');
    reloader.innerHTML = 'test all';
    reloader.onclick = function () {
      location.hash = '#doctest-testall';
      location.reload();
    };
    output.parentNode.insertBefore(reloader, output);
  }
};

doctest.autoSetup._idCount = 0;

doctest.Spy = function (name, options, extraOptions) {
  var self;
  if (doctest.spies[name]) {
     self = doctest.spies[name];
     if (! options && ! extraOptions) {
       return self;
     }
  } else {
    self = function () {
      return self.func.apply(this, arguments);
    };
  }
  name = name || 'spy';
  options = options || {};
  if (typeof options == 'function') {
    options = {applies: options};
  }
  if (extraOptions) {
    doctest.extendDefault(options, extraOptions);
  }
  doctest.extendDefault(options, doctest.defaultSpyOptions);
  self._name = name;
  self.options = options;
  self.called = false;
  self.calledWait = false;
  self.args = null;
  self.self = null;
  self.argList = [];
  self.selfList = [];
  self.writes = options.writes || false;
  self.returns = options.returns || null;
  self.applies = options.applies || null;
  self.binds = options.binds || null;
  self.throwError = options.throwError || null;
  self.ignoreThis = options.ignoreThis || false;
  self.wrapArgs = options.wrapArgs || false;
  self.func = function () {
    self.called = true;
    self.calledWait = true;
    self.args = doctest._argsToArray(arguments);
    self.self = this;
    self.argList.push(self.args);
    self.selfList.push(this);
    // It might be possible to get the caller?
    if (self.writes) {
      writeln(self.formatCall());
    }
    if (self.throwError) {
      throw self.throwError;
    }
    if (self.applies) {
      return self.applies.apply(this, arguments);
    }
    return self.returns;
  };
  self.func.toString = function () {
    return "Spy('" + self._name + "').func";
  };

  // Method definitions:
  self.formatCall = function () {
    var s = '';
    if ((! self.ignoreThis) && self.self !== window && self.self !== self) {
      s += doctest.repr(self.self) + '.';
    }
    s += self._name;
    if (self.args === null) {
      return s + ':never called';
    }
    s += '(';
    for (var i=0; i<self.args.length; i++) {
      if (i) {
        s += ', ';
      }
      if (self.wrapArgs) {
        var maxLen = 10;
      } else {
        var maxLen = undefined;
      }
      s += doctest.repr(self.args[i], '', maxLen);
    }
    s += ')';
    return s;
  };

  self.method = function (name, options, extraOptions) {
    var desc = self._name + '.' + name;
    var newSpy = Spy(desc, options, extraOptions);
    self[name] = self.func[name] = newSpy.func;
    return newSpy;
  };

  self.methods = function (props) {
    for (var i in props) {
      if (props[i] === props.prototype[i]) {
        continue;
      }
      self.method(i, props[i]);
    }
    return self;
  };

  self.wait = function (timeout) {
    var func = function () {
      var value = self.calledWait;
      if (value) {
        self.calledWait = false;
      }
      return value;
    };
    func.repr = function () {
      return 'called:'+repr(self);
    };
    doctest.wait(func, timeout);
  };

  self.repr = function () {
    return "Spy('" + self._name + "')";
  };

  if (options.methods) {
    self.methods(options.methods);
  }
  doctest.spies[name] = self;
  if (options.wait) {
    self.wait();
  }
  return self;
};

doctest._argsToArray = function (args) {
  var array = [];
  for (var i=0; i<args.length; i++) {
    array.push(args[i]);
  }
  return array;
};

Spy = doctest.Spy;

doctest.spies = {};

doctest.defaultTimeout = 2000;

doctest.defaultSpyOptions = {writes: true};

var docTestOnLoad = function () {
  var auto = false;
  if (/\bautodoctest\b/.exec(document.body.className)) {
    doctest.autoSetup();
    auto = true;
  } else {
    logDebug('No autodoctest class on <body>');
  }
  var loc = window.location.search.substring(1);
  if (auto || (/doctestRun/).exec(loc)) {
    var elements = null;
    // FIXME: we need to put the output near the specific test being tested:
    if (location.hash) {
      var el = document.getElementById(location.hash.substr(1));
      if (el) {
        if (/\btest\b/.exec(el.className)) {
          var testEls = doctest.getElementsByTagAndClassName('pre', 'doctest', el);
          elements = doctest.getElementsByTagAndClassName('pre', ['doctest', 'setup']);
          for (var i=0; i<testEls.length; i++) {
            elements.push(testEls[i]);
          }
        }
      }
    }
    doctest(0, elements);
  }
};

if (window.addEventListener) {
    window.addEventListener('load', docTestOnLoad, false);
} else if(window.attachEvent) {
    window.attachEvent('onload', docTestOnLoad);
}

