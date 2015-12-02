var parser = exports;

var ometajs = require('../ometajs'),
    util = require('util'),
    utils = ometajs.utils;

//
// ### function Parser (code)
// #### @code {String} input source code
// Parser's constructor
//
function Parser(code) {
  ometajs.compiler.ast.call(this);

  this.code = code;
  this.lexer = ometajs.lexer.create(code);
  this.state = 'empty';
  this.active = true;

  // Tokens history for lookahead
  this.tokens = [];
  this.pending = [];
}
util.inherits(Parser, ometajs.compiler.ast);

//
// ### function create (code)
// #### @code {String} input source code
// Returns Parser's instance
//
parser.create = function create(code) {
  return new Parser(code);
};

//
// ### function execute ()
// Returns AST of source code
//
Parser.prototype.execute = function execute() {
  while (this.active) {
    var code = [];
    while (this.active && !(this.lookahead('keyword', 'ometa', true) &&
           this.lookahead('space', null, true) &&
           this.lookahead('name', null, true))) {
      while (this.pending.length) {
        this.reject();
      }
      while (this.tokens.length) {
        code.push(this.lexer.stringify(this.expect(null, null, true)));
      }
    };

    if (this.active) {
      while (this.pending.length) {
        this.reject();
      }
    }

    if (code.length > 0) {
      code = code.join('').trim();
      if (code) {
        this.push(['code', code + '\n']);
        this.pop();
      }
    }

    if (this.active) this.parseGrammar();
  }

  return this.result;
};

//
// ### function parseGrammar ()
// Parser's routine
//
Parser.prototype.parseGrammar = function parseGrammar() {
  // ometa %GrammarName%
  if (!this.expect('keyword', 'ometa')) return;
  this.state = 'grammar';
  this.push(['grammar', this.expect('name').value]);

  // Nested grammars, <: %GrammarName%
  this.push(
    this.maybe('punc', '<:') ?
        this.expect('name').value
        :
        null
  );
  this.pop();

  // Prepare for rules
  this.expect('punc', '{');
  this.push([]);

  this.parseUntil('punc', '}', function parseRules() {
    this.parseRule();
  });
  this.accept();

  this.pop();
  this.pop();

  this.state = 'empty';
};

//
// ### function parseRule ()
// Parser's routine
//
Parser.prototype.parseRule = function parseRule() {
  // Handle grammar end
  if (this.maybe('punc', '}')) {
    // Leave rules and grammar
    this.pop();
    this.pop();

    // Wait for next grammar
    return;
  }

  var name;

  // Rule name
  if (name = this.maybe('name')) {
    this.push(['rule', name.value]);
  } else {
    this.push(['rule', null]);
  }
  this.push([]);

  // Parse rule's expressions
  var end = this.parseUntil('punc', ['}', ','], function parseExprs() {
    var end;

    // Parse left expression
    end = this.parseUntil('punc', ['=', ',', '}'], function parseLeft() {
      this.parseExpression();
    });

    // Parse right expression
    if (matchToken(end, 'punc', '=')) {
      this.accept();
      this.push(['choice']);
      this.list('punc', '|', function parseRight() {
        this.push([]);
        this.parseUntil('punc', ['|', ',', '}'], function() {
          this.parseExpression();
        });
        this.pop();
        this.reject();
      });
      this.reject();
      this.pop();

      end = this.lookahead(null, null);
    }

    this.reject();
  });

  this.pop();

  if (matchToken(end, 'punc', ',')) {
    // Continue parsing next rule
    this.accept();
  } else {
    // Let others parse '}'
    this.reject();
  }

  this.pop();
};

//
// ### function parseExpression ()
// Parser's routine
//
Parser.prototype.parseExpression = function parseExpression() {
  var self = this,
      token = this.expect(null, null, true),
      space = this.maybe('space'),
      matched = false;

  function parseInvocation() {
    var name;
    // grmr.rule
    if (self.maybe('punc', '.')) {
      name = [token.value, self.expect('name', null, true).value];
      space = self.maybe('space');
    } else {
      // rule
      name = [null, token.value];
    }

    // Maybe rule(...) ?
    if (!space && self.maybe('punc', '(')) {
      self.push(['call'].concat(name));
      self.push([]);
      self.parseUntil('punc', ')', true, function() {
        self.parseHostExpression([',']);
        self.maybe('punc', ',');
      });
      self.accept();
      self.pop();
      space = null;
    } else {
      self.push(['match'].concat(name));
    }
  }

  // Choice group (a | b | c)
  if (matchToken(token, 'punc', '(')) {
    // ( expr | ... | expr )
    this.push(['choice']);
    this.list('punc', '|', function() {
      this.push([]);
      this.parseUntil('punc', ['|', ')'], true, function() {
        this.parseExpression();
      });
      this.reject();
      this.pop();
    });
    this.reject();
    this.expect('punc', ')', true);
    space = null;

  // Array match [ a b c ]
  } else if (matchToken(token, 'punc', '[')) {
    this.push(['list']);
    // [ expr expr expr ]
    this.parseUntil('punc', ']', true, function() {
      this.maybe('space');
      this.parseExpression();
    });
    this.accept();
    space = null;

  // String's chars match < a b c >
  } else if (matchToken(token, 'punc', '<')) {
    this.push(['chars']);
    // [ expr expr expr ]
    this.parseUntil('punc', '>', true, function() {
      this.maybe('space');
      this.parseExpression();
    });
    this.accept();
    space = null;

  // Super call ^rule
  } else if (matchToken(token, 'punc', '^')) {
    this.push(['super']);
    token = this.expect('name', null, true);
    space = this.maybe('space');
    parseInvocation();
    this.pop();

  // Predicate ?host-language-code
  } else if (matchToken(token, 'punc', '?')) {
    this.push(['predicate']);
    this.parseHostExpression([',', '|', '->', '&', '?'], true);
    space = null;

  // Local %host-language-code
  } else if (matchToken(token, 'punc', '%')) {
    this.push(['local']);
    this.expect('punc', '(');
    this.parseHostExpression([]);
    this.expect('punc', ')', true);
    space = null;

  // Regexp @/.../
  } else if (matchToken(token, 're')) {
    this.push(['re', token.value]);

  // Lookahead
  } else if (matchToken(token, 'punc', '&')) {
    this.push(['lookahead']);
    space = this.parseExpression();

  // Not
  } else if (matchToken(token, 'punc', '~')) {
    this.push(['not']);
    space = this.parseExpression();

  // Host language result -> { ..code.. }
  } else if (matchToken(token, 'punc', '->')) {
    this.push(['result']);
    this.parseHostExpression([',', '|']);

  // Rule invocation name or name(...)
  } else if (matchToken(token, 'name')) {
    switch (token.value) {
      case 'true':
        this.push(['bool', true]);
        break;
      case 'false':
        this.push(['bool', false]);
        break;
      case 'null':
        this.push(['null']);
        break;
      default:
        parseInvocation();
        break;
    }

  // Number match 123
  } else if (matchToken(token, 'number')) {
    this.push(['number', parseFloat(token.value)]);

  // String match '123'
  } else if (matchToken(token, 'string')) {
    this.push(['string', token.value]);

  // Token match "123"
  } else if (matchToken(token, 'token')) {
    this.push(['call', null, 'token', [
      JSON.stringify(token.value)
    ]]);

  // Sequence match ``123''
  } else if (matchToken(token, 'sequence')) {
    this.push(['seq', token.value]);

  // Host language chunk
  } else if (matchToken(token, 'punc', '{')) {
    this.push(['body']);
    this.parseHostExpression(['}']);
    this.expect('punc', '}', true);
    space = null;

  // Just argument
  } else {
    this.push(['match', null, 'anything']);
  }

  if (space === null) space = this.maybe('space');

  if (!space) {
    // Modificators
    if (this.maybe('punc', '*')) {
      this.wrap(['any']);
      space = null;
    } else if (this.maybe('punc', '+')) {
      this.wrap(['many']);
      space = null;
    } else if (this.maybe('punc', '?')) {
      this.wrap(['optional']);
      space = null;
    }

    if (matchToken(token, 'punc', ':') || !space && this.maybe('punc', ':')) {
      this.wrap(['arg']);
      this.push(this.expect('name').value);
      this.pop();
      space = null;
    }

    if (space === null) space = this.maybe('space');
  }
  this.pop();

  return space;
};

//
// ### function parseHostExpression (terminals, space)
// #### @terminals {Array} terminal symbols
// #### @space {Boolean} break on space after 'name' token
// #### @expr {Boolean} true if shouldn't be wrapped in return ()
// Parser's routine
//
Parser.prototype.parseHostExpression = function parseHostExpression(terminals,
                                                                    space) {
  var token,
      depth = 0,
      dived = false,
      code = [];

  for (;;) {
    token = this.lookahead(null, null, true)

    if (matchToken(token, 'punc', ['{', '(', '['])) {
      depth++;
      dived = true;
    } else if (matchToken(token, 'punc', ['}', ')', ']'])) {
      depth--;
      dived = true;
    }

    if (dived && depth < 0 ||
        depth <= 0 && matchToken(token, 'punc', terminals)) {
      this.reject();
      break;
    }

    this.accept();

    code.push(this.lexer.stringify(token));

    if (space && depth <= 0 &&
        (matchToken(token, 'name') && this.lookahead('space') ||
         matchToken(token, 'space') &&
         (this.lookahead('punc', ['{', ':', '!', '=', '->']) ||
          this.lookahead('name') || this.lookahead('token')))) {
      this.reject();
      break;
    }
  }

  this.push(utils.expressionify(code.join('')));
  this.pop();
};

//
// ### function list (type, value, parse)
// #### @type {String}
// #### @value {String|Array}
// #### @parse {Function}
// Parser's helper
//
Parser.prototype.list = function list(type, value, parse) {
  var token;
  do {
    if (token) this.accept();
    parse.call(this);
  } while (this.active && (token = this.lookahead(type, value)));
  this.reject();
};

//
// ### function parseUntil (type, value, parse)
// #### @type {String}
// #### @value {String|Array}
// #### @space {Boolean} (optional)
// #### @parse {Function}
// Parser's helper
//
Parser.prototype.parseUntil = function parseUntil(type, value, space, parse) {
  var end;

  // Space is optional argument
  if (parse === undefined) {
    parse = space;
    space = false;
  }

  while (this.active && !(end = this.lookahead(type, value, space))) {
    parse.call(this);
  }

  return end;
};

//
// ### function matchToken (token, type, value)
// #### @type {String} (optional) required token type
// #### @value {String|Array} (optional) required token value or values
// Checks if token match type and value
//
function matchToken(token, type, value) {
  return (!type || token.type === type) &&
         (!value || (
           Array.isArray(value) ?
               value.indexOf(token.value) !== -1
               :
               token.value === value
         ));
}

//
// ### function expect (type, value, space, optional)
// #### @type {String} (optional) required token type
// #### @value {String|Array} (optional) required token value, or values
// #### @space {Boolean} (optional) do not parse spaces automatically
// #### @optional {Boolean} (optional) is that token optional?
// Demand token (may throw error)
//
Parser.prototype.expect = function expect(type, value, space, optional) {
  var token = this.tokens.length ? this.tokens.shift() : this.lexer.token();

  // End of file
  if (!token) {
    if (this.state === 'empty' || optional && type === 'space') {
      this.active = false;
      return token;
    } else {
      throw new Error('Unexpected end of file, ' + type + ' expected');
    }
  }

  // Push token to staging if we was matching optionally
  if (optional) this.pending.push(token);

  // Check if token matches our conditions
  if (matchToken(token, type, value)) {
    // Skip space automatically
    if (!space && type !== 'space') {
      if (this.tokens.length) {
        if (this.tokens[0].type === 'space') this.tokens.shift();
      } else {
        this.lexer.trim();
      }
    }

    return token;
  } else {
    if (optional) {
      this.reject();
      return false;
    } else {
      this.unexpected(token, type, value);
    }
  }
};

//
// ### function lookahead (type, value, space)
// #### @type {String} (optional) required token type
// #### @value {String} (optional) required token value
// #### @space {Boolean} (optional) do not parse spaces automatically
// Look ahead, wrapper for .token()
//
Parser.prototype.lookahead = function lookahead(type, value, space) {
  return this.expect(type, value, space, true);
};

//
// ### function maybe (type, value, space)
// #### @type {String} (optional) required token type
// #### @value {String} (optional) required token value
// #### @space {Boolean} (optional) do not parse spaces automatically
// Look ahead and commit automatically, wrapper for .lookahead()
//
Parser.prototype.maybe = function maybe(type, value, space) {
  var res = this.lookahead(type, value, space, true);

  if (res) this.accept();

  return res;
};
//
// ### function accept ()
// Removes all tokens from staging history
// Call it after .lookahead() if you matched
//
Parser.prototype.accept = function accept() {
  this.pending.pop();
  return true;
};

//
// ### function reject ()
// Adds tokens from stage to actual history
// Call it after .lookahead() if you haven't matched
//
Parser.prototype.reject = function reject() {
  if (this.pending.length > 0) {
    this.tokens.push(this.pending.shift());
  }
  return true;
};

//
// ### function unexpected (token, type, value)
// #### @token {Object} Source token
// #### @type {String} Expected type
// #### @value {String} Expected value
// Throws pretty error
//
Parser.prototype.unexpected = function unexpected(token, type, value) {
  throw new Error('Expected [type: ' + type + ' value: ' + value + '] ' +
                  'token, but found [type: ' + token.type + ' value: ' +
                  token.value + '] at \n' +
                  this.code.slice(token.offset, token.offset + 11));
};
