# The CoffeeScript language has a good deal of optional syntax, implicit syntax,
# and shorthand syntax. This can greatly complicate a grammar and bloat
# the resulting parse table. Instead of making the parser handle it all, we take
# a series of passes over the token stream, using this **Rewriter** to convert
# shorthand into the unambiguous long form, add implicit indentation and
# parentheses, and generally clean things up.

# Create a generated token: one that exists due to a use of implicit syntax.
generate = (tag, value, origin) ->
  tok = [tag, value]
  tok.generated = yes
  tok.origin = origin if origin
  tok

# The **Rewriter** class is used by the [Lexer](lexer.html), directly against
# its internal array of tokens.
class exports.Rewriter

  # Helpful snippet for debugging:
  #
  #     console.log (t[0] + '/' + t[1] for t in @tokens).join ' '

  # Rewrite the token stream in multiple passes, one logical filter at
  # a time. This could certainly be changed into a single pass through the
  # stream, with a big ol' efficient switch, but it's much nicer to work with
  # like this. The order of these passes matters -- indentation must be
  # corrected before implicit parentheses can be wrapped around blocks of code.
  rewrite: (@tokens) ->
    @removeLeadingNewlines()
    @closeOpenCalls()
    @closeOpenIndexes()
    @normalizeLines()
    @tagPostfixConditionals()
    @addImplicitBracesAndParens()
    @addLocationDataToGeneratedTokens()
    @tokens

  # Rewrite the token stream, looking one token ahead and behind.
  # Allow the return value of the block to tell us how many tokens to move
  # forwards (or backwards) in the stream, to make sure we don't miss anything
  # as tokens are inserted and removed, and the stream changes length under
  # our feet.
  scanTokens: (block) ->
    {tokens} = this
    i = 0
    i += block.call this, token, i, tokens while token = tokens[i]
    true

  detectEnd: (i, condition, action) ->
    {tokens} = this
    levels = 0
    while token = tokens[i]
      return action.call this, token, i     if levels is 0 and condition.call this, token, i
      return action.call this, token, i - 1 if not token or levels < 0
      if token[0] in EXPRESSION_START
        levels += 1
      else if token[0] in EXPRESSION_END
        levels -= 1
      i += 1
    i - 1

  # Leading newlines would introduce an ambiguity in the grammar, so we
  # dispatch them here.
  removeLeadingNewlines: ->
    break for [tag], i in @tokens when tag isnt 'TERMINATOR'
    @tokens.splice 0, i if i

  # The lexer has tagged the opening parenthesis of a method call. Match it with
  # its paired close. We have the mis-nested outdent case included here for
  # calls that close on the same line, just before their outdent.
  closeOpenCalls: ->
    condition = (token, i) ->
      token[0] in [')', 'CALL_END'] or
      token[0] is 'OUTDENT' and @tag(i - 1) is ')'

    action = (token, i) ->
      @tokens[if token[0] is 'OUTDENT' then i - 1 else i][0] = 'CALL_END'

    @scanTokens (token, i) ->
      @detectEnd i + 1, condition, action if token[0] is 'CALL_START'
      1

  # The lexer has tagged the opening parenthesis of an indexing operation call.
  # Match it with its paired close.
  closeOpenIndexes: ->
    condition = (token, i) ->
      token[0] in [']', 'INDEX_END']

    action = (token, i) ->
      token[0] = 'INDEX_END'

    @scanTokens (token, i) ->
      @detectEnd i + 1, condition, action if token[0] is 'INDEX_START'
      1

  # Match tags in token stream starting at i with pattern, skipping HERECOMMENTs
  # Pattern may consist of strings (equality), an array of strings (one of)
  # or null (wildcard)
  matchTags: (i, pattern...) ->
    fuzz = 0
    for j in [0 ... pattern.length]
      fuzz += 2 while @tag(i + j + fuzz) is 'HERECOMMENT'
      continue if not pattern[j]?
      pattern[j] = [pattern[j]] if typeof pattern[j] is 'string'
      return no if @tag(i + j + fuzz) not in pattern[j]
    yes

  # yes iff standing in front of something looking like
  # @<x>: or <x>:, skipping over 'HERECOMMENT's
  looksObjectish: (j) ->
    @matchTags(j, '@', null, ':') or @matchTags(j, null, ':')

  # yes iff current line of tokens contain an element of tags on same
  # expression level. Stop searching at LINEBREAKS or explicit start of
  # containing balanced expression.
  findTagsBackwards: (i, tags) ->
    backStack = []
    while i >= 0 and (backStack.length or
          @tag(i) not in tags and
          (@tag(i) not in EXPRESSION_START or @tokens[i].generated) and
          @tag(i) not in LINEBREAKS)
      backStack.push @tag(i) if @tag(i) in EXPRESSION_END
      backStack.pop() if @tag(i) in EXPRESSION_START and backStack.length
      i -= 1
    @tag(i) in tags

  # Look for signs of implicit calls and objects in the token stream and
  # add them.
  addImplicitBracesAndParens: ->
    # Track current balancing depth (both implicit and explicit) on stack.
    stack = []

    @scanTokens (token, i, tokens) ->
      [tag]     = token
      [prevTag] = prevToken = if i > 0 then tokens[i - 1] else []
      [nextTag] = if i < tokens.length - 1 then tokens[i + 1] else []
      stackTop  = -> stack[stack.length - 1]
      startIdx  = i

      # Helper function, used for keeping track of the number of tokens consumed
      # and spliced, when returning for getting a new token.
      forward   = (n) -> i - startIdx + n

      # Helper functions
      inImplicit        = -> stackTop()?[2]?.ours
      inImplicitCall    = -> inImplicit() and stackTop()?[0] is '('
      inImplicitObject  = -> inImplicit() and stackTop()?[0] is '{'
      # Unclosed control statement inside implicit parens (like
      # class declaration or if-conditionals)
      inImplicitControl = -> inImplicit and stackTop()?[0] is 'CONTROL'

      startImplicitCall = (j) ->
        idx = j ? i
        stack.push ['(', idx, ours: yes]
        tokens.splice idx, 0, generate 'CALL_START', '('
        i += 1 if not j?

      endImplicitCall = ->
        stack.pop()
        tokens.splice i, 0, generate 'CALL_END', ')'
        i += 1

      startImplicitObject = (j, startsLine = yes) ->
        idx = j ? i
        stack.push ['{', idx, sameLine: yes, startsLine: startsLine, ours: yes]
        tokens.splice idx, 0, generate '{', generate(new String('{')), token
        i += 1 if not j?

      endImplicitObject = (j) ->
        j = j ? i
        stack.pop()
        tokens.splice j, 0, generate '}', '}', token
        i += 1

      # Don't end an implicit call on next indent if any of these are in an argument
      if inImplicitCall() and tag in ['IF', 'TRY', 'FINALLY', 'CATCH',
        'CLASS', 'SWITCH']
        stack.push ['CONTROL', i, ours: true]
        return forward(1)

      if tag is 'INDENT' and inImplicit()

        # An `INDENT` closes an implicit call unless
        #
        #  1. We have seen a `CONTROL` argument on the line.
        #  2. The last token before the indent is part of the list below
        #
        if prevTag not in ['=>', '->', '[', '(', ',', '{', 'TRY', 'ELSE', '=']
          endImplicitCall() while inImplicitCall()
        stack.pop() if inImplicitControl()
        stack.push [tag, i]
        return forward(1)

      # Straightforward start of explicit expression
      if tag in EXPRESSION_START
        stack.push [tag, i]
        return forward(1)

      # Close all implicit expressions inside of explicitly closed expressions.
      if tag in EXPRESSION_END
        while inImplicit()
          if inImplicitCall()
            endImplicitCall()
          else if inImplicitObject()
            endImplicitObject()
          else
            stack.pop()
        stack.pop()

      # Recognize standard implicit calls like
      # f a, f() b, f? c, h[0] d etc.
      if (tag in IMPLICIT_FUNC and token.spaced and not token.stringEnd or
          tag is '?' and i > 0 and not tokens[i - 1].spaced) and
         (nextTag in IMPLICIT_CALL or
          nextTag in IMPLICIT_UNSPACED_CALL and
          not tokens[i + 1]?.spaced and not tokens[i + 1]?.newLine)
        tag = token[0] = 'FUNC_EXIST' if tag is '?'
        startImplicitCall i + 1
        return forward(2)

      # Implicit call taking an implicit indented object as first argument.
      #
      #     f
      #       a: b
      #       c: d
      #
      # and
      #
      #     f
      #       1
      #       a: b
      #       b: c
      #
      # Don't accept implicit calls of this type, when on the same line
      # as the control strucutures below as that may misinterpret constructs like:
      #
      #     if f
      #        a: 1
      # as
      #
      #     if f(a: 1)
      #
      # which is probably always unintended.
      # Furthermore don't allow this in literal arrays, as
      # that creates grammatical ambiguities.
      if tag in IMPLICIT_FUNC and @matchTags(i + 1, 'INDENT', null, ':') and
         not @findTagsBackwards(i, ['CLASS', 'EXTENDS', 'IF', 'CATCH',
          'SWITCH', 'LEADING_WHEN', 'FOR', 'WHILE', 'UNTIL'])
        startImplicitCall i + 1
        stack.push ['INDENT', i + 2]
        return forward(3)

      # Implicit objects start here
      if tag is ':'
        # Go back to the (implicit) start of the object
        if @tag(i - 2) is '@' then s = i - 2 else s = i - 1
        s -= 2 while @tag(s - 2) is 'HERECOMMENT'

        # Mark if the value is a for loop
        @insideForDeclaration = nextTag is 'FOR'

        startsLine = s is 0 or @tag(s - 1) in LINEBREAKS or tokens[s - 1].newLine
        # Are we just continuing an already declared object?
        if stackTop()
          [stackTag, stackIdx] = stackTop()
          if (stackTag is '{' or stackTag is 'INDENT' and @tag(stackIdx - 1) is '{') and
             (startsLine or @tag(s - 1) is ',' or @tag(s - 1) is '{')
            return forward(1)

        startImplicitObject(s, !!startsLine)
        return forward(2)

      # End implicit calls when chaining method calls
      # like e.g.:
      #
      #     f ->
      #       a
      #     .g b, ->
      #       c
      #     .h a
      #
      # and also
      #
      #     f a
      #     .g b
      #     .h a

      stackTop()[2].sameLine = no if inImplicitObject() and tag in LINEBREAKS

      newLine = prevTag is 'OUTDENT' or prevToken.newLine
      if tag in IMPLICIT_END or tag in CALL_CLOSERS and newLine
        while inImplicit()
          [stackTag, stackIdx, {sameLine, startsLine}] = stackTop()
          # Close implicit calls when reached end of argument list
          if inImplicitCall() and prevTag isnt ','
            endImplicitCall()
          # Close implicit objects such as:
          # return a: 1, b: 2 unless true
          else if inImplicitObject() and not @insideForDeclaration and sameLine and
                  tag isnt 'TERMINATOR' and prevTag isnt ':' and
            endImplicitObject()
          # Close implicit objects when at end of line, line didn't end with a comma
          # and the implicit object didn't start the line or the next line doesn't look like
          # the continuation of an object.
          else if inImplicitObject() and tag is 'TERMINATOR' and prevTag isnt ',' and
                  not (startsLine and @looksObjectish(i + 1))
            endImplicitObject()
          else
            break

      # Close implicit object if comma is the last character
      # and what comes after doesn't look like it belongs.
      # This is used for trailing commas and calls, like:
      #
      #     x =
      #         a: b,
      #         c: d,
      #     e = 2
      #
      # and
      #
      #     f a, b: c, d: e, f, g: h: i, j
      #
      if tag is ',' and not @looksObjectish(i + 1) and inImplicitObject() and
         not @insideForDeclaration and
         (nextTag isnt 'TERMINATOR' or not @looksObjectish(i + 2))
        # When nextTag is OUTDENT the comma is insignificant and
        # should just be ignored so embed it in the implicit object.
        #
        # When it isn't the comma go on to play a role in a call or
        # array further up the stack, so give it a chance.

        offset = if nextTag is 'OUTDENT' then 1 else 0
        while inImplicitObject()
          endImplicitObject i + offset
      return forward(1)

  # Add location data to all tokens generated by the rewriter.
  addLocationDataToGeneratedTokens: ->
    @scanTokens (token, i, tokens) ->
      return 1 if     token[2]
      return 1 unless token.generated or token.explicit
      if token[0] is '{' and nextLocation=tokens[i + 1]?[2]
        {first_line: line, first_column: column} = nextLocation
      else if prevLocation = tokens[i - 1]?[2]
        {last_line: line, last_column: column} = prevLocation
      else
        line = column = 0
      token[2] =
        first_line:   line
        first_column: column
        last_line:    line
        last_column:  column
      return 1

  # Because our grammar is LALR(1), it can't handle some single-line
  # expressions that lack ending delimiters. The **Rewriter** adds the implicit
  # blocks, so it doesn't need to. To keep the grammar clean and tidy, trailing
  # newlines within expressions are removed and the indentation tokens of empty
  # blocks are added.
  normalizeLines: ->
    starter = indent = outdent = null

    condition = (token, i) ->
      token[1] isnt ';' and token[0] in SINGLE_CLOSERS and
      not (token[0] is 'TERMINATOR' and @tag(i + 1) in EXPRESSION_CLOSE) and
      not (token[0] is 'ELSE' and starter isnt 'THEN') and
      not (token[0] in ['CATCH', 'FINALLY'] and starter in ['->', '=>']) or
      token[0] in CALL_CLOSERS and @tokens[i - 1].newLine

    action = (token, i) ->
      @tokens.splice (if @tag(i - 1) is ',' then i - 1 else i), 0, outdent

    @scanTokens (token, i, tokens) ->
      [tag] = token
      if tag is 'TERMINATOR'
        if @tag(i + 1) is 'ELSE' and @tag(i - 1) isnt 'OUTDENT'
          tokens.splice i, 1, @indentation()...
          return 1
        if @tag(i + 1) in EXPRESSION_CLOSE
          tokens.splice i, 1
          return 0
      if tag is 'CATCH'
        for j in [1..2] when @tag(i + j) in ['OUTDENT', 'TERMINATOR', 'FINALLY']
          tokens.splice i + j, 0, @indentation()...
          return 2 + j
      if tag in SINGLE_LINERS and @tag(i + 1) isnt 'INDENT' and
         not (tag is 'ELSE' and @tag(i + 1) is 'IF')
        starter = tag
        [indent, outdent] = @indentation tokens[i]
        indent.fromThen   = true if starter is 'THEN'
        tokens.splice i + 1, 0, indent
        @detectEnd i + 2, condition, action
        tokens.splice i, 1 if tag is 'THEN'
        return 1
      return 1

  # Tag postfix conditionals as such, so that we can parse them with a
  # different precedence.
  tagPostfixConditionals: ->

    original = null

    condition = (token, i) ->
      [tag] = token
      [prevTag] = @tokens[i - 1]
      tag is 'TERMINATOR' or (tag is 'INDENT' and prevTag not in SINGLE_LINERS)

    action = (token, i) ->
      if token[0] isnt 'INDENT' or (token.generated and not token.fromThen)
        original[0] = 'POST_' + original[0]

    @scanTokens (token, i) ->
      return 1 unless token[0] is 'IF'
      original = token
      @detectEnd i + 1, condition, action
      return 1

  # Generate the indentation tokens, based on another token on the same line.
  indentation: (origin) ->
    indent  = ['INDENT', 2]
    outdent = ['OUTDENT', 2]
    if origin
      indent.generated = outdent.generated = yes
      indent.origin = outdent.origin = origin
    else
      indent.explicit = outdent.explicit = yes
    [indent, outdent]

  generate: generate

  # Look up a tag by token index.
  tag: (i) -> @tokens[i]?[0]

# Constants
# ---------

# List of the token pairs that must be balanced.
BALANCED_PAIRS = [
  ['(', ')']
  ['[', ']']
  ['{', '}']
  ['INDENT', 'OUTDENT'],
  ['CALL_START', 'CALL_END']
  ['PARAM_START', 'PARAM_END']
  ['INDEX_START', 'INDEX_END']
]

# The inverse mappings of `BALANCED_PAIRS` we're trying to fix up, so we can
# look things up from either end.
exports.INVERSES = INVERSES = {}

# The tokens that signal the start/end of a balanced pair.
EXPRESSION_START = []
EXPRESSION_END   = []

for [left, rite] in BALANCED_PAIRS
  EXPRESSION_START.push INVERSES[rite] = left
  EXPRESSION_END  .push INVERSES[left] = rite

# Tokens that indicate the close of a clause of an expression.
EXPRESSION_CLOSE = ['CATCH', 'THEN', 'ELSE', 'FINALLY'].concat EXPRESSION_END

# Tokens that, if followed by an `IMPLICIT_CALL`, indicate a function invocation.
IMPLICIT_FUNC    = ['IDENTIFIER', 'SUPER', ')', 'CALL_END', ']', 'INDEX_END', '@', 'THIS']

# If preceded by an `IMPLICIT_FUNC`, indicates a function invocation.
IMPLICIT_CALL    = [
  'IDENTIFIER', 'NUMBER', 'STRING', 'JS', 'REGEX', 'NEW', 'PARAM_START', 'CLASS'
  'IF', 'TRY', 'SWITCH', 'THIS', 'BOOL', 'NULL', 'UNDEFINED', 'UNARY', 'YIELD'
  'UNARY_MATH', 'SUPER', 'THROW', '@', '->', '=>', '[', '(', '{', '--', '++'
]

IMPLICIT_UNSPACED_CALL = ['+', '-']

# Tokens that always mark the end of an implicit call for single-liners.
IMPLICIT_END     = ['POST_IF', 'FOR', 'WHILE', 'UNTIL', 'WHEN', 'BY',
  'LOOP', 'TERMINATOR']

# Single-line flavors of block expressions that have unclosed endings.
# The grammar can't disambiguate them, so we insert the implicit indentation.
SINGLE_LINERS    = ['ELSE', '->', '=>', 'TRY', 'FINALLY', 'THEN']
SINGLE_CLOSERS   = ['TERMINATOR', 'CATCH', 'FINALLY', 'ELSE', 'OUTDENT', 'LEADING_WHEN']

# Tokens that end a line.
LINEBREAKS       = ['TERMINATOR', 'INDENT', 'OUTDENT']

# Tokens that close open calls when they follow a newline.
CALL_CLOSERS     = ['.', '?.', '::', '?::']
