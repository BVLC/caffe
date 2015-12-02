{
    function merge(hash, secondHash) {
        secondHash = secondHash[0]
        for(var i in secondHash) {
       		hash[i] = merge_obj(hash[i], secondHash[i]);
        }

        return hash;
    }
    
    function merge_obj(obj, secondObj) {
    	if (!obj)
    		return secondObj;

        for(var i in secondObj)
            obj[i] = merge_obj(obj[i], secondObj[i]);

        return obj;
    }
}

/*
 *  Project: point of entry from pbxproj file
 */
Project
  = headComment:SingleLineComment? InlineComment? _ obj:Object NewLine _
    {
        var proj = Object.create(null)
        proj.project = obj

        if (headComment) {
            proj.headComment = headComment
        }

        return proj;
    }

/*
 *  Object: basic hash data structure with Assignments
 */
Object
  = "{" obj:(AssignmentList / EmptyBody) "}"
    { return obj }

EmptyBody
  = _
    { return Object.create(null) }

AssignmentList
  = _ head:Assignment _ tail:AssignmentList* _
    { 
      if (tail) return merge(head,tail)
      else return head
    }
    / _ head:DelimitedSection _ tail:AssignmentList*
    {
      if (tail) return merge(head,tail)
      else return head
    }

/*
 *  Assignments
 *  can be simple "key = value"
 *  or commented "key /* real key * / = value"
 */
Assignment
  = SimpleAssignment / CommentedAssignment

SimpleAssignment
  = id:Identifier _ "=" _ val:Value ";"
    { 
      var result = Object.create(null);
      result[id] = val
      return result
    }

CommentedAssignment
  = commentedId:CommentedIdentifier _ "=" _ val:Value ";"
    {
        var result = Object.create(null),
            commentKey = commentedId.id + '_comment';

        result[commentedId.id] = val;
        result[commentKey] = commentedId[commentKey];
        return result;

    }
    /
    id:Identifier _ "=" _ commentedVal:CommentedValue ";"
    {
        var result = Object.create(null);
        result[id] = commentedVal.value;
        result[id + "_comment"] = commentedVal.comment;
        return result;
    }

CommentedIdentifier
  = id:Identifier _ comment:InlineComment
    {
        var result = Object.create(null);
        result.id = id;
        result[id + "_comment"] = comment.trim();
        return result
    }

CommentedValue
  = literal:Value _ comment:InlineComment
    {
        var result = Object.create(null)
        result.comment = comment.trim();
        result.value = literal.trim();
        return result;
    }

InlineComment
  = InlineCommentOpen body:[^*]+ InlineCommentClose
    { return body.join('') }

InlineCommentOpen
  = "/*"

InlineCommentClose
  = "*/"

/*
 *  DelimitedSection - ad hoc project structure pbxproj files use
 */
DelimitedSection
  = begin:DelimitedSectionBegin _ fields:(AssignmentList / EmptyBody) _ DelimitedSectionEnd
    {
        var section = Object.create(null);
        section[begin.name] = fields

        return section
    }

DelimitedSectionBegin
  = "/* Begin " sectionName:Identifier " section */" NewLine
    { return { name: sectionName } }

DelimitedSectionEnd
  = "/* End " sectionName:Identifier " section */" NewLine
    { return { name: sectionName } }

/*
 * Arrays: lists of values, possible wth comments
 */
Array
  = "(" arr:(ArrayBody / EmptyArray ) ")" { return arr }

EmptyArray
  = _ { return [] }

ArrayBody
  = _ head:ArrayEntry _ tail:ArrayBody? _
    {
        if (tail) {
            tail.unshift(head);
            return tail;
        } else {
            return [head];
        }
    }

ArrayEntry
  = SimpleArrayEntry / CommentedArrayEntry

SimpleArrayEntry
  = val:Value EndArrayEntry { return val }

CommentedArrayEntry
  = val:Value _ comment:InlineComment EndArrayEntry
    {
        var result = Object.create(null);
        result.value = val.trim();
        result.comment = comment.trim();
        return result;
    }

EndArrayEntry
  = "," / _ &")"

/*
 *  Identifiers and Values
 */
Identifier
  = id:[A-Za-z0-9_.]+ { return id.join('') }
  / QuotedString

Value
  = Object / Array / NumberValue / StringValue

NumberValue
  = DecimalValue / IntegerValue

DecimalValue
  = decimal:(IntegerValue "." IntegerValue)
    { 
        // store decimals as strings
        // as JS doesn't differentiate bw strings and numbers
        return decimal.join('')
    }

IntegerValue
  = !Alpha number:Digit+ !NonTerminator
    { return parseInt(number.join(''), 10) }

StringValue
 = QuotedString / LiteralString

QuotedString
 = DoubleQuote str:QuotedBody DoubleQuote { return '"' + str + '"' }

QuotedBody
 = str:NonQuote+ { return str.join('') }

NonQuote
  = EscapedQuote / !DoubleQuote char:. { return char }

EscapedQuote
  = "\\" DoubleQuote { return '\\"' }

LiteralString
  = literal:LiteralChar+ { return literal.join('') }

LiteralChar
  = !InlineCommentOpen !LineTerminator char:NonTerminator
    { return char }

NonTerminator
  = [^;,\n]

/*
 * SingleLineComment - used for the encoding comment
 */
SingleLineComment
  = "//" _ contents:OneLineString NewLine
    { return contents }

OneLineString
  = contents:NonLine*
    { return contents.join('') }

/*
 *  Simple character checking rules
 */
Digit
  = [0-9]

Alpha
  = [A-Za-z]

DoubleQuote
  = '"'

_ "whitespace"
  = whitespace*

whitespace
  = NewLine / [\t ]

NonLine
  = !NewLine char:Char
    { return char }

LineTerminator
  = NewLine / ";"

NewLine
    = [\n\r]

Char
  = .
