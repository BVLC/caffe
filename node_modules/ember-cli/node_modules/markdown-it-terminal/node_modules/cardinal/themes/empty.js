/* 
 * Copy this file and use it as a starting point for your custom cardinal color theme.
 * Just fill in or change the entries for the tokens you want to color
 * Keep in mind that more specific configurations override less specific ones.
 */

var colors = require('ansicolors');

// Change the below definitions in order to tweak the color theme.
module.exports = {

    'Boolean': {
      'true'   :  undefined
    , 'false'  :  undefined
    , _default :  undefined
    }

  , 'Identifier': {
      _default: undefined
    }

  , 'Null': {
      _default: undefined
    }

  , 'Numeric': {
      _default: undefined
    }

  , 'String': {
      _default: undefined
    }

  , 'Keyword': {
      'break'       :  undefined

    , 'case'        :  undefined
    , 'catch'       :  undefined
    , 'class'       :  undefined
    , 'const'       :  undefined
    , 'continue'    :  undefined

    , 'debugger'    :  undefined
    , 'default'     :  undefined
    , 'delete'      :  undefined
    , 'do'          :  undefined

    , 'else'        :  undefined
    , 'export'      :  undefined
    , 'extends'     :  undefined

    , 'finally'     :  undefined
    , 'for'         :  undefined
    , 'function'    :  undefined

    , 'if'          :  undefined
    , 'import'      :  undefined
    , 'in'          :  undefined
    , 'instanceof'  :  undefined
    , 'let'         :  undefined
    , 'new'         :  undefined
    , 'return'      :  undefined
    , 'static'      :  undefined
    , 'super'       :  undefined
    , 'switch'      :  undefined

    , 'this'        :  undefined
    , 'throw'       :  undefined
    , 'try'         :  undefined
    , 'typeof'      :  undefined

    , 'var'         :  undefined
    , 'void'        :  undefined

    , 'while'       :  undefined
    , 'with'        :  undefined
    , 'yield'       :  undefined
    , _default      :  undefined
  }
  , 'Punctuator': {
      ';': undefined  
    , '.': undefined  
    , ',': undefined  

    , '{': undefined  
    , '}': undefined  
    , '(': undefined  
    , ')': undefined  
    , '[': undefined
    , ']': undefined

    , '<': undefined
    , '>': undefined
    , '+': undefined
    , '-': undefined
    , '*': undefined
    , '%': undefined
    , '&': undefined
    , '|': undefined
    , '^': undefined
    , '!': undefined
    , '~': undefined
    , '?': undefined
    , ':': undefined
    , '=': undefined

    , '<=': undefined
    , '>=': undefined
    , '==': undefined
    , '!=': undefined
    , '++': undefined
    , '--': undefined
    , '<<': undefined
    , '>>': undefined
    , '&&': undefined
    , '||': undefined
    , '+=': undefined
    , '-=': undefined
    , '*=': undefined
    , '%=': undefined
    , '&=': undefined
    , '|=': undefined
    , '^=': undefined
    , '/=': undefined
    , '=>': undefined

    , '===': undefined
    , '!==': undefined
    , '>>>': undefined
    , '<<=': undefined
    , '>>=': undefined
    , '...': undefined
    
    , '>>>=': undefined

    , _default: undefined
  }

    // line comment
  , Line: {
     _default: undefined
    }

    /* block comment */
  , Block: {
     _default: undefined
    }

  , _default: undefined
};
