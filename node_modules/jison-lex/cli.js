#!/usr/bin/env node

var version = require('./package.json').version;

var path = require('path');
var fs = require('fs');
var lexParser = require('lex-parser');
var RegExpLexer = require('./regexp-lexer.js');


var opts = require("nomnom")
  .script('jison-lex')
  .option('file', {
    flag: true,
    position: 0,
    help: 'file containing a lexical grammar'
  })
  .option('outfile', {
    abbr: 'o',
    metavar: 'FILE',
    help: 'Filename and base module name of the generated parser'
  })
  .option('module-type', {
    abbr: 't',
    default: 'commonjs',
    metavar: 'TYPE',
    help: 'The type of module to generate (commonjs, js)'
  })
  .option('version', {
    abbr: 'V',
    flag: true,
    help: 'print version and exit',
    callback: function() {
       return version;
    }
  })
  .parse();

exports.main = function () {
    if (opts.file) {
        var raw = fs.readFileSync(path.normalize(opts.file), 'utf8'),
            name = path.basename((opts.outfile||opts.file)).replace(/\..*$/g,'');

        fs.writeFileSync(opts.outfile||(name + '.js'), processGrammar(raw, name));
    } else {
        readin(function (raw) {
            console.log(processGrammar(raw));
        });
    }
};

function processGrammar (file, name) {
    var grammar;
    try {
        grammar = lexParser.parse(file);
    } catch (e) {
        try {
            grammar = JSON.parse(file);
        } catch (e2) {
            throw e;
        }
    }

    var settings = grammar.options || {};
    if (!settings.moduleType) settings.moduleType = opts.moduleType;
    if (!settings.moduleName && name) settings.moduleName = name.replace(/-\w/g, function (match){ return match.charAt(1).toUpperCase(); });

    grammar.options = settings;

    var lexer = new RegExpLexer(grammar);
    return lexer.generate(settings);
}

function readin (cb) {
    var stdin = process.openStdin(),
        data = '';

    stdin.setEncoding('utf8');
    stdin.addListener('data', function (chunk) {
        data += chunk;
    });
    stdin.addListener('end', function () {
        cb(data);
    });
}

if (require.main === module)
    exports.main();
