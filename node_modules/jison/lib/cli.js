#!/usr/bin/env node

var jison      = require('./jison.js');
var nomnom     = require('nomnom');
var fs         = require('fs');
var path       = require('path');
var ebnfParser = require('ebnf-parser');
var lexParser  = require('lex-parser');

var version = require('../package.json').version;

var opts = require("nomnom")
  .script('jison')
  .option('file', {
    flag: true,
    position: 0,
    help: 'file containing a grammar'
  })
  .option('lexfile', {
    flag: true,
    position: 1,
    help: 'file containing a lexical grammar'
  })
  .option('outfile', {
    abbr: 'o',
    metavar: 'FILE',
    help: 'Filename and base module name of the generated parser'
  })
  .option('debug', {
    abbr: 't',
    default: false,
    help: 'Debug mode'
  })
  .option('module-type', {
    abbr: 't',
    default: 'commonjs',
    metavar: 'TYPE',
    help: 'The type of module to generate (commonjs, amd, js)'
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
        var raw = fs.readFileSync(path.normalize(opts.file), 'utf8');
        var name = path.basename((opts.outfile||opts.file)).replace(/\..*$/g,'');
        var lex;

        if (opts.lexfile) {
            lex = fs.readFileSync(path.normalize(opts.lexfile), 'utf8');
        }

        fs.writeFileSync(opts.outfile||(name + '.js'), processGrammar(raw, lex, name));
    } else {
        readin(function (raw) {
            console.log(processGrammar(raw));
        });
    }
};

function processGrammar (file, lexFile, name) {
    var grammar;
    try {
        grammar = ebnfParser.parse(file);
    } catch (e) {
        try {
            grammar = JSON.parse(file);
        } catch (e2) {
            throw e;
        }
    }

    var settings = grammar.options || {};
    if (lexFile) grammar.lex = lexParser.parse(lexFile);
    settings.debug = opts.debug;
    if (!settings.moduleType) settings.moduleType = opts.moduleType;
    if (!settings.moduleName && name) settings.moduleName = name.replace(/-\w/g, function (match){ return match.charAt(1).toUpperCase(); });

    var generator = new jison.Generator(grammar, settings);
    return generator.generate(settings);
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

