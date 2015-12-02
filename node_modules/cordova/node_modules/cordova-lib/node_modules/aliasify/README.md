Aliasify is a [transform](https://github.com/substack/node-browserify#btransformtr) for [browserify](https://github.com/substack/node-browserify) which lets you rewrite calls to `require`.

Installation
============

Install with `npm install --save-dev aliasify`.

Usage
=====

To use, add a section to your package.json:

    {
        "aliasify": {
            "aliases": {
                "d3": "./shims/d3.js",
                "underscore": "lodash"
            }
        }
    }

Now if you have a file in src/browserify/index.js which looks like:

    d3 = require('d3')
    _ = require('underscore')
    ...

This will automatically be transformed to:

    d3 = require('../../shims/d3.js')
    _ = require('lodash')
    ...

Any replacement that starts with a "." will be resolved as a relative path (as "d3" above.)  Replacements that start with any other character will be replaced verbatim (as with "underscore" above.)

Configuration
=============

Configuration can be loaded in multiple ways;  You can put your configuration directly in package.json, as in the example above, or you can use an external json or js file.  In your package.json:

    {
        "aliasify": "./aliasifyConfig.js"
    }

Then in aliasifyConfig.js:

    module.exports = {
        aliases: {
            "d3": "./shims/d3.js"
        },
        verbose: false
    };

Note that using a js file means you can change your configuration based on environment variables.

Alternatively, if you're using the Browserify API, you can configure your aliasify programatically:

    aliasifyConfig = {
        aliases: {
            "d3": "./shims/d3.js"
        },
        verbose: false
    }

    var b = browserify();
    b.transform(aliasify, aliasifyConfig);

note that using the browserify API, './shims/d3.js' will be resolved against the current working
directory.

Configuration options:
* `aliases` - An object mapping aliases to their replacements.
* `replacements` - An object mapping RegExp strings with RegExp replacements, or a function that will return a replacement.
* `verbose` - If true, then aliasify will print modificiations it is making to stdout.
* `configDir` - An absolute path to resolve relative paths against.  If you're using package.json, this will automatically be filled in for you with the directory containing package.json.  If you're using a .js file for configuration, set this to `__dirname`.
* `appliesTo` - Controls which files will be transformed.  By default, only JS type files will be transformed ('.js', '.coffee', etc...).  See [browserify-trasnform-tools documentation](https://github.com/benbria/browserify-transform-tools/wiki/Transform-Configuration#common-configuration) for details.

Relative Requires
=================

When you specify:

    aliases: {
        "d3": "./shims/d3.js"
    }

The "./" means this will be resolved relative to the current working directory (or relative to the
configuration file which contains the line, in the case where configuration is loaded from
package.json.)  Sometimes it is desirable to literally replace an alias; to resolve the alias
relative to the file which is doing the `require` call.  In this case you can do:

    aliases: {
        "d3": {"relative": "./shims/d3.js"}
    }

This will cause all occurences of `require("d3")` to be replaced with `require("./shims/d3.js")`,
regardless of where those files are in the directory tree.

Regular Expression Aliasing
===========================
You can use the `replacements` configuration section to create more powerful aliasing.  This is useful if you
have a large project but don't want to manually add an alias for every single file.  It is also incredibly useful when you want to combine
aliasify with other transforms, such as hbsfy, reactify, or coffeeify.

    replacements: {
        "_components/(\\w+)": "src/react/components/$1/index.jsx
    }

Will let you replace `require('_components/SomeCoolReactComponent')` with `require('src/react/components/SomeCoolReactComponent/index.jsx')`

You can also match an alias and pass a function which can return a new file name.

`require("_coffee/delicious-coffee");`

Using this configuration:

    replacements: {
        "_coffee/(\\w+)": function (alias, regexMatch, regexObject) {
            console.log(alias); // _coffee/delicious-coffee
            console.log(regexMatch); // _coffee/(\\w+)
            return 'coffee.js'; // default behavior - won't replace
        }
    }



Alternatives
============

`aliasify` is essentially a fancy version of the [`browser` field](https://gist.github.com/defunctzombie/4339901#replace-specific-files---advanced) from package.json, which is [interpreted](https://github.com/substack/node-browserify#packagejson) by browserify.

Using the `browser` field is probably going to be faster, as it doesn't involve running a transform on each of your files.  On the other hand, `aliasify` gives you a finer degree of control and can be run before other transforms (for example, you can run `aliasify` before [debowerify](https://github.com/eugeneware/debowerify), which will let you replace certain components that debowerify would otherwise replace.)

