# grunt-cli [![Build Status](https://secure.travis-ci.org/gruntjs/grunt-cli.png?branch=master)](http://travis-ci.org/gruntjs/grunt-cli)
> The Grunt command line interface.

Install this globally and you'll have access to the `grunt` command anywhere on your system.

```shell
npm install -g grunt-cli
```

**Note:** The job of the `grunt` command is to load and run the version of Grunt you have installed locally to your project, irrespective of its version.  Starting with Grunt v0.4, you should never install Grunt itself globally.  For more information about why, [please read this](http://blog.nodejs.org/2011/03/23/npm-1-0-global-vs-local-installation).

See the [Getting Started](http://gruntjs.com/getting-started) guide for more information.

## Shell tab auto-completion
To enable tab auto-completion for Grunt, add one of the following lines to your `~/.bashrc` or `~/.zshrc` file.

```bash
# Bash, ~/.bashrc
eval "$(grunt --completion=bash)"
```

```bash
# Zsh, ~/.zshrc
eval "$(grunt --completion=zsh)"
```

## Installing grunt-cli locally
If you prefer the idiomatic Node.js method to get started with a project (`npm install && npm test`) then install grunt-cli locally with `npm install grunt-cli --save-dev`. Then add a script to your `package.json` to run the associated grunt command: `"scripts": { "test": "grunt test" } `. Now `npm test` will use the locally installed `./node_modules/.bin/grunt` executable to run your Grunt commands.

To read more about npm scripts, please visit the npm docs: [https://npmjs.org/doc/misc/npm-scripts.html](https://npmjs.org/doc/misc/npm-scripts.html).
