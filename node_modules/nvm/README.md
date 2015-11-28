# nvm

Install and managing different versions of node and linking local versions into specific directories. Very simple setup and no need for a special shell.

# Install

You do have to have at least one version of node and npm to get started, but after that you can install `nvm` with `npm`.

```bash
$ npm install -g nvm
```

# Setup
You must update your path to include ./node_modules/.bin

```bash
export PATH=./node_modules/.bin:$PATH
```

Note, this is a good thing to do anyway – it will allow you to run things like `jshint` on a per-repo basis without having to install it globally and without having to prefix the path.

# Usage

## nvm download <version>
Downloads <version> from http://nodejs.org/dist/ and unpacks to ~/.nvm

## nvm build <version>
Build a version of node.

## nvm link <version>
Link a version of `node` to the current app/library.

## nvm unlink <version>
Unlink a version of `node` to the current app/library.

## nvm install <version>
Globally install a specific version of node.

# License

MIT

```
Copyright (c) 2013 Brian J. Brennan

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
```