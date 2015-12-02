# node-properties-parser

A parser for [.properties](http://en.wikipedia.org/wiki/.properties) files written in javascript.  Properties files store key-value pairs.  They are typically used for configuration and internationalization in Java applications as well as in Actionscript projects.  Here's an example of the format:

	# You are reading the ".properties" entry.
	! The exclamation mark can also mark text as comments.
	website = http://en.wikipedia.org/
	language = English
	# The backslash below tells the application to continue reading
	# the value onto the next line.
	message = Welcome to \
	          Wikipedia!
	# Add spaces to the key
	key\ with\ spaces = This is the value that could be looked up with the key "key with spaces".
	# Unicode
	tab : \u0009
*(taken from [Wikipedia](http://en.wikipedia.org/wiki/.properties#Format))*

Currently works with any version of node.js.

## The API

- `parse(text)`: Parses `text` into key-value pairs.  Returns an object containing the key-value pairs.
- `read(path[, callback])`: Opens the file specified by `path` and calls `parse` on its content.  If the optional `callback` parameter is provided, the result is then passed to it as the second parameter.  If an error occurs, the error object is passed to `callback` as the first parameter. If `callback` is not provided, the file specified by `path` is synchronously read and calls `parse` on its contents.  The resulting object is immediately returned.
- `createEditor([path[, callback]])`:  If neither `path` or `callback` are provided an empty editor object is returned synchronously.  If only `path` is provided, the file specified by `path` is synchronously read and parsed.  An editor object with the results in then immediately returned.  If both `path` and `callback` are provided, the file specified by `path` is read and parsed asynchronously.  An editor object with the results are then passed to `callback` as the second parameters.  If an error occurs, the error object is passed to `callback` as the first parameter.
- `Editor`: The editor object is returned by `createEditor`.  Has the following API:
	- `get(key)`: Returns the value currently associated with `key`.
	- `set(key, [value[, comment]])`: Associates `key` with `value`. An optional comment can be provided. If `value` is not specified or is `null`, then `key` is unset.
	- `unset(key)`: Unsets the specified `key`.
	- `save([path][, callback]])`: Writes the current contents of this editor object to a file specified by `path`.  If `path` is not provided, then it'll be defaulted to the `path` value passed to `createEditor`.  The `callback` parameter is called when the file has been written to disk.
	- `addHeadComment`: Added a comment to the head of the file.
	- `toString`: Returns the string representation of this properties editor object.  This string will be written to a file if `save` is called.

## Getting node-properties-parser

The easiest way to get node-properties-parser is with [npm](http://npmjs.org/):

	npm install properties-parser

Alternatively you can clone this git repository:

	git://github.com/xavi-/node-properties-parser.git

## Developed by
* Xavi Ramirez

## License
This project is released under [The MIT License](http://www.opensource.org/licenses/mit-license.php).