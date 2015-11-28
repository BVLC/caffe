charm
=====

Use
[ansi terminal characters](http://www.termsys.demon.co.uk/vtansi.htm)
to write colors and cursor positions.

![me lucky charms](http://substack.net/images/charms.png)

example
=======

lucky
-----

````javascript
var charm = require('charm')();
charm.pipe(process.stdout);
charm.reset();

var colors = [ 'red', 'cyan', 'yellow', 'green', 'blue' ];
var text = 'Always after me lucky charms.';

var offset = 0;
var iv = setInterval(function () {
    var y = 0, dy = 1;
    for (var i = 0; i < 40; i++) {
        var color = colors[(i + offset) % colors.length];
        var c = text[(i + offset) % text.length];
        charm
            .move(1, dy)
            .foreground(color)
            .write(c)
        ;
        y += dy;
        if (y <= 0 || y >= 5) dy *= -1;
    }
    charm.position(0, 1);
    offset ++;
}, 150);
````

events
======

Charm objects pass along the data events from their input stream except for
events generated from querying the terminal device.

Because charm puts stdin into raw mode, charm emits two special events: "^C" and
"^D" when the user types those combos. It's super convenient with these events
to do:

````javascript
charm.on('^C', process.exit)
````

The above is set on all `charm` streams. If you want to add your own handling for these
special events simply:

````javascript
charm.removeAllListeners('^C')
charm.on('^C', function () {
  // Don't exit. Do some mad science instead.
})
````

methods
=======

var charm = require('charm')(param or stream, ...)
--------------------------------------------------

Create a new readable/writable `charm` stream.

You can pass in readable or writable streams as parameters and they will be
piped to or from accordingly. You can also pass `process` in which case
`process.stdin` and `process.stdout` will be used.

You can `pipe()` to and from the `charm` object you get back.

charm.reset()
-------------

Reset the entire screen, like the /usr/bin/reset command.

charm.destroy(), charm.end()
----------------------------

Emit an `"end"` event downstream.

charm.write(msg)
----------------

Pass along `msg` to the output stream.

charm.position(x, y)
--------------------

Set the cursor position to the absolute coordinates `x, y`.

charm.position(cb)
------------------

Query the absolute cursor position from the input stream through the output
stream (the shell does this automatically) and get the response back as
`cb(x, y)`.

charm.move(x, y)
----------------

Move the cursor position by the relative coordinates `x, y`.

charm.up(y)
-----------

Move the cursor up by `y` rows.

charm.down(y)
-------------

Move the cursor down by `y` rows.

charm.left(x)
-------------

Move the cursor left by `x` columns.

charm.right(x)
--------------

Move the cursor right by `x` columns.

charm.push(withAttributes=false)
--------------------------------

Push the cursor state and optionally the attribute state.

charm.pop(withAttributes=false)
-------------------------------

Pop the cursor state and optionally the attribute state.

charm.erase(s)
--------------

Erase a region defined by the string `s`.

`s` can be:

* end - erase from the cursor to the end of the line
* start - erase from the cursor to the start of the line
* line - erase the current line
* down - erase everything below the current line
* up - erase everything above the current line
* screen - erase the entire screen

charm.delete(mode, n)
---------------------
Delete `'line'` or `'char'`s. `delete` differs from erase
because it does not write over the deleted characters with whitesapce,
but instead removes the deleted space.

`mode` can be `'line'` or `'char'`. `n` is the number of items to be deleted.
`n` must be a positive integer.

The cursor position is not updated.

charm.insert(mode, n)
---------------------

Insert space into the terminal. `insert` is the opposite of` delete`,
and the arguments are the same.

charm.display(attr)
-------------------

Set the display mode with the string `attr`.

`attr` can be:

* reset
* bright
* dim
* underscore
* blink
* reverse
* hidden

charm.foreground(color)
-----------------------

Set the foreground color with the string `color`, which can be:

* red
* yellow
* green
* blue
* cyan
* magenta
* black
* white

or `color` can be an integer from 0 to 255, inclusive.

charm.background(color)
-----------------------

Set the background color with the string `color`, which can be:

* red
* yellow
* green
* blue
* cyan
* magenta
* black
* white

or `color` can be an integer from 0 to 255, inclusive.

charm.cursor(visible)
---------------------

Set the cursor visibility with a boolean `visible`.

install
=======

With [npm](http://npmjs.org) do:

```
npm install charm
```
