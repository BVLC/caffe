multimeter
==========

Control multiple ANSI progress bars on the terminal.

![multibar example output](http://substack.net/images/screenshots/multibar.png)

![multimeter](http://substack.net/images/multimeter.png)

example
=======

````javascript
var multimeter = require('multimeter');
var multi = multimeter(process);

multi.drop(function (bar) {
    var iv = setInterval(function () {
        var p = bar.percent();
        bar.percent(p + 1);
        
        if (p >= 100) clearInterval(iv);
    }, 25);
});
````

methods
=======

var multimeter = require('multimeter');

var multi = multimeter(stream, ...)
-----------------------------------

Create a new multimeter handle on the supplied stream/process objects, which
will be passed directly to [charm](https://github.com/substack/node-charm).

If you pass in a charm object that will be used instead of creating a new one.

var bar = multi(x, y, params)
-----------------------------

Create a new progress bar at `(x,y)` with `params` which default to:

* width : 10
* before : '['
* after : '] '
* solid : { background : 'blue', foreground : 'white', text : '|' }
* empty : { background : null, foreground : null, text : ' ' }

If `y` is negative or `'-0'` it will be treated as a relative coordinate.

var bar = multi.rel(x, y, params)
---------------------------------

Create a new progress bar at an absolute `x` and relative `y` coordinate with
respect to the present `multi.offset`.

multi.drop(params, cb)
----------------------

Create a new progress bar at the present cursor location. The `bar` object will
be passed to `cb(bar)` once the cursor location has been determined. 

multi.on(...), multi.removeListener(...), multi.destroy(...), multi.write(...)
------------------------------------------------------------------------------

Call event emitter functions on the underlying `charm` object.

multi.offset
------------

This getter/setter controls the positioning for relative progress bars.

Increment this value whenever you write a newline to the stream to prevent the
pending progress bars from drifting down from their original positions.

bar.percent(p, msg=p + ' %')
----------------------------

Update the progress bar to `p` percent, a value between 0 and 100, inclusive.

The text to the right of the progress bar will be set to `msg`.

bar.ratio(n, d, msg=n + ' / ' + d)
----------------------------------

Update the progress bar with a ratio, `n/d`.

The text to the right of the progress bar will be set to `msg`.

attributes
==========

multi.charm
-----------

The [charm](https://github.com/substack/node-charm) object used internally to
draw the progress bars.

install
=======

With [npm](http://npmjs.org) do:

    npm install multimeter
