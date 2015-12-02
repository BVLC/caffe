Lazy lists for node
===================


# Table of contents:

[Introduction](#Introduction)
  
[Documentation](#Documentation)

<a name="Introduction" />
# Introduction
Lazy comes really handy when you need to treat a stream of events like a list.
The best use case currently is returning a lazy list from an asynchronous
function, and having data pumped into it via events. In asynchronous
programming you can't just return a regular list because you don't yet have
data for it. The usual solution so far has been to provide a callback that gets
called when the data is available. But doing it this way you lose the power of
chaining functions and creating pipes, which leads to not that nice interfaces.
(See the 2nd example below to see how it improved the interface in one of my
modules.)

Check out this toy example, first you create a Lazy object:
```javascript
    var Lazy = require('lazy');

    var lazy = new Lazy;
    lazy
      .filter(function (item) {
        return item % 2 == 0
      })
      .take(5)
      .map(function (item) {
        return item*2;
      })
      .join(function (xs) {
        console.log(xs);
      });
```

This code says that 'lazy' is going to be a lazy list that filters even
numbers, takes first five of them, then multiplies all of them by 2, and then
calls the join function (think of join as in threads) on the final list.

And now you can emit 'data' events with data in them at some point later,
```javascript
    [0,1,2,3,4,5,6,7,8,9,10].forEach(function (x) {
      lazy.emit('data', x);
    });
```

The output will be produced by the 'join' function, which will output the
expected [0, 4, 8, 12, 16].

And here is a real-world example. Some time ago I wrote a hash database for
node.js called node-supermarket (think of key-value store except greater). Now
it had a similar interface as a list, you could .forEach on the stored
elements, .filter them, etc. But being asynchronous in nature it lead to the
following code, littered with callbacks and temporary lists:
```javascript
    var Store = require('supermarket');

    var db = new Store({ filename : 'users.db', json : true });

    var users_over_20 = [];
    db.filter(
      function (user, meta) {
        // predicate function
        return meta.age > 20;
      },
      function (err, user, meta) {
        // function that gets executed when predicate is true
        if (users_over_20.length < 5)
          users_over_20.push(meta);
      },
      function () {
        // done function, called when all records have been filtered

        // now do something with users_over_20
      }
    )
```
This code selects first five users who are over 20 years old and stores them
in users_over_20.

But now we changed the node-supermarket interface to return lazy lists, and
the code became:
```javascript
    var Store = require('supermarket');

    var db = new Store({ filename : 'users.db', json : true });

    db.filter(function (user, meta) {
        return meta.age > 20;
      })
      .take(5)
      .join(function (xs) {
        // xs contains the first 5 users who are over 20!
      });
```
This is so much nicer!

Here is the latest feature: .lines. Given a stream of data that has \n's in it,
.lines converts that into a list of lines.

Here is an example from node-iptables that I wrote the other week,
```javascript
    var Lazy = require('lazy');
    var spawn = require('child_process').spawn;
    var iptables = spawn('iptables', ['-L', '-n', '-v']);

    Lazy(iptables.stdout)
        .lines
        .map(String)
        .skip(2) // skips the two lines that are iptables header
        .map(function (line) {
            // packets, bytes, target, pro, opt, in, out, src, dst, opts
            var fields = line.trim().split(/\s+/, 9);
            return {
                parsed : {
                    packets : fields[0],
                    bytes : fields[1],
                    target : fields[2],
                    protocol : fields[3],
                    opt : fields[4],
                    in : fields[5],
                    out : fields[6],
                    src : fields[7],
                    dst : fields[8]
                },
                raw : line.trim()
            };
        });
```
This example takes the `iptables -L -n -v` command and uses .lines on its output.
Then it .skip's two lines from input and maps a function on all other lines that
creates a data structure from the output.

<a name="Documentation" />
# Documentation

Supports the following operations:

* lazy.filter(f)
* lazy.forEach(f)
* lazy.map(f)
* lazy.take(n)
* lazy.takeWhile(f)
* lazy.bucket(init, f)
* lazy.lines
* lazy.sum(f)
* lazy.product(f)
* lazy.foldr(op, i, f)
* lazy.skip(n)
* lazy.head(f)
* lazy.tail(f)
* lazy.join(f)

The Lazy object itself has a .range property for generating all the possible ranges.

Here are several examples:

* Lazy.range('10..') - infinite range starting from 10
* Lazy.range('(10..') - infinite range starting from 11
* Lazy.range(10) - range from 0 to 9
* Lazy.range(-10, 10) - range from -10 to 9 (-10, -9, ... 0, 1, ... 9)
* Lazy.range(-10, 10, 2) - range from -10 to 8, skipping every 2nd element (-10, -8, ... 0, 2, 4, 6, 8)
* Lazy.range(10, 0, 2) - reverse range from 10 to 1, skipping every 2nd element (10, 8, 6, 4, 2)
* Lazy.range(10, 0) - reverse range from 10 to 1
* Lazy.range('5..50') - range from 5 to 49
* Lazy.range('50..44') - range from 50 to 45
* Lazy.range('1,1.1..4') - range from 1 to 4 with increment of 0.1 (1, 1.1, 1.2, ... 3.9)
* Lazy.range('4,3.9..1') - reverse range from 4 to 1 with decerement of 0.1
* Lazy.range('[1..10]') - range from 1 to 10 (all inclusive)
* Lazy.range('[10..1]') - range from 10 to 1 (all inclusive)
* Lazy.range('[1..10)') - range grom 1 to 9
* Lazy.range('[10..1)') - range from 10 to 2
* Lazy.range('(1..10]') - range from 2 to 10
* Lazy.range('(10..1]') - range from 9 to 1
* Lazy.range('(1..10)') - range from 2 to 9
* Lazy.range('[5,10..50]') - range from 5 to 50 with a step of 5 (all inclusive)

Then you can use other lazy functions on these ranges.


