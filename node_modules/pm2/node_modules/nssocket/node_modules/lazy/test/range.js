var assert = require('assert');
var Lazy = require('..');
var expresso = expresso;

function range(i, j, s) {
    var r = [];
    var s = s || 1;
    if (j > i) for (;i<j;i+=s) r.push(i);
    else for(;i>j;i-=s) r.push(i);
    return r;
}

exports['infinite range'] = function () {
    var joinExecuted = false;
    Lazy.range('10..').take(10).join(function (xs) {
        joinExecuted = true;
        assert.deepEqual(xs, range(10, 20));
        assert.equal(xs.length, 10);
    });

    setTimeout(function () {
        assert.ok(joinExecuted, 'join didn\'t execute');
    }, 2000);
}

exports['infinite range half-open'] = function () {
    var joinExecuted = false;
    Lazy.range('(10..').take(10).join(function (xs) {
        joinExecuted = true;
        assert.deepEqual(xs, range(11, 21));
        assert.equal(xs.length, 10);
    });

    setTimeout(function () {
        assert.ok(joinExecuted, 'join didn\'t execute');
    }, 2000);
}

exports['range i'] = function () {
    var joinExecuted = false;
    Lazy.range(10).join(function (xs) {
        joinExecuted = true;
        assert.deepEqual(xs, range(0, 10));
        assert.equal(xs.length, 10);
    });

    setTimeout(function () {
        assert.ok(joinExecuted, 'join didn\'t execute');
    }, 2000);
}

exports['range i,j (i<j)'] = function () {
    var joinExecuted = false;
    Lazy.range(-10, 10).join(function (xs) {
        joinExecuted = true;
        assert.deepEqual(xs, range(-10, 10));
        assert.equal(xs.length, 20);
    });

    setTimeout(function () {
        assert.ok(joinExecuted, 'join didn\'t execute');
    }, 2000);
}

exports['range i,j,s (i<j)'] = function () {
    var joinExecuted = false;
    Lazy.range(-10, 10, 2).join(function (xs) {
        joinExecuted = true;
        assert.deepEqual(xs, range(-10, 10, 2));
        assert.equal(xs.length, 10);
    });

    setTimeout(function () {
        assert.ok(joinExecuted, 'join didn\'t execute');
    }, 2000);
}

exports['range i,j,s (i>j)'] = function () {
    var joinExecuted = false;
    Lazy.range(10, 0, 2).join(function (xs) {
        joinExecuted = true;
        assert.deepEqual(xs, range(10, 0, 2));
        assert.equal(xs.length, 5);
    });

    setTimeout(function () {
        assert.ok(joinExecuted, 'join didn\'t execute');
    }, 2000);
}

exports['range i,j (i>j)'] = function () {
    var joinExecuted = false;
    Lazy.range(10, -8).join(function (xs) {
        joinExecuted = true;
        assert.deepEqual(xs, range(10, -8));
        assert.equal(xs.length, 18);
    });

    setTimeout(function () {
        assert.ok(joinExecuted, 'join didn\'t execute');
    }, 2000);
}

exports['range i..j (i<j)'] = function () {
    var joinExecuted = false;
    Lazy.range('5..50').join(function (xs) {
        joinExecuted = true;
        assert.deepEqual(xs, range(5, 50));
        assert.equal(xs.length, 45);
    });

    setTimeout(function () {
        assert.ok(joinExecuted, 'join didn\'t execute');
    }, 2000);
}

exports['range i..j (i>j)'] = function () {
    var joinExecuted = false;
    Lazy.range('50..44').join(function (xs) {
        joinExecuted = true;
        assert.deepEqual(xs, range(50, 44));
        assert.equal(xs.length, 6);
    });

    setTimeout(function () {
        assert.ok(joinExecuted, 'join didn\'t execute');
    }, 2000);
}

exports['range i,next..j (i<j)'] = function () {
    var joinExecuted = false;
    Lazy.range('1,1.1..4').join(function (xs) {
        joinExecuted = true;
        assert.deepEqual(xs, range(1,4,0.1));
        assert.equal(xs.length, 30);
    });

    setTimeout(function () {
        assert.ok(joinExecuted, 'join didn\'t execute');
    }, 2000);
}

exports['range i,next..j (i>j)'] = function () {
    var joinExecuted = false;
    Lazy.range('4,3.9..1').join(function (xs) {
        joinExecuted = true;
        assert.deepEqual(xs, range(4,1,0.1));
        assert.equal(xs.length, 30);
    });

    setTimeout(function () {
        assert.ok(joinExecuted, 'join didn\'t execute');
    }, 2000);
}

exports['range [i..j] (i<j)'] = function () {
    var joinExecuted = false;
    Lazy.range('[1..10]').join(function (xs) {
        joinExecuted = true;
        assert.deepEqual(xs, range(1,11));
        assert.equal(xs.length, 10);
    });

    setTimeout(function () {
        assert.ok(joinExecuted, 'join didn\'t execute');
    }, 2000);
}

exports['range [i..j] (i>j)'] = function () {
    var joinExecuted = false;
    Lazy.range('[10..1]').join(function (xs) {
        joinExecuted = true;
        assert.deepEqual(xs, range(10,0));
        assert.equal(xs.length, 10);
    });

    setTimeout(function () {
        assert.ok(joinExecuted, 'join didn\'t execute');
    }, 2000);
}

exports['range [i..j) (i<j)'] = function () {
    var joinExecuted = false;
    Lazy.range('[1..10)').join(function (xs) {
        joinExecuted = true;
        assert.deepEqual(xs, range(1,10));
        assert.equal(xs.length, 9);
    });

    setTimeout(function () {
        assert.ok(joinExecuted, 'join didn\'t execute');
    }, 2000);
}

exports['range [i..j) (i>j)'] = function () {
    var joinExecuted = false;
    Lazy.range('[10..1)').join(function (xs) {
        joinExecuted = true;
        assert.deepEqual(xs, range(10,1));
        assert.equal(xs.length, 9);
    });

    setTimeout(function () {
        assert.ok(joinExecuted, 'join didn\'t execute');
    }, 2000);
}

exports['range (i..j] (i<j)'] = function () {
    var joinExecuted = false;
    Lazy.range('(1..10]').join(function (xs) {
        joinExecuted = true;
        assert.deepEqual(xs, range(2,11));
        assert.equal(xs.length, 9);
    });

    setTimeout(function () {
        assert.ok(joinExecuted, 'join didn\'t execute');
    }, 2000);
}

exports['range (i..j] (i>j)'] = function () {
    var joinExecuted = false;
    Lazy.range('(10..1]').join(function (xs) {
        joinExecuted = true;
        assert.deepEqual(xs, range(9,0));
        assert.equal(xs.length, 9);
    });

    setTimeout(function () {
        assert.ok(joinExecuted, 'join didn\'t execute');
    }, 2000);
}

exports['range (i..j) (i<j)'] = function () {
    var joinExecuted = false;
    Lazy.range('(1..10)').join(function (xs) {
        joinExecuted = true;
        assert.deepEqual(xs, range(2,10));
        assert.equal(xs.length, 8);
    });

    setTimeout(function () {
        assert.ok(joinExecuted, 'join didn\'t execute');
    }, 2000);
}

exports['range (i..j) (i>j)'] = function () {
    var joinExecuted = false;
    Lazy.range('(10..1)').join(function (xs) {
        joinExecuted = true;
        assert.deepEqual(xs, range(9,1));
        assert.equal(xs.length, 8);
    });

    setTimeout(function () {
        assert.ok(joinExecuted, 'join didn\'t execute');
    }, 2000);
}

exports['range [i,step..j]'] = function () {
    var joinExecuted = false;
    Lazy.range('[5,10..50]').join(function (xs) {
        joinExecuted = true;
        assert.deepEqual(xs, range(5,51,5));
        assert.equal(xs.length, 10);
    });

    setTimeout(function () {
        assert.ok(joinExecuted, 'join didn\'t execute');
    }, 2000);
}

