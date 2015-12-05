var c = require('../')();
c.pipe(process.stdout);
c.on('^C', process.exit);

var queue = (function () {
    var tasks = [];
    var pending = false;
    
    return {
        abort : function () {
            tasks = [];
            next();
        },
        push : function (t) {
            tasks.push(t);
            if (!pending) next();
        }
    };
    
    function next () {
        pending = true;
        process.nextTick(function () {
            if (tasks.length === 0) return;
            var t = tasks.shift();
            t();
            pending = false;
            next();
        });
    }
})();

process.stdout.on('resize', draw);
draw();
setInterval(function () {}, 1e8);

function draw () {
    var cols = process.stdout.columns;
    var rows = process.stdout.rows;
    queue.abort();
    
    queue.push(function () {
        c.reset();
        c.background('blue');
        c.position(1, 1);
        c.write(Array(cols + 1).join(' '));
    });
    
    for (var y = 1; y < rows; y++) {
        queue.push(function () {
            c.position(1, y);
            c.write(' ');
            c.position(cols, y);
            c.write(' ');
        });
    }
    
    queue.push(function () {
        c.position(1, rows);
        c.write(Array(cols + 1).join(' '));
        c.display('reset');
    });
}
