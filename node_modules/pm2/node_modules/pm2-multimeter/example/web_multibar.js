var http = require('http');
var multimeter = require('multimeter');

http.createServer(function (req, res) {
    res.setHeader('content-type', 'application/octet-stream');
    
    var multi = multimeter(res);
    multi.charm.on('^C', process.exit);
    multi.charm.reset();
    
    var bars = [];
    var progress = [];
    var deltas = [];
    
    for (var i = 0; i < 5; i++) {
        var s = 'ABCDE'[i] + ': \n';
        multi.write(s);
        
        var bar = multi(s.length, i + 1, {
            width : 20,
            solid : {
                text : '|',
                foreground : 'white',
                background : 'blue'
            },
            empty : { text : ' ' },
        });
        bars.push(bar);
        
        deltas[i] = 1 + Math.random() * 9;
        progress.push(0);
    }
    
    multi.write('\nbeep boop\n');
    
    var pending = progress.length;
    var iv = setInterval(function () {
        progress.forEach(function (p, i) {
            progress[i] += Math.random() * deltas[i];
            bars[i].percent(progress[i]);
            if (p < 100 && progress[i] >= 100) pending --;
            if (pending === 0) {
                multi.write('\nAll done.\n');
                res.end();
            }
        });
    }, 100);
    
    res.connection.on('end', function () {
        multi.destroy();
        clearInterval(iv);
    });
}).listen(8080);

console.log('curl -N localhost:8080');
