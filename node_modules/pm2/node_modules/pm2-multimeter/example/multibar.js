var multimeter = require('multimeter');

var multi = multimeter(process);
multi.on('^C', process.exit);
multi.charm.reset();
    
var bars = [];
var progress = [];
var deltas = [];

multi.write('Progress:\n\n');

for (var i = 0; i < 5; i++) {
    var s = 'ABCDE'[i] + ': \n';
    multi.write(s);
    
    var bar = multi(s.length, i + 3, {
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
            multi.destroy();
            clearInterval(iv);
            pending --;
        }
    });
}, 100);
