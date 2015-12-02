var multimeter = require('multimeter');

var multi = multimeter(process);
var charm = multi.charm;
charm.on('^C', process.exit);
charm.reset();

var xs = [];
for (var i = 0; i < 100; i++) xs.push(i);

console.log('Calculating the sum of [0..100]:\n');
charm.write('    ');

multi.drop(function (bar) {
    bar.percent(0);
    
    charm.write('\n\nResult: ');
    charm.position(function (x, y) {
        var sum = 0;
        var iv = setInterval(function () {
            sum += xs.shift();
            
            bar.percent(100 - xs.length);
            
            charm
                .position(x, y)
                .erase('end')
                .write(sum.toString())
            ;
            
            if (xs.length === 0) {
                clearInterval(iv);
                multi.destroy();
            }
        }, 50);
    });
});
