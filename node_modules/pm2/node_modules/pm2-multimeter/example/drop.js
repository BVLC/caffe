var multimeter = require('multimeter');
var multi = multimeter(process);

multi.on('^C', function () {
    multi.charm.cursor(true);
    multi.write('\n').destroy();
    
    process.exit();
});
multi.charm.cursor(false);

multi.drop(function (bar) {
    var iv = setInterval(function () {
        var p = bar.percent();
        bar.percent(p + 1);
        if (p >= 100) {
            clearInterval(iv);
            
            multi.charm.cursor(true);
            multi.write('\n').destroy();
        }
    }, 25);
});
