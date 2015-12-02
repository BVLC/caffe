
var code = Number(process.argv[2]);

process.stdout.write('stdout\n');
process.stderr.write('stderr\n');

// Instead of process.exit. See https://github.com/cowboy/node-exit
require('exit')(code);
