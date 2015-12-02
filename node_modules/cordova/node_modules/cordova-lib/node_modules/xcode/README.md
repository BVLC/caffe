# node-xcode

> parser/toolkit for xcodeproj project files

Allows you to edit xcodeproject files and write them back out.

## Example

    // API is a bit wonky right now
    var xcode = require('xcode'),
        fs = require('fs'),
        projectPath = 'myproject.xcodeproj/project.pbxproj',
        myProj = xcode.project(projectPath);

    // parsing is async, in a different process
    myProj.parse(function (err) {
        myProj.addHeaderFile('foo.h');
        myProj.addSourceFile('foo.m');
        myProj.addFramework('FooKit.framework');
        
        fs.writeFileSync(projectPath, myProj.writeSync());
        console.log('new project written');
    });

## Working on the parser

If there's a problem parsing, you will want to edit the grammar under
`lib/parser/pbxproj.pegjs`. You can test it online with the PEGjs online thingy
at http://pegjs.majda.cz/online - I have had some mixed results though.

Tests under the `test/parser` directory will compile the parser from the
grammar. Other tests will use the prebuilt parser (`lib/parser/pbxproj.js`).

To rebuild the parser js file after editing the grammar, run:

    ./node_modules/.bin/pegjs lib/parser/pbxproj.pegjs

(easier if `./node_modules/.bin` is in your path)

## License

MIT
