# infinity-agent [![Build Status](https://travis-ci.org/floatdrop/infinity-agent.svg?branch=master)](https://travis-ci.org/floatdrop/infinity-agent)

Node-core HTTP Agent for userland.

## Usage

```js
var infinityAgent = require('infinity-agent');

var http = require('http');
var https = require('https');

http.get('http://google.com', {agent: infinityAgent.http.globalAgent});
https.get('http://google.com', {agent: infinityAgent.https.globalAgent});
```

This package is a mirror of the Agent class in Node-core.

## License

MIT © [Vsevolod Strukchinsky](floatdrop@gmail.com)
