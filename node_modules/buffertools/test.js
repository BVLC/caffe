/* Copyright (c) 2010, Ben Noordhuis <info@bnoordhuis.nl>
 *
 * Permission to use, copy, modify, and/or distribute this software for any
 * purpose with or without fee is hereby granted, provided that the above
 * copyright notice and this permission notice appear in all copies.
 *
 * THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
 * WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
 * ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
 * WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
 * ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
 * OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
 */

var buffertools = require('./buffertools');
var Buffer = require('buffer').Buffer;
var assert = require('assert');

var WritableBufferStream = buffertools.WritableBufferStream;

// Extend Buffer.prototype and SlowBuffer.prototype.
buffertools.extend();

// these trigger the code paths for UnaryAction and BinaryAction
assert.throws(function() { buffertools.clear({}); });
assert.throws(function() { buffertools.equals({}, {}); });

var a = new Buffer('abcd'), b = new Buffer('abcd'),  c = new Buffer('efgh');
assert.ok(a.equals(b));
assert.ok(!a.equals(c));
assert.ok(a.equals('abcd'));
assert.ok(!a.equals('efgh'));

assert.ok(a.compare(a) == 0);
assert.ok(a.compare(c) < 0);
assert.ok(c.compare(a) > 0);

assert.ok(a.compare('abcd') == 0);
assert.ok(a.compare('efgh') < 0);
assert.ok(c.compare('abcd') > 0);

b = new Buffer('****');
assert.equal(b, b.clear());
assert.equal(b.inspect(), '<Buffer 00 00 00 00>');	// FIXME brittle test

b = new Buffer(4);
assert.equal(b, b.fill(42));
assert.equal(b.inspect(), '<Buffer 2a 2a 2a 2a>');

b = new Buffer(4);
assert.equal(b, b.fill('*'));
assert.equal(b.inspect(), '<Buffer 2a 2a 2a 2a>');

b = new Buffer(4);
assert.equal(b, b.fill('ab'));
assert.equal(b.inspect(), '<Buffer 61 62 61 62>');

b = new Buffer(4);
assert.equal(b, b.fill('abcd1234'));
assert.equal(b.inspect(), '<Buffer 61 62 63 64>');

b = new Buffer('Hello, world!');
assert.equal(-1, b.indexOf(new Buffer('foo')));
assert.equal(0,  b.indexOf(new Buffer('Hell')));
assert.equal(7,  b.indexOf(new Buffer('world')));
assert.equal(7,  b.indexOf(new Buffer('world!')));
assert.equal(-1, b.indexOf('foo'));
assert.equal(0,  b.indexOf('Hell'));
assert.equal(7,  b.indexOf('world'));
assert.equal(-1, b.indexOf(''));
assert.equal(-1, b.indexOf('x'));
assert.equal(7,  b.indexOf('w'));
assert.equal(0,  b.indexOf('Hello, world!'));
assert.equal(-1, b.indexOf('Hello, world!1'));
assert.equal(7,  b.indexOf('world', 7));
assert.equal(-1, b.indexOf('world', 8));
assert.equal(7,  b.indexOf('world', -256));
assert.equal(7,  b.indexOf('world', -6));
assert.equal(-1, b.indexOf('world', -5));
assert.equal(-1, b.indexOf('world', 256));
assert.equal(-1, b.indexOf('', 256));

b = new Buffer("\t \r\n");
assert.equal('09200d0a', b.toHex());
assert.equal(b.toString(), new Buffer('09200d0a').fromHex().toString());

// https://github.com/bnoordhuis/node-buffertools/pull/9
b = new Buffer(4);
b[0] = 0x98;
b[1] = 0x95;
b[2] = 0x60;
b[3] = 0x2f;
assert.equal('9895602f', b.toHex());

assert.equal('', buffertools.concat());
assert.equal('', buffertools.concat(''));
assert.equal('foobar', new Buffer('foo').concat('bar'));
assert.equal('foobarbaz', buffertools.concat(new Buffer('foo'), 'bar', new Buffer('baz')));
assert.throws(function() { buffertools.concat('foo', 123, 'baz'); });
// assert that the buffer is copied, not returned as-is
a = new Buffer('For great justice.'), b = buffertools.concat(a);
assert.equal(a.toString(), b.toString());
assert.notEqual(a, b);

assert.equal('', new Buffer('').reverse());
assert.equal('For great justice.', new Buffer('.ecitsuj taerg roF').reverse());

// bug fix, see http://github.com/bnoordhuis/node-buffertools/issues#issue/5
var endOfHeader = new Buffer('\r\n\r\n');
assert.equal(0, endOfHeader.indexOf(endOfHeader));
assert.equal(0, endOfHeader.indexOf('\r\n\r\n'));

// feature request, see https://github.com/bnoordhuis/node-buffertools/issues#issue/8
var closed = false;
var stream = new WritableBufferStream();

stream.on('close', function() { closed = true; });
stream.write('Hello,');
stream.write(' ');
stream.write('world!');
stream.end();

assert.equal(true, closed);
assert.equal(false, stream.writable);
assert.equal('Hello, world!', stream.toString());
assert.equal('Hello, world!', stream.getBuffer().toString());

// closed stream should throw
assert.throws(function() { stream.write('ZIG!'); });

// GH-10 indexOf sometimes incorrectly returns -1
for (var i = 0; i < 100; i++) {
	var buffer = new Buffer('9A8B3F4491734D18DEFC6D2FA96A2D3BC1020EECB811F037F977D039B4713B1984FBAB40FCB4D4833D4A31C538B76EB50F40FA672866D8F50D0A1063666721B8D8322EDEEC74B62E5F5B959393CD3FCE831CC3D1FA69D79C758853AFA3DC54D411043263596BAD1C9652970B80869DD411E82301DF93D47DCD32421A950EF3E555152E051C6943CC3CA71ED0461B37EC97C5A00EBACADAA55B9A7835F148DEF8906914617C6BD3A38E08C14735FC2EFE075CC61DFE5F2F9686AB0D0A3926604E320160FDC1A4488A323CB4308CDCA4FD9701D87CE689AF999C5C409854B268D00B063A89C2EEF6673C80A4F4D8D0A00163082EDD20A2F1861512F6FE9BB479A22A3D4ACDD2AA848254BA74613190957C7FCD106BF7441946D0E1A562DA68BC37752B1551B8855C8DA08DFE588902D44B2CAB163F3D7D7706B9CC78900D0AFD5DAE5492535A17DB17E24389F3BAA6F5A95B9F6FE955193D40932B5988BC53E49CAC81955A28B81F7B36A1EDA3B4063CBC187B0488FCD51FAE71E4FBAEE56059D847591B960921247A6B7C5C2A7A757EC62A2A2A2A2A2A2A25552591C03EF48994BD9F594A5E14672F55359EF1B38BF2976D1216C86A59847A6B7C4A5C585A0D0A2A6D9C8F8B9E999C2A836F786D577A79816F7C577A797D7E576B506B57A05B5B8C4A8D99989E8B8D9E644A6B9D9D8F9C9E4A504A6B968B93984A93984A988FA19D919C999F9A4A8B969E588C93988B9C938F9D588D8B9C9E9999989D58909C8F988D92588E0D0A3D79656E642073697A653D373035393620706172743D31207063726333323D33616230646235300D0A2E0D0A').fromHex();
	assert.equal(551, buffer.indexOf('=yend'));
}
