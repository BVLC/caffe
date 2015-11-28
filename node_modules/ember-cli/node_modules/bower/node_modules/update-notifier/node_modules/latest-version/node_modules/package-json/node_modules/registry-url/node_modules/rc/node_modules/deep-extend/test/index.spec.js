'use strict';

var should = require('should');
var extend = require('../index'); // it must be ./lib/deep-extend.js

describe('deep-extend', function () {

	it('can extend on 1 level', function () {
		var a = { hello: 1 };
		var b = { world: 2 };
		extend(a, b);
		a.should.eql({
			hello: 1,
			world: 2
		});
	});

	it('can extend on 2 levels', function () {
		var a = { person: { name: 'John' } };
		var b = { person: { age: 30 } };
		extend(a, b);
		a.should.eql({
			person: { name: 'John', age: 30 }
		});
	});

	it('can extend with Buffer values', function () {
		var a = { hello: 1 };
		var b = { value: new Buffer('world') };
		extend(a, b);
		a.should.eql({
			hello: 1,
			value: new Buffer('world')
		});
	});

	it('Buffer is cloned', function () {
		var a = { };
		var b = { value: new Buffer('foo') };
		extend(a, b);
		a.value.write('bar');
		a.value.toString().should.eql('bar');
		b.value.toString().should.eql('foo');
	});

	it('Date objects', function () {
		var a = { d: new Date() };
		var b = extend({}, a);
		b.d.should.instanceOf(Date);
	});

	it('Date object is cloned', function () {
		var a = { d: new Date() };
		var b = extend({}, a);
		b.d.setTime( (new Date()).getTime() + 100000 );
		b.d.getTime().should.not.eql( a.d.getTime() );
	});

	it('RegExp objects', function () {
		var a = { d: new RegExp() };
		var b = extend({}, a);
		b.d.should.instanceOf(RegExp);
	});

	it('RegExp object is cloned', function () {
		var a = { d: new RegExp('b', 'g') };
		var b = extend({}, a);
		b.d.test('abc');
		b.d.lastIndex.should.not.eql( a.d.lastIndex );
	});

	it('doesn\'t change sources', function () {
		var a = {a: [1]};
		var b = {a: [2]};
		var c = {c: 3};
		var d = extend({}, a, b, c);

		a.should.eql({a: [1]});
		b.should.eql({a: [2]});
		c.should.eql({c: 3});
	});

	it('example from README.md', function () {
		var obj1 = {
			a: 1,
			b: 2,
			d: {
				a: 1,
				b: [],
				c: { test1: 123, test2: 321 }
			},
			f: 5,
			g: 123,
			i: 321,
			j: [1, 2]
		};
		var obj2 = {
			b: 3,
			c: 5,
			d: {
				b: { first: 'one', second: 'two' },
				c: { test2: 222 }
			},
			e: { one: 1, two: 2 },
			f: [],
			g: (void 0),
			h: /abc/g,
			i: null,
			j: [3, 4]
		};

		extend(obj1, obj2);

		obj1.should.eql({
			a: 1,
			b: 3,
			d: {
				a: 1,
				b: { first: 'one', second: 'two' },
				c: { test1: 123, test2: 222 }
			},
			f: [],
			g: undefined,
			c: 5,
			e: { one: 1, two: 2 },
			h: /abc/g,
			i: null,
			j: [3, 4]
		});

		('g' in obj1).should.eql(true);
		('x' in obj1).should.eql(false);
	});

	it('clone arrays instead of extend', function () {
		extend({a: [1, 2, 3]}, {a: [2, 3]}).should.eql({a: [2, 3]});
	});

	it('recursive clone objects and special objects in cloned arrays', function () {
		var obj1 = {
			x: 1,
			y: new Buffer('foo')
		};
		var b = new Buffer('bar');
		var obj2 = {
			x: 1,
			y: [2, 4, obj1, b],
			z: new Buffer('test')
		};
		var foo = {
			a: [obj2, obj2]
		};
		var bar = extend({}, foo);
		bar.a[0].x = 2;
		bar.a[0].z.write('text', 'utf-8');
		bar.a[1].x = 3;
		bar.a[1].z.write('lel', 'utf-8');
		bar.a[0].y[0] = 3;
		bar.a[0].y[2].x = 5;
		bar.a[0].y[2].y.write('heh', 'utf-8');
		bar.a[0].y[3].write('ho', 'utf-8');
		bar.a[1].y[1] = 3;
		bar.a[1].y[2].y.write('nah', 'utf-8');
		bar.a[1].y[3].write('he', 'utf-8');

		obj2.x.should.eql(1);
		obj2.z.toString().should.eql('test');
		bar.a[0].x.should.eql(2);
		bar.a[0].z.toString().should.eql('text');
		bar.a[1].x.should.eql(3);
		bar.a[1].z.toString().should.eql('lelt');
		obj1.x.should.eql(1);
		obj1.y.toString().should.eql('foo');
		b.toString().should.eql('bar');

		bar.a[0].y[0].should.eql(3);
		bar.a[0].y[1].should.eql(4);
		bar.a[0].y[2].x.should.eql(5);
		bar.a[0].y[2].y.toString().should.eql('heh');
		bar.a[0].y[3].toString().should.eql('hor');

		bar.a[1].y[0].should.eql(2);
		bar.a[1].y[1].should.eql(3);
		bar.a[1].y[2].x.should.eql(1);
		bar.a[1].y[2].y.toString().should.eql('nah');
		bar.a[1].y[3].toString().should.eql('her');

		foo.a.length.should.eql(2);
		bar.a.length.should.eql(2);
		Object.keys(obj2).should.eql(['x', 'y', 'z']);
		Object.keys(bar.a[0]).should.eql(['x', 'y', 'z']);
		Object.keys(bar.a[1]).should.eql(['x', 'y', 'z']);
		obj2.y.length.should.eql(4);
		bar.a[0].y.length.should.eql(4);
		bar.a[1].y.length.should.eql(4);
		Object.keys(obj2.y[2]).should.eql(['x', 'y']);
		Object.keys(bar.a[0].y[2]).should.eql(['x', 'y']);
		Object.keys(bar.a[1].y[2]).should.eql(['x', 'y']);
	});

	it('checking keys for hasOwnPrototype', function () {
		var A = function () {
			this.x = 1;
			this.y = 2;
		};
		A.prototype.z = 3;
		var foo = new A();
		extend({x: 123}, foo).should.eql({
			x: 1,
			y: 2
		});
		foo.z = 5;
		extend({x: 123}, foo, {y: 22}).should.eql({
			x: 1,
			y: 22,
			z: 5
		});
	});

});
