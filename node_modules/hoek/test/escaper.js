'use strict';

// Load modules

const Code = require('code');
const Hoek = require('../lib');
const Lab = require('lab');


// Declare internals

const internals = {};


// Test shortcuts

const lab = exports.lab = Lab.script();
const describe = lab.experiment;
const it = lab.test;
const expect = Code.expect;


describe('escapeJavaScript()', () => {

    it('encodes / characters', (done) => {

        const encoded = Hoek.escapeJavaScript('<script>alert(1)</script>');
        expect(encoded).to.equal('\\x3cscript\\x3ealert\\x281\\x29\\x3c\\x2fscript\\x3e');
        done();
    });

    it('encodes \' characters', (done) => {

        const encoded = Hoek.escapeJavaScript('something(\'param\')');
        expect(encoded).to.equal('something\\x28\\x27param\\x27\\x29');
        done();
    });

    it('encodes large unicode characters with the correct padding', (done) => {

        const encoded = Hoek.escapeJavaScript(String.fromCharCode(500) + String.fromCharCode(1000));
        expect(encoded).to.equal('\\u0500\\u1000');
        done();
    });

    it('doesn\'t throw an exception when passed null', (done) => {

        const encoded = Hoek.escapeJavaScript(null);
        expect(encoded).to.equal('');
        done();
    });
});

describe('escapeHtml()', () => {

    it('encodes / characters', (done) => {

        const encoded = Hoek.escapeHtml('<script>alert(1)</script>');
        expect(encoded).to.equal('&lt;script&gt;alert&#x28;1&#x29;&lt;&#x2f;script&gt;');
        done();
    });

    it('encodes < and > as named characters', (done) => {

        const encoded = Hoek.escapeHtml('<script><>');
        expect(encoded).to.equal('&lt;script&gt;&lt;&gt;');
        done();
    });

    it('encodes large unicode characters', (done) => {

        const encoded = Hoek.escapeHtml(String.fromCharCode(500) + String.fromCharCode(1000));
        expect(encoded).to.equal('&#500;&#1000;');
        done();
    });

    it('doesn\'t throw an exception when passed null', (done) => {

        const encoded = Hoek.escapeHtml(null);
        expect(encoded).to.equal('');
        done();
    });

    it('encodes {} characters', (done) => {

        const encoded = Hoek.escapeHtml('{}');
        expect(encoded).to.equal('&#x7b;&#x7d;');
        done();
    });
});
