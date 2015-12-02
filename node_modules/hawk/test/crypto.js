'use strict';

// Load modules

const Code = require('code');
const Hawk = require('../lib');
const Lab = require('lab');


// Declare internals

const internals = {};


// Test shortcuts

const lab = exports.lab = Lab.script();
const describe = lab.experiment;
const it = lab.test;
const expect = Code.expect;


describe('Crypto', () => {

    describe('generateNormalizedString()', () => {

        it('should return a valid normalized string', (done) => {

            expect(Hawk.crypto.generateNormalizedString('header', {
                ts: 1357747017,
                nonce: 'k3k4j5',
                method: 'GET',
                resource: '/resource/something',
                host: 'example.com',
                port: 8080
            })).to.equal('hawk.1.header\n1357747017\nk3k4j5\nGET\n/resource/something\nexample.com\n8080\n\n\n');

            done();
        });

        it('should return a valid normalized string (ext)', (done) => {

            expect(Hawk.crypto.generateNormalizedString('header', {
                ts: 1357747017,
                nonce: 'k3k4j5',
                method: 'GET',
                resource: '/resource/something',
                host: 'example.com',
                port: 8080,
                ext: 'this is some app data'
            })).to.equal('hawk.1.header\n1357747017\nk3k4j5\nGET\n/resource/something\nexample.com\n8080\n\nthis is some app data\n');

            done();
        });

        it('should return a valid normalized string (payload + ext)', (done) => {

            expect(Hawk.crypto.generateNormalizedString('header', {
                ts: 1357747017,
                nonce: 'k3k4j5',
                method: 'GET',
                resource: '/resource/something',
                host: 'example.com',
                port: 8080,
                hash: 'U4MKKSmiVxk37JCCrAVIjV/OhB3y+NdwoCr6RShbVkE=',
                ext: 'this is some app data'
            })).to.equal('hawk.1.header\n1357747017\nk3k4j5\nGET\n/resource/something\nexample.com\n8080\nU4MKKSmiVxk37JCCrAVIjV/OhB3y+NdwoCr6RShbVkE=\nthis is some app data\n');

            done();
        });
    });
});
