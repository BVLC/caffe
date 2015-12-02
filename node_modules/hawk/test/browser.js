'use strict';

// Load modules

const Code = require('code');
const Hawk = require('../lib');
const Hoek = require('hoek');
const Lab = require('lab');
const Browser = require('../lib/browser');


// Declare internals

const internals = {};


// Test shortcuts

const lab = exports.lab = Lab.script();
const describe = lab.experiment;
const it = lab.test;
const expect = Code.expect;


describe('Browser', () => {

    const credentialsFunc = function (id, callback) {

        const credentials = {
            id: id,
            key: 'werxhqb98rpaxn39848xrunpaw3489ruxnpa98w4rxn',
            algorithm: (id === '1' ? 'sha1' : 'sha256'),
            user: 'steve'
        };

        return callback(null, credentials);
    };

    it('should generate a bewit then successfully authenticate it', (done) => {

        const req = {
            method: 'GET',
            url: '/resource/4?a=1&b=2',
            host: 'example.com',
            port: 80
        };

        credentialsFunc('123456', (err, credentials1) => {

            const bewit = Browser.client.bewit('http://example.com/resource/4?a=1&b=2', { credentials: credentials1, ttlSec: 60 * 60 * 24 * 365 * 100, ext: 'some-app-data' });
            req.url += '&bewit=' + bewit;

            Hawk.uri.authenticate(req, credentialsFunc, {}, (err, credentials2, attributes) => {

                expect(err).to.not.exist();
                expect(credentials2.user).to.equal('steve');
                expect(attributes.ext).to.equal('some-app-data');
                done();
            });
        });
    });

    it('should generate a bewit then successfully authenticate it (no ext)', (done) => {

        const req = {
            method: 'GET',
            url: '/resource/4?a=1&b=2',
            host: 'example.com',
            port: 80
        };

        credentialsFunc('123456', (err, credentials1) => {

            const bewit = Browser.client.bewit('http://example.com/resource/4?a=1&b=2', { credentials: credentials1, ttlSec: 60 * 60 * 24 * 365 * 100 });
            req.url += '&bewit=' + bewit;

            Hawk.uri.authenticate(req, credentialsFunc, {}, (err, credentials2, attributes) => {

                expect(err).to.not.exist();
                expect(credentials2.user).to.equal('steve');
                done();
            });
        });
    });

    describe('bewit()', () => {

        it('returns a valid bewit value', (done) => {

            const credentials = {
                id: '123456',
                key: '2983d45yun89q',
                algorithm: 'sha256'
            };

            const bewit = Browser.client.bewit('https://example.com/somewhere/over/the/rainbow', { credentials: credentials, ttlSec: 300, localtimeOffsetMsec: 1356420407232 - Hawk.utils.now(), ext: 'xandyandz' });
            expect(bewit).to.equal('MTIzNDU2XDEzNTY0MjA3MDdca3NjeHdOUjJ0SnBQMVQxekRMTlBiQjVVaUtJVTl0T1NKWFRVZEc3WDloOD1ceGFuZHlhbmR6');
            done();
        });

        it('returns a valid bewit value (explicit HTTP port)', (done) => {

            const credentials = {
                id: '123456',
                key: '2983d45yun89q',
                algorithm: 'sha256'
            };

            const bewit = Browser.client.bewit('http://example.com:8080/somewhere/over/the/rainbow', { credentials: credentials, ttlSec: 300, localtimeOffsetMsec: 1356420407232 - Hawk.utils.now(), ext: 'xandyandz' });
            expect(bewit).to.equal('MTIzNDU2XDEzNTY0MjA3MDdcaFpiSjNQMmNLRW80a3kwQzhqa1pBa1J5Q1p1ZWc0V1NOYnhWN3ZxM3hIVT1ceGFuZHlhbmR6');
            done();
        });

        it('returns a valid bewit value (explicit HTTPS port)', (done) => {

            const credentials = {
                id: '123456',
                key: '2983d45yun89q',
                algorithm: 'sha256'
            };

            const bewit = Browser.client.bewit('https://example.com:8043/somewhere/over/the/rainbow', { credentials: credentials, ttlSec: 300, localtimeOffsetMsec: 1356420407232 - Hawk.utils.now(), ext: 'xandyandz' });
            expect(bewit).to.equal('MTIzNDU2XDEzNTY0MjA3MDdcL2t4UjhwK0xSaTdvQTRnUXc3cWlxa3BiVHRKYkR4OEtRMC9HRUwvVytTUT1ceGFuZHlhbmR6');
            done();
        });

        it('returns a valid bewit value (null ext)', (done) => {

            const credentials = {
                id: '123456',
                key: '2983d45yun89q',
                algorithm: 'sha256'
            };

            const bewit = Browser.client.bewit('https://example.com/somewhere/over/the/rainbow', { credentials: credentials, ttlSec: 300, localtimeOffsetMsec: 1356420407232 - Hawk.utils.now(), ext: null });
            expect(bewit).to.equal('MTIzNDU2XDEzNTY0MjA3MDdcSUdZbUxnSXFMckNlOEN4dktQczRKbFdJQStValdKSm91d2dBUmlWaENBZz1c');
            done();
        });

        it('errors on invalid options', (done) => {

            const bewit = Browser.client.bewit('https://example.com/somewhere/over/the/rainbow', 4);
            expect(bewit).to.equal('');
            done();
        });

        it('errors on missing uri', (done) => {

            const credentials = {
                id: '123456',
                key: '2983d45yun89q',
                algorithm: 'sha256'
            };

            const bewit = Browser.client.bewit('', { credentials: credentials, ttlSec: 300, localtimeOffsetMsec: 1356420407232 - Hawk.utils.now(), ext: 'xandyandz' });
            expect(bewit).to.equal('');
            done();
        });

        it('errors on invalid uri', (done) => {

            const credentials = {
                id: '123456',
                key: '2983d45yun89q',
                algorithm: 'sha256'
            };

            const bewit = Browser.client.bewit(5, { credentials: credentials, ttlSec: 300, localtimeOffsetMsec: 1356420407232 - Hawk.utils.now(), ext: 'xandyandz' });
            expect(bewit).to.equal('');
            done();
        });

        it('errors on invalid credentials (id)', (done) => {

            const credentials = {
                key: '2983d45yun89q',
                algorithm: 'sha256'
            };

            const bewit = Browser.client.bewit('https://example.com/somewhere/over/the/rainbow', { credentials: credentials, ttlSec: 3000, ext: 'xandyandz' });
            expect(bewit).to.equal('');
            done();
        });

        it('errors on missing credentials', (done) => {

            const bewit = Browser.client.bewit('https://example.com/somewhere/over/the/rainbow', { ttlSec: 3000, ext: 'xandyandz' });
            expect(bewit).to.equal('');
            done();
        });

        it('errors on invalid credentials (key)', (done) => {

            const credentials = {
                id: '123456',
                algorithm: 'sha256'
            };

            const bewit = Browser.client.bewit('https://example.com/somewhere/over/the/rainbow', { credentials: credentials, ttlSec: 3000, ext: 'xandyandz' });
            expect(bewit).to.equal('');
            done();
        });

        it('errors on invalid algorithm', (done) => {

            const credentials = {
                id: '123456',
                key: '2983d45yun89q',
                algorithm: 'hmac-sha-0'
            };

            const bewit = Browser.client.bewit('https://example.com/somewhere/over/the/rainbow', { credentials: credentials, ttlSec: 300, ext: 'xandyandz' });
            expect(bewit).to.equal('');
            done();
        });

        it('errors on missing options', (done) => {

            const bewit = Browser.client.bewit('https://example.com/somewhere/over/the/rainbow');
            expect(bewit).to.equal('');
            done();
        });
    });

    it('generates a header then successfully parse it (configuration)', (done) => {

        const req = {
            method: 'GET',
            url: '/resource/4?filter=a',
            host: 'example.com',
            port: 8080
        };

        credentialsFunc('123456', (err, credentials1) => {

            req.authorization = Browser.client.header('http://example.com:8080/resource/4?filter=a', req.method, { credentials: credentials1, ext: 'some-app-data' }).field;
            expect(req.authorization).to.exist();

            Hawk.server.authenticate(req, credentialsFunc, {}, (err, credentials2, artifacts) => {

                expect(err).to.not.exist();
                expect(credentials2.user).to.equal('steve');
                expect(artifacts.ext).to.equal('some-app-data');
                done();
            });
        });
    });

    it('generates a header then successfully parse it (node request)', (done) => {

        const req = {
            method: 'POST',
            url: '/resource/4?filter=a',
            headers: {
                host: 'example.com:8080',
                'content-type': 'text/plain;x=y'
            }
        };

        const payload = 'some not so random text';

        credentialsFunc('123456', (err, credentials1) => {

            const reqHeader = Browser.client.header('http://example.com:8080/resource/4?filter=a', req.method, { credentials: credentials1, ext: 'some-app-data', payload: payload, contentType: req.headers['content-type'] });
            req.headers.authorization = reqHeader.field;

            Hawk.server.authenticate(req, credentialsFunc, {}, (err, credentials2, artifacts) => {

                expect(err).to.not.exist();
                expect(credentials2.user).to.equal('steve');
                expect(artifacts.ext).to.equal('some-app-data');
                expect(Hawk.server.authenticatePayload(payload, credentials2, artifacts, req.headers['content-type'])).to.equal(true);

                const res = {
                    headers: {
                        'content-type': 'text/plain'
                    },
                    getResponseHeader: function (header) {

                        return res.headers[header.toLowerCase()];
                    }
                };

                res.headers['server-authorization'] = Hawk.server.header(credentials2, artifacts, { payload: 'some reply', contentType: 'text/plain', ext: 'response-specific' });
                expect(res.headers['server-authorization']).to.exist();

                expect(Browser.client.authenticate(res, credentials2, artifacts, { payload: 'some reply' })).to.equal(true);
                done();
            });
        });
    });

    it('generates a header then successfully parse it (browserify)', (done) => {

        const req = {
            method: 'POST',
            url: '/resource/4?filter=a',
            headers: {
                host: 'example.com:8080',
                'content-type': 'text/plain;x=y'
            }
        };

        const payload = 'some not so random text';

        credentialsFunc('123456', (err, credentials1) => {

            const reqHeader = Browser.client.header('http://example.com:8080/resource/4?filter=a', req.method, { credentials: credentials1, ext: 'some-app-data', payload: payload, contentType: req.headers['content-type'] });
            req.headers.authorization = reqHeader.field;

            Hawk.server.authenticate(req, credentialsFunc, {}, (err, credentials2, artifacts) => {

                expect(err).to.not.exist();
                expect(credentials2.user).to.equal('steve');
                expect(artifacts.ext).to.equal('some-app-data');
                expect(Hawk.server.authenticatePayload(payload, credentials2, artifacts, req.headers['content-type'])).to.equal(true);

                const res = {
                    headers: {
                        'content-type': 'text/plain'
                    },
                    getHeader: function (header) {

                        return res.headers[header.toLowerCase()];
                    }
                };

                res.headers['server-authorization'] = Hawk.server.header(credentials2, artifacts, { payload: 'some reply', contentType: 'text/plain', ext: 'response-specific' });
                expect(res.headers['server-authorization']).to.exist();

                expect(Browser.client.authenticate(res, credentials2, artifacts, { payload: 'some reply' })).to.equal(true);
                done();
            });
        });
    });

    it('generates a header then successfully parse it (time offset)', (done) => {

        const req = {
            method: 'GET',
            url: '/resource/4?filter=a',
            host: 'example.com',
            port: 8080
        };

        credentialsFunc('123456', (err, credentials1) => {

            req.authorization = Browser.client.header('http://example.com:8080/resource/4?filter=a', req.method, { credentials: credentials1, ext: 'some-app-data', localtimeOffsetMsec: 100000 }).field;
            expect(req.authorization).to.exist();

            Hawk.server.authenticate(req, credentialsFunc, { localtimeOffsetMsec: 100000 }, (err, credentials2, artifacts) => {

                expect(err).to.not.exist();
                expect(credentials2.user).to.equal('steve');
                expect(artifacts.ext).to.equal('some-app-data');
                done();
            });
        });
    });

    it('generates a header then successfully parse it (no server header options)', (done) => {

        const req = {
            method: 'POST',
            url: '/resource/4?filter=a',
            headers: {
                host: 'example.com:8080',
                'content-type': 'text/plain;x=y'
            }
        };

        const payload = 'some not so random text';

        credentialsFunc('123456', (err, credentials1) => {

            const reqHeader = Browser.client.header('http://example.com:8080/resource/4?filter=a', req.method, { credentials: credentials1, ext: 'some-app-data', payload: payload, contentType: req.headers['content-type'] });
            req.headers.authorization = reqHeader.field;

            Hawk.server.authenticate(req, credentialsFunc, {}, (err, credentials2, artifacts) => {

                expect(err).to.not.exist();
                expect(credentials2.user).to.equal('steve');
                expect(artifacts.ext).to.equal('some-app-data');
                expect(Hawk.server.authenticatePayload(payload, credentials2, artifacts, req.headers['content-type'])).to.equal(true);

                const res = {
                    headers: {
                        'content-type': 'text/plain'
                    },
                    getResponseHeader: function (header) {

                        return res.headers[header.toLowerCase()];
                    }
                };

                res.headers['server-authorization'] = Hawk.server.header(credentials2, artifacts);
                expect(res.headers['server-authorization']).to.exist();

                expect(Browser.client.authenticate(res, credentials2, artifacts)).to.equal(true);
                done();
            });
        });
    });

    it('generates a header then successfully parse it (no server header)', (done) => {

        const req = {
            method: 'POST',
            url: '/resource/4?filter=a',
            headers: {
                host: 'example.com:8080',
                'content-type': 'text/plain;x=y'
            }
        };

        const payload = 'some not so random text';

        credentialsFunc('123456', (err, credentials1) => {

            const reqHeader = Browser.client.header('http://example.com:8080/resource/4?filter=a', req.method, { credentials: credentials1, ext: 'some-app-data', payload: payload, contentType: req.headers['content-type'] });
            req.headers.authorization = reqHeader.field;

            Hawk.server.authenticate(req, credentialsFunc, {}, (err, credentials2, artifacts) => {

                expect(err).to.not.exist();
                expect(credentials2.user).to.equal('steve');
                expect(artifacts.ext).to.equal('some-app-data');
                expect(Hawk.server.authenticatePayload(payload, credentials2, artifacts, req.headers['content-type'])).to.equal(true);

                const res = {
                    headers: {
                        'content-type': 'text/plain'
                    },
                    getResponseHeader: function (header) {

                        return res.headers[header.toLowerCase()];
                    }
                };

                expect(Browser.client.authenticate(res, credentials2, artifacts)).to.equal(true);
                done();
            });
        });
    });

    it('generates a header with stale ts and successfully authenticate on second call', (done) => {

        const req = {
            method: 'GET',
            url: '/resource/4?filter=a',
            host: 'example.com',
            port: 8080
        };

        credentialsFunc('123456', (err, credentials1) => {

            Browser.utils.setNtpOffset(60 * 60 * 1000);
            const header = Browser.client.header('http://example.com:8080/resource/4?filter=a', req.method, { credentials: credentials1, ext: 'some-app-data' });
            req.authorization = header.field;
            expect(req.authorization).to.exist();

            Hawk.server.authenticate(req, credentialsFunc, {}, (err, credentials2, artifacts2) => {

                expect(err).to.exist();
                expect(err.message).to.equal('Stale timestamp');

                const res = {
                    headers: {
                        'www-authenticate': err.output.headers['WWW-Authenticate']
                    },
                    getResponseHeader: function (lookup) {

                        return res.headers[lookup.toLowerCase()];
                    }
                };

                expect(Browser.utils.getNtpOffset()).to.equal(60 * 60 * 1000);
                expect(Browser.client.authenticate(res, credentials2, header.artifacts)).to.equal(true);
                expect(Browser.utils.getNtpOffset()).to.equal(0);

                req.authorization = Browser.client.header('http://example.com:8080/resource/4?filter=a', req.method, { credentials: credentials2, ext: 'some-app-data' }).field;
                expect(req.authorization).to.exist();

                Hawk.server.authenticate(req, credentialsFunc, {}, (err, credentials3, artifacts3) => {

                    expect(err).to.not.exist();
                    expect(credentials3.user).to.equal('steve');
                    expect(artifacts3.ext).to.equal('some-app-data');
                    done();
                });
            });
        });
    });

    it('generates a header with stale ts and successfully authenticate on second call (manual localStorage)', (done) => {

        const req = {
            method: 'GET',
            url: '/resource/4?filter=a',
            host: 'example.com',
            port: 8080
        };

        credentialsFunc('123456', (err, credentials1) => {

            const localStorage = new Browser.internals.LocalStorage();

            Browser.utils.setStorage(localStorage);

            Browser.utils.setNtpOffset(60 * 60 * 1000);
            const header = Browser.client.header('http://example.com:8080/resource/4?filter=a', req.method, { credentials: credentials1, ext: 'some-app-data' });
            req.authorization = header.field;
            expect(req.authorization).to.exist();

            Hawk.server.authenticate(req, credentialsFunc, {}, (err, credentials2, artifacts2) => {

                expect(err).to.exist();
                expect(err.message).to.equal('Stale timestamp');

                const res = {
                    headers: {
                        'www-authenticate': err.output.headers['WWW-Authenticate']
                    },
                    getResponseHeader: function (lookup) {

                        return res.headers[lookup.toLowerCase()];
                    }
                };

                expect(parseInt(localStorage.getItem('hawk_ntp_offset'))).to.equal(60 * 60 * 1000);
                expect(Browser.utils.getNtpOffset()).to.equal(60 * 60 * 1000);
                expect(Browser.client.authenticate(res, credentials2, header.artifacts)).to.equal(true);
                expect(Browser.utils.getNtpOffset()).to.equal(0);
                expect(parseInt(localStorage.getItem('hawk_ntp_offset'))).to.equal(0);

                req.authorization = Browser.client.header('http://example.com:8080/resource/4?filter=a', req.method, { credentials: credentials2, ext: 'some-app-data' }).field;
                expect(req.authorization).to.exist();

                Hawk.server.authenticate(req, credentialsFunc, {}, (err, credentials3, artifacts3) => {

                    expect(err).to.not.exist();
                    expect(credentials3.user).to.equal('steve');
                    expect(artifacts3.ext).to.equal('some-app-data');
                    done();
                });
            });
        });
    });

    it('generates a header then fails to parse it (missing server header hash)', (done) => {

        const req = {
            method: 'POST',
            url: '/resource/4?filter=a',
            headers: {
                host: 'example.com:8080',
                'content-type': 'text/plain;x=y'
            }
        };

        const payload = 'some not so random text';

        credentialsFunc('123456', (err, credentials1) => {

            const reqHeader = Browser.client.header('http://example.com:8080/resource/4?filter=a', req.method, { credentials: credentials1, ext: 'some-app-data', payload: payload, contentType: req.headers['content-type'] });
            req.headers.authorization = reqHeader.field;

            Hawk.server.authenticate(req, credentialsFunc, {}, (err, credentials2, artifacts) => {

                expect(err).to.not.exist();
                expect(credentials2.user).to.equal('steve');
                expect(artifacts.ext).to.equal('some-app-data');
                expect(Hawk.server.authenticatePayload(payload, credentials2, artifacts, req.headers['content-type'])).to.equal(true);

                const res = {
                    headers: {
                        'content-type': 'text/plain'
                    },
                    getResponseHeader: function (header) {

                        return res.headers[header.toLowerCase()];
                    }
                };

                res.headers['server-authorization'] = Hawk.server.header(credentials2, artifacts);
                expect(res.headers['server-authorization']).to.exist();

                expect(Browser.client.authenticate(res, credentials2, artifacts, { payload: 'some reply' })).to.equal(false);
                done();
            });
        });
    });

    it('generates a header then successfully parse it (with hash)', (done) => {

        const req = {
            method: 'GET',
            url: '/resource/4?filter=a',
            host: 'example.com',
            port: 8080
        };

        credentialsFunc('123456', (err, credentials1) => {

            req.authorization = Browser.client.header('http://example.com:8080/resource/4?filter=a', req.method, { credentials: credentials1, payload: 'hola!', ext: 'some-app-data' }).field;
            Hawk.server.authenticate(req, credentialsFunc, {}, (err, credentials2, artifacts) => {

                expect(err).to.not.exist();
                expect(credentials2.user).to.equal('steve');
                expect(artifacts.ext).to.equal('some-app-data');
                done();
            });
        });
    });

    it('generates a header then successfully parse it then validate payload', (done) => {

        const req = {
            method: 'GET',
            url: '/resource/4?filter=a',
            host: 'example.com',
            port: 8080
        };

        credentialsFunc('123456', (err, credentials1) => {

            req.authorization = Browser.client.header('http://example.com:8080/resource/4?filter=a', req.method, { credentials: credentials1, payload: 'hola!', ext: 'some-app-data' }).field;
            Hawk.server.authenticate(req, credentialsFunc, {}, (err, credentials2, artifacts) => {

                expect(err).to.not.exist();
                expect(credentials2.user).to.equal('steve');
                expect(artifacts.ext).to.equal('some-app-data');
                expect(Hawk.server.authenticatePayload('hola!', credentials2, artifacts)).to.be.true();
                expect(Hawk.server.authenticatePayload('hello!', credentials2, artifacts)).to.be.false();
                done();
            });
        });
    });

    it('generates a header then successfully parse it (app)', (done) => {

        const req = {
            method: 'GET',
            url: '/resource/4?filter=a',
            host: 'example.com',
            port: 8080
        };

        credentialsFunc('123456', (err, credentials1) => {

            req.authorization = Browser.client.header('http://example.com:8080/resource/4?filter=a', req.method, { credentials: credentials1, ext: 'some-app-data', app: 'asd23ased' }).field;
            Hawk.server.authenticate(req, credentialsFunc, {}, (err, credentials2, artifacts) => {

                expect(err).to.not.exist();
                expect(credentials2.user).to.equal('steve');
                expect(artifacts.ext).to.equal('some-app-data');
                expect(artifacts.app).to.equal('asd23ased');
                done();
            });
        });
    });

    it('generates a header then successfully parse it (app, dlg)', (done) => {

        const req = {
            method: 'GET',
            url: '/resource/4?filter=a',
            host: 'example.com',
            port: 8080
        };

        credentialsFunc('123456', (err, credentials1) => {

            req.authorization = Browser.client.header('http://example.com:8080/resource/4?filter=a', req.method, { credentials: credentials1, ext: 'some-app-data', app: 'asd23ased', dlg: '23434szr3q4d' }).field;
            Hawk.server.authenticate(req, credentialsFunc, {}, (err, credentials2, artifacts) => {

                expect(err).to.not.exist();
                expect(credentials2.user).to.equal('steve');
                expect(artifacts.ext).to.equal('some-app-data');
                expect(artifacts.app).to.equal('asd23ased');
                expect(artifacts.dlg).to.equal('23434szr3q4d');
                done();
            });
        });
    });

    it('generates a header then fail authentication due to bad hash', (done) => {

        const req = {
            method: 'GET',
            url: '/resource/4?filter=a',
            host: 'example.com',
            port: 8080
        };

        credentialsFunc('123456', (err, credentials1) => {

            req.authorization = Browser.client.header('http://example.com:8080/resource/4?filter=a', req.method, { credentials: credentials1, payload: 'hola!', ext: 'some-app-data' }).field;
            Hawk.server.authenticate(req, credentialsFunc, { payload: 'byebye!' }, (err, credentials2, artifacts) => {

                expect(err).to.exist();
                expect(err.output.payload.message).to.equal('Bad payload hash');
                done();
            });
        });
    });

    it('generates a header for one resource then fail to authenticate another', (done) => {

        const req = {
            method: 'GET',
            url: '/resource/4?filter=a',
            host: 'example.com',
            port: 8080
        };

        credentialsFunc('123456', (err, credentials1) => {

            req.authorization = Browser.client.header('http://example.com:8080/resource/4?filter=a', req.method, { credentials: credentials1, ext: 'some-app-data' }).field;
            req.url = '/something/else';

            Hawk.server.authenticate(req, credentialsFunc, {}, (err, credentials2, artifacts) => {

                expect(err).to.exist();
                expect(credentials2).to.exist();
                done();
            });
        });
    });

    describe('client', () => {

        describe('header()', () => {

            it('returns a valid authorization header (sha1)', (done) => {

                const credentials = {
                    id: '123456',
                    key: '2983d45yun89q',
                    algorithm: 'sha1'
                };

                const header = Browser.client.header('http://example.net/somewhere/over/the/rainbow', 'POST', { credentials: credentials, ext: 'Bazinga!', timestamp: 1353809207, nonce: 'Ygvqdz', payload: 'something to write about' }).field;
                expect(header).to.equal('Hawk id="123456", ts="1353809207", nonce="Ygvqdz", hash="bsvY3IfUllw6V5rvk4tStEvpBhE=", ext="Bazinga!", mac="qbf1ZPG/r/e06F4ht+T77LXi5vw="');
                done();
            });

            it('returns a valid authorization header (sha256)', (done) => {

                const credentials = {
                    id: '123456',
                    key: '2983d45yun89q',
                    algorithm: 'sha256'
                };

                const header = Browser.client.header('https://example.net/somewhere/over/the/rainbow', 'POST', { credentials: credentials, ext: 'Bazinga!', timestamp: 1353809207, nonce: 'Ygvqdz', payload: 'something to write about', contentType: 'text/plain' }).field;
                expect(header).to.equal('Hawk id="123456", ts="1353809207", nonce="Ygvqdz", hash="2QfCt3GuY9HQnHWyWD3wX68ZOKbynqlfYmuO2ZBRqtY=", ext="Bazinga!", mac="q1CwFoSHzPZSkbIvl0oYlD+91rBUEvFk763nMjMndj8="');
                done();
            });

            it('returns a valid authorization header (empty payload)', (done) => {

                const credentials = {
                    id: '123456',
                    key: '2983d45yun89q',
                    algorithm: 'sha1'
                };

                const header = Browser.client.header('http://example.net/somewhere/over/the/rainbow', 'POST', { credentials: credentials, ext: 'Bazinga!', timestamp: 1353809207, nonce: 'Ygvqdz', payload: '' }).field;
                expect(header).to.equal('Hawk id=\"123456\", ts=\"1353809207\", nonce=\"Ygvqdz\", hash=\"404ghL7K+hfyhByKKejFBRGgTjU=\", ext=\"Bazinga!\", mac=\"Bh1sj1DOfFRWOdi3ww52nLCJdBE=\"');
                done();
            });

            it('returns a valid authorization header (no ext)', (done) => {

                const credentials = {
                    id: '123456',
                    key: '2983d45yun89q',
                    algorithm: 'sha256'
                };

                const header = Browser.client.header('https://example.net/somewhere/over/the/rainbow', 'POST', { credentials: credentials, timestamp: 1353809207, nonce: 'Ygvqdz', payload: 'something to write about', contentType: 'text/plain' }).field;
                expect(header).to.equal('Hawk id="123456", ts="1353809207", nonce="Ygvqdz", hash="2QfCt3GuY9HQnHWyWD3wX68ZOKbynqlfYmuO2ZBRqtY=", mac="HTgtd0jPI6E4izx8e4OHdO36q00xFCU0FolNq3RiCYs="');
                done();
            });

            it('returns a valid authorization header (null ext)', (done) => {

                const credentials = {
                    id: '123456',
                    key: '2983d45yun89q',
                    algorithm: 'sha256'
                };

                const header = Browser.client.header('https://example.net/somewhere/over/the/rainbow', 'POST', { credentials: credentials, timestamp: 1353809207, nonce: 'Ygvqdz', payload: 'something to write about', contentType: 'text/plain', ext: null }).field;
                expect(header).to.equal('Hawk id="123456", ts="1353809207", nonce="Ygvqdz", hash="2QfCt3GuY9HQnHWyWD3wX68ZOKbynqlfYmuO2ZBRqtY=", mac="HTgtd0jPI6E4izx8e4OHdO36q00xFCU0FolNq3RiCYs="');
                done();
            });

            it('returns a valid authorization header (uri object)', (done) => {

                const credentials = {
                    id: '123456',
                    key: '2983d45yun89q',
                    algorithm: 'sha256'
                };

                const uri = Browser.utils.parseUri('https://example.net/somewhere/over/the/rainbow');
                const header = Browser.client.header(uri, 'POST', { credentials: credentials, timestamp: 1353809207, nonce: 'Ygvqdz', payload: 'something to write about', contentType: 'text/plain' }).field;
                expect(header).to.equal('Hawk id="123456", ts="1353809207", nonce="Ygvqdz", hash="2QfCt3GuY9HQnHWyWD3wX68ZOKbynqlfYmuO2ZBRqtY=", mac="HTgtd0jPI6E4izx8e4OHdO36q00xFCU0FolNq3RiCYs="');
                done();
            });

            it('errors on missing options', (done) => {

                const header = Browser.client.header('https://example.net/somewhere/over/the/rainbow', 'POST');
                expect(header.field).to.equal('');
                expect(header.err).to.equal('Invalid argument type');
                done();
            });

            it('errors on empty uri', (done) => {

                const credentials = {
                    id: '123456',
                    key: '2983d45yun89q',
                    algorithm: 'sha256'
                };

                const header = Browser.client.header('', 'POST', { credentials: credentials, timestamp: 1353809207, nonce: 'Ygvqdz', payload: 'something to write about', contentType: 'text/plain' });
                expect(header.field).to.equal('');
                expect(header.err).to.equal('Invalid argument type');
                done();
            });

            it('errors on invalid uri', (done) => {

                const credentials = {
                    id: '123456',
                    key: '2983d45yun89q',
                    algorithm: 'sha256'
                };

                const header = Browser.client.header(4, 'POST', { credentials: credentials, timestamp: 1353809207, nonce: 'Ygvqdz', payload: 'something to write about', contentType: 'text/plain' });
                expect(header.field).to.equal('');
                expect(header.err).to.equal('Invalid argument type');
                done();
            });

            it('errors on missing method', (done) => {

                const credentials = {
                    id: '123456',
                    key: '2983d45yun89q',
                    algorithm: 'sha256'
                };

                const header = Browser.client.header('https://example.net/somewhere/over/the/rainbow', '', { credentials: credentials, timestamp: 1353809207, nonce: 'Ygvqdz', payload: 'something to write about', contentType: 'text/plain' });
                expect(header.field).to.equal('');
                expect(header.err).to.equal('Invalid argument type');
                done();
            });

            it('errors on invalid method', (done) => {

                const credentials = {
                    id: '123456',
                    key: '2983d45yun89q',
                    algorithm: 'sha256'
                };

                const header = Browser.client.header('https://example.net/somewhere/over/the/rainbow', 5, { credentials: credentials, timestamp: 1353809207, nonce: 'Ygvqdz', payload: 'something to write about', contentType: 'text/plain' });
                expect(header.field).to.equal('');
                expect(header.err).to.equal('Invalid argument type');
                done();
            });

            it('errors on missing credentials', (done) => {

                const header = Browser.client.header('https://example.net/somewhere/over/the/rainbow', 'POST', { ext: 'Bazinga!', timestamp: 1353809207 });
                expect(header.field).to.equal('');
                expect(header.err).to.equal('Invalid credentials object');
                done();
            });

            it('errors on invalid credentials (id)', (done) => {

                const credentials = {
                    key: '2983d45yun89q',
                    algorithm: 'sha256'
                };

                const header = Browser.client.header('https://example.net/somewhere/over/the/rainbow', 'POST', { credentials: credentials, ext: 'Bazinga!', timestamp: 1353809207 });
                expect(header.field).to.equal('');
                expect(header.err).to.equal('Invalid credentials object');
                done();
            });

            it('errors on invalid credentials (key)', (done) => {

                const credentials = {
                    id: '123456',
                    algorithm: 'sha256'
                };

                const header = Browser.client.header('https://example.net/somewhere/over/the/rainbow', 'POST', { credentials: credentials, ext: 'Bazinga!', timestamp: 1353809207 });
                expect(header.field).to.equal('');
                expect(header.err).to.equal('Invalid credentials object');
                done();
            });

            it('errors on invalid algorithm', (done) => {

                const credentials = {
                    id: '123456',
                    key: '2983d45yun89q',
                    algorithm: 'hmac-sha-0'
                };

                const header = Browser.client.header('https://example.net/somewhere/over/the/rainbow', 'POST', { credentials: credentials, payload: 'something, anything!', ext: 'Bazinga!', timestamp: 1353809207 });
                expect(header.field).to.equal('');
                expect(header.err).to.equal('Unknown algorithm');
                done();
            });

            it('uses a pre-calculated payload hash', (done) => {

                const credentials = {
                    id: '123456',
                    key: '2983d45yun89q',
                    algorithm: 'sha256'
                };

                const options = { credentials: credentials, ext: 'Bazinga!', timestamp: 1353809207, nonce: 'Ygvqdz', payload: 'something to write about', contentType: 'text/plain' };
                options.hash = Browser.crypto.calculatePayloadHash(options.payload, credentials.algorithm, options.contentType);
                const header = Browser.client.header('https://example.net/somewhere/over/the/rainbow', 'POST', options).field;
                expect(header).to.equal('Hawk id="123456", ts="1353809207", nonce="Ygvqdz", hash="2QfCt3GuY9HQnHWyWD3wX68ZOKbynqlfYmuO2ZBRqtY=", ext="Bazinga!", mac="q1CwFoSHzPZSkbIvl0oYlD+91rBUEvFk763nMjMndj8="');
                done();
            });
        });

        describe('authenticate()', () => {

            it('skips tsm validation when missing ts', (done) => {

                const res = {
                    headers: {
                        'www-authenticate': 'Hawk error="Stale timestamp"'
                    },
                    getResponseHeader: function (header) {

                        return res.headers[header.toLowerCase()];
                    }
                };

                const credentials = {
                    id: '123456',
                    key: 'werxhqb98rpaxn39848xrunpaw3489ruxnpa98w4rxn',
                    algorithm: 'sha256',
                    user: 'steve'
                };

                const artifacts = {
                    ts: 1402135580,
                    nonce: 'iBRB6t',
                    method: 'GET',
                    resource: '/resource/4?filter=a',
                    host: 'example.com',
                    port: '8080',
                    ext: 'some-app-data'
                };

                expect(Browser.client.authenticate(res, credentials, artifacts)).to.equal(true);
                done();
            });

            it('returns false on invalid header', (done) => {

                const res = {
                    headers: {
                        'server-authorization': 'Hawk mac="abc", bad="xyz"'
                    },
                    getResponseHeader: function (header) {

                        return res.headers[header.toLowerCase()];
                    }
                };

                expect(Browser.client.authenticate(res, {})).to.equal(false);
                done();
            });

            it('returns false on invalid mac', (done) => {

                const res = {
                    headers: {
                        'content-type': 'text/plain',
                        'server-authorization': 'Hawk mac="_IJRsMl/4oL+nn+vKoeVZPdCHXB4yJkNnBbTbHFZUYE=", hash="f9cDF/TDm7TkYRLnGwRMfeDzT6LixQVLvrIKhh0vgmM=", ext="response-specific"'
                    },
                    getResponseHeader: function (header) {

                        return res.headers[header.toLowerCase()];
                    }
                };

                const artifacts = {
                    method: 'POST',
                    host: 'example.com',
                    port: '8080',
                    resource: '/resource/4?filter=a',
                    ts: '1362336900',
                    nonce: 'eb5S_L',
                    hash: 'nJjkVtBE5Y/Bk38Aiokwn0jiJxt/0S2WRSUwWLCf5xk=',
                    ext: 'some-app-data',
                    app: undefined,
                    dlg: undefined,
                    mac: 'BlmSe8K+pbKIb6YsZCnt4E1GrYvY1AaYayNR82dGpIk=',
                    id: '123456'
                };

                const credentials = {
                    id: '123456',
                    key: 'werxhqb98rpaxn39848xrunpaw3489ruxnpa98w4rxn',
                    algorithm: 'sha256',
                    user: 'steve'
                };

                expect(Browser.client.authenticate(res, credentials, artifacts)).to.equal(false);
                done();
            });

            it('returns true on ignoring hash', (done) => {

                const res = {
                    headers: {
                        'content-type': 'text/plain',
                        'server-authorization': 'Hawk mac="XIJRsMl/4oL+nn+vKoeVZPdCHXB4yJkNnBbTbHFZUYE=", hash="f9cDF/TDm7TkYRLnGwRMfeDzT6LixQVLvrIKhh0vgmM=", ext="response-specific"'
                    },
                    getResponseHeader: function (header) {

                        return res.headers[header.toLowerCase()];
                    }
                };

                const artifacts = {
                    method: 'POST',
                    host: 'example.com',
                    port: '8080',
                    resource: '/resource/4?filter=a',
                    ts: '1362336900',
                    nonce: 'eb5S_L',
                    hash: 'nJjkVtBE5Y/Bk38Aiokwn0jiJxt/0S2WRSUwWLCf5xk=',
                    ext: 'some-app-data',
                    app: undefined,
                    dlg: undefined,
                    mac: 'BlmSe8K+pbKIb6YsZCnt4E1GrYvY1AaYayNR82dGpIk=',
                    id: '123456'
                };

                const credentials = {
                    id: '123456',
                    key: 'werxhqb98rpaxn39848xrunpaw3489ruxnpa98w4rxn',
                    algorithm: 'sha256',
                    user: 'steve'
                };

                expect(Browser.client.authenticate(res, credentials, artifacts)).to.equal(true);
                done();
            });

            it('errors on invalid WWW-Authenticate header format', (done) => {

                const res = {
                    headers: {
                        'www-authenticate': 'Hawk ts="1362346425875", tsm="PhwayS28vtnn3qbv0mqRBYSXebN/zggEtucfeZ620Zo=", x="Stale timestamp"'
                    },
                    getResponseHeader: function (header) {

                        return res.headers[header.toLowerCase()];
                    }
                };

                expect(Browser.client.authenticate(res, {})).to.equal(false);
                done();
            });

            it('errors on invalid WWW-Authenticate header format', (done) => {

                const credentials = {
                    id: '123456',
                    key: 'werxhqb98rpaxn39848xrunpaw3489ruxnpa98w4rxn',
                    algorithm: 'sha256',
                    user: 'steve'
                };

                const res = {
                    headers: {
                        'www-authenticate': 'Hawk ts="1362346425875", tsm="hwayS28vtnn3qbv0mqRBYSXebN/zggEtucfeZ620Zo=", error="Stale timestamp"'
                    },
                    getResponseHeader: function (header) {

                        return res.headers[header.toLowerCase()];
                    }
                };

                expect(Browser.client.authenticate(res, credentials)).to.equal(false);
                done();
            });
        });

        describe('message()', () => {

            it('generates an authorization then successfully parse it', (done) => {

                credentialsFunc('123456', (err, credentials1) => {

                    const auth = Browser.client.message('example.com', 8080, 'some message', { credentials: credentials1 });
                    expect(auth).to.exist();

                    Hawk.server.authenticateMessage('example.com', 8080, 'some message', auth, credentialsFunc, {}, (err, credentials2) => {

                        expect(err).to.not.exist();
                        expect(credentials2.user).to.equal('steve');
                        done();
                    });
                });
            });

            it('generates an authorization using custom nonce/timestamp', (done) => {

                credentialsFunc('123456', (err, credentials) => {

                    const auth = Browser.client.message('example.com', 8080, 'some message', { credentials: credentials, nonce: 'abc123', timestamp: 1398536270957 });
                    expect(auth).to.exist();
                    expect(auth.nonce).to.equal('abc123');
                    expect(auth.ts).to.equal(1398536270957);
                    done();
                });
            });

            it('errors on missing host', (done) => {

                credentialsFunc('123456', (err, credentials) => {

                    const auth = Browser.client.message(null, 8080, 'some message', { credentials: credentials });
                    expect(auth).to.not.exist();
                    done();
                });
            });

            it('errors on invalid host', (done) => {

                credentialsFunc('123456', (err, credentials) => {

                    const auth = Browser.client.message(5, 8080, 'some message', { credentials: credentials });
                    expect(auth).to.not.exist();
                    done();
                });
            });

            it('errors on missing port', (done) => {

                credentialsFunc('123456', (err, credentials) => {

                    const auth = Browser.client.message('example.com', 0, 'some message', { credentials: credentials });
                    expect(auth).to.not.exist();
                    done();
                });
            });

            it('errors on invalid port', (done) => {

                credentialsFunc('123456', (err, credentials) => {

                    const auth = Browser.client.message('example.com', 'a', 'some message', { credentials: credentials });
                    expect(auth).to.not.exist();
                    done();
                });
            });

            it('errors on missing message', (done) => {

                credentialsFunc('123456', (err, credentials) => {

                    const auth = Browser.client.message('example.com', 8080, undefined, { credentials: credentials });
                    expect(auth).to.not.exist();
                    done();
                });
            });

            it('errors on null message', (done) => {

                credentialsFunc('123456', (err, credentials) => {

                    const auth = Browser.client.message('example.com', 8080, null, { credentials: credentials });
                    expect(auth).to.not.exist();
                    done();
                });
            });

            it('errors on invalid message', (done) => {

                credentialsFunc('123456', (err, credentials) => {

                    const auth = Browser.client.message('example.com', 8080, 5, { credentials: credentials });
                    expect(auth).to.not.exist();
                    done();
                });
            });

            it('errors on missing credentials', (done) => {

                const auth = Browser.client.message('example.com', 8080, 'some message', {});
                expect(auth).to.not.exist();
                done();
            });

            it('errors on missing options', (done) => {

                const auth = Browser.client.message('example.com', 8080, 'some message');
                expect(auth).to.not.exist();
                done();
            });

            it('errors on invalid credentials (id)', (done) => {

                credentialsFunc('123456', (err, credentials) => {

                    const creds = Hoek.clone(credentials);
                    delete creds.id;
                    const auth = Browser.client.message('example.com', 8080, 'some message', { credentials: creds });
                    expect(auth).to.not.exist();
                    done();
                });
            });

            it('errors on invalid credentials (key)', (done) => {

                credentialsFunc('123456', (err, credentials) => {

                    const creds = Hoek.clone(credentials);
                    delete creds.key;
                    const auth = Browser.client.message('example.com', 8080, 'some message', { credentials: creds });
                    expect(auth).to.not.exist();
                    done();
                });
            });

            it('errors on invalid algorithm', (done) => {

                credentialsFunc('123456', (err, credentials) => {

                    const creds = Hoek.clone(credentials);
                    creds.algorithm = 'blah';
                    const auth = Browser.client.message('example.com', 8080, 'some message', { credentials: creds });
                    expect(auth).to.not.exist();
                    done();
                });
            });
        });

        describe('authenticateTimestamp()', (done) => {

            it('validates a timestamp', (done) => {

                credentialsFunc('123456', (err, credentials) => {

                    const tsm = Hawk.crypto.timestampMessage(credentials);
                    expect(Browser.client.authenticateTimestamp(tsm, credentials)).to.equal(true);
                    done();
                });
            });

            it('validates a timestamp without updating local time', (done) => {

                credentialsFunc('123456', (err, credentials) => {

                    const offset = Browser.utils.getNtpOffset();
                    const tsm = Hawk.crypto.timestampMessage(credentials, 10000);
                    expect(Browser.client.authenticateTimestamp(tsm, credentials, false)).to.equal(true);
                    expect(offset).to.equal(Browser.utils.getNtpOffset());
                    done();
                });
            });

            it('detects a bad timestamp', (done) => {

                credentialsFunc('123456', (err, credentials) => {

                    const tsm = Hawk.crypto.timestampMessage(credentials);
                    tsm.ts = 4;
                    expect(Browser.client.authenticateTimestamp(tsm, credentials)).to.equal(false);
                    done();
                });
            });
        });
    });

    describe('internals', () => {

        describe('LocalStorage', () => {

            it('goes through the full lifecycle', (done) => {

                const storage = new Browser.internals.LocalStorage();
                expect(storage.length).to.equal(0);
                expect(storage.getItem('a')).to.equal(null);
                storage.setItem('a', 5);
                expect(storage.length).to.equal(1);
                expect(storage.key()).to.equal('a');
                expect(storage.key(0)).to.equal('a');
                expect(storage.getItem('a')).to.equal('5');
                storage.setItem('b', 'test');
                expect(storage.key()).to.equal('a');
                expect(storage.key(0)).to.equal('a');
                expect(storage.key(1)).to.equal('b');
                expect(storage.length).to.equal(2);
                expect(storage.getItem('b')).to.equal('test');
                storage.removeItem('a');
                expect(storage.length).to.equal(1);
                expect(storage.getItem('a')).to.equal(null);
                expect(storage.getItem('b')).to.equal('test');
                storage.clear();
                expect(storage.length).to.equal(0);
                expect(storage.getItem('a')).to.equal(null);
                expect(storage.getItem('b')).to.equal(null);
                done();
            });
        });
    });

    describe('utils', () => {

        describe('setStorage()', () => {

            it('sets storage for the first time', (done) => {

                Browser.utils.storage = new Browser.internals.LocalStorage();        // Reset state

                expect(Browser.utils.storage.getItem('hawk_ntp_offset')).to.not.exist();
                Browser.utils.storage.setItem('test', '1');
                Browser.utils.setStorage(new Browser.internals.LocalStorage());
                expect(Browser.utils.storage.getItem('test')).to.not.exist();
                Browser.utils.storage.setItem('test', '2');
                expect(Browser.utils.storage.getItem('test')).to.equal('2');
                done();
            });
        });

        describe('setNtpOffset()', (done) => {

            it('catches localStorage errors', { parallel: false }, (done) => {

                const orig = Browser.utils.storage.setItem;
                const consoleOrig = console.error;
                let count = 0;
                console.error = function () {

                    if (count++ === 2) {

                        console.error = consoleOrig;
                    }
                };

                Browser.utils.storage.setItem = function () {

                    Browser.utils.storage.setItem = orig;
                    throw new Error();
                };

                expect(() => {

                    Browser.utils.setNtpOffset(100);
                }).not.to.throw();

                done();
            });
        });

        describe('parseAuthorizationHeader()', (done) => {

            it('returns null on missing header', (done) => {

                expect(Browser.utils.parseAuthorizationHeader()).to.equal(null);
                done();
            });

            it('returns null on bad header syntax (structure)', (done) => {

                expect(Browser.utils.parseAuthorizationHeader('Hawk')).to.equal(null);
                done();
            });

            it('returns null on bad header syntax (parts)', (done) => {

                expect(Browser.utils.parseAuthorizationHeader(' ')).to.equal(null);
                done();
            });

            it('returns null on bad scheme name', (done) => {

                expect(Browser.utils.parseAuthorizationHeader('Basic asdasd')).to.equal(null);
                done();
            });

            it('returns null on bad attribute value', (done) => {

                expect(Browser.utils.parseAuthorizationHeader('Hawk test="\t"', ['test'])).to.equal(null);
                done();
            });

            it('returns null on duplicated attribute', (done) => {

                expect(Browser.utils.parseAuthorizationHeader('Hawk test="a", test="b"', ['test'])).to.equal(null);
                done();
            });
        });

        describe('parseUri()', () => {

            it('returns empty object on invalid', (done) => {

                const uri = Browser.utils.parseUri('ftp');
                expect(uri).to.deep.equal({ host: '', port: '', resource: '' });
                done();
            });

            it('returns empty port when unknown scheme', (done) => {

                const uri = Browser.utils.parseUri('ftp://example.com');
                expect(uri.port).to.equal('');
                done();
            });

            it('returns default port when missing', (done) => {

                const uri = Browser.utils.parseUri('http://example.com');
                expect(uri.port).to.equal('80');
                done();
            });

            it('handles unusual characters correctly', (done) => {

                const parts = {
                    protocol: 'http+vnd.my-extension',
                    user: 'user!$&\'()*+,;=%40my-domain.com',
                    password: 'pass!$&\'()*+,;=%40:word',
                    hostname: 'foo-bar.com',
                    port: '99',
                    pathname: '/path/%40/!$&\'()*+,;=:@/',
                    query: 'query%40/!$&\'()*+,;=:@/?',
                    fragment: 'fragm%40/!$&\'()*+,;=:@/?'
                };

                parts.userInfo = parts.user + ':' + parts.password;
                parts.authority = parts.userInfo + '@' + parts.hostname + ':' + parts.port;
                parts.relative = parts.pathname + '?' + parts.query;
                parts.resource = parts.relative + '#' + parts.fragment;
                parts.source = parts.protocol + '://' + parts.authority + parts.resource;

                const uri = Browser.utils.parseUri(parts.source);
                expect(uri.host).to.equal('foo-bar.com');
                expect(uri.port).to.equal('99');
                expect(uri.resource).to.equal(parts.pathname + '?' + parts.query);
                done();
            });
        });

        const str = 'https://www.google.ca/webhp?sourceid=chrome-instant&ion=1&espv=2&ie=UTF-8#q=url';
        const base64str = 'aHR0cHM6Ly93d3cuZ29vZ2xlLmNhL3dlYmhwP3NvdXJjZWlkPWNocm9tZS1pbnN0YW50Jmlvbj0xJmVzcHY9MiZpZT1VVEYtOCNxPXVybA';

        describe('base64urlEncode()', () => {

            it('should base64 URL-safe decode a string', (done) => {

                expect(Browser.utils.base64urlEncode(str)).to.equal(base64str);
                done();
            });
        });
    });
});
