'use strict';

// Load modules

const Dgram = require('dgram');
const Code = require('code');
const Lab = require('lab');
const Sntp = require('../lib');


// Declare internals

const internals = {};


// Test shortcuts

const lab = exports.lab = Lab.script();
const describe = lab.describe;
const it = lab.it;
const expect = Code.expect;


describe('SNTP', () => {

    describe('time()', () => {

        it('returns consistent result over multiple tries', (done) => {

            Sntp.time((err, time1) => {

                expect(err).to.not.exist();
                expect(time1).to.exist();
                const t1 = time1.t;

                Sntp.time((err, time2) => {

                    expect(err).to.not.exist();
                    expect(time2).to.exist();
                    const t2 = time2.t;
                    expect(Math.abs(t1 - t2)).to.be.below(200);
                    done();
                });
            });
        });

        it('resolves reference IP', (done) => {

            Sntp.time({ host: 'ntp.exnet.com', resolveReference: true }, (err, time) => {

                expect(err).to.not.exist();
                expect(time).to.exist();
                expect(time.referenceHost).to.exist();
                done();
            });
        });

        it('times out on no response', (done) => {

            Sntp.time({ port: 124, timeout: 100 }, (err, time) => {

                expect(err).to.exist();
                expect(time).to.not.exist();
                expect(err.message).to.equal('Timeout');
                done();
            });
        });

        it('errors on error event', { parallel: false }, (done) => {

            const orig = Dgram.createSocket;
            Dgram.createSocket = function (type) {

                Dgram.createSocket = orig;
                const socket = Dgram.createSocket(type);
                setImmediate(() => {

                    socket.emit('error', new Error('Fake'));
                });
                return socket;
            };

            Sntp.time((err, time) => {

                expect(err).to.exist();
                expect(time).to.not.exist();
                expect(err.message).to.equal('Fake');
                done();
            });
        });

        it('errors on incorrect sent size', { parallel: false }, (done) => {

            const orig = Dgram.Socket.prototype.send;
            Dgram.Socket.prototype.send = function (buf, offset, length, port, address, callback) {

                Dgram.Socket.prototype.send = orig;
                return callback(null, 40);
            };

            Sntp.time((err, time) => {

                expect(err).to.exist();
                expect(time).to.not.exist();
                expect(err.message).to.equal('Could not send entire message');
                done();
            });
        });

        it('times out on invalid host', (done) => {

            Sntp.time({ host: 'error', timeout: 10000 }, (err, time) => {

                expect(err).to.exist();
                expect(time).to.not.exist();
                expect(err.message).to.contain('getaddrinfo');
                done();
            });
        });

        it('fails on bad response buffer size', (done) => {

            const server = Dgram.createSocket('udp4');
            server.on('message', (message, remote) => {

                const msg = new Buffer(10);
                server.send(msg, 0, msg.length, remote.port, remote.address, (err, bytes) => {

                    server.close();
                });
            });

            server.bind(49123);

            Sntp.time({ host: 'localhost', port: 49123 }, (err, time) => {

                expect(err).to.exist();
                expect(err.message).to.equal('Invalid server response');
                done();
            });
        });

        const messup = function (bytes) {

            const server = Dgram.createSocket('udp4');
            server.on('message', (message, remote) => {

                const msg = new Buffer([
                    0x24, 0x01, 0x00, 0xe3,
                    0x00, 0x00, 0x00, 0x00,
                    0x00, 0x00, 0x00, 0x00,
                    0x41, 0x43, 0x54, 0x53,
                    0xd4, 0xa8, 0x2d, 0xc7,
                    0x1c, 0x5d, 0x49, 0x1b,
                    0xd4, 0xa8, 0x2d, 0xe6,
                    0x67, 0xef, 0x9d, 0xb2,
                    0xd4, 0xa8, 0x2d, 0xe6,
                    0x71, 0xed, 0xb5, 0xfb,
                    0xd4, 0xa8, 0x2d, 0xe6,
                    0x71, 0xee, 0x6c, 0xc5
                ]);

                for (let i = 0; i < bytes.length; ++i) {
                    msg[bytes[i][0]] = bytes[i][1];
                }

                server.send(msg, 0, msg.length, remote.port, remote.address, (err, bytes2) => {

                    server.close();
                });
            });

            server.bind(49123);
        };

        it('fails on bad version', (done) => {

            messup([[0, (0 << 6) + (3 << 3) + (4 << 0)]]);

            Sntp.time({ host: 'localhost', port: 49123 }, (err, time) => {

                expect(err).to.exist();
                expect(time.version).to.equal(3);
                expect(err.message).to.equal('Invalid server response');
                done();
            });
        });

        it('fails on bad originateTimestamp', (done) => {

            messup([[24, 0x83], [25, 0xaa], [26, 0x7e], [27, 0x80], [28, 0], [29, 0], [30, 0], [31, 0]]);

            Sntp.time({ host: 'localhost', port: 49123 }, (err, time) => {

                expect(err).to.exist();
                expect(err.message).to.equal('Invalid server response');
                done();
            });
        });

        it('fails on bad receiveTimestamp', (done) => {

            messup([[32, 0x83], [33, 0xaa], [34, 0x7e], [35, 0x80], [36, 0], [37, 0], [38, 0], [39, 0]]);

            Sntp.time({ host: 'localhost', port: 49123 }, (err, time) => {

                expect(err).to.exist();
                expect(err.message).to.equal('Invalid server response');
                done();
            });
        });

        it('fails on bad originate timestamp and alarm li', (done) => {

            messup([[0, (3 << 6) + (4 << 3) + (4 << 0)]]);

            Sntp.time({ host: 'localhost', port: 49123 }, (err, time) => {

                expect(err).to.exist();
                expect(err.message).to.equal('Wrong originate timestamp');
                expect(time.leapIndicator).to.equal('alarm');
                done();
            });
        });

        it('returns time with death stratum and last61 li', (done) => {

            messup([[0, (1 << 6) + (4 << 3) + (4 << 0)], [1, 0]]);

            Sntp.time({ host: 'localhost', port: 49123 }, (err, time) => {

                expect(time.stratum).to.equal('death');
                expect(time.leapIndicator).to.equal('last-minute-61');
                done();
            });
        });

        it('returns time with reserved stratum and last59 li', (done) => {

            messup([[0, (2 << 6) + (4 << 3) + (4 << 0)], [1, 0x1f]]);

            Sntp.time({ host: 'localhost', port: 49123 }, (err, time) => {

                expect(time.stratum).to.equal('reserved');
                expect(time.leapIndicator).to.equal('last-minute-59');
                done();
            });
        });

        it('fails on bad mode (symmetric-active)', (done) => {

            messup([[0, (0 << 6) + (4 << 3) + (1 << 0)]]);

            Sntp.time({ host: 'localhost', port: 49123 }, (err, time) => {

                expect(err).to.exist();
                expect(time.mode).to.equal('symmetric-active');
                done();
            });
        });

        it('fails on bad mode (symmetric-passive)', (done) => {

            messup([[0, (0 << 6) + (4 << 3) + (2 << 0)]]);

            Sntp.time({ host: 'localhost', port: 49123 }, (err, time) => {

                expect(err).to.exist();
                expect(time.mode).to.equal('symmetric-passive');
                done();
            });
        });

        it('fails on bad mode (client)', (done) => {

            messup([[0, (0 << 6) + (4 << 3) + (3 << 0)]]);

            Sntp.time({ host: 'localhost', port: 49123 }, (err, time) => {

                expect(err).to.exist();
                expect(time.mode).to.equal('client');
                done();
            });
        });

        it('fails on bad mode (broadcast)', (done) => {

            messup([[0, (0 << 6) + (4 << 3) + (5 << 0)]]);

            Sntp.time({ host: 'localhost', port: 49123 }, (err, time) => {

                expect(err).to.exist();
                expect(time.mode).to.equal('broadcast');
                done();
            });
        });

        it('fails on bad mode (reserved)', (done) => {

            messup([[0, (0 << 6) + (4 << 3) + (6 << 0)]]);

            Sntp.time({ host: 'localhost', port: 49123 }, (err, time) => {

                expect(err).to.exist();
                expect(time.mode).to.equal('reserved');
                done();
            });
        });
    });

    describe('offset()', () => {

        it('gets the current offset', (done) => {

            Sntp.offset((err, offset) => {

                expect(err).to.not.exist();
                expect(offset).to.not.equal(0);
                done();
            });
        });

        it('gets the current offset from cache', (done) => {

            Sntp.offset((err, offset1) => {

                expect(err).to.not.exist();
                expect(offset1).to.not.equal(0);

                Sntp.offset({}, (err, offset2) => {

                    expect(err).to.not.exist();
                    expect(offset2).to.equal(offset1);
                    done();
                });
            });
        });

        it('gets the new offset on different server', (done) => {

            Sntp.offset((err, offset1) => {

                expect(err).to.not.exist();
                expect(offset1).to.not.equal(0);

                Sntp.offset({ host: 'us.pool.ntp.org' }, (err, offset2) => {

                    expect(err).to.not.exist();
                    expect(offset2).to.not.equal(offset1);
                    done();
                });
            });
        });

        it('gets the new offset on different server', (done) => {

            Sntp.offset((err, offset1) => {

                expect(err).to.not.exist();
                expect(offset1).to.not.equal(0);

                Sntp.offset({ port: 123 }, (err, offset2) => {

                    expect(err).to.not.exist();
                    expect(offset2).to.not.equal(offset1);
                    done();
                });
            });
        });

        it('fails getting the current offset on invalid server', (done) => {

            Sntp.offset({ host: 'error' }, (err, offset) => {

                expect(err).to.exist();
                expect(offset).to.equal(0);
                done();
            });
        });
    });

    describe('now()', () => {

        it('starts auto-sync, gets now, then stops', (done) => {

            Sntp.stop();

            const before = Sntp.now();
            expect(before).to.be.about(Date.now(), 5);

            Sntp.start(() => {

                const now = Sntp.now();
                expect(now).to.not.equal(Date.now());
                Sntp.stop();

                done();
            });
        });

        it('starts twice', (done) => {

            Sntp.start(() => {

                Sntp.start(() => {

                    const now = Sntp.now();
                    expect(now).to.not.equal(Date.now());
                    Sntp.stop();

                    done();
                });
            });
        });

        it('starts auto-sync, gets now, waits, gets again after timeout', (done) => {

            Sntp.stop();

            const before = Sntp.now();
            expect(before).to.be.about(Date.now(), 5);

            Sntp.start({ clockSyncRefresh: 100 }, () => {

                const now = Sntp.now();
                expect(now).to.not.equal(Date.now());
                expect(now).to.be.about(Sntp.now(), 5);

                setTimeout(() => {

                    expect(Sntp.now()).to.not.equal(now);
                    Sntp.stop();
                    done();
                }, 110);
            });
        });
    });
});

