/*
 *
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 *
*/

describe('urlutil', function () {
    var urlutil = require('cordova/urlutil');
    if (typeof process != 'undefined') {
        // Tests don't work under jsdom.
        return;
    }

    it('can handle absolute URLs', function () {
        expect(urlutil.makeAbsolute('http://www.foo.com')).toBe('http://www.foo.com/');
        expect(urlutil.makeAbsolute('http://www.foo.com?foo#bar')).toBe('http://www.foo.com/?foo#bar');
        expect(urlutil.makeAbsolute('http://www.foo.com/%20hi')).toBe('http://www.foo.com/%20hi');
    });

    function testRootRelative(url) {
        var rootUrl = url.slice(0, 10) + url.slice(10).replace(/[?#].*/, '').replace(/\/.*/, '') + '/';
        expect(urlutil.makeAbsolute('/')).toBe(rootUrl);
        expect(urlutil.makeAbsolute('/foo?a#b')).toBe(rootUrl + 'foo?a#b');
        expect(urlutil.makeAbsolute('/foo/b%20ar')).toBe(rootUrl + 'foo/b%20ar');
    }
    it('can handle root-relative URLs', function () {
        testRootRelative(window.location.href);
    });

    it('can handle relative URLs', function () {
        var rootUrl = window.location.href.replace(/[?#].*/, '').replace(/[^\/]*$/, '');
        expect(urlutil.makeAbsolute('foo?a#b')).toBe(rootUrl + 'foo?a#b');
        expect(urlutil.makeAbsolute('foo/b%20ar')).toBe(rootUrl + 'foo/b%20ar');
    });

    it('can handle relative URLs with base tags', function () {
        var rootUrl = 'http://base.com/esab/';
        var baseTag = document.createElement('base');
        baseTag.href = rootUrl;
        document.head.appendChild(baseTag);
        this.after(function() {
            document.head.removeChild(baseTag);
        });
        expect(urlutil.makeAbsolute('foo?a#b')).toBe(rootUrl + 'foo?a#b');
        expect(urlutil.makeAbsolute('foo/b%20ar')).toBe(rootUrl + 'foo/b%20ar');
        testRootRelative(rootUrl);
    });

    it('can handle scheme-relative URLs', function () {
        var rootUrl = window.location.href.replace(/:.*/, '');
        expect(urlutil.makeAbsolute('//www.foo.com/baz%20?foo#bar')).toBe(rootUrl + '://www.foo.com/baz%20?foo#bar');
    });

});
