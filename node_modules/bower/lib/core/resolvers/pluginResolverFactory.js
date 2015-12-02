var Q = require('q');
var path = require('path');
var fs = require('../../util/fs');
var object = require('mout/object');

var semver = require('../../util/semver');
var createError = require('../../util/createError');
var readJson = require('../../util/readJson');
var removeIgnores = require('../../util/removeIgnores');

function pluginResolverFactory(resolverFactory, bower) {
    bower = bower || {};

    if (typeof resolverFactory !== 'function') {
        throw createError('Resolver has "' + typeof resolverFactory + '" type instead of "function" type.', 'ERESOLERAPI');
    }

    var resolver = resolverFactory(bower);

    function maxSatisfyingVersion(versions, target) {
        var versionsArr, index;

        versionsArr = versions.map(function (obj) { return obj.version; });

        // Find a satisfying version, enabling strict match so that pre-releases
        // have lower priority over normal ones when target is *
        index = semver.maxSatisfyingIndex(versionsArr, target, true);

        if (index !== -1) {
            return versions[index];
        }
    }

    function PluginResolver(decEndpoint) {
        this._decEndpoint = decEndpoint;
    }

    // @private
    PluginResolver.prototype.getEndpoint = function () {
        return object.merge(this._decEndpoint, {
            name: this.getName(),
            source: this.getSource(),
            target: this.getTarget()
        });
    };

    PluginResolver.prototype.getSource = function () {
        return this._decEndpoint.source;
    };

    PluginResolver.prototype.getTarget = function () {
        return this._decEndpoint.target || '*';
    };

    PluginResolver.prototype.getName = function() {
        if (!this._decEndpoint.name && typeof resolver.getName === 'function') {
            return resolver.getName.call(resolver, this.getSource());
        } else if (!this._decEndpoint.name) {
            return path.basename(this.getSource());
        } else {
            return this._decEndpoint.name;
        }
    };

    PluginResolver.prototype.getPkgMeta = function () {
        return this._pkgMeta;
    };

    // -----------------

    // Plugin Resolver is always considered potentially cacheable
    // The "resolve" method decides whether to use cached or fetch new version.
    PluginResolver.prototype.isCacheable = function() {
        return true;
    };

    // Not only it's always potentially cacheable, but also always potenially new.
    // The "resolve" handles logic of re-downloading target if needed.
    PluginResolver.prototype.hasNew = function (pkgMeta) {
        if (this.hasNewPromise) {
            return this.hasNewPromise;
        }

        this._pkgMeta = pkgMeta;

        return this.hasNewPromise = this.resolve().then(function (result) {
            return result !== undefined;
        });
    };

    PluginResolver.prototype.resolve = function () {
        if (this.resolvePromise) {
            return this.resolvePromise;
        }

        var that = this;

        return this.resolvePromise = Q.fcall(function() {
            var target = that.getTarget();

            // It means that we can accept ranges as targets
            if(that.constructor.isTargetable()) {
                that._release = target;

                if (semver.validRange(target)) {
                    return Q.fcall(resolver.releases.bind(resolver), that.getSource())
                    .then(function (result) {
                        if (!result) {
                            throw createError('Resolver did not provide releases of package.');
                        }

                        var releases = this._releases = result;

                        var versions = releases.filter(function (target) {
                            return semver.clean(target.version);
                        });

                        var maxRelease = maxSatisfyingVersion(versions, target);

                        if (maxRelease) {
                            that._version = maxRelease.version;
                            that._release = that._decEndpoint.target = maxRelease.target;
                        } else {
                            throw createError('No version found that was able to satisfy ' + target, 'ENORESTARGET', {
                                details: !versions.length ?
                                    'No versions found in ' + that.getSource() :
                                    'Available versions: ' + versions.map(function (version) { return version.version; }).join(', ')
                            });
                        }
                    });
                }
            } else {
                if (semver.validRange(target) && target !== '*') {
                    return Q.reject(createError('Resolver does not accept version ranges (' + target + ')'));
                }
            }
        })
        .then(function () {

            // We pass old _resolution (if hasNew has been called before contents).
            // So resolver can decide wheter use cached version of contents new one.
            if (typeof resolver.fetch !== 'function') {
                throw createError('Resolver does not implement the "fetch" method.');
            }

            var cached = {};

            if (that._releases) {
                cached.releases = that._releases;
            }

            if (that._pkgMeta) {
                cached.endpoint = {
                    name: that._pkgMeta.name,
                    source: that._pkgMeta._source,
                    target: that._pkgMeta._target
                };

                cached.release = that._pkgMeta._release;

                cached.version = that._pkgMeta.version;

                cached.resolution = that._pkgMeta._resolution || {};
            }

            return Q.fcall(resolver.fetch.bind(resolver), that.getEndpoint(), cached);
        })
        .then(function (result) {
            // Empty result means to re-use existing resolution
            if (!result) {
                return;
            } else {
                if (!result.tempPath) {
                    throw createError('Resolver did not provide path to extracted contents of package.');
                }

                that._tempDir = result.tempPath;

                return that._readJson(that._tempDir).then(function (meta) {
                    return that._applyPkgMeta(meta, result)
                    .then(that._savePkgMeta.bind(that, meta, result))
                    .then(function () {
                        return that._tempDir;
                    });
                });
            }
        });
    };

    PluginResolver.prototype._readJson = function (dir) {
        var that = this;

        return readJson(dir, {
            assume: { name: that.getName() }
        })
        .spread(function (json, deprecated) {
            if (deprecated) {
                bower.logger.warn('deprecated', 'Package ' + that.getName() + ' is using the deprecated ' + deprecated);
            }

            return json;
        });
    };

    PluginResolver.prototype._applyPkgMeta = function (meta, result) {
        // Check if name defined in the json is different
        // If so and if the name was "guessed", assume the json name
        if (meta.name !== this._name) {
            this._name = meta.name;
        }

        // Handle ignore property, deleting all files from the temporary directory
        // If no ignores were specified, simply resolve
        if (result.removeIgnores === false || !meta.ignore || !meta.ignore.length) {
            return Q.resolve(meta);
        }

        // Otherwise remove them from the temp dir
        return removeIgnores(this._tempDir, meta).then(function () {
            return meta;
        });
    };

    PluginResolver.prototype._savePkgMeta = function (meta, result) {
        var that = this;

        meta._source = that.getSource();
        meta._target = that.getTarget();

        if (result.resolution) {
            meta._resolution = result.resolution;
        }

        if (that._release) {
            meta._release = that._release;
        }

        if (that._version) {
            meta.version = that._version;
        } else {
            delete meta.version;
        }

        // Stringify contents
        var contents = JSON.stringify(meta, null, 2);

        return Q.nfcall(fs.writeFile, path.join(this._tempDir, '.bower.json'), contents)
        .then(function () {
            return that._pkgMeta = meta;
        });
    };

    // It is used only by "bower info". It returns all semver versions.
    PluginResolver.versions = function (source) {
        return Q.fcall(resolver.releases.bind(resolver), source).then(function (result) {
            if (!result) {
                throw createError('Resolver did not provide releases of package.');
            }

            var releases = this._releases = result;

            var versions = releases.map(function (version) {
                return semver.clean(version.version);
            });

            versions = versions.filter(function (version) {
                return version;
            });

            versions.sort(function (a, b) {
                return semver.rcompare(a, b);
            });

            return versions;
        });
    };

    PluginResolver.isTargetable = function() {
        // If resolver doesn't define versions function, it's not targetable..
        return typeof resolver.releases === 'function';
    };

    PluginResolver.clearRuntimeCache = function () {
        resolver = resolverFactory(bower);
    };

    PluginResolver.match = function (source) {
        if (typeof resolver.match !== 'function') {
            throw createError('Resolver is missing "match" method.', 'ERESOLVERAPI');
        }

        var match = resolver.match.bind(resolver);

        return Q.fcall(match, source).then(function (result) {
            if (typeof result !== 'boolean') {
                throw createError('Resolver\'s "match" method should return a boolean', 'ERESOLVERAPI');
            }

            return result;
        });
    };

    PluginResolver.locate = function (source) {
        if (typeof resolver.locate !== 'function') {
            return source;
        }

        return Q.fcall(resolver.locate.bind(resolver), source).then(function (result) {
            if (typeof result !== 'string') {
                throw createError('Resolver\'s "locate" method should return a string', 'ERESOLVERAPI');
            }

            return result;
        });
    };

    return PluginResolver;
}


module.exports = pluginResolverFactory;

