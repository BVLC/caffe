var Q = require('q');
var fs = require('../util/fs');
var path = require('path');
var mout = require('mout');
var resolvers = require('./resolvers');
var createError = require('../util/createError');

var pluginResolverFactory = require('./resolvers/pluginResolverFactory');

function createInstance(decEndpoint, options, registryClient) {
    decEndpoint = mout.object.pick(decEndpoint, ['name', 'target', 'source']);

    options.version = require('../../package.json').version;

    return getConstructor(decEndpoint, options, registryClient)
    .spread(function (ConcreteResolver, decEndpoint) {
        return new ConcreteResolver(decEndpoint, options.config, options.logger);
    });
}

function getConstructor(decEndpoint, options, registryClient) {
    var source = decEndpoint.source;
    var config = options.config;

    // Below we try a series of async tests to guess the type of resolver to use
    // If a step was unable to guess the resolver, it returns undefined
    // If a step can guess the resolver, it returns with construcotor of resolver

    var promise = Q.resolve();

    var addResolver = function (resolverFactory) {
        promise = promise.then(function (result) {
            if (result === undefined) {
                return resolverFactory(decEndpoint, options);
            } else {
                return result;
            }
        });
    };

    // Plugin resolvers.
    //
    // It requires each resolver defined in config.resolvers and calls
    // its "match" to check if given resolves supports given decEndpoint
    addResolver(function () {
        var selectedResolver;
        var resolverNames;

        if (Array.isArray(config.resolvers)) {
            resolverNames = config.resolvers;
        } else if (!!config.resolvers) {
            resolverNames = config.resolvers.split(',');
        } else {
            resolverNames = [];
        }

        var resolverPromises = resolverNames.map(function (resolverName) {
            var resolver = resolvers[resolverName]
                || pluginResolverFactory(require(resolverName), options);

            return function () {
                if (selectedResolver === undefined) {
                    var match = resolver.match.bind(resolver);

                    return Q.fcall(match, source).then(function (result) {
                        if (result) {
                            return selectedResolver = resolver;
                        }
                    });
                } else {
                    return selectedResolver;
                }
            };
        });

        return resolverPromises.reduce(Q.when, new Q(undefined)).then(function (resolver) {
            if (resolver) {
                return Q.fcall(resolver.locate.bind(resolver), decEndpoint.source).then(function (result) {
                    if (result && result !== decEndpoint.source) {
                        decEndpoint.source = result;
                        decEndpoint.registry = true;
                        return getConstructor(decEndpoint, options, registryClient);
                    } else {
                        return [resolver, decEndpoint];
                    }
                });
            }
        });
    });

    // Git case: git git+ssh, git+http, git+https
    //           .git at the end (probably ssh shorthand)
    //           git@ at the start
    addResolver(function() {
        if (/^git(\+(ssh|https?))?:\/\//i.test(source) || /\.git\/?$/i.test(source) || /^git@/i.test(source)) {
            decEndpoint.source = source.replace(/^git\+/, '');

            // If it's a GitHub repository, return the specialized resolver
            if (resolvers.GitHub.getOrgRepoPair(source)) {
                return [resolvers.GitHub, decEndpoint];
            }

            return [resolvers.GitRemote, decEndpoint];
        }
    });

    // SVN case: svn, svn+ssh, svn+http, svn+https, svn+file
    addResolver(function () {
        if (/^svn(\+(ssh|https?|file))?:\/\//i.test(source)) {
            return [resolvers.Svn, decEndpoint];
        }
    });

    // URL case
    addResolver(function () {
        if (/^https?:\/\//i.exec(source)) {
            return [resolvers.Url, decEndpoint];
        }
    });


    // If source is ./ or ../ or an absolute path

    addResolver(function () {
        var absolutePath = path.resolve(config.cwd, source);

        if (/^\.\.?[\/\\]/.test(source) || /^~\//.test(source) ||
            path.normalize(source).replace(/[\/\\]+$/, '') === absolutePath
        ) {
            return Q.nfcall(fs.stat, path.join(absolutePath, '.git'))
            .then(function (stats) {
                decEndpoint.source = absolutePath;

                if (stats.isDirectory()) {
                    return Q.resolve([resolvers.GitFs, decEndpoint]);
                }

                throw new Error('Not a Git repository');
            })
            // If not, check if source is a valid Subversion repository
            .fail(function () {
                return Q.nfcall(fs.stat, path.join(absolutePath, '.svn'))
                .then(function (stats) {
                    decEndpoint.source = absolutePath;

                    if (stats.isDirectory()) {
                        return Q.resolve([resolvers.Svn, decEndpoint]);
                    }

                    throw new Error('Not a Subversion repository');
                });
            })
            // If not, check if source is a valid file/folder
            .fail(function () {
                return Q.nfcall(fs.stat, absolutePath)
                .then(function () {
                    decEndpoint.source = absolutePath;

                    return Q.resolve([resolvers.Fs, decEndpoint]);
                });
            });
        }
    });

    // Check if is a shorthand and expand it
    addResolver(function () {
        // Skip ssh and/or URL with auth
        if (/[:@]/.test(source)) {
            return;
        }

        // Ensure exactly only one "/"
        var parts = source.split('/');
        if (parts.length === 2) {
            decEndpoint.source = mout.string.interpolate(config.shorthandResolver, {
                shorthand: source,
                owner: parts[0],
                package: parts[1]
            });

            return getConstructor(decEndpoint, options, registryClient);
        }
    });

    // As last resort, we try the registry
    addResolver(function () {
        if (!registryClient) {
            return;
        }

        return Q.nfcall(registryClient.lookup.bind(registryClient), source)
        .then(function (entry) {
            if (!entry) {
                throw createError('Package ' + source + ' not found', 'ENOTFOUND');
            }

            decEndpoint.registry = true;

            if (!decEndpoint.name) {
                decEndpoint.name = decEndpoint.source;
            }

            decEndpoint.source = entry.url;

            return getConstructor(decEndpoint, options);
        });
    });

    addResolver(function () {
        throw createError('Could not find appropriate resolver for ' + source, 'ENORESOLVER');
    });

    return promise;
}

function clearRuntimeCache() {
    mout.object.values(resolvers).forEach(function (ConcreteResolver) {
        ConcreteResolver.clearRuntimeCache();
    });
}

module.exports = createInstance;
module.exports.getConstructor = getConstructor;
module.exports.clearRuntimeCache = clearRuntimeCache;
