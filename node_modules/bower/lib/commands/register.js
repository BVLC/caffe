var Q = require('q');
var chalk = require('chalk');
var PackageRepository = require('../core/PackageRepository');
var Tracker = require('../util/analytics').Tracker;
var createError = require('../util/createError');
var defaultConfig = require('../config');

function register(logger, name, url, config) {
    var repository;
    var registryClient;
    var tracker;
    var force;

    config = defaultConfig(config);
    force = config.force;
    tracker = new Tracker(config);

    name = (name || '').trim();
    url = (url || '').trim();

    // Bypass any cache
    config.offline = false;
    config.force = true;

    return Q.try(function () {
        // Verify name and url
        if (!name || !url) {
            throw createError('Usage: bower register <name> <url>', 'EINVFORMAT');
        }

        tracker.track('register');

        // Attempt to resolve the package referenced by the URL to ensure
        // everything is ok before registering
        repository = new PackageRepository(config, logger);
        return repository.fetch({ name: name, source: url, target: '*' });
    })
    .spread(function (canonicalDir, pkgMeta) {
        if (pkgMeta.private) {
            throw createError('The package you are trying to register is marked as private', 'EPRIV');
        }

        // If non interactive or user forced, bypass confirmation
        if (!config.interactive || force) {
            return true;
        }

        // Confirm if the user really wants to register
        return Q.nfcall(logger.prompt.bind(logger), {
            type: 'confirm',
            message: 'Registering a package will make it installable via the registry (' +
                chalk.cyan.underline(config.registry.register) + '), continue?',
            default: true
        });
    })
    .then(function (result) {
        // If user response was negative, abort
        if (!result) {
            return;
        }

        // Register
        registryClient = repository.getRegistryClient();

        logger.action('register', url, {
            name: name,
            url: url
        });

        return Q.nfcall(registryClient.register.bind(registryClient), name, url);
    });
}

// -------------------

register.readOptions = function (argv) {
    var cli = require('../util/cli');

    var options = cli.readOptions(argv);
    var name = options.argv.remain[1];
    var url = options.argv.remain[2];

    return [name, url];
};

module.exports = register;
