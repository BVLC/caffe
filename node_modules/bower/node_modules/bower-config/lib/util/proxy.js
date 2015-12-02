// EnvProxy uses the proxy vaiables passed to it in set and sets the
// process.env uppercase proxy variables to them with the ability
// to restore the original values later
var EnvProxy = function() {
  this.restoreFrom = {};
};

EnvProxy.prototype.set = function (config) {
  this.config = config;

  // Override environment defaults if proxy config options are set
  // This will make requests.js follow the proxies in config
  if (Object.prototype.hasOwnProperty.call(config, 'noProxy')) {
    this.restoreFrom.NO_PROXY = process.env.NO_PROXY;
    this.restoreFrom.no_proxy = process.env.no_proxy;
    delete process.env.no_proxy;
    process.env.NO_PROXY = config.noProxy;
  }

  if (Object.prototype.hasOwnProperty.call(config, 'proxy')) {
    this.restoreFrom.HTTP_PROXY = process.env.HTTP_PROXY;
    this.restoreFrom.http_proxy = process.env.http_proxy;
    delete process.env.http_proxy;
    process.env.HTTP_PROXY = config.proxy;
  }

  if (Object.prototype.hasOwnProperty.call(config, 'httpsProxy')) {
    this.restoreFrom.HTTPS_PROXY = process.env.HTTPS_PROXY;
    this.restoreFrom.https_proxy = process.env.https_proxy;
    delete process.env.https_proxy;
    process.env.HTTPS_PROXY = config.httpsProxy;
  }
};

EnvProxy.prototype.restore = function () {
  if (Object.prototype.hasOwnProperty.call(this.config, 'noProxy')) {
    if (this.restoreFrom.NO_PROXY !== undefined) {
      process.env.NO_PROXY = this.restoreFrom.NO_PROXY;
    } else {
      delete process.env.NO_PROXY;
    }

    if (this.restoreFrom.no_proxy !== undefined) {
      process.env.no_proxy = this.restoreFrom.no_proxy;
    } else {
      delete process.env.no_proxy;
    }
  }

  if (Object.prototype.hasOwnProperty.call(this.config, 'proxy')) {
    if (this.restoreFrom.HTTP_PROXY !== undefined) {
      process.env.HTTP_PROXY = this.restoreFrom.HTTP_PROXY;
    } else {
      delete process.env.HTTP_PROXY;
    }

    if (this.restoreFrom.http_proxy !== undefined) {
      process.env.http_proxy = this.restoreFrom.http_proxy;
    } else {
      delete process.env.http_proxy;
    }
  }

  if (Object.prototype.hasOwnProperty.call(this.config, 'httpsProxy')) {
    if (this.restoreFrom.HTTPS_PROXY !== undefined) {
      process.env.HTTPS_PROXY = this.restoreFrom.HTTPS_PROXY;
    } else {
      delete process.env.HTTPS_PROXY;
    }

    if (this.restoreFrom.https_proxy !== undefined) {
      process.env.https_proxy = this.restoreFrom.https_proxy;
    } else {
      delete process.env.https_proxy;
    }
  }
};

module.exports = EnvProxy;
