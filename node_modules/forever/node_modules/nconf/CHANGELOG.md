# CHANGELOG

### Version 0.5.0

* `nconf.stores.*` is now `nconf.*`
* `nconf.stores` now represents the set of nconf.* Store instances on the nconf object.
* Added `nconf.argv()`, `nconf.env()`, `nconf.file()`, `nconf.overrides()`, `nconf.defaults()`. 
* `nconf.system` no longer exists. The `nconf.System` store has been broken into `nconf.Argv`, `nconf.Env` and `nconf.Literal`
* Fixed bugs in hierarchical configuration loading.