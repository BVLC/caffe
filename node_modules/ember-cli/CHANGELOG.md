# ember-cli Changelog

### 1.13.13

The following changes are required if you are upgrading from the previous
version:

- Users
  + default Ember.js version is now at `1.13.11`
  + Ember CLI SRI version was bumped to `^1.2.0`
  + Ember `loader.js` version was bumped to `3.4.0`
  + Testem version was bumped to `0.9.11`

#### Community Contributions

- [#5061](https://github.com/ember-cli/ember-cli/pull/5061) Testem 0.9.11, [@johanneswuerbach](https://github.com/johanneswuerbach)
- [#5094](https://github.com/ember-cli/ember-cli/pull/5094) Remove deprecated `this.Funnel` usage, [@stefanpenner](https://github.com/stefanpenner)
- [#5075](https://github.com/ember-cli/ember-cli/pull/5075) Disable bundling, npm client seems to have issues, [@stefanpenner](https://github.com/stefanpenner)
- [#5104](https://github.com/ember-cli/ember-cli/pull/5104) Workaround for babel includePolyfill exception, [@ef4](https://github.com/ef4)
- [#5116](https://github.com/ember-cli/ember-cli/pull/5116) Bumps Ember version to `1.13.11`, [@stefanpenner](https://github.com/stefanpenner)
- [#5116](https://github.com/ember-cli/ember-cli/pull/5116) Bumps Ember CLI SRI to `^1.2.0`, [@rwjblue](https://github.com/rwjblue)

### 1.13.12

The following changes are required if you are upgrading from the previous
version:

- Users
  + changes to `tests/index.html` file. All tests are now in a separate file, [diff](https://github.com/twokul/ember-cli-release-notes/commit/bd5ac542c0d6dd8e095553d6528ec40ae4be6b4e).
  + default Ember.js version is now at `1.13.10`
  + default Ember Data version is now at `1.13.14`
  + Upgrade your project's ember-cli version - [docs](http://www.ember-cli.com/user-guide/#upgrading)
  + [`ember new` diff](https://github.com/twokul/ember-cli-release-notes/commit/1cee41eb91bf7c534917fdd4cd42a1cd1a481c75)
- Addon Developers
  + [`ember addon` diff](https://github.com/twokul/ember-cli-release-notes/commit/733bb1a30d7ec5ef7f8d6fc021c052633ae66bb3)
- Core Contributors
  + No changes required

#### Community Contributions

- [#4838](https://github.com/ember-cli/ember-cli/pull/4838) Add `npm test` to Addon README [@elwayman02](https://github.com/elwayman02)
- [#4756](https://github.com/ember-cli/ember-cli/pull/4756) Discard runCommand stdout/stderr unless we have a test failure [@joliss](https://github.com/joliss)
- [#4753](https://github.com/ember-cli/ember-cli/pull/4753) Upgraded ember-cli-app-version to 1.0.0 [@ember-cli](https://github.com/ember-cli)
- [#4235](https://github.com/ember-cli/ember-cli/pull/4235) Reintroduce comment regarding bodyParser on http-mock [@joostdevries](https://github.com/joostdevries)
- [#4728](https://github.com/ember-cli/ember-cli/pull/4728) Revert "Do not pack ember-cli-build.js" [@ember-cli](https://github.com/ember-cli)
- [#4846](https://github.com/ember-cli/ember-cli/pull/4846) Update ember-cli-htmlbars-inline-precompile dependency [@joliss](https://github.com/joliss)
- [#4757](https://github.com/ember-cli/ember-cli/pull/4757) Remove last use of broccoli-writer in test suite [@joliss](https://github.com/joliss)
- [#4765](https://github.com/ember-cli/ember-cli/pull/4765) Fix typo Brocolli → Broccoli [@lancedikson](https://github.com/lancedikson)
- [#4770](https://github.com/ember-cli/ember-cli/pull/4770) Updates configstore [@twokul](https://github.com/twokul)
- [#4774](https://github.com/ember-cli/ember-cli/pull/4774) Update ember-cli-shims to prevent errors on Ember < 1.13. [@rwjblue](https://github.com/rwjblue)
- [#4772](https://github.com/ember-cli/ember-cli/pull/4772) Implement a destroyApp helper. [@blimmer](https://github.com/blimmer)
- [#4771](https://github.com/ember-cli/ember-cli/pull/4771) Make default generated tests pass. [@blimmer](https://github.com/blimmer)
- [#4854](https://github.com/ember-cli/ember-cli/pull/4854) command unit test syncing [@kellyselden](https://github.com/kellyselden)
- [#4801](https://github.com/ember-cli/ember-cli/pull/4801) add a missing curved brackets [@dukex](https://github.com/dukex)
- [#4799](https://github.com/ember-cli/ember-cli/pull/4799) Bump ember-cli-dependency-checker to v1.1.0 [@quaertym](https://github.com/quaertym)
- [#4792](https://github.com/ember-cli/ember-cli/pull/4792) Replace Esperanto With Babel [@ember-cli](https://github.com/ember-cli)
- [#4783](https://github.com/ember-cli/ember-cli/pull/4783) bump viz [@ember-cli](https://github.com/ember-cli)
- [#4788](https://github.com/ember-cli/ember-cli/pull/4788) include FS usage monitoring [@ember-cli](https://github.com/ember-cli)
- [#4785](https://github.com/ember-cli/ember-cli/pull/4785) some safe runCommand removals [@kellyselden](https://github.com/kellyselden)
- [#4781](https://github.com/ember-cli/ember-cli/pull/4781) Set `ember serve --host` default to `undefined`. [@buschtoens](https://github.com/buschtoens)
- [#4796](https://github.com/ember-cli/ember-cli/pull/4796) tested needs null integrity value (since it always chan… [@ember-cli](https://github.com/ember-cli)
- [#4880](https://github.com/ember-cli/ember-cli/pull/4880) consolidate test setup to not use fixtures, instead use mocking [@kellyselden](https://github.com/kellyselden)
- [#4816](https://github.com/ember-cli/ember-cli/pull/4816) Allow OS to choose ephemeral port if --test-port=0 [@williamsbdev](https://github.com/williamsbdev)
- [#4815](https://github.com/ember-cli/ember-cli/pull/4815) Add Node.js 4.0 as valid platform version [@szines](https://github.com/szines)
- [#4839](https://github.com/ember-cli/ember-cli/pull/4839) addAddonsToProject for blueprints [@elwayman02](https://github.com/elwayman02)
- [#4826](https://github.com/ember-cli/ember-cli/pull/4826) Update bower deps. [@rwjblue](https://github.com/rwjblue)
- [#4844](https://github.com/ember-cli/ember-cli/pull/4844) fix failing tests [@ember-cli](https://github.com/ember-cli)
- [#4836](https://github.com/ember-cli/ember-cli/pull/4836) Make config replace cache [@ember-cli](https://github.com/ember-cli)
- [#4837](https://github.com/ember-cli/ember-cli/pull/4837) Add `Blueprint.prototype.filesPath`. [@rwjblue](https://github.com/rwjblue)
- [#4827](https://github.com/ember-cli/ember-cli/pull/4827) Update to broccoli-caching-writer 2.0.0 [@joliss](https://github.com/joliss)
- [#4829](https://github.com/ember-cli/ember-cli/pull/4829) broccoli-plugin{description -> annotation} [@ember-cli](https://github.com/ember-cli)
- [#4874](https://github.com/ember-cli/ember-cli/pull/4874) Add ability to specify a build path for running tests [@trentmwillis](https://github.com/trentmwillis)
- [#4863](https://github.com/ember-cli/ember-cli/pull/4863) remove duplicate in package.json [@lazybensch](https://github.com/lazybensch)
- [#4849](https://github.com/ember-cli/ember-cli/pull/4849) cleanup whitespace in the commands [@kellyselden](https://github.com/kellyselden)
- [#4857](https://github.com/ember-cli/ember-cli/pull/4857) Only exclude node_modules at root [@joliss](https://github.com/joliss)
- [#4881](https://github.com/ember-cli/ember-cli/pull/4881) add unit tests for blueprint help printing [@kellyselden](https://github.com/kellyselden)
- [#4913](https://github.com/ember-cli/ember-cli/pull/4913) adding addtional help tests [@kellyselden](https://github.com/kellyselden)
- [#4946](https://github.com/ember-cli/ember-cli/pull/4946) Fix beforeEach/afterEach callbacks with moduleForAcceptance. [@rwjblue](https://github.com/rwjblue)
- [#4954](https://github.com/ember-cli/ember-cli/pull/4954) Add smoke test for `moduleForAcceptance` [@seanpdoyle](https://github.com/seanpdoyle)
- [#4970](https://github.com/ember-cli/ember-cli/pull/4970) Upgrade testem to 0.9.8 [@rzurad](https://github.com/rzurad)
- [#4973](https://github.com/ember-cli/ember-cli/pull/4973) Lock down ember-router-generator@1.0.0 [@ember-cli](https://github.com/ember-cli)
- [#4989](https://github.com/ember-cli/ember-cli/pull/4989) added my "Why is CI broken?" tool to the readme [@ember-cli](https://github.com/ember-cli)
- [#5039](https://github.com/ember-cli/ember-cli/pull/5039) Add `UI.errorStream` [@seanpdoyle](https://github.com/seanpdoyle)
- [#5036](https://github.com/ember-cli/ember-cli/pull/5036) Remove native bundled dependencies [@patocallaghan](https://github.com/patocallaghan)
- [#5026](https://github.com/ember-cli/ember-cli/pull/5026) Add support for watchman 4 [@jcope2013](https://github.com/jcope2013)
- [#5020](https://github.com/ember-cli/ember-cli/pull/5020) explicitly test node 5.0 [@stefanpenner](https://github.com/stefanpenner)

Thank you to all who took the time to contribute!

### 1.13.8

The following changes are required if you are upgrading from the previous
version:

- Users
  + [`ember new` diff](https://github.com/kellyselden/ember-cli-output/commit/db09559dc922eafdfb6723969b53dfff1a8f5331)
  + default ember is now at 1.13.7 (but feel free to upgrade/downgrade as desired)
  + default ember-data is now at 1.13.8 (but feel free to upgrade/downgrade as desired)
  + for users with very large bower_components directories, rebuild times should improve
  + Upgrade your project's ember-cli version - [docs](http://www.ember-cli.com/user-guide/#upgrading)
  + If you haven't already, please remember to transition your Brocfile.js to ember-cli-build.js. [more details](https://github.com/ember-cli/ember-cli/blob/master/TRANSITION.md#brocfile-transition)
- Addon Developers
  + [`ember addon` diff](https://github.com/kellyselden/ember-addon-output/commit/aaf7faebf1ca721382d281dd3125b24c7a752c7e)
  + No changes required
- Core Contributors
  + No changes required

#### Community Contributions

- [#4599](https://github.com/ember-cli/ember-cli/pull/4599) Update valid-platform-version.js [@stefanpenner](https://github.com/stefanpenner)
- [#4590](https://github.com/ember-cli/ember-cli/pull/4590) Remove `ember update` mention in update-checker [@quaertym](https://github.com/quaertym)
- [#4582](https://github.com/ember-cli/ember-cli/pull/4582) blueprints/app/package.json: Sort scripts alphabetically [@Turbo87](https://github.com/Turbo87)
- [#4577](https://github.com/ember-cli/ember-cli/pull/4577) Adding more help acceptance tests [@kellyselden](https://github.com/kellyselden)
- [#4621](https://github.com/ember-cli/ember-cli/pull/4621) bump funnel, and prefer globs for includes. [@stefanpenner](https://github.com/stefanpenner)
- [#4598](https://github.com/ember-cli/ember-cli/pull/4598) bump timeout, see if iojs on CI becomes happy again [@stefanpenner](https://github.com/stefanpenner)
- [#4596](https://github.com/ember-cli/ember-cli/pull/4596) Bump version of ember-cli-app-version to 0.5.0 [@taras](https://github.com/taras)
- [#4591](https://github.com/ember-cli/ember-cli/pull/4591) Revert "Remove `ember update` mention in update-checker" [@stefanpenner](https://github.com/stefanpenner)
- [#4597](https://github.com/ember-cli/ember-cli/pull/4597) update to ember-cli-qunit v1.0.0 [@stefanpenner](https://github.com/stefanpenner)
- [#4593](https://github.com/ember-cli/ember-cli/pull/4593) Remove ember update mention in update-checker [@quaertym](https://github.com/quaertym)
- [#4628](https://github.com/ember-cli/ember-cli/pull/4628) a couple more tests that weren't being run [@kellyselden](https://github.com/kellyselden)
- [#4606](https://github.com/ember-cli/ember-cli/pull/4606) [TYPO] SRI changelog typo fix [@jonathanKingston](https://github.com/jonathanKingston)
- [#4601](https://github.com/ember-cli/ember-cli/pull/4601) update broccoli-caching-writer [@stefanpenner](https://github.com/stefanpenner);
- [#4630](https://github.com/ember-cli/ember-cli/pull/4630) Update blueprint dependencies [@btecu](https://github.com/btecu)
- [#4611](https://github.com/ember-cli/ember-cli/pull/4611) Update Ember Data dependency to 1.13.8 [@bmac](https://github.com/bmac)
- [#4638](https://github.com/ember-cli/ember-cli/pull/4638) broccoli-plugin now uses annotation, rather then our own convention o… [@stefanpenner](https://github.com/stefanpenner)
- [#4622](https://github.com/ember-cli/ember-cli/pull/4622) Upgrade merge trees [@stefanpenner](https://github.com/stefanpenner)
- [#4625](https://github.com/ember-cli/ember-cli/pull/4625) bump broccoli-caching-writer to 1.1.0 [@kellyselden](https://github.com/kellyselden)
- [#4627](https://github.com/ember-cli/ember-cli/pull/4627) fix test that was never being run [@kellyselden](https://github.com/kellyselden)
- [#4629](https://github.com/ember-cli/ember-cli/pull/4629) broccoli-asset-rev to 2.1.2 [@kellyselden](https://github.com/kellyselden)
- [#4632](https://github.com/ember-cli/ember-cli/pull/4632) Windows CI: Removed npm upgrade [@johanneswuerbach](https://github.com/johanneswuerbach)
- [#4636](https://github.com/ember-cli/ember-cli/pull/4636) Update Ember version to 1.13.7. [@rwjblue](https://github.com/rwjblue)
- [#4637](https://github.com/ember-cli/ember-cli/pull/4637) [INTERNAL] Prefer globs over RegExps as funnel arguments [@dschmidt](https://github.com/dschmidt)
- [#4642](https://github.com/ember-cli/ember-cli/pull/4642) bump broccoli-funnel [@stefanpenner](https://github.com/stefanpenner)
- [#4643](https://github.com/ember-cli/ember-cli/pull/4643) Support a (now deprecated) single-argument use of addBowerPackageToProject [@mike-north](https://github.com/mike-north)

Thank you to all who took the time to contribute!

### 1.13.7

The following changes are required if you are upgrading from the previous
version:

- Users
  + [`ember new` diff](https://github.com/kellyselden/ember-cli-output/commit/6a41c5cd7f0f68e7cf710268376d0349c5b57171)
  + Upgrade your project's ember-cli version - [docs](http://www.ember-cli.com/user-guide/#upgrading)
  + If you haven't already, please remember to transition your Brocfile.js to ember-cli-build.js. [more details](https://github.com/ember-cli/ember-cli/blob/master/TRANSITION.md#brocfile-transition)
- Addon Developers
  + [`ember addon` diff](https://github.com/kellyselden/ember-addon-output/commit/f6f61d55c31d631203bc5491432b435e2cc807c2)
  + No changes required
- Core Contributors
  + No changes required

#### Community Contributions

- [#4558](https://github.com/ember-cli/ember-cli/pull/4558) ensure we apply patches at the right part of the release. [@stefanpenner](https://github.com/ember-cli)
- [#4559](https://github.com/ember-cli/ember-cli/pull/4559) bundle testem [@stefanpenner](https://github.com/ember-cli)
- [#4560](https://github.com/ember-cli/ember-cli/pull/4560) Update ember-qunit to 0.4.9. [@rwjblue](https://github.com/rwjblue)
- [#4561](https://github.com/ember-cli/ember-cli/pull/4561) Upgrade to Broccoli 0.16.5 [@joliss](https://github.com/joliss)
- [#4564](https://github.com/ember-cli/ember-cli/pull/4564) add 1.13.6 diffs to changelog [@kellyselden](https://github.com/kellyselden)
- [#4569](https://github.com/ember-cli/ember-cli/pull/4569) Update Ember to v1.13.6. [@rwjblue](https://github.com/rwjblue)
- [#4572](https://github.com/ember-cli/ember-cli/pull/4572) Update QUnit version to 1.18.0. [@rwjblue](https://github.com/rwjblue)
- [#4589](https://github.com/ember-cli/ember-cli/pull/4589) Fixes issue with smoke test failure. [@rickharrison](https://github.com/rickharrison)

Thank you to all who took the time to contribute!

### 1.13.6

The following changes are required if you are upgrading from the previous
version:

- Users
  + [`ember new` diff](https://github.com/kellyselden/ember-cli-output/commit/c36b2e35b9ef2a66d6f01f360831c6ec9707c5d7)
  + Upgrade your project's ember-cli version - [docs](http://www.ember-cli.com/user-guide/#upgrading)
  + If you haven't already, please remember to transition your Brocfile.js to ember-cli-build.js. [more details](https://github.com/ember-cli/ember-cli/blob/master/TRANSITION.md#brocfile-transition)
- Addon Developers
  + [`ember addon` diff](https://github.com/kellyselden/ember-addon-output/commit/d77330079ca14a1d0e39383cce87565c1c2d742f)
  + No changes required
- Core Contributors
  + No changes required

#### Community Contributions

- [#3239](https://github.com/ember-cli/ember-cli/pull/3239) ENHANCEMENT: Added `--test-port`/`testPort` option to configure test port [@patocallaghan](https://github.com/patocallaghan)
- [#4545](https://github.com/ember-cli/ember-cli/pull/4545) bump es6modules to fix IE8 issue [@stefanpenner](https://github.com/ember-cli)
- [#4549](https://github.com/ember-cli/ember-cli/pull/4549) adding 1.13.5 diff to changelog [@kellyselden](https://github.com/kellyselden)
- [#4553](https://github.com/ember-cli/ember-cli/pull/4553) [Bugfix] addAddonToProject fix for 1.13.5 [@jasonmit](https://github.com/jasonmit)

Thank you to all who took the time to contribute!

### 1.13.5

The following changes are required if you are upgrading from the previous
version:

- Users
  + [`ember new` diff](https://github.com/kellyselden/ember-cli-output/commit/750a6ba374fc8bb2bbb6102fcb9db399dd1c2472)
  + Upgrade your project's ember-cli version - [docs](http://www.ember-cli.com/user-guide/#upgrading)
  + If you haven't already, please remember to transition your Brocfile.js to ember-cli-build.js. [more details](https://github.com/ember-cli/ember-cli/blob/master/TRANSITION.md#brocfile-transition)
  + We now bundle ember.js 1.13.5 and ember-data 1.13.7 by default, but please note you can change these by updating bower.json
  + We have included support for [Subresource Integrity (SRI)](http://www.w3.org/TR/SRI) by default, to find out more checkout our site's [SRI section](http://www.ember-cli.com/user-guide/#subresource-integrity)
  + Please note: Testem will now error if a specified runner is missing.
  + When installing ember-cli, one can use `npm install ember-cli --no-optional` to skip all native dependencies.
- Addon Developers
  + [`ember addon` diff](https://github.com/kellyselden/ember-addon-output/commit/ace30d3fecafee7e27ae5d75254096a08ede2a6c)
  + No changes required
- Core Contributors
  + No changes required

#### Community Contributions

- [#4471](https://github.com/ember-cli/ember-cli/pull/4471) Make the Examples less confusing [@mdragon](https://github.com/mdragon)
- [#4367](https://github.com/ember-cli/ember-cli/pull/4367) [BUGFIX] Adding check for undefined inRepoAddon option in Blueprint.prototype.\_locals. [@gmurphey](https://github.com/gmurphey)
- [#4411](https://github.com/ember-cli/ember-cli/pull/4411) Removing unknown command from `ember help`. [@gmurphey](https://github.com/gmurphey)
- [#4405](https://github.com/ember-cli/ember-cli/pull/4405) Fix default generated integration test. [@blimmer](https://github.com/blimmer)
- [#4406](https://github.com/ember-cli/ember-cli/pull/4406) [ENHANCEMENT] Let ember install take multiple addons [@DanielOchoa](https://github.com/DanielOchoa)
- [#4478](https://github.com/ember-cli/ember-cli/pull/4478) Store acceptance test application in test context. [@rwjblue](https://github.com/rwjblue)
- [#4413](https://github.com/ember-cli/ember-cli/pull/4413) Add clarity for 0.2.7 to 1.13.x transition (build) [@pixelhandler](https://github.com/pixelhandler)
- [#4416](https://github.com/ember-cli/ember-cli/pull/4416) bump sourcemap-concat version [@kwikPRs](https://github.com/kwikPRs)
- [#4419](https://github.com/ember-cli/ember-cli/pull/4419) Nuke 'ember update' [@jonnii](https://github.com/jonnii)
- [#4488](https://github.com/ember-cli/ember-cli/pull/4488) Update Ember to 1.13.5. [@rwjblue](https://github.com/rwjblue)
- [#4440](https://github.com/ember-cli/ember-cli/pull/4440) EmberAddon Should Merge Defaults [@chadhietala](https://github.com/chadhietala)
- [#4438](https://github.com/ember-cli/ember-cli/pull/4438) Fix Ember CLI project update link [@balinterdi](https://github.com/balinterdi)
- [#4428](https://github.com/ember-cli/ember-cli/pull/4428) Update #watchman message to point to correct url [@supabok](https://github.com/supabok)
- [#4430](https://github.com/ember-cli/ember-cli/pull/4430) Handle a wide variety of bower endpoints [@truenorth](https://github.com/truenorth)
- [#4454](https://github.com/ember-cli/ember-cli/pull/4454) Always use Brocfile (with deprecation messaging) if it exists [@gmurphey](https://github.com/gmurphey)
- [#4452](https://github.com/ember-cli/ember-cli/pull/4452) [ENHANCEMENT] Detect & skip lib install on http-mock gen [@sivakumar-kailasam](https://github.com/sivakumar-kailasam)
- [#4456](https://github.com/ember-cli/ember-cli/pull/4456) Updating getPort logic to use liveReloadHost. Fixes #4455. [@gmurphey](https://github.com/gmurphey)
- [#4443](https://github.com/ember-cli/ember-cli/pull/4443) Prevent live-reload-port collisions (by default) [@stefanpenner](https://github.com/stefanpenner)
- [#4457](https://github.com/ember-cli/ember-cli/pull/4457) Removing extraneous newline from the component-test blueprint [@gmurphey](https://github.com/gmurphey)
- [#4447](https://github.com/ember-cli/ember-cli/pull/4447) Update to Ember 1.13.4. [@rwjblue](https://github.com/rwjblue)
- [#4449](https://github.com/ember-cli/ember-cli/pull/4449) [ENHANCEMENT] Add --skip-router flag to route blueprint generator [@sivakumar-kailasam](https://github.com/sivakumar-kailasam)
- [#4484](https://github.com/ember-cli/ember-cli/pull/4484) Updating route blueprint to write to router when dummy flag is used [@gmurphey](https://github.com/gmurphey)
- [#4468](https://github.com/ember-cli/ember-cli/pull/4468) [fixes #4467] ensure commands that fail do to being run in the wrong … [@stefanpenner](https://github.com/stefanpenner)
- [#4474](https://github.com/ember-cli/ember-cli/pull/4474) [fixes #4328] patch engion.io-client to use any XMLHTTPRequest, but t… [@stefanpenner](https://github.com/stefanpenner)
- [#4462](https://github.com/ember-cli/ember-cli/pull/4462) Make ember destroy and ember generate give a better error when called… [@marcioj](https://github.com/marcioj)
- [#4466](https://github.com/ember-cli/ember-cli/pull/4466) Native deps now gone [@stefanpenner](https://github.com/stefanpenner)
- [#4465](https://github.com/ember-cli/ember-cli/pull/4465) ensure reexporter doesn't introduce instability during builds [@stefanpenner](https://github.com/stefanpenner)
- [#4516](https://github.com/ember-cli/ember-cli/pull/4516) update ember-cli-qunit to mitigate some leaks [@stefanpenner](https://github.com/stefanpenner)
- [#4489](https://github.com/ember-cli/ember-cli/pull/4489) [ENHANCEMENT] Testem v0.9.0 [@johanneswuerbach](https://github.com/johanneswuerbach)
- [#4483](https://github.com/ember-cli/ember-cli/pull/4483) Adding in ember-cli-sri into application package.json by default [@jonathanKingston](https://github.com/jonathanKingston)
- [#4524](https://github.com/ember-cli/ember-cli/pull/4524) Update ember-qunit to 0.4.6. [@rwjblue](https://github.com/rwjblue)
- [#4490](https://github.com/ember-cli/ember-cli/pull/4490) Fix code of conduct markdown formatting [@max](https://github.com/max)
- [#4495](https://github.com/ember-cli/ember-cli/pull/4495) bump ember-export-application-global [@stefanpenner](https://github.com/stefanpenner)
- [#4498](https://github.com/ember-cli/ember-cli/pull/4498) Testem v0.9.0 [@johanneswuerbach](https://github.com/johanneswuerbach)
- [#4514](https://github.com/ember-cli/ember-cli/pull/4514) component-unit blueprint - trim component content by default [@ramybenaroya](https://github.com/ramybenaroya)
- [#4534](https://github.com/ember-cli/ember-cli/pull/4534) add os to version output [@kellyselden](https://github.com/kellyselden)
- [#4536](https://github.com/ember-cli/ember-cli/pull/4536) Remove unnecessary `"use strict";`s in "app.js" [@nathanhammond](https://github.com/nathanhammond)
- [#4538](https://github.com/ember-cli/ember-cli/pull/4540) Updated ember-qunit to 0.4.7 [@stefanpenner](https://github.com/stefanpenner)

Thank you to all who took the time to contribute!

### 1.13.1

The following changes are required if you are upgrading from the previous
version:

- Users
  + [`ember new` diff](https://github.com/kellyselden/ember-cli-output/commit/f1425c5073a33dfb7ff60d5254fd340046f578bd)
  + Upgrade your project's ember-cli version - [docs](http://www.ember-cli.com/user-guide/#upgrading)
- Addon Developers
  + [`ember addon` diff](https://github.com/kellyselden/ember-addon-output/commit/dc309f7655a2cde4cd81bb75d8f274087e9d82f8)
  + No changes required
- Core Contributors
  + No changes required

#### Community Contributions

- [#4398](https://github.com/ember-cli/ember-cli/pull/4398) [BUGFIX] Fixes #4397 add silentError with deprecation [@trabus](https://github.com/trabus)

Thank you to all who took the time to contribute!
### 1.13.0

The following changes are required if you are upgrading from the previous
version:

- Users
  + [`ember new` diff](https://github.com/kellyselden/ember-cli-output/commit/e83bf78f9be69a6dcedd5a7e16402c6b874efceb)
  + Upgrade your project's ember-cli version - [docs](http://www.ember-cli.com/user-guide/#upgrading)
  + `Brocfile.js` has been deprecated in favor of `ember-cli-build.js`. See [TRANSITION.md](https://github.com/ember-cli/ember-cli/blob/master/TRANSITION.md) for details on how to transition your `Brocfile.js` code to `ember-cli-build.js`.
  + Components are now generated with integration tests by default instead of unit tests. Component unit tests can still be generated separately with: `ember g component-test foo-bar -unit`.
  + Services can now be generated into pod structure.
- Addon Developers
  + [`ember addon` diff](https://github.com/kellyselden/ember-addon-output/commit/c8496e7574826520ead230009068a6313a9712e4)
  + `Brocfile.js` has been deprecated in favor of `ember-cli-build.js`. See [TRANSITION.md](https://github.com/ember-cli/ember-cli/blob/master/TRANSITION.md) for details on how to transition your `Brocfile.js` code to `ember-cli-build.js`.
  + Blueprints can now be generated into the `tests/dummy/app` folder with the `--dummy` flag.
  + Scoped npm dependencies are now supported.
- Core Contributors
  + fs.existsSync is deprecated, use exists-sync instead.

#### Community Contributions
- [#4378](https://github.com/ember-cli/ember-cli/pull/4378) Update Ember to 1.13.3 [@rwjblue](https://github.com/rwjblue)
- [#4395](https://github.com/ember-cli/ember-cli/pull/4395) Update ember-data to 1.13.5 [@trabus](https://github.com/trabus)
- [#4217](https://github.com/ember-cli/ember-cli/pull/4217) [BUGFIX] generating tests inside addons no longer generates addon export file [@trabus](https://github.com/trabus)
- [#4212](https://github.com/ember-cli/ember-cli/pull/4212) fix friendly test description for transforms [@csantero](https://github.com/csantero)
- [#4214](https://github.com/ember-cli/ember-cli/pull/4214) [BUGFIX] correct relative import path for nested adapters [@trabus](https://github.com/trabus)
- [#4215](https://github.com/ember-cli/ember-cli/pull/4215) extract clean-base-url to its own module [@stefanpenner](https://github.com/stefanpenner)
- [#4197](https://github.com/ember-cli/ember-cli/pull/4197) [BUGFIX] add default for path option in component blueprint locals [@trabus](https://github.com/trabus)
- [#4316](https://github.com/ember-cli/ember-cli/pull/4316) fs.existsSync deprecated, replace with exists-sync [@jasonmit](https://github.com/jasonmit)
- [#4224](https://github.com/ember-cli/ember-cli/pull/4224) extract silent-error to its own addon [@stefanpenner](https://github.com/stefanpenner)
- [#4319](https://github.com/ember-cli/ember-cli/pull/4319) Update tmp.js [@jjmiv](https://github.com/jjmiv)
- [#4228](https://github.com/ember-cli/ember-cli/pull/4228) add 0.2.7 diffs [@kellyselden](https://github.com/kellyselden)
- [#4227](https://github.com/ember-cli/ember-cli/pull/4227) Extract process relative require [@stefanpenner](https://github.com/stefanpenner)
- [#4226](https://github.com/ember-cli/ember-cli/pull/4226) extract node-modules-path as its own module [@stefanpenner](https://github.com/stefanpenner)
- [#4326](https://github.com/ember-cli/ember-cli/pull/4326) Drop unused line from app blueprint [@ef4](https://github.com/ef4)
- [#4254](https://github.com/ember-cli/ember-cli/pull/4254) [BUGFIX] Closes #4253. Add `skipHelp` as an available option to commands. [@DanielOchoa](https://github.com/DanielOchoa)
- [#4249](https://github.com/ember-cli/ember-cli/pull/4249) Passing options to tiny-lr (live reload) for HTTPS support [@dosco](https://github.com/dosco)
- [#4239](https://github.com/ember-cli/ember-cli/pull/4239) Fix JSDoc issues [@Turbo87](https://github.com/Turbo87)
- [#4242](https://github.com/ember-cli/ember-cli/pull/4242) Add devDependencies "up to date" badge to README [@truenorth](https://github.com/truenorth)
- [#4251](https://github.com/ember-cli/ember-cli/pull/4251) [BUGFIX] Fix generated addon acceptance test [@trabus](https://github.com/trabus)
- [#4240](https://github.com/ember-cli/ember-cli/pull/4240) Added ember-cli-release to app/addon devDeps blueprint for simple release cutting [@jayphelps](https://github.com/jayphelps)
- [#4286](https://github.com/ember-cli/ember-cli/pull/4286) [Deprecation] Introduce new build file [@chadhietala](https://github.com/chadhietala)
- [#4280](https://github.com/ember-cli/ember-cli/pull/4280) [ENHANCEMENT] Add pod support for services blueprint [@trabus](https://github.com/trabus)
- [#4272](https://github.com/ember-cli/ember-cli/pull/4272) [ENHANCEMENT] Generate component-tests into `tests/integration` by default [@trabus](https://github.com/trabus)
- [#4261](https://github.com/ember-cli/ember-cli/pull/4261) bump ember-cli-htmlbars [@stefanpenner](https://github.com/stefanpenner)
- [#4266](https://github.com/ember-cli/ember-cli/pull/4266) [fixes #4264] [@stefanpenner](https://github.com/stefanpenner)
- [#4270](https://github.com/ember-cli/ember-cli/pull/4270) [BUGFIX] don't allow ember init to create an application without project name [@dukex/bugfix](https://github.com/dukex/bugfix)
- [#4278](https://github.com/ember-cli/ember-cli/pull/4278) Fix 2 typos in livereload-server-test [@jrobeson](https://github.com/jrobeson)
- [#4271](https://github.com/ember-cli/ember-cli/pull/4271) Adding support for private npm modules in blueprints. Closes #4256. [@gmurphey](https://github.com/gmurphey)
- [#4263](https://github.com/ember-cli/ember-cli/pull/4263) [fixes #4260] postprocessTree hook for templates [@stefanpenner](https://github.com/stefanpenner)
- [#4347](https://github.com/ember-cli/ember-cli/pull/4347) ES3+ and for ES5+ deprecation free keys + forEach [@stefanpenner](https://github.com/stefanpenner)
- [#4292](https://github.com/ember-cli/ember-cli/pull/4292) Cleanup pr4283 [@stefanpenner](https://github.com/stefanpenner)
- [#4284](https://github.com/ember-cli/ember-cli/pull/4284) Update ember-resolver to 0.1.17. [@rwjblue](https://github.com/rwjblue)
- [#4287](https://github.com/ember-cli/ember-cli/pull/4287) [ENHANCEMENT] Add ability to generate blueprints into addon `tests/dummy/app` [@trabus](https://github.com/trabus)
- [#4290](https://github.com/ember-cli/ember-cli/pull/4290) Pass the correct port property to LiveReload server [@jrobeson](https://github.com/jrobeson)
- [#4282](https://github.com/ember-cli/ember-cli/pull/4282) Display the LiveReload server address as url [@jrobeson](https://github.com/jrobeson)
- [#4288](https://github.com/ember-cli/ember-cli/pull/4288) Update ember-cli-htmlbars to 0.7.9. [@rwjblue](https://github.com/rwjblue)
- [#4376](https://github.com/ember-cli/ember-cli/pull/4376) Update initializer-test blueprint [@quaertym](https://github.com/quaertym)
- [#4306](https://github.com/ember-cli/ember-cli/pull/4306) [ENHANCEMENT] Print notification when modifying router.js [@trabus](https://github.com/trabus)
- [#4377](https://github.com/ember-cli/ember-cli/pull/4377) Add a test-page option to the test command [@jrjohnson](https://github.com/jrjohnson)
- [#4341](https://github.com/ember-cli/ember-cli/pull/4341) Update ember-load-initializers to 0.1.5 [@jmurphyau](https://github.com/jmurphyau)
- [#4322](https://github.com/ember-cli/ember-cli/pull/4322) Update ADDON_HOOKS.md [@jjmiv](https://github.com/jjmiv)
- [#4334](https://github.com/ember-cli/ember-cli/pull/4334) Cache processed styles tree to prevent double style builds. [@rwjblue](https://github.com/rwjblue)
- [#4309](https://github.com/ember-cli/ember-cli/pull/4309) [ENHANCEMENT] Name blueprint in generate and destroy output message [@trabus](https://github.com/trabus)
- [#4327](https://github.com/ember-cli/ember-cli/pull/4327) Bring tests jshintrc closer to app jshintrc [@ef4](https://github.com/ef4)
- [#4344](https://github.com/ember-cli/ember-cli/pull/4344) [ENHANCEMENT] Fix typo in test command description. [@fabianrbz](https://github.com/fabianrbz)
- [#4362](https://github.com/ember-cli/ember-cli/pull/4362) extract preprocessor-registry -> ember-cli-preprocessor-registry [@stefanpenner](https://github.com/stefanpenner)
- [#4348](https://github.com/ember-cli/ember-cli/pull/4348) Do not pack ember-cli-build.js [@chadhietala](https://github.com/chadhietala)
- [#4343](https://github.com/ember-cli/ember-cli/pull/4343) bump to ember-resolve 0.1.18 – which fixes deprecations while continu… [@stefanpenner](https://github.com/stefanpenner)
- [#4349](https://github.com/ember-cli/ember-cli/pull/4349) enable both relative and absolute treePaths (npm v3 fix) [@stefanpenner](https://github.com/stefanpenner)
- [#4359](https://github.com/ember-cli/ember-cli/pull/4359) Update helper blueprint to use `Ember.Helper.helper` [@balinterdi](https://github.com/balinterdi)
- [#4354](https://github.com/ember-cli/ember-cli/pull/4354) Upgrade to ember-cli-app-version 0.4.0 [@taras](https://github.com/taras)
- [#4370](https://github.com/ember-cli/ember-cli/pull/4370) Remove unexpected final newline [@treyhunner](https://github.com/treyhunner)
- [#4374](https://github.com/ember-cli/ember-cli/pull/4374) Update appveyor.yml [@stefanpenner](https://github.com/stefanpenner)
- [#4382](https://github.com/ember-cli/ember-cli/pull/4382) Update ember-cli-qunit to 0.3.15. [@rwjblue](https://github.com/rwjblue)
- [#4384](https://github.com/ember-cli/ember-cli/pull/4384) [ENHANCEMENT] Add block-template assertion to generated component integration test [@trabus](https://github.com/trabus)
- [#4385](https://github.com/ember-cli/ember-cli/pull/4385) Dependency updates [@truenorth](https://github.com/truenorth)

Thank you to all who took the time to contribute!

### 0.2.7

The following changes are required if you are upgrading from the previous
version:

- Users
  + [`ember new` diff](https://github.com/kellyselden/ember-cli-output/commit/8cd6f5ee0012d3e4960dd9204c9e459f05babd15)
  + Upgrade your project's ember-cli version - [docs](http://www.ember-cli.com/user-guide/#upgrading)
- Addon Developers
  + [`ember addon` diff](https://github.com/kellyselden/ember-addon-output/commit/d59788f7a175376a16e8f4890ac40e6eabb7b9dd)
  + No changes required
- Core Contributors
  + No changes required

#### Community Contributions

- [#4196](https://github.com/ember-cli/ember-cli/pull/4196) [BUGFIX] Adding fileMapToken __name__ for route-addon blueprint. [@gmurphey](https://github.com/gmurphey)
- [#4203](https://github.com/ember-cli/ember-cli/pull/4203) [BUGFIX] acceptance-test blueprint no longer generates addon re-export in app folder [@trabus](https://github.com/trabus)
- [#4206](https://github.com/ember-cli/ember-cli/pull/4206) [Bugfix] Addon.prototype.compileTemplates should not use deprecated t… [@stefanpenner](https://github.com/stefanpenner)
- [#4207](https://github.com/ember-cli/ember-cli/pull/4207) [fixes #4205] allow null addonTemplates. [@stefanpenner](https://github.com/stefanpenner)
- [#4208](https://github.com/ember-cli/ember-cli/pull/4208) Drop ncp for cpr [@stefanpenner](https://github.com/stefanpenner)
- [#4210](https://github.com/ember-cli/ember-cli/pull/4210) Upgrade ember-try dependency in addon blueprint [@kategengler](https://github.com/kategengler)

Thank you to all who took the time to contribute!

### 0.2.6

The following changes are required if you are upgrading from the previous
version:

- Users
  + [`ember new` diff](https://github.com/kellyselden/ember-cli-output/commit/734a6b49d4c88ea6431d2793b49477aed70fc220)
  + Upgrade your project's ember-cli version - [docs](http://www.ember-cli.com/user-guide/#upgrading)
  + `ember server` can now be started over `https`. Default ssl
    certificate and ssl key paths are `ssl/server.crt` and
    `ssl/server.key` respectively. Custom paths can be added with
    `--ssl-cert` and `--ssl-key` [#3550](https://github.com/ember-cli/ember-cli/issues/3550).
  + `ember test` now accepts a `reporter` option, it passes this option to Testem with the reporter to use `[tap|dot|xunit]` [#4106](https://github.com/ember-cli/ember-cli/pull/4106).
  + `app/views` is not longer included in the default project blueprint [#4083](https://github.com/ember-cli/ember-cli/pull/4083).
  + Added again `podModulePrefix` to `app.js`. We still need
    podModulePrefix for the time being, it can be removed again when
    the state of pods has been finalized.
  + New apps include a `.watchmanconfig` which tells `watchman` to ignoring `tmp` dir [#4101](https://github.com/ember-cli/ember-cli/issues/4101).
  + Updated `ember-data` to `1.0.0-beta.18`. Install with `npm install --save-dev ember-data@1.0.0-beta.18`.
  + Unit tests for components are now flagged as such [#4177](https://github.com/ember-cli/ember-cli/pull/4177).
- Addon Developers
  + [`ember addon` diff](https://github.com/kellyselden/ember-addon-output/commit/c5db8de5351628f532535f5f6e76e6da8d259299)
  + A new hook is available: `treeForAddonTemplates` which allows you to specify the templates tree. For more info on how to use this hook see [the following issue](https://github.com/yapplabs/ember-modal-dialog/issues/34).
  + Route blueprint now works within addons [#4152](https://github.com/ember-cli/ember-cli/pull/4152).
  + A new generator is available, `ember g route-addon` allows you to create import wrappers for your addon's routes.
- Core Contributors
  + We started to merge pull-request as part of the quest to improve code quality, keep them coming! [#3730](https://github.com/ember-cli/ember-cli/issues/3730).

#### Community Contributions

- [#4143](https://github.com/ember-cli/ember-cli/pull/4143) [BUGFIX] Blueprint.load verify blueprint is in a directory [@trabus](https://github.com/trabus)
- [#4035](https://github.com/ember-cli/ember-cli/pull/4035) Add a verification step to fail the build when tests are filtered with .only [@marcioj](https://github.com/marcioj)
- [#4091](https://github.com/ember-cli/ember-cli/pull/4091) fix name of ember-cli-dependency-checker [@bantic](https://github.com/bantic)
- [#3854](https://github.com/ember-cli/ember-cli/pull/3854) [ENHANCEMENT] install:addon command will show a deprecation message before running the install command. [@DanielOchoa](https://github.com/DanielOchoa)
- [#3550](https://github.com/ember-cli/ember-cli/pull/3550) Add ability to start ember serve on https [@drogus](https://github.com/drogus)
- [#3786](https://github.com/ember-cli/ember-cli/pull/3786) Throw if templating a file fails [@davewasmer](https://github.com/davewasmer)
- [#4026](https://github.com/ember-cli/ember-cli/pull/4026) Revert "Test powershell for appveyor builds" [@stefanpenner](https://github.com/stefanpenner)
- [#4148](https://github.com/ember-cli/ember-cli/pull/4148) extract common SilentError debug/throw logic [@stefanpenner](https://github.com/stefanpenner)
- [#4104](https://github.com/ember-cli/ember-cli/pull/4104) [BUGFIX] Fix custom blueprint options for destroy command [@trabus](https://github.com/trabus)
- [#4106](https://github.com/ember-cli/ember-cli/pull/4106) [ENHANCEMENT] Adding Report option to 'ember test' [@step2yeung](https://github.com/step2yeung)
- [#4155](https://github.com/ember-cli/ember-cli/pull/4155) Updating in-addon and in-repo-addon adapters [@gmurphey](https://github.com/gmurphey)
- [#4123](https://github.com/ember-cli/ember-cli/pull/4123) Remove duplication in lib/utilities/test-info [@quaertym](https://github.com/quaertym)
- [#4114](https://github.com/ember-cli/ember-cli/pull/4114) [Bugfix] 1.4 diff displayed removal before addition. [@stefanpenner](https://github.com/stefanpenner)
- [#4108](https://github.com/ember-cli/ember-cli/pull/4108) Update ember-disable-proxy-controller to 1.0.0 [@cibernox](https://github.com/cibernox)
- [#4116](https://github.com/ember-cli/ember-cli/pull/4116) gzip served files. [@stefanpenner](https://github.com/stefanpenner)
- [#4120](https://github.com/ember-cli/ember-cli/pull/4120) [fixes #4083] remove views dir by default [@stefanpenner](https://github.com/stefanpenner)
- [#4159](https://github.com/ember-cli/ember-cli/pull/4159) Add `treeForAddonTemplates` hook. [@lukemelia](https://github.com/lukemelia)
- [#4142](https://github.com/ember-cli/ember-cli/pull/4142) Installation checker [@stefanpenner](https://github.com/stefanpenner)
- [#4132](https://github.com/ember-cli/ember-cli/pull/4132) Revert "Remove podModulePrefix from app.js" [@trabus](https://github.com/trabus)
- [#4141](https://github.com/ember-cli/ember-cli/pull/4141) ENHANCEMENT More advanced detection of whether outputPath is a parent of the project directory [@catbieber](https://github.com/catbieber)
- [#4138](https://github.com/ember-cli/ember-cli/pull/4138) Extract unknown command [@quaertym](https://github.com/quaertym)
- [#4139](https://github.com/ember-cli/ember-cli/pull/4139) [fixes #4133] warn if helper without `-` is generated [@stefanpenner](https://github.com/stefanpenner)
- [#4124](https://github.com/ember-cli/ember-cli/pull/4124) [ENHANCEMENT] Add watchmanconfig file to blueprints [@mikegrassotti](https://github.com/mikegrassotti)
- [#4152](https://github.com/ember-cli/ember-cli/pull/4152) [ENHANCEMENT] Updating route blueprint to work within addons and create route-addon… [@stefanpenner](https://github.com/stefanpenner)
- [#4157](https://github.com/ember-cli/ember-cli/pull/4157) Add Code Climate config [@chrislopresto](https://github.com/chrislopresto)
- [#4150](https://github.com/ember-cli/ember-cli/pull/4150) Code Quality: npm-install.js, npm-uninstall.js D -> A [@jkarsrud](https://github.com/jkarsrud)
- [#4146](https://github.com/ember-cli/ember-cli/pull/4146) Code Quality: addon.js, project.js D -> C [@jkarsrud](https://github.com/jkarsrud)
- [#4147](https://github.com/ember-cli/ember-cli/pull/4147) Detect ember-cli from deps as well as devDeps [@searls](https://github.com/searls)
- [#4154](https://github.com/ember-cli/ember-cli/pull/4154) remove duplication from normalize entity name [@tyleriguchi](https://github.com/tyleriguchi)
- [#4158](https://github.com/ember-cli/ember-cli/pull/4158) Allow addons to have pod based templates [@pzuraq](https://github.com/pzuraq)
- [#4160](https://github.com/ember-cli/ember-cli/pull/4160) Friendlier comments for Brocfile in addons [@igorT](https://github.com/igorT)
- [#4162](https://github.com/ember-cli/ember-cli/pull/4162) Remove unused variables [@quaertym](https://github.com/quaertym)
- [#4163](https://github.com/ember-cli/ember-cli/pull/4163) Bump ember-data to v1.0.0-beta.18 [@quaertym](https://github.com/quaertym)
- [#4166](https://github.com/ember-cli/ember-cli/pull/4166) upgrade node-require-timings [@stefanpenner](https://github.com/stefanpenner)
- [#4168](https://github.com/ember-cli/ember-cli/pull/4168) Allow internal cli parameters to be configurable by other cli tools [@rodyhaddad](https://github.com/rodyhaddad )
- [#4177](https://github.com/ember-cli/ember-cli/pull/4177) Flag component unit tests as such [@mixonic](https://github.com/mixonic)
- [#4187](https://github.com/ember-cli/ember-cli/pull/4187) isbinaryfile is used in more the just development [@stefanpenner](https://github.com/ember-clistefanpenner)
- [#4188](https://github.com/ember-cli/ember-cli/pull/4188) Fixed type annotations [@Turbo87](https://github.com/Turbo87)

Thank you to all who took the time to contribute!

### 0.2.5

The following changes are required if you are upgrading from the previous
version:

- Users
  + [`ember new` diff](https://github.com/kellyselden/ember-cli-output/commit/f49b35bbb243b6e3b8e20fb2a9c69a2fa13a6aec)
  + Upgrade your project's ember-cli version - [docs](http://www.ember-cli.com/user-guide/#upgrading)
  + package.json
     + Upgrade `ember-cli-qunit` to `0.3.13`.
     + Make sure that `ember-cli-dependency-checker` is using caret `^1.0.0`.
  + bower.json
     + Upgrade `ember-qunit` to `0.3.3`.
- Addon Developers
  + [`ember addon` diff](https://github.com/kellyselden/ember-addon-output/commit/8ef831d2df8abad6445ca7bfa732518c6d8777af)
  + No changes required
- Core Contributors
+ No changes required


#### Community Contributions

- [#4076](https://github.com/ember-cli/ember-cli/pull/4076) Use caret version for stable dependencies in project blueprint. [@abuiles](https://github.com/abuiles)
- [#4087](https://github.com/ember-cli/ember-cli/pull/4087) Bump ember-cli-qunit to v0.3.13 (ember-qunit@0.3.3). [@rwjblue](https://github.com/rwjblue)

### 0.2.4

The following changes are required if you are upgrading from the previous
version:

- Users
  + [`ember new` diff](https://github.com/kellyselden/ember-cli-output/commit/964c80924d665adfde3ce31acaac5c26b95a1bc0)
  + Upgrade your project's ember-cli version - [docs](http://www.ember-cli.com/user-guide/#upgrading)
  + Apps now have [ember-disable-proxy-controllers](https://github.com/cibernox/ember-disable-proxy-controllers)
    included by default, this ensures that autogenerated controllers
    always are regular `Ember.Controller` instead of the deprecated
    proxy ones. This does not affect explicitly created controllers.
  + Generated routes always use `this.route` (`this.resource` is no longer used).
  + The command `ember install:bower` has been removed.
  + Pod components can now be generated outside the
    `app/pods/components` (or `app/components` sans podModulePrefix)
    folder with the `--path` option. `ember g component foo-bar -p
    -path foo` generates into `app/foo/foo-bar/component.js`
  + The `ember new` command now has a `--directory` option, allowing
    you to generate into a directory that differs from your app
    name. `ember new foo -dir bar` generates an app named `foo` into a
    directory named `bar`.
  + Generated apps no longer have `podModulePrefix` in the config.
  + All blueprints have been updated to use shorthand ES6 syntax for importing and exporting.
  + package.json
     + Upgrade `ember-cli-qunit` to `0.3.12`
     + Upgrade `ember-cli-dependency-checker` to `1.0.0`
  + bower.json
     + Bundled ember `v1.12`
     + Upgrade bower.json `ember-qunit` to `0.3.2` for glimmer support.
- Addon Developers
  + [`ember addon` diff](https://github.com/kellyselden/ember-addon-output/commit/4175b66d0911c9ea454daaefb219d11b334f1bab)
  + No changes required
- Core Contributors
  + No changes required

#### Community Contributions

- [#3965](https://github.com/ember-cli/ember-cli/pull/3965) fixup doc generator test [@stefanpenner](https://github.com/stefanpenner)
- [#3822](https://github.com/ember-cli/ember-cli/pull/3822) adding 0.2.3 diffs [@kellyselden](https://github.com/kellyselden)
- [#3384](https://github.com/ember-cli/ember-cli/pull/3384) Test powershell for appveyor builds [@stefanpenner](https://github.com/stefanpenner)
- [#3771](https://github.com/ember-cli/ember-cli/pull/3771) [ENHANCEMENT] Support custom node_module paths [@jakehow](https://github.com/jakehow)
- [#3820](https://github.com/ember-cli/ember-cli/pull/3820) [ENHANCEMENT] Change blueprint command options to type String to avoid nopt transformations [@rodyhaddad](https://github.com/rodyhaddad)
- [#3698](https://github.com/ember-cli/ember-cli/pull/3698) adding docker for linux testing/debugging [@kellyselden](https://github.com/kellyselden)
- [#3973](https://github.com/ember-cli/ember-cli/pull/3973) Config cache unc share [@stefanpenner](https://github.com/stefanpenner)
- [#3836](https://github.com/ember-cli/ember-cli/pull/3836) Suggestion: Adding test coverage to pull requests [@kellyselden](https://github.com/kellyselden)
- [#3827](https://github.com/ember-cli/ember-cli/pull/3827) [BUGFIX] Fixes availableOptions in custom blueprints [@trabus](https://github.com/trabus)
- [#3825](https://github.com/ember-cli/ember-cli/pull/3825) Exclude dist/ from addon npm publishes by default [@jayphelps](https://github.com/jayphelps)
- [#3826](https://github.com/ember-cli/ember-cli/pull/3826) [fixes #3712] rethrow errors in build task [@marcioj](https://github.com/marcioj)
- [#3978](https://github.com/ember-cli/ember-cli/pull/3978) Update broccoli-es6modules [@marcioj](https://github.com/marcioj)
- [#3882](https://github.com/ember-cli/ember-cli/pull/3882) Removes bower install command. [@willrax](https://github.com/willrax)
- [#3869](https://github.com/ember-cli/ember-cli/pull/3869) [BUGFIX] Use posix path for in-repo-addons in package.json [@trabus](https://github.com/trabus)
- [#3846](https://github.com/ember-cli/ember-cli/pull/3846) [BUGFIX] Prevent addon-import blueprint from generating if entity name is undefined [@trabus](https://github.com/trabus)
- [#3851](https://github.com/ember-cli/ember-cli/pull/3851) Use shorthand ES6 syntax for addon -> app re-exports [@jayphelps](https://github.com/jayphelps)
- [#3848](https://github.com/ember-cli/ember-cli/pull/3848) writeError now looks for filename as well as file [@wagenet](https://github.com/wagenet)
- [#3858](https://github.com/ember-cli/ember-cli/pull/3858) move github to normal dependencies to hack around: https://github.com/np... [@stefanpenner](https://github.com/stefanpenner)
- [#3856](https://github.com/ember-cli/ember-cli/pull/3856) Use shorthand ES6 re-export for addon-imports as well, which landed in #3690 [@jayphelps](https://github.com/jayphelps)
- [#3842](https://github.com/ember-cli/ember-cli/pull/3842) Add remove packages [@jonathanKingston](https://github.com/jonathanKingston)
- [#3845](https://github.com/ember-cli/ember-cli/pull/3845) [BUGFIX] Fix ability to generate blueprints (blueprint, http-mock, http-proxy, and tests) inside addons [@trabus](https://github.com/trabus)
- [#3853](https://github.com/ember-cli/ember-cli/pull/3853) Cache `node_modules` and `bower_components` in CI [@seanpdoyle](https://github.com/seanpdoyle)
- [#3994](https://github.com/ember-cli/ember-cli/pull/3994) update sane + broccoli-sane-watcher [@stefanpenner](https://github.com/stefanpenner)
- [#3949](https://github.com/ember-cli/ember-cli/pull/3949) adding a shared folder with host, and fixing git PATH [@kellyselden](https://github.com/kellyselden)
- [#3937](https://github.com/ember-cli/ember-cli/pull/3937) [BUGFIX] Merge app/styles from addons with overwrite: true. Fixes #3930. [@yapplabs](https://github.com/yapplabs)
- [#3895](https://github.com/ember-cli/ember-cli/pull/3895) [Enhancement] PhantomJS 2.0 running on travis-ci [@truenorth](https://github.com/truenorth)
- [#3921](https://github.com/ember-cli/ember-cli/pull/3921) [Enhancement] Ember-try & parallel travis-ci scenario tests for addons [@truenorth](https://github.com/truenorth)
- [#3936](https://github.com/ember-cli/ember-cli/pull/3936) Replace 'this.resource' with 'this.route' in generators [@HeroicEric](https://github.com/HeroicEric)
- [#3946](https://github.com/ember-cli/ember-cli/pull/3946) [ENHANCEMENT] Add host option to `ember test`. [@wangjohn](https://github.com/wangjohn)
- [#3909](https://github.com/ember-cli/ember-cli/pull/3909) test blueprints now use consistent, less-opinionated import style [@jayphelps](https://github.com/jayphelps)
- [#3891](https://github.com/ember-cli/ember-cli/pull/3891) Fixes problem in initializer tests generated in addons [@marcioj](https://github.com/marcioj)
- [#3889](https://github.com/ember-cli/ember-cli/pull/3889) Updating dev folder to help with debugging [@kellyselden](https://github.com/kellyselden)
- [#3916](https://github.com/ember-cli/ember-cli/pull/3916) Disable directory listings on development server [@joliss](https://github.com/joliss)
- [#3887](https://github.com/ember-cli/ember-cli/pull/3887) Bump ember router generator and allow index routes [@abuiles](https://github.com/abuiles)
- [#3915](https://github.com/ember-cli/ember-cli/pull/3915) Always create addon trees when developing an addon [@marcioj](https://github.com/marcioj)
- [#3901](https://github.com/ember-cli/ember-cli/pull/3901) bump blueprints to latest released ember [@stefanpenner](https://github.com/stefanpenner)
- [#3913](https://github.com/ember-cli/ember-cli/pull/3913) Remove podModulePrefix from app.js [@knownasilya](https://github.com/knownasilya)
- [#3945](https://github.com/ember-cli/ember-cli/pull/3945) [ENHANCEMENT] Friendly test names and descriptions [@eccegordo/feature](https://github.com/eccegordo/feature)
- [#3922](https://github.com/ember-cli/ember-cli/pull/3922) Export test output dir via ENV [@ef4](https://github.com/ef4)
- [#3919](https://github.com/ember-cli/ember-cli/pull/3919) Remove connect-restreamer. [@abuiles](https://github.com/abuiles)
- [#4021](https://github.com/ember-cli/ember-cli/pull/4021) Allow custom history location types. [@stefanpenner](https://github.com/stefanpenner)
- [#3950](https://github.com/ember-cli/ember-cli/pull/3950) allow override of os.EOL in tests [@stefanpenner](https://github.com/stefanpenner)
- [#3962](https://github.com/ember-cli/ember-cli/pull/3962) Disable any file watching done by testem [@johanneswuerbach](https://github.com/johanneswuerbach)
- [#3968](https://github.com/ember-cli/ember-cli/pull/3968) node-glob doesn’t work with windows shares… [@stefanpenner](https://github.com/stefanpenner)
- [#3951](https://github.com/ember-cli/ember-cli/pull/3951) [ENHANCEMENT] Add --directory flag to `ember new` [@HeroicEric](https://github.com/HeroicEric)
- [#3956](https://github.com/ember-cli/ember-cli/pull/3956) Bump ember-route-generator to match #3936. [@abuiles](https://github.com/abuiles)
- [#3954](https://github.com/ember-cli/ember-cli/pull/3954) [ENHANCEMENT] Generate component pods outside components folder [@trabus](https://github.com/trabus)
- [#3958](https://github.com/ember-cli/ember-cli/pull/3958) Fix indentation in `crossdomain.xml` [@arthurvr](https://github.com/arthurvr)
- [#3967](https://github.com/ember-cli/ember-cli/pull/3967) allow override of os.EOL in tests [@stefanpenner](https://github.com/stefanpenner)
- [#3959](https://github.com/ember-cli/ember-cli/pull/3959) Add `Disallow:` to robots.txt [@arthurvr](https://github.com/arthurvr)
- [#3966](https://github.com/ember-cli/ember-cli/pull/3966) increase timeouts, and use mocha’s inheriting config [@stefanpenner](https://github.com/stefanpenner)
- [#4039](https://github.com/ember-cli/ember-cli/pull/4039) Update ember-qunit to support glimmer [@knownasilya](https://github.com/knownasilya)
- [#4000](https://github.com/ember-cli/ember-cli/pull/4000) use `escape-string-regexp` module [@sindresorhus](https://github.com/sindresorhus)
- [#3975](https://github.com/ember-cli/ember-cli/pull/3975) included modules is no longer needed [@stefanpenner](https://github.com/stefanpenner)
- [#3982](https://github.com/ember-cli/ember-cli/pull/3982) Changed markdown-color blue to bright-blue to be the same on all platforms [@trabus](https://github.com/trabus)
- [#3984](https://github.com/ember-cli/ember-cli/pull/3984) Allow io.js-next in development in 'valid-platform-version' [@laiso](https://github.com/laiso)
- [#3974](https://github.com/ember-cli/ember-cli/pull/3974) Resolve sync [@stefanpenner](https://github.com/stefanpenner)
- [#3976](https://github.com/ember-cli/ember-cli/pull/3976) Relative require [@stefanpenner](https://github.com/stefanpenner)
- [#3996](https://github.com/ember-cli/ember-cli/pull/3996) Warns when npm or bower dependencies aren't installed [@marcioj](https://github.com/marcioj)
- [#4024](https://github.com/ember-cli/ember-cli/pull/4024) Appveyor: Use `run` instead of `run-script` [@knownasilya](https://github.com/knownasilya)
- [#4033](https://github.com/ember-cli/ember-cli/pull/4033) Bump ember-cli-dependency-checker to v0.1.0 [@quaertym](https://github.com/quaertym)
- [#4027](https://github.com/ember-cli/ember-cli/pull/4027) Re-order postBuild hook [@chadhietala](https://github.com/chadhietala)
- [#4008](https://github.com/ember-cli/ember-cli/pull/4008) Disable leek for `ember -v` [@twokul](https://github.com/twokul)
- [#4020](https://github.com/ember-cli/ember-cli/pull/4020) Allowed failures [@stefanpenner](https://github.com/stefanpenner)
- [#4007](https://github.com/ember-cli/ember-cli/pull/4007) Hide Python on appveyor so npm won't build native extentions [@raytiley](https://github.com/raytiley)
- [#4022](https://github.com/ember-cli/ember-cli/pull/4022) Run all tests again [@marcioj](https://github.com/marcioj)
- [#4032](https://github.com/ember-cli/ember-cli/pull/4032) Update ember-cli-qunit to v0.3.2 [@HeroicEric](https://github.com/HeroicEric)
- [#4037](https://github.com/ember-cli/ember-cli/pull/4037) Add ember-disable-proxy-controllers to app blueprint [@cibernox](https://github.com/cibernox)
- [#4046](https://github.com/ember-cli/ember-cli/pull/4046) Upgrade ember-cli-htmlbars to 0.7.6 [@teddyzeenny](https://github.com/teddyzeenny)
- [#4057](https://github.com/ember-cli/ember-cli/pull/4057) [INTERNAL] Fix tests to expect single line qunit import [@trabus](https://github.com/trabus)
- [#4058](https://github.com/ember-cli/ember-cli/pull/4058) Bump ember-cli-dependency-checker to v1.0.0 [@quaertym](https://github.com/quaertym)
- [#4059](https://github.com/ember-cli/ember-cli/pull/4059) Update Ember-data to beta 17 [@cibernox](https://github.com/cibernox)
- [#4065](https://github.com/ember-cli/ember-cli/pull/4065) Update to Ember 1.12.0. [@rwjblue](https://github.com/rwjblue)

Thank you to all who took the time to contribute!


### 0.2.3

The following changes are required if you are upgrading from the previous
version:

- Users
  + [`ember new` diff](https://github.com/kellyselden/ember-cli-output/commit/0aaabc98378600e116da0fcc5b75c1a8b00ce541)
  + Upgrade your project's ember-cli version - [docs](http://www.ember-cli.com/user-guide/#upgrading)
  + `ember install <addon-name>` now is the correct way to install an add-on (not `ember install:npm <addon-name>`)
  + babel has been upgraded to `5.0.0`, be sure any configuration to babel is updated accordingly
  + bundled ember is now 1.11.1
  + when existing test --server, tmp files should once again be correctly cleaned up.
- Addon Developers
  + [`ember addon` diff](https://github.com/kellyselden/ember-addon-output/commit/567f9f2db157ce835c116ffde1567cc8c709ae0c)
  + No changes required
- Core Contributors
  + No changes required
  + Code Climate was added: https://codeclimate.com/github/ember-cli/ember-cli,
    we have been making steady progress in improving our code quality and coverage.
    As new code enters the system, we should ensure we continue to improve.

#### Community Contributions

- [#3782](https://github.com/ember-cli/ember-cli/pull/3782) add OS X as a CI target for travis [@stefanpenner](https://github.com/stefanpenner)
- [#3711](https://github.com/ember-cli/ember-cli/pull/3711) adding changelog diffs [@kellyselden](https://github.com/kellyselden)
- [#3703](https://github.com/ember-cli/ember-cli/pull/3703) [ENHANCEMENT] Add testem --launch option to ember test command [@jrjohnson](https://github.com/jrjohnson)
- [#3598](https://github.com/ember-cli/ember-cli/pull/3598) [ENHANCEMENT] Replace install:addon with install, remove install:bower and install:npm [@DanielOchoa](https://github.com/DanielOchoa)
- [#3690](https://github.com/ember-cli/ember-cli/pull/3690) [ENHANCEMENT] Addon-import support for built-in blueprints [@trabus](https://github.com/trabus)
- [#3700](https://github.com/ember-cli/ember-cli/pull/3700) fixes #3613 - added path.normalize [@swelham](https://github.com/swelham)
- [#3412](https://github.com/ember-cli/ember-cli/pull/3412) Changes match application regex [@twokul](https://github.com/twokul)
- [#3789](https://github.com/ember-cli/ember-cli/pull/3789) Code Quality: ember-app.js F -> D [@kellyselden](https://github.com/kellyselden)
- [#3731](https://github.com/ember-cli/ember-cli/pull/3731) Promise cleanup [@stefanpenner](https://github.com/stefanpenner)
- [#3722](https://github.com/ember-cli/ember-cli/pull/3722) Updated included hook example [@RSSchermer](https://github.com/RSSchermer)
- [#3713](https://github.com/ember-cli/ember-cli/pull/3713) The unbundling [@stefanpenner](https://github.com/stefanpenner)
- [#3725](https://github.com/ember-cli/ember-cli/pull/3725) increase timeouts, and use mocha’s inheriting config strategy to prevent... [@stefanpenner](https://github.com/stefanpenner)
- [#3727](https://github.com/ember-cli/ember-cli/pull/3727) misc cleanup [@stefanpenner](https://github.com/stefanpenner)
- [#3794](https://github.com/ember-cli/ember-cli/pull/3794) BUGFIX fixes vars-on-top error in ESLint [@jonathanKingston](https://github.com/jonathanKingston)
- [#3759](https://github.com/ember-cli/ember-cli/pull/3759) Order bower dependencies alphabetically [@pmdarrow](https://github.com/pmdarrow)
- [#3736](https://github.com/ember-cli/ember-cli/pull/3736) [fixes #3732] configure YAM with the Project.root. [@stefanpenner](https://github.com/stefanpenner)
- [#3743](https://github.com/ember-cli/ember-cli/pull/3743) no longer bundle testem, allow it to drift along semver [@stefanpenner](https://github.com/stefanpenner)
- [#3756](https://github.com/ember-cli/ember-cli/pull/3756) adding a blueprint uninstall test [@kellyselden](https://github.com/kellyselden)
- [#3750](https://github.com/ember-cli/ember-cli/pull/3750) add developer requirements to CONTRIBUTING.md [@jakehow](https://github.com/jakehow)
- [#3748](https://github.com/ember-cli/ember-cli/pull/3748) Fix wording [@jbrown](https://github.com/jbrown)
- [#3740](https://github.com/ember-cli/ember-cli/pull/3740) Remove dead code [@IanVS](https://github.com/IanVS)
- [#3747](https://github.com/ember-cli/ember-cli/pull/3747) code quality refactor of blueprint model [@kellyselden](https://github.com/kellyselden)
- [#3755](https://github.com/ember-cli/ember-cli/pull/3755) return currentURL() rather than path, ref #3719 [@mariogintili](https://github.com/mariogintili)
- [#3800](https://github.com/ember-cli/ember-cli/pull/3800) [fixes #3799] fix jshint [@stefanpenner](https://github.com/stefanpenner)
- [#3780](https://github.com/ember-cli/ember-cli/pull/3780) Upgrade to npm 2.7.6 [@davewasmer](https://github.com/davewasmer)
- [#3775](https://github.com/ember-cli/ember-cli/pull/3775) Code quality blueprint duplicates [@kellyselden](https://github.com/kellyselden)
- [#3762](https://github.com/ember-cli/ember-cli/pull/3762) Improved serializer-test blueprint [@bmac](https://github.com/bmac)
- [#3778](https://github.com/ember-cli/ember-cli/pull/3778) bump to babel 5.0 [@stefanpenner](https://github.com/stefanpenner)
- [#3764](https://github.com/ember-cli/ember-cli/pull/3764) Version bump ember-load-initializers to handle instance initializers [@jasonmit](https://github.com/jasonmit)
- [#3769](https://github.com/ember-cli/ember-cli/pull/3769) Babel 5.0 now separates codeFrame from error.{message, stack} [@stefanpenner](https://github.com/stefanpenner)
- [#3804](https://github.com/ember-cli/ember-cli/pull/3804) increase some timeouts and prefer mocha’s inheriting timers [@stefanpenner](https://github.com/stefanpenner)
- [#3798](https://github.com/ember-cli/ember-cli/pull/3798) adding coverage badge to readme [@stefanpenner](https://github.com/stefanpenner)
- [#3781](https://github.com/ember-cli/ember-cli/pull/3781) bump to a non-vulnerable semver module [@stefanpenner](https://github.com/stefanpenner)
- [#3784](https://github.com/ember-cli/ember-cli/pull/3784) Add ember-try for addons. [@rwjblue](https://github.com/rwjblue)
- [#3795](https://github.com/ember-cli/ember-cli/pull/3795) Code Climate: adding test coverage [@kellyselden](https://github.com/kellyselden)
- [#3797](https://github.com/ember-cli/ember-cli/pull/3797) Update to ember-qunit 0.3.1. [@rwjblue](https://github.com/rwjblue)
- [#3801](https://github.com/ember-cli/ember-cli/pull/3801) moving coverage repo key to travis env variable [@kellyselden](https://github.com/kellyselden)
- [#3802](https://github.com/ember-cli/ember-cli/pull/3802) remove pre-mature process.exit when existing `ember test —server` [@stefanpenner](https://github.com/stefanpenner)
- [#3803](https://github.com/ember-cli/ember-cli/pull/3803) update testem [@stefanpenner](https://github.com/stefanpenner)
- [#3809](https://github.com/ember-cli/ember-cli/pull/3809) Fix url format of isGitRepo [@stefanpenner](https://github.com/stefanpenner)
- [#3812](https://github.com/ember-cli/ember-cli/pull/3812) update ember to 1.11.1 in the default blueprint [@stefanpenner](https://github.com/stefanpenner)

Thank you to all who took the time to contribute!

### 0.2.2

The following changes are required if you are upgrading from the previous
version:

- Users
  + [`ember new` diff](https://github.com/kellyselden/ember-cli-output/commit/1c47557e629d88ec399786bd3f06995a251e6f0f)
  + updated to ember 1.11.0
  + Upgrade your project's ember-cli version - [docs](http://www.ember-cli.com/user-guide/#upgrading)
  + `ember init` once again works inside an addon.
  + error live-reloading now actually works!
  + npm WARN for `makeError` and `tmpl` have been fixed
  + ember-qunit was updated from `0.2.8` -> `0.3.0`, `this.render()` in a test now no-longer returns a jQuery object.

- Addon Developers
  + [`ember addon` diff](https://github.com/kellyselden/ember-addon-output/commit/4bb6c82e5411560c6d21517755d6a2276bad9a39)
  + Addons now have `ember-disable-prototype-extensions` included by default,
    this ensures add-ons are written in a way that works regardless of the
    consumers prototype extension preference.

  + the following addon API's in has been deprecated:
    * `this.mergeTrees`       -> `require('mergeTrees');`
    * `this.Funnel`           -> `require('broccoli-funnel');`
    * `this.pickFiles`        -> `require('broccoli-funnel');`
    * `this.walkSync`         -> `require('walk-sync');`
    * `this.transpileModules` -> `require('broccoli-es6modules');`

  Rather then relying on them from ember-cli, add-ons should require them via NPM.

  + We now are using broccoli v0.15.3, which is a backwards compatible upgrade,
    but it does expose the new `rebuild` api, that will soon superseed the `read`
    api. TL;DR among other things, this paves the path to having a configurable
    tmp directory.

    We recommend broccoli-plugin authors to update as soon as they are able to.

    For more information checkout: [new rebuild api](https://github.com/broccolijs/broccoli/blob/master/docs/new-rebuild-api.md)

- Core Contributors

  + Keep being awesome!

#### Community Contributions

- [#3560](https://github.com/ember-cli/ember-cli/pull/3560) fixing the formatting from one line to two [@kellyselden](https://github.com/kellyselden)
- [#3622](https://github.com/ember-cli/ember-cli/pull/3622) [BUGFIX] Fix ember init inside an existing addon [@johanneswuerbach](https://github.com/johanneswuerbach)
- [#3469](https://github.com/ember-cli/ember-cli/pull/3469) [ENHANCEMENT] Update component-test test.js blueprint [@simonprev](https://github.com/simonprev)
- [#3565](https://github.com/ember-cli/ember-cli/pull/3565) [BUGFIX] temporarily disable podModulePrefix deprecation [@trabus](https://github.com/trabus)
- [#3601](https://github.com/ember-cli/ember-cli/pull/3601) Allow Node 0.13 in platform deprecation check. [@rwjblue](https://github.com/rwjblue)
- [#3585](https://github.com/ember-cli/ember-cli/pull/3585) [ENHANCEMENT] Add in-repo-addon generate and destroy support [@trabus](https://github.com/trabus)
- [#3674](https://github.com/ember-cli/ember-cli/pull/3674) Update nock dependency [@btecu](https://github.com/btecu)
- [#3636](https://github.com/ember-cli/ember-cli/pull/3636) [fixes #3618] we will add some acceptance tests in this area soon (rushi... [@stefanpenner](https://github.com/stefanpenner)
- [#3634](https://github.com/ember-cli/ember-cli/pull/3634) Resolves #3628 postprocessTree for styles with vendor + app [@jschilli](https://github.com/jschilli)
- [#3630](https://github.com/ember-cli/ember-cli/pull/3630) Fix minor typo's [@QuantumInformation](https://github.com/QuantumInformation)
- [#3631](https://github.com/ember-cli/ember-cli/pull/3631) [Documentation] adding new ember new and ember addon diffs [@kellyselden](https://github.com/kellyselden)
- [#3680](https://github.com/ember-cli/ember-cli/pull/3680) Updates [@stefanpenner](https://github.com/stefanpenner)
- [#3645](https://github.com/ember-cli/ember-cli/pull/3645) Add ember-disable-prototype-extensions to addons by default. [@rwjblue](https://github.com/rwjblue)
- [#3642](https://github.com/ember-cli/ember-cli/pull/3642) Check if style file with project name exists [@btecu](https://github.com/btecu)
- [#3639](https://github.com/ember-cli/ember-cli/pull/3639) Bump ember-data to beta-16.1 [@bmac](https://github.com/bmac)
- [#3682](https://github.com/ember-cli/ember-cli/pull/3682) strip ansi from babel errors for now. [@stefanpenner](https://github.com/stefanpenner)
- [#3655](https://github.com/ember-cli/ember-cli/pull/3655) Uses Ember.keys instead of Object.keys in reexport [@danmcclain](https://github.com/danmcclain)
- [#3646](https://github.com/ember-cli/ember-cli/pull/3646) Add `chai` as dependency. [@rwjblue](https://github.com/rwjblue)
- [#3647](https://github.com/ember-cli/ember-cli/pull/3647) add timeouts until we improve the mocha <-> custom runner timeout stuff [@stefanpenner](https://github.com/stefanpenner)
- [#3648](https://github.com/ember-cli/ember-cli/pull/3648) Update broccoli-sane-watcher. [@rwjblue](https://github.com/rwjblue)
- [#3691](https://github.com/ember-cli/ember-cli/pull/3691) Update Ember to 1.11.0. [@rwjblue](https://github.com/rwjblue)
- [#3675](https://github.com/ember-cli/ember-cli/pull/3675) Restore addon pick files [@stefanpenner](https://github.com/stefanpenner)
- [#3673](https://github.com/ember-cli/ember-cli/pull/3673) Update Broccoli to 0.15.3 [@joliss](https://github.com/joliss)
- [#3672](https://github.com/ember-cli/ember-cli/pull/3672) Use broccoli-funnel instead of broccoli-static-compiler [@joliss](https://github.com/joliss)
- [#3666](https://github.com/ember-cli/ember-cli/pull/3666) Tweaks [@stefanpenner](https://github.com/stefanpenner)
- [#3669](https://github.com/ember-cli/ember-cli/pull/3669) Update dependencies [@btecu](https://github.com/btecu)
- [#3677](https://github.com/ember-cli/ember-cli/pull/3677) Export return value from Router.map (closes #3676). [@abuiles](https://github.com/abuiles)
- [#3681](https://github.com/ember-cli/ember-cli/pull/3681) Deprecate funnel and pickfiles [@stefanpenner](https://github.com/stefanpenner)
- [#3692](https://github.com/ember-cli/ember-cli/pull/3692) Replace lodash-node with lodash [@btecu](https://github.com/btecu)
- [#3696](https://github.com/ember-cli/ember-cli/pull/3696) Update markdown-it and markdown-it-terminal [@stefanpenner](https://github.com/stefanpenner)
- [#3704](https://github.com/ember-cli/ember-cli/pull/3704) Live reload fix [@stefanpenner](https://github.com/stefanpenner)
- [#3705](https://github.com/ember-cli/ember-cli/pull/3705) Fix initial commit message [@xymbol](https://github.com/xymbol)

Thank you to all who took the time to contribute!

### 0.2.1

The following changes are required if you are upgrading from the previous
version:

- Users
  + [`ember new` diff](https://github.com/kellyselden/ember-cli-output/commit/e4d36aa2ce99ebb288cd596270e7b38da90f535e)
  + Upgrade your project's ember-cli version - [docs](http://www.ember-cli.com/user-guide/#upgrading)
  + Mostly just bug-fixes and "Nice things"
  + build errors now live-reload and correctly display build failure in the browser. [more-details](https://github.com/ember-cli/ember-cli/pull/3576)
- Addon Developers
  + [`ember addon` diff](https://github.com/kellyselden/ember-addon-output/commit/5d87ed789651b1fbecf9a30d7b82eb86e0629bd2)
  + UI is now provided to the AddonDiscovery
  + ember-cli-babel is now included in the default blueprint, this is still optional but enabled by default

#### Community Contributions

- [#3555](https://github.com/ember-cli/ember-cli/pull/3555) [BUGFIX] Generate mixin in addon/mixins when inside an addon project [@trabus](https://github.com/trabus)
- [#3476](https://github.com/ember-cli/ember-cli/pull/3476) Removes initializer mention from service generator help text [@corpulentcoffee](https://github.com/corpulentcoffee)
- [#3433](https://github.com/ember-cli/ember-cli/pull/3433) [ENHANCEMENT] Prevent addon generation in existing ember-cli project [@cbrock](https://github.com/cbrock)
- [#3463](https://github.com/ember-cli/ember-cli/pull/3463) disable visual progress effect in dumb terminals [@jesse-black](https://github.com/jesse-black)
- [#3440](https://github.com/ember-cli/ember-cli/pull/3440) Enforcing newlines in template files results in unwanted Nodes [@jclem](https://github.com/jclem)
- [#3484](https://github.com/ember-cli/ember-cli/pull/3484) Component blueprint only import layout when generated inside addon [@trabus](https://github.com/trabus)
- [#3505](https://github.com/ember-cli/ember-cli/pull/3505) Update testem to 0.7.5 [@johanneswuerbach](https://github.com/johanneswuerbach)
- [#3481](https://github.com/ember-cli/ember-cli/pull/3481) BUGFIX Fixes #3472 Check for 'usePods' instead of 'pod'. [@jankrueger](https://github.com/jankrueger)
- [#3493](https://github.com/ember-cli/ember-cli/pull/3493) Fix helper test failing by default [@kimroen](https://github.com/kimroen)
- [#3488](https://github.com/ember-cli/ember-cli/pull/3488) ENHANCEMENT: update-checker.js should use environment http_proxy if detected [@xomaczar](https://github.com/xomaczar)
- [#3483](https://github.com/ember-cli/ember-cli/pull/3483) Ensure that addons pass the `ui` into their AddonDiscovery. [@rwjblue](https://github.com/rwjblue)
- [#3501](https://github.com/ember-cli/ember-cli/pull/3501) [Enhancement] Architecture Diagram [@visheshjoshi](https://github.com/visheshjoshi)
- [#3562](https://github.com/ember-cli/ember-cli/pull/3562) dist can be watched, it really is just tmp that matters. This prevents p... [@stefanpenner](https://github.com/stefanpenner)
- [#3540](https://github.com/ember-cli/ember-cli/pull/3540) [fixes #3520, #3174] disable ES3SafeFilter if babel is present, as babel... [@stefanpenner](https://github.com/stefanpenner)
- [#3508](https://github.com/ember-cli/ember-cli/pull/3508) Update ember-cli-app-version [@btecu](https://github.com/btecu)
- [#3518](https://github.com/ember-cli/ember-cli/pull/3518) [BUGFIX] Add missing bind when server already in use [@bdvholmes](https://github.com/bdvholmes)
- [#3539](https://github.com/ember-cli/ember-cli/pull/3539) add tmp dir to npmignore [@ahmadsoe](https://github.com/ahmadsoe)
- [#3515](https://github.com/ember-cli/ember-cli/pull/3515) [BUGFIX] Fixes nested component generation in addons with correct relative path for template import [@trabus](https://github.com/trabus)
- [#3517](https://github.com/ember-cli/ember-cli/pull/3517) Use node 0.12 on Windows CI [@johanneswuerbach](https://github.com/johanneswuerbach)
- [#3535](https://github.com/ember-cli/ember-cli/pull/3535) Update ADDON_HOOKS.md [@ahmadsoe](https://github.com/ahmadsoe)
- [#3533](https://github.com/ember-cli/ember-cli/pull/3533) [BUGFIX] Replace marked with markdown-it [@trabus](https://github.com/trabus)
- [#3583](https://github.com/ember-cli/ember-cli/pull/3583) Updated license copyright date range [@jayphelps](https://github.com/jayphelps)
- [#3546](https://github.com/ember-cli/ember-cli/pull/3546) [ENHANCEMENT] Add podModulePrefix deprecation for generate and destroy commands [@trabus](https://github.com/trabus)
- [#3544](https://github.com/ember-cli/ember-cli/pull/3544) Add links with watchman info to cli output [@felixbuenemann](https://github.com/felixbuenemann)
- [#3545](https://github.com/ember-cli/ember-cli/pull/3545) [BUGFIX] Ensure `package.json` `main` entry point is used for addon lookup. [@rwjblue](https://github.com/rwjblue)
- [#3541](https://github.com/ember-cli/ember-cli/pull/3541) [fixes #3520, #3174] bump es3-safe-recast [@stefanpenner](https://github.com/stefanpenner)
- [#3594](https://github.com/ember-cli/ember-cli/pull/3594) fix broken link [@kellyselden](https://github.com/kellyselden)
- [#3564](https://github.com/ember-cli/ember-cli/pull/3564) Added babel to addons package.json dependencies by default [@jayphelps](https://github.com/jayphelps)
- [#3559](https://github.com/ember-cli/ember-cli/pull/3559) [Documentation] add ref to ember-cli-output and ember-addon-output [@kellyselden](https://github.com/kellyselden)
- [#3571](https://github.com/ember-cli/ember-cli/pull/3571) [BREAKING ENHANCEMENT] Update ember-cli-content-security-policy to v0.4.0 [@sir-dunxalot/enhancement](https://github.com/sir-dunxalot/enhancement)
- [#3572](https://github.com/ember-cli/ember-cli/pull/3572) Specify node version (0.12) for CI [@quaertym](https://github.com/quaertym)
- [#3576](https://github.com/ember-cli/ember-cli/pull/3576) ensure a build-failure is “live-reloaded” to the user. [@stefanpenner](https://github.com/stefanpenner)
- [#3578](https://github.com/ember-cli/ember-cli/pull/3578) Teaches updateChecker about dev builds [@twokul](https://github.com/twokul)
- [#3579](https://github.com/ember-cli/ember-cli/pull/3579) allow for ./server to export express app [@calvinmetcalf](https://github.com/calvinmetcalf)
- [#3581](https://github.com/ember-cli/ember-cli/pull/3581) Resolves #3534 - addon postprocessTrees for styles [@jschilli](https://github.com/jschilli)
- [#3593](https://github.com/ember-cli/ember-cli/pull/3593) add command uninstall:npm [@kellyselden](https://github.com/kellyselden)
- [#3604](https://github.com/ember-cli/ember-cli/pull/3604) Testem update [@johanneswuerbach](https://github.com/johanneswuerbach)
- [#3611](https://github.com/ember-cli/ember-cli/pull/3611) Bump ember-data to beta-16 [@bmac](https://github.com/bmac)

Thank you to all who took the time to contribute!

### 0.2.0

#### Addon Formatting

Support for addon's without an entry point script (either `index.js` by default or the script specified by ember-addon main in the addon's `package.json`
has been removed. An addon must have at least the following:

```javascript
module.exports = {
  name: "addons-name-here"
};
```

This should *not* pose a problem for the vast majority of addons.

#### Addon Nesting

This release updates the way that addons can be nested, and contains some breaking changes in non-default addon configurations.

Prior versions of Ember CLI maintained a flat addon structure, so that all addons (of any depth) would be added to the consuming
application. This has led to many issues, like the inability to use preprocessors (i.e. ember-cli-htmlbars, ember-cli-sass, etc)
in nested addons.

For the majority of apps, the update from 0.1.15 to 0.2.0 is non-breaking and should not cause significant concern.

For addon creators, make sure to update to use the `setupPreprocessorRegistry` hook (documented [here](https://github.com/ember-cli/ember-cli/blob/master/ADDON_HOOKS.md))
if you need to add a preprocessor to the registry.  You can review the update process in
[ember-cli-htmlbars#38](https://github.com/ember-cli/ember-cli-htmlbars/pull/38) or [ember-cli-coffeescript#60](https://github.com/kimroen/ember-cli-coffeescript/pull/60)
which show how to maintain support for both 0.1.x and 0.2.0 in an addon.

The following changes are required if you are upgrading from the previous version:

- Users
  + [`ember new` diff](https://github.com/kellyselden/ember-cli-output/commit/d3080cd44b2b62cef45e7f723c18c862b7789f9d)
  + Upgrade your project's ember-cli version - [docs](http://www.ember-cli.com/user-guide/#upgrading)
  + The 6to5 project has been renamed to Babel.  See [the blog post](http://babeljs.io/blog/2015/02/15/not-born-to-die/) for more details.
  + The default blueprint has been updated to work with Ember 1.10 by default.
  + Update the following packages in your `package.json`:
    * Remove `broccoli-ember-hbs-template-compiler`. Uninstall with `npm uninstall --save-dev broccoli-ember-hbs-template-compiler`.
    * Remove `ember-cli-6to5`. Uninstall with `npm uninstall --save-dev ember-cli-6to5`.
    * Add `ember-cli-babel`. Install with `npm install --save-dev ember-cli-babel`.
    * Add `ember-cli-htmlbars`. Install with `npm install --save-dev ember-cli-htmlbars`.
    * Updated `ember-cli-qunit` to 0.3.9.  Install with `npm install --save-dev ember-cli-qunit@0.3.9`.
    * Updated `ember-data` to 1.0.0-beta.15. Install with `npm install --save-dev ember-data@1.0.0-beta.15`.
    * Updated `ember-cli-dependency-checker` to 0.0.8. Install with `npm install --save-dev ember-cli-dependency-checker@0.0.8`.
    * Updated `ember-cli-app-version` to 0.3.2. Install with `npm install --save-dev ember-cli-app-version@0.3.2`.
  + Update the following packages in your `bower.json`:
    * Removed `handlebars`. Uninstall with `bower uninstall --save handlebars`.
    * Updated `ember` to 1.10.0. Install with `bower install --save ember#1.10.0`.
    * Updated `ember-data` to 1.0.0-beta.15. Install with `bower install --save ember-data#1.0.0-beta.15`.
    * Updated `ember-cli-test-loader` to 0.1.3.  Install with `bower install --save ember-cli-test-loader#0.1.3`.
    * Updated `ember-resolver` to 0.1.12. Install with `bower install --save ember-resolver`.
    * Updated `loader.js` to 3.2.0.
- Addon Developers
  + [`ember addon` diff](https://github.com/kellyselden/ember-addon-output/commit/c78af207563593e5cb33a9a79d5d249cb134c1f9)
  + Usage of the `included` hook to add items to the `registry` will need to be refactored to use the newly added `setupPreprocessorRegistry` hook instead.
- Core Contributors
  + No changes required

#### Community Contributions

- [#3246](https://github.com/ember-cli/ember-cli/pull/3246) [ENHANCEMENT] Update the service blueprint to use `Ember.Service` (and remove usage of an initializer). [@ohcibi](https://github.com/ohcibi)
- [#3054](https://github.com/ember-cli/ember-cli/pull/3054) [ENHANCEMENT] Updated `loader.js` to the latest version. [@stefanpenner](https://github.com/stefanpenner)
- [#3216](https://github.com/ember-cli/ember-cli/pull/3216) [BUGFIX] Do not default to development asset [@martndemus](https://github.com/martndemus)
- [#3237](https://github.com/ember-cli/ember-cli/pull/3237) [BUGFIX] Blueprint templates with undefined variables should fallback to raw text [@davewasmer](https://github.com/davewasmer)
- [#3288](https://github.com/ember-cli/ember-cli/pull/3288) [ENHANCEMENT] Override default port with `PORT` env var [@knownasilya](https://github.com/knownasilya)
- [#3158](https://github.com/ember-cli/ember-cli/pull/3158) [INTERNAL] add more steps to release.md [@raytiley](https://github.com/raytiley)
- [#3160](https://github.com/ember-cli/ember-cli/pull/3160) [BUGFIX] Don't override the request's path [@dmathieu](https://github.com/dmathieu)
- [#3367](https://github.com/ember-cli/ember-cli/pull/3367) [ENHANCEMENT] Prevent spotlight from indexing `tmp`. [@stefanpenner](https://github.com/stefanpenner)
- [#3336](https://github.com/ember-cli/ember-cli/pull/3336) [ENHANCEMENT] Nested addons should be overrideable from parent. [@rwjblue](https://github.com/rwjblue)
- [#3335](https://github.com/ember-cli/ember-cli/pull/3335) [ENHANCEMENT] Allow shared nested addons to be properly discovered. [@rwjblue](https://github.com/rwjblue)
- [#3312](https://github.com/ember-cli/ember-cli/pull/3312) [BUGFIX] ADDON_HOOKS.md - fixed broken and outdated links [@leandrocp](https://github.com/leandrocp)
- [#3326](https://github.com/ember-cli/ember-cli/pull/3326) [ENHANCEMENT] Print deprecation warning for Node 0.10. [@rwjblue](https://github.com/rwjblue)
- [#3317](https://github.com/ember-cli/ember-cli/pull/3317) [ENHANCEMENT] Remove express & glob from default app package.json. [@rwjblue](https://github.com/rwjblue)
- [#3383](https://github.com/ember-cli/ember-cli/pull/3383) [ENHANCEMENT] Use Ember.HTMLBars by default in new helpers. [@maxwerr](https://github.com/maxwerr)
- [#3355](https://github.com/ember-cli/ember-cli/pull/3355) [ENHANCEMENT] Add `ui` to `Project` and `Addon` instances. [@rwjblue](https://github.com/rwjblue)
- [#3341](https://github.com/ember-cli/ember-cli/pull/3341) [ENHANCEMENT] Improve blueprint help output method (markdown support) [@trabus](https://github.com/trabus)
- [#3349](https://github.com/ember-cli/ember-cli/pull/3349) [BUGFIX] Allow deprecated lookup of invalid packages. [@rwjblue](https://github.com/rwjblue)
- [#3353](https://github.com/ember-cli/ember-cli/pull/3353) [BUGFIX] Allow generated acceptance tests to be in directories [@koriroys](https://github.com/koriroys)
- [#3345](https://github.com/ember-cli/ember-cli/pull/3345) [ENHANCEMENT] Check if blueprint exists before printing help [@trabus](https://github.com/trabus)
- [#3338](https://github.com/ember-cli/ember-cli/pull/3338) [ENHANCEMENT] Update resolver to 0.1.12 [@teddyzeenny](https://github.com/teddyzeenny)
- [#3401](https://github.com/ember-cli/ember-cli/pull/3401) [BUGFIX] Fixes accidental global Error object pollution. [@stefanpenner](https://github.com/stefanpenner)
- [#3363](https://github.com/ember-cli/ember-cli/pull/3363) [ENHANCEMENT] Bump ember-cli-dependency-checker to v0.0.8 [@quaertym](https://github.com/quaertym)
- [#3358](https://github.com/ember-cli/ember-cli/pull/3358) [ENHANCEMENT] CI=true puts the UI into `silent` writeLevel [@stefanpenner](https://github.com/stefanpenner)
- [#3361](https://github.com/ember-cli/ember-cli/pull/3361) [ENHANCEMENT] Update `loader.js` to 3.0.2 [@stefanpenner](https://github.com/stefanpenner)
- [#3356](https://github.com/ember-cli/ember-cli/pull/3356) [ENHANCEMENT] Generate blueprint inside addon generates into addon folder with re-export in app folder [@trabus](https://github.com/trabus)
- [#3378](https://github.com/ember-cli/ember-cli/pull/3378) [ENHANCEMENT] Only generate JSHint warnings for the addon being developed [@teddyzeenny](https://github.com/teddyzeenny)
- [#3375](https://github.com/ember-cli/ember-cli/pull/3375) [ENHANCEMENT] JSHint addon before preprocessing the JS [@teddyzeenny](https://github.com/teddyzeenny)
- [#3373](https://github.com/ember-cli/ember-cli/pull/3373) [ENHANCEMENT] Provide a helpful error when an addon does not have a template compiler. [@rwjblue](https://github.com/rwjblue)
- [#3386](https://github.com/ember-cli/ember-cli/pull/3386) [ENHANCEMENT] Display localhost in console instead of 0.0.0.0. [@rwjblue](https://github.com/rwjblue)
- [#3391](https://github.com/ember-cli/ember-cli/pull/3391) [ENHANCEMENT] Update ember-cli-qunit to 0.3.9. [@rwjblue](https://github.com/rwjblue)
- [#3410](https://github.com/ember-cli/ember-cli/pull/3410) [ENHANCEMENT] Use correct bound helper params for HTMLBars [@jbrown](https://github.com/jbrown)
- [#3428](https://github.com/ember-cli/ember-cli/pull/3428) [BUGFIX] Lock glob and rimraf to prevent EEXISTS errors. [@raytiley](https://github.com/raytiley)
- [#3435](https://github.com/ember-cli/ember-cli/pull/3435) [ENHANCEMENT] Update bundled npm [@stefanpenner](https://github.com/stefanpenner)
- [#3436](https://github.com/ember-cli/ember-cli/pull/3436) [ENHANCEMENT] Update Broccoli to 0.13.6 to provide errors on new API. [@rwjblue](https://github.com/rwjblue)
- [#3438](https://github.com/ember-cli/ember-cli/pull/3438) [BUGFIX] Ensure nested addon registry matches addon order. [@rwjblue](https://github.com/rwjblue)
- [#3456](https://github.com/ember-cli/ember-cli/pull/3456) [BUGFIX] Update ember-cli-app-version to 0.3.2 [@taras](https://github.com/taras)

Thank you to all who took the time to contribute!

### 0.2.0-beta.1

This release updates the way that addons can be nested, and contains some breaking changes in non-default addon configurations.

Prior versions of Ember CLI maintained a flat addon structure, so that all addons (of any depth) would be added to the consuming
application. This has led to many issues, like the inability to use preprocessors (i.e. ember-cli-htmlbars, ember-cli-sass, etc)
in nested addons.

For the majority of apps, the update from 0.1.15 to 0.2.0 is non-breaking and should not cause significant concern.

For addon creators, make sure to update to use the `setupPreprocessorRegistry` hook (documented [here](https://github.com/ember-cli/ember-cli/blob/master/ADDON_HOOKS.md))
if you need to add a preprocessor to the registry.  You can review the update process in
[ember-cli-htmlbars#38](https://github.com/ember-cli/ember-cli-htmlbars/pull/38) or [ember-cli-coffeescript#60](https://github.com/kimroen/ember-cli-coffeescript/pull/60)
which show how to maintain support for both 0.1.x and 0.2.0 in an addon.

The following changes are required if you are upgrading from the previous version:

- Users
  + [`ember new` diff](https://github.com/kellyselden/ember-cli-output/commit/d717009d95d75cee1800e8ba9f52c24d117acb12)
  + Upgrade your project's ember-cli version - [docs](http://www.ember-cli.com/user-guide/#upgrading)
  + The 6to5 project has been renamed to Babel.  See [the blog post](http://babeljs.io/blog/2015/02/15/not-born-to-die/) for more details.
  + The default blueprint has been updated to work with Ember 1.10 by default.
  + Update the following packages in your `package.json`:
    * Remove `broccoli-ember-hbs-template-compiler`. Uninstall with `npm uninstall --save-dev broccoli-ember-hbs-template-compiler`.
    * Remove `ember-cli-6to5`. Uninstall with `npm uninstall --save-dev ember-cli-6to5`.
    * Add `ember-cli-babel`. Install with `npm install --save-dev ember-cli-babel`.
    * Add `ember-cli-htmlbars`. Install with `npm install --save-dev ember-cli-htmlbars`.
    * Updated `ember-cli-qunit` to 0.3.8.  Install with `npm install --save-dev ember-cli-qunit@0.3.8`.
    * Updated `ember-data` to 1.0.0-beta.15. Install with `npm install --save-dev ember-data@1.0.0-beta.15`.
  + Update the following packages in your `bower.json`:
    * Removed `handlebars`. Uninstall with `bower uninstall --save handlebars`.
    * Updated `ember` to 1.10.0. Install with `bower install --save ember#1.10.0`.
    * Updated `ember-data` to 1.0.0-beta.15. Install with `bower install --save ember-data#1.0.0-beta.15`.
    * Updated `ember-cli-test-loader` to 0.1.3.  Install with `bower install --save ember-cli-test-loader#0.1.3`.
- Addon Developers
  + [`ember addon` diff](https://github.com/kellyselden/ember-addon-output/commit/c7e8a2a97ab5d508ea3f586bc97fedffa5763a75)
  + Usage of the `included` hook to add items to the `registry` will need to be refactored to use the newly added `setupPreprocessorRegistry` hook instead.
- Core Contributors
  + No changes required

#### Community Contributions

- [#3166](https://github.com/ember-cli/ember-cli/pull/3166) [BREAKING ENHANCEMENT] Addon discovery and isolation [@lukemelia](https://github.com/lukemelia) / [@chrislopresto](https://github.com/chrislopresto) / [@rwjblue](https://github.com/rwjblue)
- [#3285](https://github.com/ember-cli/ember-cli/pull/3285) [INTERNAL ENHANCEMENT] Update to Testem 0.7 [@johanneswuerbach](https://github.com/johanneswuerbach)
- [#3295](https://github.com/ember-cli/ember-cli/pull/3295) [ENHANCEMENT] Update ember-data to 1.0.0-beta.15 [@bmac](https://github.com/bmac)
- [#3297](https://github.com/ember-cli/ember-cli/pull/3297) [ENHANCEMENT] Use ember-cli-babel instead of ember-cli-6to5 [@fivetanley](https://github.com/fivetanley)
- [#3298](https://github.com/ember-cli/ember-cli/pull/3298) [BUGFIX] Update ember-cli-qunit to v0.3.8. [@rwjblue](https://github.com/rwjblue)
- [#3301](https://github.com/ember-cli/ember-cli/pull/3301) [BUGFIX] Only add Handlebars to `vendor.js` if present in `bower.json`. [@rwjblue](https://github.com/rwjblue)

Thank you to all who took the time to contribute!

### 0.1.15

This release fixes a regression in 0.1.13. See [#3271](https://github.com/ember-cli/ember-cli/issues/3271) for details.

The following changes are required if you are upgrading from the previous version:

- Users
  + [`ember new` diff](https://github.com/kellyselden/ember-cli-output/commit/1f0bc0414b460da9c924e7e750d7bc5639b62f42)
  + Upgrade your project's ember-cli version - [docs](http://www.ember-cli.com/user-guide/#upgrading)
- Addon Developers
  + [`ember addon` diff](https://github.com/kellyselden/ember-addon-output/commit/0ba9b5980684c48c063a3d320914db90498f684a)
  + No changes required
- Core Contributors
  + No changes required

- [#3271](https://github.com/ember-cli/ember-cli/pull/3271) [HOTFIX] Update broccoli-funnel to v0.2.2.  [@rwjblue](https://github.com/rwjblue)


### 0.1.14

This release fixes a regression in 0.1.13. See [#3267](https://github.com/ember-cli/ember-cli/issues/3267) for details.

The following changes are required if you are upgrading from the previous version:

- Users
  + [`ember new` diff](https://github.com/kellyselden/ember-cli-output/commit/1f5c865c5979d35f1aac72d00f97bda86864667f)
  + Upgrade your project's ember-cli version - [docs](http://www.ember-cli.com/user-guide/#upgrading)
- Addon Developers
  + [`ember addon` diff](https://github.com/kellyselden/ember-addon-output/commit/cec7a598854db05f9190ebb6ef68d570592b8e6e)
  + No changes required
- Core Contributors
  + No changes required

- [#3267](https://github.com/ember-cli/ember-cli/pull/3267) [HOTFIX] Ensure reexports work to not cause an error on rebuild. [@rwjblue](https://github.com/rwjblue)


### 0.1.13

The following changes are required if you are upgrading from the previous version:

- Users
  + [`ember new` diff](https://github.com/kellyselden/ember-cli-output/commit/15a28d18f13b68d32b635535b168d1aa7c3f6d4d)
  + Upgrade your project's ember-cli version - [docs](http://www.ember-cli.com/user-guide/#upgrading)
  + Update the following packages in your `package.json`:
    * Updated `ember-cli-qunit` to 0.3.7.  Install with `npm install --save-dev ember-cli-qunit@0.3.7`.
    * Updated `ember-data` to 1.0.0-beta.14.1. Install with `npm install --save-dev ember-data@1.0.0-beta.14.1`.
    * Updated `ember-export-application-global` to 1.0.2. Install with `npm install --save-dev ember-export-application-global@^1.0.2`.
  + Update the following packages in your `bower.json`:
    * Updated `ember-data` to 1.0.0-beta.14.1. Install with `bower install --save ember-data#1.0.0-beta.14.1`.
    * Updated `ember-cli-test-loader` to 0.1.1.  Install with `bower install --save ember-cli-test-loader#0.1.1`.
    * Updated `ember-qunit` to 0.2.8. Install with `bower install --save ember-qunit#0.2.8`. Please review [Ember QUnit 0.2.x](http://reefpoints.dockyard.com/2015/02/06/ember-qunit-0-2.html) for background and impact.
    * Updated `ember-qunit-notifications` to 0.0.7. Install with `bower install --save ember-qunit-notifications#0.0.7`.
- Addon Developers
  + [`ember addon` diff](https://github.com/kellyselden/ember-addon-output/commit/8c1a672e0ccf0fe3c8f709191ff130cd20abb03e)
  + No changes required
- Core Contributors
  + No changes required

#### Community Contributions

- [#3218](https://github.com/ember-cli/ember-cli/pull/3218) [ENHANCEMENT] Add JS context {{content-for}} hooks. This allows addons to inject things into `vendor.js`/`my-app-name.js` without violating CSP or having to do crazy hacks. [@rwjblue](https://github.com/rwjblue)
- [#3156](https://github.com/ember-cli/ember-cli/pull/3156) [BUGFIX] Serve static files from `/test` if they exist. [@trek](https://github.com/trek)
- [#3155](https://github.com/ember-cli/ember-cli/pull/3155) [BUGFIX] Guard against rawArgs being `undefined` [@chadhietala](https://github.com/chadhietala)
- [#3183](https://github.com/ember-cli/ember-cli/pull/3183) [BUGFIX] Use recent Esperanto update to allow ES3 safe output. [@rwjblue](https://github.com/rwjblue)
- [#3170](https://github.com/ember-cli/ember-cli/pull/3170) / [#3184](https://github.com/ember-cli/ember-cli/pull/3184) [#3255](https://github.com/ember-cli/ember-cli/pull/3255) [ENHANCEMENT] Update ember-qunit to 0.2.8. [@rwjblue](https://github.com/rwjblue) / [@jbrown](https://github.com/jbrown)
- [#3165](https://github.com/ember-cli/ember-cli/pull/3165) [BUGFIX] Fix `npm install --save-dev` ordering of default `package.json`. [@kellyselden](https://github.com/kellyselden)
- [#3164](https://github.com/ember-cli/ember-cli/pull/3164) [ENHANCEMENT] Enable asynchronous `Addon.prototype.serverMiddleware` hooks by returning a promise from the hook. [@taras](https://github.com/taras)
- [#3182](https://github.com/ember-cli/ember-cli/pull/3182) [INTERNAL ENHANCEMENT] Update `ember-router-generator` to ensure routes are injected into `router.js` with single quotes. [@abuiles](https://github.com/abuiles)
- [#3232](https://github.com/ember-cli/ember-cli/pull/3232) / [#3212](https://github.com/ember-cli/ember-cli/pull/3212) / [#3243](https://github.com/ember-cli/ember-cli/pull/3243) [INTERNAL ENHANCEMENT] Update testem to 0.6.39. [@joostdevries](https://github.com/joostdevries) / [@johanneswuerbach](https://github.com/johanneswuerbach)
- [#3203](https://github.com/ember-cli/ember-cli/pull/3203) / [#3252](https://github.com/ember-cli/ember-cli/pull/3252) [INTERNAL ENHANCEMENT] Bump broccoli-es6modules to v0.5.0. [@rwjblue](https://github.com/rwjblue)
- [#3197](https://github.com/ember-cli/ember-cli/pull/3197) [ENHANCEMENT] Update test blueprints to use [QUnit 2.0 compatible](http://qunitjs.com/upgrade-guide-2.x/) output. [@rwjblue](https://github.com/rwjblue)
- [#3199](https://github.com/ember-cli/ember-cli/pull/3199) [ENHANCEMENT] Provide locals to `Blueprint.prototype.beforeInstall`/`Blueprint.prototype.beforeUninstall` hooks. [@mattmarcum](https://github.com/mattmarcum)
- [#3188](https://github.com/ember-cli/ember-cli/pull/3188) [ENHANCEMENT] Update Ember Data version to 1.0.0-beta.14.1. [@abuiles](https://github.com/abuiles)
- [#3245](https://github.com/ember-cli/ember-cli/pull/3245) [ENHANCEMENT] Update ember-cli-qunit to v0.3.7. [@rwjblue](https://github.com/rwjblue)
- [#3231](https://github.com/ember-cli/ember-cli/pull/3231) [INTERNAL ENHANCEMENT] Remove extra Addon build steps. [@rwjblue](https://github.com/rwjblue)
- [#3236](https://github.com/ember-cli/ember-cli/pull/3236) [INTERNAL ENHANCEMENT] Remove module transpilation from Addon model. [@rwjblue](https://github.com/rwjblue)
- [#3242](https://github.com/ember-cli/ember-cli/pull/3242) [DOCS] Add `isDevelopingAddon` to `ADDON_HOOKS.md`. [@matthiasleitner](https://github.com/matthiasleitner)
- [#3244](https://github.com/ember-cli/ember-cli/pull/3244) [BUGFIX] Ensure that Blueprints are returned in a consistent order when looking them up. [@nathanpalmer](https://github.com/nathanpalmer)
- [#3251](https://github.com/ember-cli/ember-cli/pull/3251) Update ember-export-application-global to v1.0.2. [@rwjblue](https://github.com/rwjblue)
- [#3167](https://github.com/ember-cli/ember-cli/pull/3167) [ENHANCEMENT]`usePodsByDefault` in app config deprecated in favor of `usePods` in .ember-cli [@trabus](https://github.com/trabus)
- [#3260](https://github.com/ember-cli/ember-cli/pull/3260) [BUGFIX] Ensure newly generated project has an `app/styles/app.css` file (prevents a 404 on a newly generated project). [@rwjblue](https://github.com/rwjblue)

Thank you to all who took the time to contribute!

### 0.1.12

The following changes are required if you are upgrading from the previous version:

- Users
  + [`ember new` diff](https://github.com/kellyselden/ember-cli-output/commit/a9bbe9c3cebc9768bf3e239ae8b2e5b5387335bf)
  + Upgrade your project's ember-cli version - [docs](http://www.ember-cli.com/user-guide/#upgrading)
  + `package.json` changes:
    + Update `ember-cli-qunit` to 0.3.1.
    + Update `ember-cli-app-version` to 0.3.1.
- Addon Developers
  + [`ember addon` diff](https://github.com/kellyselden/ember-addon-output/commit/fb04f954b345a1f5a1d891b64d7a596b2f566a57)
  + No changes required
- Core Contributors
  + No changes required

#### Community Contributions

- [#3118](https://github.com/ember-cli/ember-cli/pull/3118) [BUGFIX] Fix conflicting aliases. The `serve` command `host` alias is now `H` [@taddeimania](https://github.com/taddeimania)
- [#3130](https://github.com/ember-cli/ember-cli/pull/3130) [ENHANCEMENT] Tomster looks fabulous without breaking `ember new`[@johnnyshields](https://github.com/johnnyshields)
- [#3132](https://github.com/ember-cli/ember-cli/pull/3132) [BUGFIX] Update ember-cli-qunit to v0.3.1. Fixes `tests/.jshintrc` being used instead of app `.jshintrc`. [@rwjblue](https://github.com/rwjblue)
- [#3133](https://github.com/ember-cli/ember-cli/pull/3133) [BUGFIX] Fix analytics being disabled by default. Users can opt out of anylytics with `--disable-analytics` [@stefanpenner](https://github.com/stefanpenner)
- [#3153](https://github.com/ember-cli/ember-cli/pull/3153) [ENHANCEMENT]  Remove deafult css from `app/styles/app.css` [@mattjmorrison](https://github.com/mattjmorrison)
- [#3132](https://github.com/ember-cli/ember-cli/pull/3157) [BUGFIX] Ensure `ember test --environment=production` runs JSHint. [@rwjblue](https://github.com/rwjblue)

Thank you to all who took the time to contribute!


### 0.1.11

* [`ember new` diff](https://github.com/kellyselden/ember-cli-output/commit/1f0fe5089efd1a28be810f261d6cd17a342fce7b)
* [`ember addon` diff](https://github.com/kellyselden/ember-addon-output/commit/52c9eca14e3d498786fc93a17e08a92688cd43a5)
* [#3126](https://github.com/ember-cli/ember-cli/pull/3126) hot-fix tomster ` -> ., prevents breaking the initial git commit

### 0.1.10

The following changes are required if you are upgrading from the previous version:

- Users
  + [`ember new` diff](https://github.com/kellyselden/ember-cli-output/commit/bc9e076e0bb2c00f183e479bf025cdce84eeca1a)
  + Upgrade your project's ember-cli version - [docs](http://www.ember-cli.com/user-guide/#upgrading)
  + `package.json` changes:
    + Add `ember-cli-app-version` at 0.3.0.
    + Add `ember-cli-uglify` at 1.0.1.
    + Update `ember-cli-qunit` to 0.3.0.
    + Update `ember-cli-6to5` to 3.0.0.
  + `bower.json` changes:
    + Update `ember-cli-test-loader` to 0.1.0.
- Addon Developers
  + [`ember addon` diff](https://github.com/kellyselden/ember-addon-output/commit/88b8bf1f22e47298cd79b91bf2ccc6e054d5354b)
  + No changes required
- Core Contributors
  + No changes required

#### Community Contributions

- [#2970](https://github.com/ember-cli/ember-cli/pull/2970) [ENHANCEMENT] - Added ember-cli-app-version to app blueprint - Close 2524 [@taras](https://github.com/taras)
- [#3086](https://github.com/ember-cli/ember-cli/pull/3086) [BUGFIX] Ensure that addon test-support trees are not JSHinted in the app. [@rwjblue](https://github.com/rwjblue)
- [#3085](https://github.com/ember-cli/ember-cli/pull/3085) [ENHANCEMENT] Better ASCII art [@johnnyshields](https://github.com/johnnyshields)
- [#3084](https://github.com/ember-cli/ember-cli/pull/3084) [ENHANCEMENT] Add `ember b` as `ember build` command alias. [@cbrock](https://github.com/cbrock)
- [#3092](https://github.com/ember-cli/ember-cli/pull/3092) [BUGFIX] Fix issues with running ember-cli in iojs. [@stefanpenner](https://github.com/stefanpenner)
- [#3096](https://github.com/ember-cli/ember-cli/pull/3096) [BUGFIX] Ensure that `ember g resource` uses custom blueprints (i.e. ember-cli-coffeescript or ember-cli-mocha) properly.  [@jluckyiv](https://github.com/jluckyiv)
- [#3106](https://github.com/ember-cli/ember-cli/pull/3106) [BUGFIX] Fixes file stat related crashes (i.e. using Sublime Text with atomic save enabled). [@raytiley](https://github.com/raytiley)
- [#3114](https://github.com/ember-cli/ember-cli/pull/3114) [BUGFIX] Update version of ES2015 module transpiler (Esperanto). Fixes many transpilation issues (including shadowed declarations and re-exports). [@rwjblue](https://github.com/rwjblue)
- [#3116](https://github.com/ember-cli/ember-cli/pull/3116) [Bugfix] Ensure that files starting with `app/styles*` and `app/templates*` are still available in the app tree. [@stefanpenner](https://github.com/stefanpenner)
- [#3119](https://github.com/ember-cli/ember-cli/pull/3110) [ENHANCEMENT] Update ember-cli-qunit to 0.2.0. [@rwjblue](https://github.com/rwjblue)
- [#3117](https://github.com/ember-cli/ember-cli/pull/3117) [ENHANCEMENT] Replace builtin minification with ember-cli-uglify [@jkarsrud](https://github.com/jkarsrud)
- [#3119](https://github.com/ember-cli/ember-cli/pull/3119) & [#3121](https://github.com/ember-cli/ember-cli/pull/3121) [ENHANCEMENT] Update ember-cli-qunit to v0.3.0. [@rwjblue](https://github.com/rwjblue)
- [#3122](https://github.com/ember-cli/ember-cli/pull/3122) [ENHANCEMENT] Make linting pluggable. [@ef4](https://github.com/ef4)
- [#3123](https://github.com/ember-cli/ember-cli/pull/3123) [ENHANCEMENT] Update ember-cli-6to5 to v3.0.0. [@stefanpenner](https://github.com/stefanpenner)


Thank you to all who took the time to contribute!

### 0.1.9

This release fixes a regression in 0.1.8. See [#3075](https://github.com/ember-cli/ember-cli/issues/3075) for details.

The following changes are required if you are upgrading from the previous version:

- Users
  + [`ember new` diff](https://github.com/kellyselden/ember-cli-output/commit/2275ca51593bae2f6fa91568869f36cd84c264c4)
  + Upgrade your project's ember-cli version - [docs](http://www.ember-cli.com/user-guide/#upgrading)
- Addon Developers
  + [`ember addon` diff](https://github.com/kellyselden/ember-addon-output/commit/d41dcf806a55056a5591dd4e81d712988be1e22f)
  + No changes required
- Core Contributors
  + No changes required

#### Community Contributions

- [#3077](https://github.com/ember-cli/ember-cli/pull/3077) [BUGFIX] Fix error `Cannot call method 'bind' of undefined` [@stefanpenner](https://github.com/stefanpenner)

Thank you to all who took the time to contribute!

### 0.1.8

The following changes are required if you are upgrading from the previous version:

- Users
  + [`ember new` diff](https://github.com/kellyselden/ember-cli-output/commit/ed4f5bcbff0641dba8eca8fe3a3ab96f1347a7cf)
  + Upgrade your project's ember-cli version - [docs](http://www.ember-cli.com/user-guide/#upgrading)
- Addon Developers
  + [`ember addon` diff](https://github.com/kellyselden/ember-addon-output/commit/237c74c0b5972c8ee1ef9b427809aaf93c53db6a)
  + No changes required
- Core Contributors
  + No changes required

#### Community Contributions

- [#3072](https://github.com/ember-cli/ember-cli/pull/3072) [BUGFIX] Update to app blueprint to use QUnit 1.17.1 [@rwjblue](https://github.com/rwjblue)
- [#3069](https://github.com/ember-cli/ember-cli/pull/3069) [BUGFIX] Fix style preprocessors for included addons [@pzuraq](https://github.com/pzuraq)
- [#3068](https://github.com/ember-cli/ember-cli/pull/3068) [ENHANCEMENT] Hide passed tests by default [@rwjblue](https://github.com/rwjblue)
- [#3036](https://github.com/ember-cli/ember-cli/pull/3036) [BUGFIX] Fix platform dependent path separator [@KarimBaaba](https://github.com/KarimBaaba)
- [#2754](https://github.com/ember-cli/ember-cli/pull/2754) [FEATURE] Allow addon commands to be classes [@chadhietala](https://github.com/chadhietala)
- [#2923](https://github.com/ember-cli/ember-cli/pull/2923) [ENHANCEMENT] Add disable-analytics option to all commands [@twokul](https://github.com/twokul)
- [#2901](https://github.com/ember-cli/ember-cli/pull/2901) [ENHANCEMENT] Improve boot by 300–400ms [@stefanpenner](https://github.com/stefanpenner)
- [#3049](https://github.com/ember-cli/ember-cli/pull/3049) [FEATURE] Add Test helper blueprint [@stefanpenner](https://github.com/stefanpenner)
- [#2826](https://github.com/ember-cli/ember-cli/pull/2826) [BUGFIX] Remove path.join from http-mock bluerint urls [@knownasilya](https://github.com/knownasilya)
- [#2983](https://github.com/ember-cli/ember-cli/pull/2983) [ENHANCEMENT] Update QUnit version [@wagenet](https://github.com/wagenet)
- [#2814](https://github.com/ember-cli/ember-cli/pull/2814) [ENHANCEMENT] Add listing of available addons [@rondale-sc](https://github.com/rondale-sc)
- [#3007](https://github.com/ember-cli/ember-cli/pull/3007) [FEATURE] Add a watcher option to the build command [@rauhryan](https://github.com/rauhryan)
- [#3039](https://github.com/ember-cli/ember-cli/pull/3039) [BUGFIX] Move static file check earlier so it only affects the default value [@krisselden](https://github.com/krisselden)
- [#3028](https://github.com/ember-cli/ember-cli/pull/3028) [BUGFIX] Update Testem (fixes timeouts and reloads with Pretender) [@johanneswuerbach](https://github.com/johanneswuerbach)
- [#3026](https://github.com/ember-cli/ember-cli/pull/3026) [BUGFIX] Correct comment in server blueprint [@ohcibi](https://github.com/ohcibi)
- [#3008](https://github.com/ember-cli/ember-cli/pull/3008) [BUGFIX] Clarify error message for ensuring hyphen presence in component name [@artfuldodger](https://github.com/artfuldodger)
- [#3009](https://github.com/ember-cli/ember-cli/pull/3009) [BUGFIX] Tweak error message for executing `ember unknownCommand` [@artfuldodger](https://github.com/artfuldodger)
- [#2996](https://github.com/ember-cli/ember-cli/pull/2996) [BUGFIX] Rename .npmignore in addon blueprint (fixes broken package) [@jgwhite](https://github.com/jgwhite)
- [#2995](https://github.com/ember-cli/ember-cli/pull/2995) [BUGFIX] Correct package.json ordering in app blueprint [@kellyselden](https://github.com/kellyselden)
- [#2984](https://github.com/ember-cli/ember-cli/pull/2984) [ENHANCEMENT] Add `"strict": false` to blueprint .jshintrc [@quaertym](https://github.com/quaertym)

Thank you to all who took the time to contribute!

### 0.1.7

- Users
  + [`ember new` diff](https://github.com/kellyselden/ember-cli-output/commit/22b868fb064631d9ed16e208db982ee808f05296)
  + Upgrade your project's ember-cli version - [docs](http://www.ember-cli.com/user-guide/#upgrading)
  + Uninstall ember-cli-esnext. It has been replaced by ember-cli-6to5 which will be added to your project from the upgrade steps.
- Addon Developers
  + [`ember addon` diff](https://github.com/kellyselden/ember-addon-output/commit/eb5319eb7fd626cc53bf0b5aee639167a031a4c5)
  + No changes required
- Core Contributors
  + No changes required

#### Using 6to5.

The module transpile step is still es3 safe, if you are concern about using
6to5 you can remove it from package.json.

#### Community Contributions

- [#2841](https://github.com/ember-cli/ember-cli/pull/2841) Add insecure-proxy option to `serve` command [@matthewlehner](https://github.com/matthewlehner)
- [#2922](https://github.com/ember-cli/ember-cli/pull/2922) 0.1.6 changelog [@stefanpenner](https://github.com/stefanpenner)
- [#2927](https://github.com/ember-cli/ember-cli/pull/2927) Cleanup badges [@knownasilya](https://github.com/knownasilya)
- [#2937](https://github.com/ember-cli/ember-cli/pull/2937) [BUGFIX] only include dependencies(not devDependencies) of included addons [@jakecraige](https://github.com/jakecraige)
- [#2951](https://github.com/ember-cli/ember-cli/pull/2951) Fix for aliases for hyphenated options [@marcioj](https://github.com/marcioj)
- [#2952](https://github.com/ember-cli/ember-cli/pull/2952) Fixes validation of shorthand commandOptions [@trabus](https://github.com/trabus)
- [#2960](https://github.com/ember-cli/ember-cli/pull/2960) Move from esnext to 6to5 [@stefanpenner](https://github.com/stefanpenner)
- [#2964](https://github.com/ember-cli/ember-cli/pull/2964) [Bugfix] when developing an add-on, it’s jshint tests would be duplicate... [@stefanpenner](https://github.com/stefanpenner)
- [#2965](https://github.com/ember-cli/ember-cli/pull/2965) Remove Hardcoded test-loader [@chadhietala](https://github.com/chadhietala)
- [#2976](https://github.com/ember-cli/ember-cli/pull/2976) debug logging for add-ons + projects [@stefanpenner](https://github.com/stefanpenner)

Thank you to all who took the time to contribute!

### 0.1.6

The following changes are required if you are upgrading from the previous version:

- Users
  + [`ember new` diff](https://github.com/kellyselden/ember-cli-output/commit/5c08ee64df8df8bd40c3e1fd9541e19c9db9b214)
  + Upgrade your project's ember-cli version - [docs](http://www.ember-cli.com/user-guide/#upgrading)
- Addon Developers
  + [`ember addon` diff](https://github.com/kellyselden/ember-addon-output/commit/82652c474424a118268383d429f4b054567bb966)
  + No changes required
- Core Contributors
  + Use `expect` over `assert` in tests going forward

#### Community Contributions

- [#2885](https://github.com/ember-cli/ember-cli/pull/2885) [ENHANCEMENT] NPM should use save-exact flags [@chadhietala](https://github.com/chadhietala)
- [#2840](https://github.com/ember-cli/ember-cli/pull/2840) [INTERNAL ]using 'expect' vs. assert. [@Mawaheb](https://github.com/Mawaheb)
- [#2669](https://github.com/ember-cli/ember-cli/pull/2669) [ENHANCEMENT] add .npmignore to addon blueprint [@pogopaule](https://github.com/pogopaule)
- [#2909](https://github.com/ember-cli/ember-cli/pull/2909) [INTERNAL] Use lib/ext/promise instead of RSVP directly [@zeppelin](https://github.com/zeppelin)
- [#2857](https://github.com/ember-cli/ember-cli/pull/2857) [ENHANCEMENT] Add descriptions to more Broccoli trees. [@rwjblue](https://github.com/rwjblue)
- [#2842](https://github.com/ember-cli/ember-cli/pull/2842) [INTERNAL] Prefer `expect` over `assert` for testing [@stavarotti](https://github.com/stavarotti)
- [#2847](https://github.com/ember-cli/ember-cli/pull/2847) [BUGFIX] Bump ember-router-generator (fixes WARN on description not present). [@abuiles](https://github.com/abuiles)
- [#2843](https://github.com/ember-cli/ember-cli/pull/2843) [INTERNAL] Unify using chai.expect [@twokul](https://github.com/twokul)
- [#2900](https://github.com/ember-cli/ember-cli/pull/2900) [INTERNAL] update some CI stuff [@stefanpenner](https://github.com/stefanpenner)
- [#2876](https://github.com/ember-cli/ember-cli/pull/2876) [BUGFIX] make sure adapter cannot extend from itself [@jakecraige](https://github.com/jakecraige)
- [#2869](https://github.com/ember-cli/ember-cli/pull/2869) [BUGFIX] Tolerate before & after references to missing addons [@ef4](https://github.com/ef4)
- [#2864](https://github.com/ember-cli/ember-cli/pull/2864) [ENHANCEMENT] the .gitkeep in /public can now be removed [@kellyselden](https://github.com/kellyselden)
- [#2887](https://github.com/ember-cli/ember-cli/pull/2887) [INTERNAL] I don’t think we need this anymore. [@stefanpenner](https://github.com/stefanpenner)
- [#2910](https://github.com/ember-cli/ember-cli/pull/2910) [DOCS] Update org references to ember-cli [@zeppelin](https://github.com/zeppelin)
- [#2911](https://github.com/ember-cli/ember-cli/pull/2911) [DOCS] More org updates to reference ember-cli [@Dhaulagiri](https://github.com/Dhaulagiri)
- [#2916](https://github.com/ember-cli/ember-cli/pull/2916) [BUGFIX] findAddonByName returning incorrect matches [@jakecraige](https://github.com/jakecraige)
- [#2918](https://github.com/ember-cli/ember-cli/pull/2918) [ENHANCEMENT] Updated testem [@johanneswuerbach](https://github.com/johanneswuerbach)
- [#2919](https://github.com/ember-cli/ember-cli/pull/2919) [ENHANCEMENT] implement Blueprint.prototype.addAddonToProject [@jakecraige](https://github.com/jakecraige)
- [#2920](https://github.com/ember-cli/ember-cli/pull/2920) [BUGFIX] explicitly bump broccoli-sourcemap-concat to fix #2890 [@krisselden](https://github.com/krisselden)
- [#2929](https://github.com/ember-cli/ember-cli/pull/2929) [BUGFIX] Bump ember-router-generator. [@abuiles](https://github.com/abuiles)
- [#2939](https://github.com/ember-cli/ember-cli/pull/2939) [ENHANCEMENT] Add a hook for postprocessing tests tree [@ef4](https://github.com/ef4)
- [#2941](https://github.com/ember-cli/ember-cli/pull/2941) [ENHANCEMENT] Bumped testem [@johanneswuerbach](https://github.com/johanneswuerbach)
- [#2944](https://github.com/ember-cli/ember-cli/pull/2944) [INTERNAL] update CONTRIBUTING.md [@jakecraige](https://github.com/jakecraige)

Thank you to all who took the time to contribute!

### 0.1.5

### Applications

- [`ember new` diff](https://github.com/kellyselden/ember-cli-output/commit/feb4dd773e2d68c8576b060c2973062ba83ed66a)

- [#2727](https://github.com/ember-cli/ember-cli/pull/2727) Added
  sourcemap support to the JS concatenation and minification steps of
  the build. This eliminates the need for the wrapInEval hack. Any
  Javascript preprocessors that produce sourcemaps will also be
  automatically incorporated into the final result. Sourcemaps are
  enabled by default in dev, to enable them in production pass
  `{sourcemaps: { enabled: true, extensions: ['js']}}` to your
  EmberApp constructor.

- [#2777](https://github.com/ember-cli/ember-cli/pull/2777) allowed
  the creation of components with slashes in their names since this is
  supported in Handlebars 2.0.

- [#2800](https://github.com/ember-cli/ember-cli/pull/2800) Added 3 new commands

  ```
  ember install
  ember install:bower moment
  ember install:npm ember-browserify
  ```

  They behave exactly as you'd expect. Install runs npm and bower
  install on the project. The last two simply pass in the package names
  you give it to the underlying task to do it.

- [#2805](https://github.com/ember-cli/ember-cli/pull/2805) Added the
  `install:addon` command, which installs an addon with NPM and then
  runs the included generator of the same name  if it provides one.

  If the blueprint for the installed addon requires arguments, then
  you can pass them too, for example, the `ember-cli-cordova` addon
  needs an extra argument which you can pass running the command as
  follows: `ember install:addon ember-cli-cordova com.myapp.app`.

- [#2565](https://github.com/ember-cli/ember-cli/pull/2565) added
  support for command options aliases, as well as aliases for
  predefined options, this means that some commands can use aliases
  for their existing options, for example, instead of running `ember g
  route foo --type route` we can now use the -route alias: `ember g
  route foo -route`.

  You can see available aliases for each command running `ember help`,
  they will show as `aliases: ` follow by the alias.

- [#2668](https://github.com/ember-cli/ember-cli/pull/2668) added the
  `prepend` flag to `app.import` in `Brocfile.js`, allowing to prepend
  a file to the vendor bundle rather than appended which is the
  default behaviour.

   ```
   // Brocfile.js
   app.import('bower_components/es5-shim/es5-shim.js', {
     type: 'vendor',
     prepend: true
   });
   ```

- [#2694](https://github.com/ember-cli/ember-cli/pull/2694) disabled
  default lookup & active generation logging in
  `config/environment.js`.

- [#2748](https://github.com/ember-cli/ember-cli/pull/2748) improved
  the router generator to support properly nested routes and
  resources, previously if you had a route similar like:

  ```
  Router.map(function() {
    this.route("foo");
  });
  ```

  And you did `ember g route foo/bar` the generated routes would be

  ```
  Router.map(function() {
    this.route("foo");
    this.route("foo/bar");
  });
  ```

  Now it keeps manages nested routes properly so the result would be:

  ```
  Router.map(function() {
    this.route("foo", function() {
      this.route("bar");
    });
  });
  ```

  Additionally the option `--path` was added so you can do things like
  `ember g route friends/edit --path=:friend_id/id` creating a nested
  route under `friends` like: `this.route('edit', {path: ':friend_id/edit'})`

- [#2734](https://github.com/ember-cli/ember-cli/pull/2734) changed
  the options for editorconfig so it won't remove trailing whitespace
  on .diff files.

- [#2788](https://github.com/ember-cli/ember-cli/pull/2788) added an
  `on('error')` handler to the proxy blueprint, with this your `ember
  server` won't be killed when receiving `socket hang up` from the
  `http-proxy`.

- [#2741](https://github.com/stefanpenner/ember-cli/pull/2741) updated `broccoli-asset-rev` to 2.0.0.

- [#2779](https://github.com/ember-cli/ember-cli/pull/2779) fixed a
  bug in your `.ember-cli` file, if you had a liveReloadPort of say
  "4200" it would not actually end up as that port. This casts the
  string to a number so that the port is set correctly.

- [#2817](https://github.com/ember-cli/ember-cli/pull/2817) added a
  new feature so [Leek](https://github.com/twokul/leek) can be
  configured through your `.ember-cli` file. It means you will be able
  to configure the URLs Leek sends requests to, with this you can plug
  internal tools and track usage patterns.

- [#2828](https://github.com/ember-cli/ember-cli/pull/2828) added the
  option to consume `app.env` before app instance creation in your
  Brocfile, this is useful if you want to pass environment-dependent
  options to the EmberApp constructor in your Brocfile:

  ```
  new EmberApp({
    someOption: EmberApp.env() === 'production' ? 'foo' : 'bar';
  });
  ```
- [#2829](https://github.com/ember-cli/ember-cli/pull/2829) fixed an
  issue on the model-test blueprint which was causing the build to fail
  when the options `needs` wasn't present.

- [#2832](https://github.com/ember-cli/ember-cli/pull/2832) added a
  buildError hook which will be called when an error occurs during the
  `preBuild` or `postBuild` hooks for addons, or when `builder#build`
  fails hook.

- [#2836](https://github.com/ember-cli/ember-cli/pull/2836) added a
  check when passing `--proxy` to `ember server`. If the URL doesn't
  include `http` or `https` then the command will fail since it
  requires the protocol in order to get the proxy working correctly.

#### Addons

- [`ember addon` diff](https://github.com/kellyselden/ember-addon-output/commit/1df573947070214b058e830e962a673fc9819925)

- [#2693](https://github.com/ember-cli/ember-cli/pull/2693) fixed an
  issue with blueprints ensuring that last loaded blueprint takes
  precedence.

- [#2805](https://github.com/ember-cli/ember-cli/pull/2805) Added the
  `install:addon` command, which installs an addon with NPM and then
  runs the included generator of the same name if it provides one,
  additionally if you addon generator's name is different to the addon
  name, you can pass the option `defaultBlueprint` in your
  `package.json` and the command will run the generator after
  installed. The following will run `ember g cordova-starter-kit`
  after it has successfully installed `ember-cli-cordova`

  ```
  name: 'ember-cli-cordova',
  'ember-addon': {
    defaultBlueprint: 'cordova-starter-kit'
  }
  ```
- [#2775](https://github.com/ember-cli/ember-cli/pull/2775) added a
  default `.jshintrc` for `in-repo-addons` so they are treated as
  `Node` applications.

### 0.1.4

#### Applications

* [`ember new` diff](https://github.com/kellyselden/ember-cli-output/commit/94439eb22e76e7e71be472c264bef40235583fa9)
* [BUGFIX] Use the container from the created Ember.Application for initializer tests. [#2582](https://github.com/stefanpenner/ember-cli/pull/2582)
* [ENHANCEMENT] Add extra contentFor hooks [#2588](https://github.com/stefanpenner/ember-cli/pull/2592)
  * `{{content-for 'head-footer'}}`
  * `{{content-for 'test-head-footer'}}`
  * `{{content-for 'body-footer'}}`
  * `{{content-for 'test-body-footer'}}`
* [BUGFIX] Create separate server blueprint to stop http-{mock,proxy} removing files [#2610](https://github.com/stefanpenner/ember-cli/pull/2610)
* [BUGFIX] Fixes `--proxy` so it proxies correctly to API's under subdomains [#2615](https://github.com/stefanpenner/ember-cli/pull/2615)
* [BUGFIX] Ensure `watchman` does not conflict with NPM's `watchman` package. [#2645](https://github.com/stefanpenner/ember-cli/pull/2645)
* [BUGFIX] Ensure that the generated meta tag is now self closing. [#2661](https://github.com/stefanpenner/ember-cli/pull/2661)

#### Addons

* [`ember addon` diff](https://github.com/kellyselden/ember-addon-output/commit/1ba17bff6003dd176a038113f412fc24a26b03d2)

### 0.1.3

#### Applications

  * [`ember new` diff](https://github.com/kellyselden/ember-cli-output/commit/25323aef6cd8a4e277dc451aa4fa0d80a1715acd)
  * [#2586](https://github.com/stefanpenner/ember-cli/pull/2586) Set locationType to none in tests.
  * [#2573](https://github.com/stefanpenner/ember-cli/pull/2574) Added --silent option for quieter UI
  * [#2458](https://github.com/stefanpenner/ember-cli/pull/2458) Added additional file watching mechanism: [Watchman](https://facebook.github.io/watchman/)
    This helps resolve the recent Node + Yosemite file watching issues, but also improves file watching (when available) for all `*nix` systems

    What is Watchman?

    Watchman runs as a standalone service, this allows it to manage file-watching for multiple consumers (in our case ember-cli apps)

    How do I used it?
      homebrew: `brew install watchman`
      other: https://facebook.github.io/watchman/docs/install.html
      windows: not supported yet, but it [may happen](https://github.com/facebook/watchman/issues/19)

    What happens if its not installed?

    We fall back to the existing watcher NodeWatcher

    How do I force it to fallback to NodeWatch

    ```sh
    ember <command> --watcher=node
    ```

    Common problem: `invalid watchman found, version: [2.9.8] did not satisfy [^3.0.0]` this basically means you have an older version of watchman installed. Be sure to install `3.0.0` and run `watchman shutdown-server` before re-starting your ember server.

  * [#2265](https://github.com/stefanpenner/ember-cli/pull/2265) Added auto-restarting of server and triggering of LR on `server/*` file changes
  * [#2535](https://github.com/stefanpenner/ember-cli/pull/2535) Updated broccoli-asset-rev to 1.0.0
  * [#2452](https://github.com/stefanpenner/ember-cli/pull/2452) Including [esnext](https://github.com/esnext/esnext) via `ember-cli-esnext` per default
  * [#2518](https://github.com/stefanpenner/ember-cli/pull/2518) improved HTTP logging when using http-mocks and proxy by using [morgan](https://www.npmjs.org/package/morgan)
  * [#2532](https://github.com/stefanpenner/ember-cli/pull/2532) Added support to run specific tests via `ember test --module` and `ember test --filter`
  * [#2514](https://github.com/stefanpenner/ember-cli/pull/2514) Added config.usePodsByDefault for users who wish to have blueprints run in `pod` mode all the time
  * Warn on invalid command options
  * Allow array of paths to the preprocessCss phase
  * Adding --pods support for adapters, serializers, and transforms
  * As part of the Ember 2.0 push  remove controller types.
  * http-mock now follows ember-data conventions
  * many of ember-cli internals now are instrumented with [debug]
    usage: `DEBUG=ember-cli:* ember <command>` to see ember-cli specific verbose logging.
  * Added ember-cli-dependency-checker to app's package.json
  * Added option to disable auto-start of ember app.
  * Added optional globbing to init with `ember init <glob-pattern>`, this allows you to re-blueprint a single file like: `ember init app/index.html`
  * Added support to test the app when built with `--env production`.
  * Update to Ember 1.8.1
  * Update to Ember Data v1.0.0-beta.11
  * [#2351](https://github.com/stefanpenner/ember-cli/pull/2351) Fix automatic generated model belongs-to and has-many relations to resolve test lookup.
  * [#1888](https://github.com/stefanpenner/ember-cli/pull/1888) Allow multiple SASS/LESS files to be built by populating `outputPaths.app.css` option
  * [#2523](https://github.com/stefanpenner/ember-cli/pull/2523) Added `outputPaths.app.html` option
  * [#2472](https://github.com/stefanpenner/ember-cli/pull/2472) Added Pod support for test blueprints.

  Add much more: [view entire diff](https://github.com/stefanpenner/ember-cli/compare/v0.1.2...v0.1.3)

#### Addons

  * [`ember addon` diff](https://github.com/kellyselden/ember-addon-output/commit/b75bc2c5e0d8d5f954d6c7adcb108f71fdef9ebf)
  * [#2505](https://github.com/stefanpenner/ember-cli/pull/2505) Added ability to dynamic add/remove module whitelist entries so that the [ember-browserify](https://github.com/ef4/ember-browserify) addon can work
  * [#2505](https://github.com/stefanpenner/ember-cli/pull/2505) Added an addon postprocess hook for all javascript
  * [#2271](https://github.com/stefanpenner/ember-cli/pull/2271) Added Addon.prototype.isEnabled for an addon to exclude itself from the project at runtime.
  * [#2451](https://github.com/stefanpenner/ember-cli/pull/2451) Ensure that in-repo addons are watched.
  * [#2411](https://github.com/stefanpenner/ember-cli/pull/2411) Add preBuild hook for addons.

### 0.1.2

#### Applications

* [`ember new` diff](https://github.com/kellyselden/ember-cli-output/commit/0e6a6834b46df72658225490073980c44413892f)
* [BREAKING ENHANCEMENT] Remove hard-coded support for `broccoli-less-single`, use `ember-cli-less` for `.less` support now. [#2210](https://github.com/stefanpenner/ember-cli/pull/2210)
* [ENHANCEMENT] Provide a helpful error if the configuration info cannot be read from the proper `<meta>` tag. [#2219](https://github.com/stefanpenner/ember-cli/pull/2219)
* [ENHANCEMENT] Allow test filtering from the command line. Running `ember test --filter "foo bar"` or `ember test --server --filter "foo bar"` will limit test runs
  to tests that contain "foo bar" in their module name or test name. [#2223](https://github.com/stefanpenner/ember-cli/pull/2223)
* [ENHANCEMENT] Add a few more `content-for` hooks to `index.html` and `tests/index.html`. [#2236](https://github.com/stefanpenner/ember-cli/pull/2236)
* [ENHANCEMENT] Properly display the file causing build errors in `ember build --watch` and `ember serve` commands. [#2237](https://github.com/stefanpenner/ember-cli/pull/2237), [#2246](https://github.com/stefanpenner/ember-cli/pull/2246), and [#2297](https://github.com/stefanpenner/ember-cli/pull/2297)
* [ENHANCEMENT] Update `broccoli-asset-rev` to 0.3.1. [#2250](https://github.com/stefanpenner/ember-cli/pull/2250)
* [ENHANCEMENT] Add `ember-export-application-global` to allow easier debugging. [#2270](https://github.com/stefanpenner/ember-cli/pull/2270)
* [BUGFIX] Fix default `.gitignore` to properly match `bower_components`. [#2285](https://github.com/stefanpenner/ember-cli/pull/2285)
* [ENHANCEMENT] Display `baseURL` in `ember serve` startup messages. [#2291](https://github.com/stefanpenner/ember-cli/pull/2291)
* [BUGFIX] Fix issues resulting in files outside of `tmp/` being removed due to following of symlinks. [#2290](https://github.com/stefanpenner/ember-cli/pull/2290) and [#2301](https://github.com/stefanpenner/ember-cli/pull/2301)
* [ENHANCEMENT] Add --watcher=polling option to `ember test --server`. This provides a work around for folks having `EMFILE` errors in some scenarios. [#2296](https://github.com/stefanpenner/ember-cli/pull/2296)
* [ENHANCEMENT] Allow opting out of storing the applications configuration in the generated `index.html` via `storeConfigInMeta` option in the `Brocfile.js`. [#2298](https://github.com/stefanpenner/ember-cli/pull/2298)
* [BUGFIX] Update ember-cli-content-security-policy and ember-cli-inject-live-reload packages to latest version. Allows livereload to function properly regardless
  of host (0.1.0 always assumed `localhost` for the livereload server). [#2306](https://github.com/stefanpenner/ember-cli/pull/2306)
* [ENHANCEMENT] Update internal dependencies to latest versions. [#2307](https://github.com/stefanpenner/ember-cli/pull/2307)
* [BUGFIX] Allow overriding of vendor files to not loose required ordering. [#2312](https://github.com/stefanpenner/ember-cli/pull/2312)
* [ENHANCEMENT] Add `bowerDirectory` to `Project` model (discovered on initialization). [#2287](https://github.com/stefanpenner/ember-cli/pull/2287)

#### Addons

* [`ember addon` diff](https://github.com/kellyselden/ember-addon-output/commit/333ec703fba364bf5d2dcb3cc1d04a7b65c246f0)
* [ENHANCEMENT] Allow addons to inject middleware into testem. [#2128](https://github.com/stefanpenner/ember-cli/pull/2128)
* [ENHANCEMENT] Add {{content-for 'body'}} to `app/index.html` and `tests/index.html`. [#2236](https://github.com/stefanpenner/ember-cli/pull/2236)
* [ENHANCEMENT] Add {{content-for 'test-head'}} to `tests/index.html`. [#2236](https://github.com/stefanpenner/ember-cli/pull/2236)
* [ENHANCEMENT] Add {{content-for 'test-body'}} to `tests/index.html`. [#2236](https://github.com/stefanpenner/ember-cli/pull/2236)
* [ENHANCEMENT] Allow adding multiple bower packages at once via `Blueprint.prototype.addBowerPackagesToProject`. [#2222](https://github.com/stefanpenner/ember-cli/pull/2222)
* [ENHANCEMENT] Allow adding multiple NPM packages at once via `Blueprint.prototype.addPackagesToProject`. [#2245](https://github.com/stefanpenner/ember-cli/pull/2245)
* [ENHANCEMENT] Ensure generated addons are in strict mode. [#2295](https://github.com/stefanpenner/ember-cli/pull/2295)
* [BUGFIX] Ensure that addon's with `addon/styles/app.css` are able to compile properly (copying contents of `addon/styles/app.css` into `vendor.css`). [#2301](https://github.com/stefanpenner/ember-cli/pull/2301)
* [ENHANCEMENT] Provide the `httpServer` instance to `serverMiddleware` (and `./server/index.js`). [#2302](https://github.com/stefanpenner/ember-cli/issues/2302)

#### Blueprints

* [ENHANCEMENT] Tweak helper blueprint to make it easier to test. [#2257](https://github.com/stefanpenner/ember-cli/pull/2257)
* [ENHANCEMENT] Streamline initializer and service blueprints. [#2260](https://github.com/stefanpenner/ember-cli/pull/2260)

### 0.1.1

#### Applications

* [`ember new` diff](https://github.com/kellyselden/ember-cli-output/commit/c97633f87717074986403e3ad87149d3bd8d1ee3)
* [BUGFIX] Fix symlink regression in Windows (update broccoli-file-remover to 0.3.1). [#2204](https://github.com/stefanpenner/ember-cli/pull/2204)

#### Addons

* [`ember addon` diff](https://github.com/kellyselden/ember-addon-output/commit/decc6c6c87071d271ee4b86dc292b7a353ead0e1)

### 0.1.0

#### Applications

* [`ember new` diff](https://github.com/kellyselden/ember-cli-output/commit/d35102254aeac097405167f2abeef57b92def518)
* [ENHANCEMENT] Add symlinking to speed up Broccoli builds. [#2125](https://github.com/stefanpenner/ember-cli/pull/2125)
* [BUGFIX] Fix issue with livereload in 0.0.47. [#2176](https://github.com/stefanpenner/ember-cli/pull/2176)
* [BUGFIX] Change content security policy addon to use report only mode by default. [#2190](https://github.com/stefanpenner/ember-cli/pull/2190)
* [ENHANCEMENT] Allow addons to customize their ES6 module prefix (for `addon` tree). [#2189](https://github.com/stefanpenner/ember-cli/pull/2189)
* [BUGFIX] Ensure all addon hooks are executed in addon test harness. [#2195](https://github.com/stefanpenner/ember-cli/pull/2195)

#### Addons

* [`ember addon` diff](https://github.com/kellyselden/ember-addon-output/commit/84fd1c523407e9fa7df7d2a664a14ddb543ea5e0)

### 0.0.47

#### Applications

* [`ember new` diff](https://github.com/kellyselden/ember-cli-output/commit/7c59cf59fa42a9bf585b52455e424e2553aeb2aa)
* [ENHANCEMENT] Add `--pod` option to blueprints for generate and destroy. Add `fileMapTokens` hook to blueprints, and optional
  blueprint file tokens `__path__` and `__test__` for pod support. [#1994](https://github.com/stefanpenner/ember-cli/pull/1994)
* [ENHANCEMENT] Provide better error messages when uncaught errors occur during `ember build` and `ember serve`. [#2043](https://github.com/stefanpenner/ember-cli/pull/2043)
* [ENHANCEMENT] Do not use inline `<script>` tags.  Set the stage for enabling content security policy. [#2058](https://github.com/stefanpenner/ember-cli/pull/2058)
* [ENHANCEMENT] Add [ember-cli-content-security-policy](https://github.com/rwjblue/ember-cli-content-security-policy) addon
  when running development server (see [content-security-policy.com](http://content-security-policy.com/) for details). [#2065](https://github.com/stefanpenner/ember-cli/pull/2065)
* [BREAKING] Remove `environment` and `getJSON` options to `EmberApp` (in the `Brocfile.js`).
* [ENHANCEMENT] Add `configPath` option to `EmberApp` (in the `Brocfile.js`) to allow using a custom file for obtaining configuration
  information. [#2068](https://github.com/stefanpenner/ember-cli/pull/2068)
* [BUGFIX] Use url.parse instead of manually checking baseURL. This allows `app://localhost/` URLs needed for node-webkit. [#2088](https://github.com/stefanpenner/ember-cli/pull/2088)
* [BUGFIX] Remove duplicate warning when generating controllers. [#2066](https://github.com/stefanpenner/ember-cli/pull/2066)
* [BREAKING ENHANCEMENT] Move `config` information out of the `assets/my-app-name.js` file and into a `<meta>` tag in the document `head`. [#2086](https://github.com/stefanpenner/ember-cli/pull/2086)
  * Removes `<my-app-name>/config/environments/*` from module system output.
  * Makes build output the same regardless of environment config.
  * Makes injection of custom config information as simple as adding/modifying/customizing the meta contents.
* [BREAKING BUGFIX] Update `loader.js` entry in `bower.json` to use the proper name.

  This requires editing `bower.json` to change:

```
  "loader": "stefanpenner/loader.js#1.0.1",
```

To:

```
  "loader.js": "stefanpenner/loader.js#1.0.1",
```
* [BREAKING ENHANCEMENT] Replace `{{BASE_TAG}}` in `index.html` with `{{content-for 'head'}}`. [#2153](https://github.com/stefanpenner/ember-cli/pull/2153)

#### Addons

* [`ember addon` diff](https://github.com/kellyselden/ember-addon-output/commit/de0caa4385998250786a975a32208ebff7fcbf87)
* [BUGFIX] `addon/` directory is no longer required when running local development server inside an addon. [#2044](https://github.com/stefanpenner/ember-cli/pull/2044)
* [BUGFIX] Use the specified name for the addon (was previously using `dummy` for all addon's names). [#2042](https://github.com/stefanpenner/ember-cli/pull/2042)
* [ENHANCEMENT] Add `Registry.prototype.remove` to make it easier to remove preprocessor plugins. [#2048](https://github.com/stefanpenner/ember-cli/pull/2048)
* [ENHANCEMENT] Add `Registry.prototype.extensionsForType` to make it easier to detect what extensions are support for a given type
  of file (`js`, `css`, or `template` files). [#2050](https://github.com/stefanpenner/ember-cli/pull/2050)
* [BUGFIX] Allow `addon` tree to contain any filetype that is known by the JS preprocessor registry. [#2054](https://github.com/stefanpenner/ember-cli/pull/2054)
* [BUGFIX] Ensure that addons cannot override the application configuration (in the `config` hook). [#2133](https://github.com/stefanpenner/ember-cli/pull/2133)
* [ENHANCEMENT] Allow addons to implement `contentFor` method to add string output into the associated
  `{{content-for 'foo'}}` section in `index.html`. [#2153](https://github.com/stefanpenner/ember-cli/pull/2153)

#### Blueprints

* [ENHANCEMENT] Add `description` for blueprints created by `ember generate blueprint`. [#2062](https://github.com/stefanpenner/ember-cli/pull/2062)
* [ENHANCEMENT] Add `in-repo-addon` generator. [#2072](https://github.com/stefanpenner/ember-cli/pull/2072)
* [ENHANCEMENT] Add `before` and `after` options to `Blueprint.prototype.insertIntoFile`. [#2122](https://github.com/stefanpenner/ember-cli/pull/2122)
* [ENHANCEMENT] Allow `git` based application blueprints. [#2103](https://github.com/stefanpenner/ember-cli/pull/2103)

### 0.0.46

#### Applications

* [`ember new` diff](https://github.com/kellyselden/ember-cli-output/commit/4127ceb7851f28dc32fe50eaeb618754b7fa6027)
* [BUGFIX] Addons shared the same `treePaths` and `treeForMethods` listing. This meant that an addon changing `this.treePaths.vendor` (for example) would modify where
  ALL addons looked for their vendor trees. [#2035](https://github.com/stefanpenner/ember-cli/pull/2035)

#### Addons

* [`ember addon` diff](https://github.com/kellyselden/ember-addon-output/commit/176915ef4c5b01e898db8a85cd2e11a83b7117c3)

### 0.0.45

#### Applications

* [`ember new` diff](https://github.com/kellyselden/ember-cli-output/commit/02106d8aa7873a07c05bf05f5c6ed08f7bb30311)
* [BREAKING ENHANCEMENT] Moved `modulePrefix` to `config/environment.js`. [#1933](https://github.com/stefanpenner/ember-cli/pull/1933)
* [BREAKING ENHANCEMENT] Remove `window.MyAppNameENV`. You will now need to import the configuration instead of relying on using the global.  [#1903](https://github.com/stefanpenner/ember-cli/pull/1903).

```javascript
import ENV from '<appName>/config/environment';
ENV.API_HOST; // example.com
```

* [ENHANCEMENT] Allowing config of asset output paths. [#1904](https://github.com/stefanpenner/ember-cli/pull/1904)
* [ENHANCEMENT] Add a default `.ember-cli` file and document disableAnalytics. [#1801](https://github.com/stefanpenner/ember-cli/pull/1801)
* [BUGFIX] Add location type for test environment. This generally makes using `ember test` with a custom baseURL work properly. [#1915](https://github.com/stefanpenner/ember-cli/pull/1915)
* [ENHANCEMENT] Allow multiple pre-processors per type (for example, using `broccoli-sass` AND `broccoli-less` is now possible). [#1918](https://github.com/stefanpenner/ember-cli/pull/1918)
* [ENHANCEMENT] Update `startApp` to provide app configuration. [#1329](https://github.com/stefanpenner/ember-cli/pull/1329)
* [BUGFIX] Remove manual `env === 'production'` checks. [#1929](https://github.com/stefanpenner/ember-cli/pull/1929)
* [BUGFIX] Fixed an issue where project.config() could be called with undefined environment when starting express server. [#1959](https://github.com/stefanpenner/ember-cli/pull/1959)
* [ENHANCEMENT] Improve blueprint self-documentation by adding additional details to `ember help generate`. [#1279](https://github.com/stefanpenner/ember-cli/pull/1279)
* [ENHANCEMENT] Update `broccoli-asset-rev`to `0.1.1`. [#1945](https://github.com/stefanpenner/ember-cli/pull/1945)
* [ENHANCEMENT] Update app blueprint's `package.json`/`bower.json` to depend on ember-data. [#1873](https://github.com/stefanpenner/ember-cli/pull/1873)
* [BUGFIX] Ensure that things loaded by server/index.js override addons. This changes the middleware ordering so that the app's middlewares are loaded *before*
  the internal middlewares. [#2008](https://github.com/stefanpenner/ember-cli/pull/2008)
* [BUGFIX] Removed broccoli-sweetjs from the internal preprocessor registry. [#2014](https://github.com/stefanpenner/ember-cli/pull/2014)
* [ENHANCEMENT] Pull `podModulePrefix` from config/environment.js. [#2024](https://github.com/stefanpenner/ember-cli/pull/2024)
* [BUGFIX] Exit with a non-zero exit code (to indicate failure), and provide a nice error message if `ember test` runs nothing. [#2025](https://github.com/stefanpenner/ember-cli/pull/2025)

#### Addons

* [`ember addon` diff](https://github.com/kellyselden/ember-addon-output/commit/d217efb35368ae89897896d4e4817711e01b2f6c)
* [ENHANCEMENT] Allow addons to return a public tree. By default anything in an addon's `public/` folder will be copied into a folder for that addon's name
in the output path. [#1930](https://github.com/stefanpenner/ember-cli/pull/1930)
* [BUGFIX] Remove extra nesting of `addon/styles` tree. Previously, the addon styles tree was looking for `addon/styles/styles/`. [#1964](https://github.com/stefanpenner/ember-cli/pull/1964)
* [ENHANCEMENT] Add `config` hook for addons. [#1972](https://github.com/stefanpenner/ember-cli/pull/1972)
* [BUGFIX] Ensure we do not add `ember-addon` twice when running `ember init` (to upgrade an addon). [#1982](https://github.com/stefanpenner/ember-cli/pull/1982)
* [BUGFIX] Allow `addon/templates` to be properly available inside the `my-addon-name.js` file with the correct module name. [#1983](https://github.com/stefanpenner/ember-cli/pull/1983)

#### Blueprints

* [ENHANCEMENT] Add empty function to resource blueprint when resource is singular. [#1946](https://github.com/stefanpenner/ember-cli/pull/1946)
* [BUGFIX] Do not inject application route into app/router.js. [#1953](https://github.com/stefanpenner/ember-cli/pull/1953)
* [ENHANCEMENT] Add `Blueprint.prototype.lookupBlueprint` which allows a blueprint to lookup other Blueprints. This makes it much easier to provide Blueprints that
  augment existing internal blueprints without having to copy and override them completely. [#2016](https://github.com/stefanpenner/ember-cli/pull/2016)

### 0.0.44

#### Applications

* [`ember new` diff](https://github.com/kellyselden/ember-cli-output/commit/d2bf0aa8e1868c45cff88b379b79d062646d1a62)
* [BUGFIX] Provide useful error message when `app/styles/app.ext` is not found. [#1866](https://github.com/stefanpenner/ember-cli/pull/1866) and [#1894](https://github.com/stefanpenner/ember-cli/pull/1894)
* [ENHANCEMENT] Updated dependency broccoli-es3-safe-recast. [#1891](https://github.com/stefanpenner/ember-cli/pull/1898) and [#1898](https://github.com/stefanpenner/ember-cli/pull/1898)
* [ENHANCEMENT] Updated dependency broccoli-merge-trees. [#1891](https://github.com/stefanpenner/ember-cli/pull/1898) and [#1898](https://github.com/stefanpenner/ember-cli/pull/1898)
* [ENHANCEMENT] Updated dependency fs-extra. [#1891](https://github.com/stefanpenner/ember-cli/pull/1898) and [#1898](https://github.com/stefanpenner/ember-cli/pull/1898)
* [ENHANCEMENT] Updated dependency proxy-middleware. [#1891](https://github.com/stefanpenner/ember-cli/pull/1898) and [#1898](https://github.com/stefanpenner/ember-cli/pull/1898)
* [ENHANCEMENT] Updated dependency tiny-lr. [#1891](https://github.com/stefanpenner/ember-cli/pull/1898) and [#1898](https://github.com/stefanpenner/ember-cli/pull/1898)
* [BUGFIX] Update `broccoli-caching-writer` to fix performance regression. [#1901](https://github.com/stefanpenner/ember-cli/pull/1901)
* [BUGFIX] Ensure that a `.bowerrc` without `directory` specified does not error. [#1902](https://github.com/stefanpenner/ember-cli/pull/1902)

#### Addons

* [`ember addon` diff](https://github.com/kellyselden/ember-addon-output/commit/56bc291bde44a0907284da0bfe10de69e3c0f876)
* [BUGFIX] Allow addons with styles to function properly. [#1892](https://github.com/stefanpenner/ember-cli/pull/1892)

#### Blueprints

* [BUGFIX] Fix `ember g http-mock foo` output to pass JSHint. [#1896](https://github.com/stefanpenner/ember-cli/pull/1896)

### 0.0.43

#### Applications

* [`ember new` diff](https://github.com/kellyselden/ember-cli-output/commit/a84279b975a550dc1f0befd2b066d4b1e7ff5c07)
* [BUGFIX] Fix ember init command in empty directory. [#1779](https://github.com/stefanpenner/ember-cli/pull/1779)
* [ENHANCEMENT] Add triggerEvent to `tests/.jshintrc`. [#1782](https://github.com/stefanpenner/ember-cli/pull/1782)
* [ENHANCEMENT] Allow opting out of analytics via `.ember-cli` config file. [#1797](https://github.com/stefanpenner/ember-cli/pull/1797)
* Bump `ember-cli-qunit` version.
* [BUGFIX] Update broccoli-caching-writer dependents to allow linking fallback (enables easier usage of ember-cli from within Vagrant). [#1799](https://github.com/stefanpenner/ember-cli/pull/1799)
* [BUGFIX] Avoid issue where `ember init` stalls on fresh system due to `bower install` prompting for permission to use analytics. [#1805](https://github.com/stefanpenner/ember-cli/pull/1805)
* [BUGFIX] Allow usage of standard Node.js functionality in `config/environments.js` (fixes a regression in 0.0.42). [#1809](https://github.com/stefanpenner/ember-cli/pull/1809)
* [ENHANCEMENT] Make current environment available modules. [#1820](https://github.com/stefanpenner/ember-cli/pull/1820)
* [BUGFIX] Ensures that AppNameENV and EmberENV are setup before the vendor files have been loaded (changes in 0.0.42 caused enabling Ember feature flags impossible from `config/environments.js`). [#1825](https://github.com/stefanpenner/ember-cli/pull/1825)
* [ENHANCEMENT] Ensures that the `<base>` tag changes when the config file is updated. [#1825](https://github.com/stefanpenner/ember-cli/pull/1825)
* [ENHANCEMENT] Injects the `/tests/index.html` with the test environment configuration (was previously whatever server was running). [#1825](https://github.com/stefanpenner/ember-cli/pull/1825)
* [ENHANCEMENT] `bower_components` and `vendor` are kept separate for import purposes. When we moved to separate directories for
  `bower_components/` and `vendor/` in 0.0.41, to allow for users to upgrade easier we merged those two folders into one single `vendor`
  tree.  This meant that you would still `app.import('vendor/baz/foo.js')` and `import Foo from 'vendor';` even if the file actually resides in
  `bower_components/`. This lead to much confusion and forced users to understand the internals that are going on (merging the two directories into `vendor/`).
  Now you would import things from `bower_components/` or `vendor/` if that is where they were on disk. [#1836](https://github.com/stefanpenner/ember-cli/pull/1836)
* [BUGFIX] Allow nested output path, if path does not previously exist. [#1872](https://github.com/stefanpenner/ember-cli/pull/1872)
* [ENHANCEMENT] Update `ember-cli-qunit` to 0.1.0. To avoid vendoring files in the addon and prevent having to run `bower install` within the addon
  itself (in a `postinstall` hook) the addon now installs its required packages directly into the applications `bower.json` file.
  This speeds up the build and make addon development much easier.  [#1877](https://github.com/stefanpenner/ember-cli/pull/1877)

#### Blueprints

* [BUGFIX] `ember g http-proxy` does not truncate the base path on proxied requests. [#1874](https://github.com/stefanpenner/ember-cli/pull/1874)
* [ENHANCEMENT] Add empty function to `ember g resource` generator. [#1817](https://github.com/stefanpenner/ember-cli/pull/1817)
* [ENHANCEMENT] Add {{outlet}} by default when generating a route template. [#1819](https://github.com/stefanpenner/ember-cli/pull/1819)
* [ENHANCEMENT] Remove use of deprecated `view.state` property. [#1826](https://github.com/stefanpenner/ember-cli/pull/1826)
* [BUGFIX] Allow blueprints without files. [#1829](https://github.com/stefanpenner/ember-cli/pull/1829)
* [ENHANCEMENT] Make `ember g adapter` extend from application adapter if present. [#1831](https://github.com/stefanpenner/ember-cli/pull/1831)
* [ENHANCEMENT] Add --base-class options to `ember g adapter`. [#1831](https://github.com/stefanpenner/ember-cli/pull/1831)
* [BUGFIX] Quote module name in object literal for `ember g http-mock`. [#1823](https://github.com/stefanpenner/ember-cli/pull/1823)
* [ENHANCEMENT] Add `Blueprint.prototype.addBowerPackageToProject`. [#1830](https://github.com/stefanpenner/ember-cli/pull/1830)
* [ENHANCEMENT] Add `Blueprint.prototype.insertIntoFile`. [#1857](https://github.com/stefanpenner/ember-cli/pull/1857)

#### Addons

* [`ember addon` diff](https://github.com/kellyselden/ember-addon-output/commit/165ee8069d32fc41bacb943bcb82acc691d0c628)
* [ENHANCEMENT] Expose Addon.prototype.isDevelopingAddon function. [#1785](https://github.com/stefanpenner/ember-cli/pull/1785)
* [ENHANCEMENT] Expose Addon.prototype.treeGenerator function, that automatically handles the
  returning an unwatchedTree vs the bare directory (therefore causing it to be watched). [#1785](https://github.com/stefanpenner/ember-cli/pull/1785)
* [ENHANCEMENT] Add default `Addon.prototype.blueprintsPath` implementation. Now if an addon has a `blueprints/` folder, it will be automatically used
  as the `blueprintsPath`. [#1876](https://github.com/stefanpenner/ember-cli/pull/1876)

### 0.0.42

#### Applications

* [`ember new` diff](https://github.com/kellyselden/ember-cli-output/commit/0e09ae86aa742450d4d88e72fc875a82c524e11f)
* [ENHANCEMENT] Throw an error if an Addon does not specify a name. [#1741](https://github.com/stefanpenner/ember-cli/pull/1741)
* [ENHANCEMENT] Extract `CoreObject` into a standalone package (`core-object`). [#1752](https://github.com/stefanpenner/ember-cli/pull/1752)
* [ENHANCEMENT] Set a default `baseURL` in `test` to allow `testem` to function properly with a custom `baseURL` specified. [#1748](https://github.com/stefanpenner/ember-cli/pull/1748)
* [BUGFIX] Update `broccoli-concat` to solve a performance issue with the recent addon changes (allows better caching when no changes are detected). [#1757](https://github.com/stefanpenner/ember-cli/pull/1757) and [#1766](https://github.com/stefanpenner/ember-cli/pull/1766)
* [BUGFIX] Bring `.bowerrc` file back for `app` blueprint. Helps alleviate upgrade issues, and ensures a parent directories `.bowerrc` cannot break an ember-cli app. [#1761](https://github.com/stefanpenner/ember-cli/pull/1761)
* [ENHANCEMENT] Update and clarify the default project README. [#1768](https://github.com/stefanpenner/ember-cli/pull/1768)
* [BUGFIX] Ensure that `app.import`'ed assets can be properly watched (and trigger a reload upon change). [#1774](https://github.com/stefanpenner/ember-cli/pull/1774)
* [BUGFIX] Ensure that `postBuild` hook is called on addons during `ember build`. [#1775](https://github.com/stefanpenner/ember-cli/pull/1775)
* [BREAKING ENHANCEMENT] Enabled automatic reloads on `config/environment.js` changes. [#1777](https://github.com/stefanpenner/ember-cli/pull/1777)
* [BREAKING ENHANCEMENT] Export the current configuration to a module (`my-app-name/config/environment'). [#1777](https://github.com/stefanpenner/ember-cli/pull/1777)

#### Addons

* [`ember addon` diff](https://github.com/kellyselden/ember-addon-output/commit/e716909c49c3b017385f4e128d3e5aa73a4558b6)

### 0.0.41

* [ENHANCEMENT] Allow calling `this._super.someMethodName()` in subclasses of CoreObject. [#1721](https://github.com/stefanpenner/ember-cli/pull/1721)
* [ENHANCEMENT] `.jshintrc`: disable esnext Promise global (prevents issues when RSVP Promise was intended but
  (non-universally-implemented) global Promise was used instead. [#1723](https://github.com/stefanpenner/ember-cli/pull/1723)
* [BUGFIX] Prevent deletion of files when invalid output-path is provided. [#1649](https://github.com/stefanpenner/ember-cli/pull/1649)
* [BUGFIX] Fix the /tests URL in IE8. [#1707](https://github.com/stefanpenner/ember-cli/pull/1707)
* [ENHANCEMENT] Remove `.bowerrc` file from application blueprint (will still use directory specified in `.bowerrc`, but uses the default
  of `bower_components/` if no `.bowerrc` exists). [#1679](https://github.com/stefanpenner/ember-cli/pull/1679)
* [BUGFIX] Fixes support for `.ember-cli` settings file. [#1676](https://github.com/stefanpenner/ember-cli/pull/1676)
* [BUGFIX] Blueprint: replace multiple occurences of `__name__` with module name. [#1658](https://github.com/stefanpenner/ember-cli/pull/1658)
* [ENHANCEMENT] Replace internal live-reload middleware with addon. [#1643](https://github.com/stefanpenner/ember-cli/pull/1643)
* [ENHANCEMENT] Add .travis.yml to app blueprint. [#1636](https://github.com/stefanpenner/ember-cli/pull/1636)
* [ENHANCEMENT] Allow individual Blueprints to determine if an entity name is required. [#1631](https://github.com/stefanpenner/ember-cli/pull/1631)
* [ENHANCEMENT] Move `qunit` support into an addon. [#1295](https://github.com/stefanpenner/ember-cli/pull/1295)
* [BUGFIX] Running `ember new foo-bar --dry-run` does not create new directory. [#1602](https://github.com/stefanpenner/ember-cli/pull/1602)
* [ENHANCEMENT] Allow addons to return an `addon` tree that will be namespaced with the addons name. [#1544](https://github.com/stefanpenner/ember-cli/pull/1544)
* [BUGFIX] Ensure non `assets/` files can be served from `public/` or when added via `app.import` (using the `destDir`). [#1549](https://github.com/stefanpenner/ember-cli/pull/1549)
* [ENHANCEMENT] Update `ember-resolver` version (allows for components and their templates to be grouped together). [#1540](https://github.com/stefanpenner/ember-cli/pull/1540)
* [ENHANCEMENT] Update `testem` version. [#1539](https://github.com/stefanpenner/ember-cli/pull/1539)
* [ENHANCEMENT] Remove `originate` from application blueprint.
* [ENHANCEMENT] Add EditorConfig file to blueprints. [#1507](https://github.com/stefanpenner/ember-cli/pull/1507)
* [ENHANCEMENT] Add `Blueprint#beforeInstall". [#1498](https://github.com/stefanpenner/ember-cli/pull/1498)
* [ENHANCEMENT] Add `--type` option (and check) to `controller` and `route` generators. [#1498](https://github.com/stefanpenner/ember-cli/pull/1498)
* [BUGFIX] Call `normalizeEntityName` hook before `locals` hook [#1717](https://github.com/stefanpenner/ember-cli/pull/1717)
* [ENHANCEMENT] replace multiple instances of __name__ in blueprints.
* [ENHANCEMENT] adds http-proxy for explicit, multi proxy use[#1474](https://github.com/stefanpenner/ember-cli/pull/1530)
* [BREAKING ENHANCEMENT] renames apiStub to http-mock to match other http-related generators [#1474] (https://github.com/stefanpenner/ember-cli/pull/1530)
* [ENHANCEMENT] Log proxy server traffic when using `ember serve --proxy` [#1583](https://github.com/stefanpenner/ember-cli/pull/1583)
* [ENHANCEMENT] Remove chain from express server [#1474](https://github.com/stefanpenner/ember-cli/pull/1474)
* [ENHANCEMENT] Remove Blueprint lookup failure stacktrace [#1476](https://github.com/stefanpenner/ember-cli/pull/1476)
* [ENHANCEMENT] --verbose errors option to have SilentError output stacktrace [#1480](https://github.com/stefanpenner/ember-cli/pull/1480)
* [BUGFIX] Modify service blueprint to create explicit injection [#1493](https://github.com/stefanpenner/ember-cli/pull/1493)
* [ENHANCEMENT] Generating a helper now also generates a test [#1503](https://github.com/stefanpenner/ember-cli/pull/1503)
* [BUGFIX] Do not run JSHint against trees returned from an addon.
* [BREAKING ENHANCEMENT] Addons can pull in test assets into test tree [#1453](https://github.com/stefanpenner/ember-cli/pull/1453)
* [BREAKING ENHANCEMENT] Addon model's \_root renamed to root [#1537](https://github.com/stefanpenner/ember-cli/pull/1537)
* [ENHANCEMENT] Addons can recursively add other addons [#1509](https://github.com/stefanpenner/ember-cli/pull/1509)
* [ENHANCEMENT] Upgrade `loader.js` to `1.0.1`. [#1543](https://github.com/stefanpenner/ember-cli/pull/1543)
* [BUGFIX] Allow `public/` to contain files in the root of the project. [#1549](https://github.com/stefanpenner/ember-cli/pull/1549)
* [ENHANCEMENT] Add `robots.txt` and `crossdomain.xml` files in the root of the project. [#1550](https://github.com/stefanpenner/ember-cli/pull/1550)
* [BUGFIX] Generating mixins and utils with several levels of nesting no longer produces a failing test. [#1551](https://github.com/stefanpenner/ember-cli/pull/1551)
* [BREAKING ENHANCEMENT] bower assets moved to bower_components instead of vendor [#1436](https://github.com/stefanpenner/ember-cli/pull/1436)
* [ENHANCEMENT] Move history support into a separate internal addon. [#1552](https://github.com/stefanpenner/ember-cli/pull/1552)
* [ENHANCEMENT] don't assume value of bowerrc.directory [#1553](https://github.com/stefanpenner/ember-cli/pull/1553)
* [ENHANCEMENT] es6 namespaced addons [#1544](https://github.com/stefanpenner/ember-cli/pull/1544)
* [ENHANCEMENT] Removed use of `memoize` from EmberApp. Allows multiple EmberApps to be instantiated [#1361](https://github.com/stefanpenner/ember-cli/issues/1361)
* [ENHANCEMENT] Add `ember destroy` command (removes files added by `generate` command). [#1547](https://github.com/stefanpenner/ember-cli/pull/1547)
* [BUGFIX] Ensure router.js is not modified when ember g route foo --dry-run [#1570](https://github.com/stefanpenner/ember-cli/pull/1570)
* [ENHANCEMENT] Add possibility to hide #ember-testing-container while testing [#1579](https://github.com/stefanpenner/ember-cli/pull/1579)
* [BUGFIX] Fix EmberAddon vendor tree [#1606](https://github.com/stefanpenner/ember-cli/pull/1606)
* [ENHANCEMENT] Addon blueprint [#1374](https://github.com/stefanpenner/ember-cli/pull/1374)
* [BUGFIX] Fix addons with empty directories [#]()
* [BUGFIX] Fix tests/helpers/start-app.js location from addon generator [#1626](https://github.com/stefanpenner/ember-cli/pull/1626)
* [BUGFIX] Allow addons to use history support middleware [#1632](https://github.com/stefanpenner/ember-cli/pull/1632)
* [ENHANCEMENT] Upgrade `broccoli-ember-hbs-template-compiler` to `1.6.1`.
* [ENHANCEMENT] Allow file patterns to be ignored by LiveReload [#1706](https://github.com/stefanpenner/ember-cli/pull/1706)
* [BUGFIX] Switch to OS-friendly line endings [#1718](https://github.com/stefanpenner/ember-cli/pull/1718)
* [BUGFIX] Prevent file deletions when the build `--output-path` is a parent directory [#1730](https://github.com/stefanpenner/ember-cli/pull/1730)

### 0.0.40

* [BUGFIX] fix detection of static files to allow periods in urls [#1399](https://github.com/stefanpenner/ember-cli/pull/1399)
* [BUGFIX] fix processing of import statements in css [#1400](https://github.com/stefanpenner/ember-cli/pull/1400)
* [BUGFIX] fix detection of requests to be proxied [#1263](https://github.com/stefanpenner/ember-cli/pull/1263)
* [BUGFIX] fix ember update (broken promises) [#1265](https://github.com/stefanpenner/ember-cli/pull/1265)
* [BUGFIX] eagerly requireing inquirer was costing ~100ms -> 150ms on boot [https://github.com/stefanpenner/ember-cli/commit/0ae78df5b4772b126facfed1d3203e9c695e80a1)
* [BUGFIX] Fix issue with invalid warnings (regarding files in the root of `vendor/`) on Windows. [#1264](https://github.com/stefanpenner/ember-cli/issues/1264)
* [BUGFIX] Fix addons being unable to use `app.import` to pull in non-js/css assets from their own `vendor/` tree. [#1159](https://github.com/stefanpenner/ember-cli/pull/1159)
* [ENHANCEMENT] When using `app.import` to import non-js/css assets, you can now specify the destination of the asset. [#1159](https://github.com/stefanpenner/ember-cli/pull/1159)
* [BUGFIX] Fix issue with `ember build` failing if the public/ folder was deleted. [#1270](https://github.com/stefanpenner/ember-cli/issues/1270)
* [BREAKING ENHANCEMENT] CoffeeScript support is now brought in by `ember-cli-coffeescript`. To use CoffeeScript with future versions run `npm install --save-dev ember-cli-coffeescript` (and `broccoli-coffee` is no longer needed as a direct dependency). [#1289](https://github.com/stefanpenner/ember-cli/pull/1289)
* [BUGFIX] `Blueprint.prototype.normalizeEntityName`'s return value should update the entity name. [#1283](https://github.com/stefanpenner/ember-cli/issues/1283)
* [BREAKING ENHANCEMENT] Move test only js/css assets into test-vendor.js and test-vendor.css respectively. [#1288](https://github.com/stefanpenner/ember-cli/pull/1288)
* [ENHANCEMENT] Update default Ember version to 1.6.0.
* [ENHANCEMENT] Display friendly error message when the server fails to start (e.g. address in use). [#1306](https://github.com/stefanpenner/ember-cli/pull/1306)
* [BREAKING ENHANCEMENT] Rename test-vendor.{css,js} to test-support.{css,js} to better reflect its role. [#1320](https://github.com/stefanpenner/ember-cli/pull/1320)
* [BUGFIX] Store version check information correctly, and only change the `lastVersionCheckAt` timestamp when the version is checked from npm. [#1323](https://github.com/stefanpenner/ember-cli/pull/1323)
* [BUGFIX] Update `broccoli-es3-safe-recast` to fix bugs with incorrectly replaced segments. [#1340](https://github.com/stefanpenner/ember-cli/pull/1340)
* [ENHANCEMENT] EmberApp can take jshintrc path options for app and test jshintrc files. [#1341](https://github.com/stefanpenner/ember-cli/pull/1341)
* [ENHANCEMENT] Using broccoli-sass > 0.2.0 now allows you to use .sass files. [#1367](https://github.com/stefanpenner/ember-cli/pull/1367)
* [ENHANCEMENT] EmberAddon constructor to build an EmberApp object with defaults for addon projects. [#1343](https://github.com/stefanpenner/ember-cli/pull/1343)
* [ENHANCEMENT] Allow addons to be vendored outside of node modules [#1370](https://github.com/stefanpenner/ember-cli/pull/1370)
* [ENHANCEMENT] Make "ember version" show NPM and Node version (versions of all loaded modules with "--verbose" switch). [#1307](https://github.com/stefanpenner/ember-cli/pull/1307)
* [BUGFIX] Duplicate-checking for generating routes now accounts for `"`-syntax. [#1371](https://github.com/stefanpenner/ember-cli/pull/1371)
* [BREAKING BUGFIX] Standard variables passed in to Blueprints now handle slashes better. Breaking if you relied on the old behavior. [#1278](https://github.com/stefanpenner/ember-cli/pull/1278)
* [BUGFIX] Generating a route named 'basic' no longer adds it to router.js. [#1390](https://github.com/stefanpenner/ember-cli/pull/1390)
* [ENHANCEMENT] EmberAddon constructor defaults `process.env.EMBER_ADDON_ENV` to "development". [#]()
* [ENHANCEMENT] Tests now run with the "test" environment by default, `config/environment.js` contains an (empty) section for the "test" environment [#1401](https://github.com/stefanpenner/ember-cli/pull/1401)
* [ENHANCEMENT] Add Git initialization to `ember new` command [#1369](https://github.com/stefanpenner/ember-cli/pull/1369)
* [ENHANCEMENT] Addons can export an object instead of a function [#1377](https://github.com/stefanpenner/ember-cli/pull/1377)
* [ENHANCEMENT] Addons will automatically load a generic addon constructor that includes app/vendor trees based on treesFor property if no main key is specified in package.json. [#1377](https://github.com/stefanpenner/ember-cli/pull/1377)
* [ENHANCEMENT] Disable `LOG_RESOLVER` flag to reduce console.log noise by default. [#1431](https://github.com/stefanpenner/ember-cli/pull/1431)
* [ENHANCEMENT] Update `broccoli-asset-rev`to `0.0.17`
* [ENHANCEMENT] Upgrade `ember-qunit` to `0.1.8`. [#1427](https://github.com/stefanpenner/ember-cli/pull/1427)
* [BUGFIX] Fix pod based templates (was broken with the advent of the `templates` tree). [#4138](https://github.com/stefanpenner/ember-cli/pull/1438)
* [ENHANCEMENT] ExpressServer middleware extracted to addons that are always pulled into every Project first [#1446](https://github.com/stefanpenner/ember-cli/pull/1446)

### 0.0.39

* [BUGFIX] `ember build --watch` should run until SIGTERM. [#1197](https://github.com/stefanpenner/ember-cli/issues/1197)
* [BUGFIX] Failed build should return non-zero exit code. [#1169](https://github.com/stefanpenner/ember-cli/pull/1169)
* [BUGFIX] improve startup time by up to 3x
* [BUGFIX] Ensure `ember generate` always operate in relation to project root. [#1165](https://github.com/stefanpenner/ember-cli/pull/1165)
* [ENHANCEMENT] Upgrade `ember-cli-ember-data` to `0.1.0`. [#1178](https://github.com/stefanpenner/ember-cli/pull/1178)
* [BUGFIX] Update `ember-cli-ic-ajax` to prevent warnings. [#1180](https://github.com/stefanpenner/ember-cli/pull/1180)
* [BUGFIX] Throw error when trailing slash present in argument to `ember generate`. [#1184](https://github.com/stefanpenner/ember-cli/pull/1184)
* [ENHANCEMENT] Don't expect `Ember` or `Em` to be global in tests. `Ember` or `Em` needs to be imported. [#1201](https://github.com/stefanpenner/ember-cli/pull/1201)
* [BUGFIX] Make behaviour of `--dry-run` more obvious & add `--skip-npm` and `--skip-bower`. [#1205](https://github.com/stefanpenner/ember-cli/pull/1205)
* [ENHANCEMENT] Remove .gitkeep files from `ember init` inside an existing project [#1209](https://github.com/stefanpenner/ember-cli/pull/1209)
* [ENHANCEMENT] Addons can add commands to the local `ember` command. [#1196](https://github.com/stefanpenner/ember-cli/pull/1196)
* [ENHANCEMENT] Addons can implement a postBuild hook. [#1215](https://github.com/stefanpenner/ember-cli/pull/1215)
* [ENHANCEMENT] Addons can add post-processing steps to the `Brocfile.js` process. [#1214](https://github.com/stefanpenner/ember-cli/pull/1214)
* [ENHANCEMENT] `broccoli-asset-rev` has been moved to an addon using the standard addon post-processing hooks. [#1214](https://github.com/stefanpenner/ember-cli/pull/1214)
* [ENHANCEMENT] Allow `app.toTree` to accept an array of additional trees to merge in the final output. [#1214](https://github.com/stefanpenner/ember-cli/pull/1214)
* [BUGFIX] Only run JSHint after preprocessing. [#1221](https://github.com/stefanpenner/ember-cli/pull/1221)
* [ENHANCEMENT] Addons can add blueprints. [#1222](https://github.com/stefanpenner/ember-cli/pull/1222)
* [ENHANCEMENT] Allow testing of production assets. [#1230](https://github.com/stefanpenner/ember-cli/pull/1230)
* [ENHANCEMENT] Provide Ember CLI version to Project model. [#1239](https://github.com/stefanpenner/ember-cli/pull/1239)
* [BREAKING ENHANCEMENT] Split `app/templates` into its own tree to prevent preprocessing template files as if they were JavaScript. [#1238](https://github.com/stefanpenner/ember-cli/pull/1238)
* [ENHANCEMENT] Print a warning when using `app.import` for assets in the root of `vendor/` (this is a significant performance penalty).
* [ENHANCEMENT] Model generation no longer requires an attribute type. [#1252](https://github.com/stefanpenner/ember-cli/pull/1252)
* [ENHANCEMENT] Allow vendor files to be configurable. [#1187](https://github.com/stefanpenner/ember-cli/pull/1187)


### 0.0.38

* accidentally deploy with node v0.0.11 which builds an invalid package

### 0.0.37

* [BREAKING BUGFIX] ensure the CLI exits with the correct status, fixes hanging tests and some non-graceful exit cleanups [#1150](https://github.com/stefanpenner/ember-cli/pull/1150)
* [BUGFIX] Ensure EDITOR is set before allowing edit in ember init. [#1090](https://github.com/stefanpenner/ember-cli/pull/1090)
* [BUGFIX] Display message to user when diff cannot be applied cleanly [#1091](https://github.com/stefanpenner/ember-cli/pull/1091)
* [ENHANCEMENT] Notify when an ember-cli update is available, and add `ember update` command. [#899](https://github.com/stefanpenner/ember-cli/pull/899)
* [BUGFIX] Ensure that build output directory is cleaned up properly. [#1122](https://github.com/stefanpenner/ember-cli/pull/1122)
* [BUGFIX] Ensure that non-zero exit code is used when running `ember test` with failing tests. [#1123](https://github.com/stefanpenner/ember-cli/pull/1123)
* [BREAKING ENHANCEMENT] Change the expected interface for the `./server/index.js` file. It now receives the instantiated `express` server. [#1097](https://github.com/stefanpenner/ember-cli/pull/1097)
* [ENHANCEMENT] Allow addons to provide server side middlewares. [#1097](https://github.com/stefanpenner/ember-cli/pull/1097)
* [ENHANCEMENT] Automatically pluralize the attribute when generating a model. [#1120](https://github.com/stefanpenner/ember-cli/pull/1120)
* [BUGFIX] Make sure non-dasherized model attributes are also added to generated tests. [#1120](https://github.com/stefanpenner/ember-cli/pull/1120)
* [ENHANCEMENT] Upgrade `ember-qunit-notifications` to `0.0.3`. [#1117](https://github.com/stefanpenner/ember-cli/pull/1117)
* [ENHANCEMENT] Allow addons to specify load ordering. [#1132](https://github.com/stefanpenner/ember-cli/pull/1132)
* [ENHANCEMENT] Adds `ember build --watch` [#1131](https://github.com/stefanpenner/ember-cli/pull/1131)
* [BREAKING ENHANCEMENT] Accept options as second parameter of ember-app#import. Pass modules as exports. [#1121](https://github.com/stefanpenner/ember-cli/pull/1121)

### 0.0.36

* deployed bundled package with outdated bundled depds... Likely user
  error (by @stefanpenner)

### 0.0.35

* [BUGFIX] Ensure that vendored JS files are concatted in a safe way (to prevent issues with ASI). [#988](https://github.com/stefanpenner/ember-cli/pull/988)
* [ENHANCEMENT] Use the `Project` model to load the project name and environment configuration (removes boilerplate from `Brocfile.js`). [#989](https://github.com/stefanpenner/ember-cli/pull/989)
* [BUGFIX] Pass `--port` option through when calling `ember test --port 8987` (allows overriding the port when running concurrent `ember test` commands). [#991](https://github.com/stefanpenner/ember-cli/pull/991)
* [ENHANCEMENT] Add `.ember-cli` configuration file. [#563](https://github.com/stefanpenner/ember-cli/pull/563)
* [ENHANCEMENT] Add edit capability to `ember init`. [#1000](https://github.com/stefanpenner/ember-cli/pull/1000)
* [ENHANCEMENT] Add the current environment to the application config (the `MyApplicationENV` global). [#1017](https://github.com/stefanpenner/ember-cli/pull/1017)
* [BUGFIX] Ensure that the project `.jshintrc` file is looked up in the project's root. [#1019](https://github.com/stefanpenner/ember-cli/pull/1019)
* [ENHANCEMENT] Allow addons to hook into the application build process. [#1025](https://github.com/stefanpenner/ember-cli/pull/1025)
* [ENHANCEMENT] Allow addons to register custom preprocessors. [#1030](https://github.com/stefanpenner/ember-cli/pull/1030)
* [BUGFIX] Prevent route blueprint adding duplicate entries to router.js [#1042](https://github.com/stefanpenner/ember-cli/pull/1042)
* [ENHANCEMENT] Add blueprint listing in ember help generate. [#952](https://github.com/stefanpenner/ember-cli/pull/952)
* [BUGFIX] Add missing descriptions for `build`, `serve`, and `test` commands. [#1045](https://github.com/stefanpenner/ember-cli/issues/1045)
* [ENHANCEMENT] Do not remove output directory. This allows easier cross-project symlinking (previous behavior broke the link when the output path was destroyed). [#1034](https://github.com/stefanpenner/ember-cli/pull/1034)
* [ENHANCEMENT] Keep output path (`/dist` by default) up to date with both `ember server` and `ember build`. [#1034](https://github.com/stefanpenner/ember-cli/pull/1034)
* [ENHANCEMENT] Use the `ember-cli-ic-ajax` addon to bring in ic-ajax. [#1047](https://github.com/stefanpenner/ember-cli/issues/1047)
* [ENHANCEMENT] Use the `ember-cli-ember-data` addon to bring in ember-data. [#1047](https://github.com/stefanpenner/ember-cli/issues/1047)
* [BUGFIX] Allow fingerprinting to be enabled/disabled in a more custom way. [#1066](https://github.com/stefanpenner/ember-cli/pull/1066)
* [ENHANCEMENT] Use `ember-addon` as the "addon" keyword. [#1071](https://github.com/stefanpenner/ember-cli/pull/1071)
* [ENHANCEMENT] loader should now support CJS mode of AMD.
* [ENHANCEMENT] Upgrade broccoli-asset-rev to 0.0.6 and allow passing a `customHash` in fingerprint options. [#1024](https://github.com/stefanpenner/ember-cli/pull/1024)

### 0.0.34

* [BUGFIX] broccoli-es6-safe-recast now once again has one-at-a-time semantics this improves incremental rebuild performance
* [BUGFIX] upgrade broccoli-sane-watcher to include better error messages when attempting to watch non-existent files
* [ENHANCEMENT] Allow opting out of `ES3SafeFilter`. [#966](https://github.com/stefanpenner/ember-cli/pull/966)
* [ENHANCEMENT] Provide `--watcher` option for switching between polling and events-based file watching. [#970](https://github.com/stefanpenner/ember-cli/pull/970)
* [BUGFIX] Ensure that tmp/ is cleaned up after running `ember server` or `ember test --server`. [#971](https://github.com/stefanpenner/ember-cli/pull/971)
* [BUGFIX] Fix errors with certain `generate` commands that depend on `inflection`. [f016820](https://github.com/stefanpenner/ember-cli/commit/f016820)
* [BUGFIX] Do not wrap `vendor` assets in eval when `wrapInEval` is set. [#983](https://github.com/stefanpenner/ember-cli/pull/983)
* [ENHANCEMENT] Use `wrapInEval` by default for application assets when running in development. [#983](https://github.com/stefanpenner/ember-cli/pull/983)
* [ENHANCEMENT] Add integration-test blueprint [#985](https://github.com/stefanpenner/ember-cli/pull/985)

### 0.0.33

* [BUGFIX] broccoli-sane-watcher now recovers after filters throw [#940](https://github.com/stefanpenner/ember-cli/pull/940)
* [ENHANCEMENT] Use ember-data.prod.js when ENV=production [#909](https://github.com/stefanpenner/ember-cli/pull/909).
* [BUGFIX] Ensure that config/environment is findable and required when setting up baseURL for server. [#916](https://github.com/stefanpenner/ember-cli/pull/916)
* [BUGFIX] Fix importing of non-JS/CSS [#915](https://github.com/stefanpenner/ember-cli/pull/915)
* [ENHANCEMENT] Use `window.MyProjectNameENV` instead of `window.ENV`. [#922](https://github.com/stefanpenner/ember-cli/pull/922)
* [BUGFIX] Disallow projects with periods in their name. [#927](https://github.com/stefanpenner/ember-cli/pull/927)
* [ENHANCEMENT] Allow customization of Javascript minification options. [#928](https://github.com/stefanpenner/ember-cli/pull/928)
* [BUGFIX] TestServer now waits until the build is done before starting. [#932](https://github.com/stefanpenner/ember-cli/pull/932)
* [ENHANCEMENT] Upgrade `leek` to `0.0.6`. [#934](https://github.com/stefanpenner/ember-cli/pull/934)
* [BUGFIX] `leek` upgrade fixes [#642](https://github.com/stefanpenner/ember-cli/issues/642), [#709](https://github.com/stefanpenner/ember-cli/issues/709)
* [ENHANCEMENT] Allow disabling of automatic fingerprinting. [#930](https://github.com/stefanpenner/ember-cli/pull/930)
* [ENHANCEMENT] Update ember-cli-shims to add `ember-data` shim. [#941](https://github.com/stefanpenner/ember-cli/pull/941)
* [ENHANCEMENT] Update default jshint settings to require importing Ember. [#941](https://github.com/stefanpenner/ember-cli/pull/941)
* [ENHANCEMENT] Bring generators in-house via blueprints. [#747](https://github.com/stefanpenner/ember-cli/pull/747)
* [BUGFIX] Only process application code with ES3SafeFilter. [#949](https://github.com/stefanpenner/ember-cli/pull/949)
* [ENHANCEMENT] Separate application code from vendor code. Generate `/assets/vendor.js` for vendored code. [#949](https://github.com/stefanpenner/ember-cli/pull/949)
* [ENHANCEMENT] Provide `registry` access from `EmberApp`. [#955](https://github.com/stefanpenner/ember-cli/pull/955)
* [BUGFIX] Ensure that `EmberENV` is setup (to allow enabling flagged features). [#958](https://github.com/stefanpenner/ember-cli/pull/958)

### 0.0.29

* [ENHANCEMENT] less CPU intensive watching thanks to @krisselden's https://github.com/krisselden/broccoli-sane-watcher and @amasad's https://github.com/amasad/sane
* [BUGFIX] Upgrade broccoli-es6-concatenator to 0.1.6 to fix a concatenation issue. [broccoli-es6-concatenator#17](https://github.com/joliss/broccoli-es6-concatenator/pull/17)
* [BUGFIX] prevent pointless event emitter memory leak warning [#850](https://github.com/stefanpenner/ember-cli/pull/850)
* [ENHANCEMENT] add and es3 safe transpile step: specifically promise.catch and promise.finally -> promise['catch'] & promise['finally']. In addition we cover afew more variables see: https://github.com/stefanpenner/es3-safe-recast [#823](https://github.com/stefanpenner/ember-cli/pull/823)
* [ENHANCEMENT] Load the vendor.css in the rendered HTML. [#728](http://github.com/stefanpenner/ember-cli/pull/728)
* [ENHANCEMENT] Allow `testem` port to be specified when running `ember test --server`. [#729](https://github.com/stefanpenner/ember-cli/pull/729)
* [BUGFIX] Use EMBER_ENV if specified in ENV_VARIABLES `EMBER_ENV=production ember build`. [#753](https://github.com/stefanpenner/ember-cli/pull/753)
* [ENHANCEMENT] If both EMBER_ENV and --environment are specified, use EMBER_ENV. [#753](https://github.com/stefanpenner/ember-cli/pull/753)
* [ENHANCEMENT] Update broccoli-jshint to 0.5.0 (more efficient caching for faster rebuilds). [#758](https://github.com/stefanpenner/ember-cli/pull/758)
* [ENHANCEMENT] Ensure that the `app/templates/components` directory is created automatically. [#761](https://github.com/stefanpenner/ember-cli/pull/761)
* [BUGFIX] For `ember-init`, Use app name if specified, over package.json or cwd name. [#792](https://github.com/stefanpenner/ember-cli/pull/792)
* [ENHANCEMENT] Add support for Web Notifications for QUnit test suite with ember-qunit-notifications. [#804](https://github.com/stefanpenner/ember-cli/pull/804)
* [BUGFIX] Ensure that files in app/ are JSHinted properly. [#832](https://github.com/stefanpenner/ember-cli/pull/832)
* [ENHANCEMENT] Update ember-load-initializers to 0.0.2.
* [ENHANCEMENT] Add broccoli-asset-rev for fingerprinting + source re-writing. [#814](https://github.com/stefanpenner/ember-cli/pull/814)
* [BUGFIX] Prevent broccoli from watching `node_modules/ember-cli/lib/broccoli/`. [#857](https://github.com/stefanpenner/ember-cli/pull/857)
* [BUGFIX] Prevent collision between running `ember server` and `ember test --server` simultaneously. [#862](https://github.com/stefanpenner/ember-cli/pull/862)
* [ENHANCEMENT] Show timing and slow tree listing for each rebuild. [#860](https://github.com/stefanpenner/ember-cli/pull/860) & [#865](https://github.com/stefanpenner/ember-cli/pull/865)
* [BUGFIX] Disable `wrapInEval` by default. [#866](//github.com/stefanpenner/ember-cli/pull/866)
* [ENHANCEMENT] Allow passing `tests` and `hinting` to `new EmberApp()`. [#876](https://github.com/stefanpenner/ember-cli/pull/876)
* [BUGFIX] Prevent slow tree printout during `ember test --server` from bleeding through `testem` UI.[#877](https://github.com/stefanpenner/ember-cli/pull/877)
* [ENHANCEMENT] Remove unused `vendor/_loader.js` file. [#880](https://github.com/stefanpenner/ember-cli/pull/880)
* [ENHANCEMENT] Allow disabling JSHint tests from within QUnit UI. [#878](https://github.com/stefanpenner/ember-cli/pull/878)
* [ENHANCEMENT] Upgrade `ember-resolver` to `0.1.1` (and lock down version in `bower.json`). [#885](https://github.com/stefanpenner/ember-cli/pull/885)

### 0.0.28

* [FEATURE] The `baseURL` in your `environment.js` now gets the leading and trailing slash automatically if you omit them. [#683](https://github.com/stefanpenner/ember-cli/pull/683)
* [FEATURE] The development server now serves the site under the specified `baseURL`. [#683](https://github.com/stefanpenner/ember-cli/pull/683)
* [FEATURE] Expose server: Bring back the API stub's functionality, give users the opportunity to add their own middleware. [#683](https://github.com/stefanpenner/ember-cli/pull/683)
* [ENHANCEMENT] `project.require()` can now be used to require files from the user's project. [#683](https://github.com/stefanpenner/ember-cli/pull/683)
* [ENHANCEMENT] Plugins can fall back to alternate file extensions (i.e scss, sass)
* [BUGFIX] Fix incorrect generation of all `vendor/` assets in build output. [#645](https://github.com/stefanpenner/ember-cli/pull/645)
* [ENHANCEMENT] Update to Broccoli 0.12. Prevents double initial rebuilds when running `ember server`. [#648](https://github.com/stefanpenner/ember-cli/pull/648)
* [BREAKING ENHANCEMENT] The generated `app.js` and `app.css` files are now named for your application name. [#638](https://github.com/stefanpenner/ember-cli/pull/638)
* [ENHANCEMENT] added first iteration of a slow but thorough acceptance
  test. A new app is generated, depedencies resolve, and the test for
  that base app are run.  [#614](https://github.com/stefanpenner/ember-cli/pull/614)
* [ENHANCEMENT] Use handlebars-runtime in production. [#675](https://github.com/stefanpenner/ember-cli/pull/675)
* [BUGFIX] Do not watch `vendor/` for changes (watching vendor dramatically increases CPU usage). [#693](https://github.com/stefanpenner/ember-cli/pull/693)
* [ENHANCEMENT] Minify CSS [#688](https://github.com/stefanpenner/ember-cli/pull/688)
* [ENHANCEMENT] Allows using app.import for things other than JS and CSS (i.e. fonts, images, json, etc). [#699](https://github.com/stefanpenner/ember-cli/pull/699)
* [BUGFIX] Fix `ember --help` output for test and version commands. [#701](https://github.com/stefanpenner/ember-cli/pull/701)
* [BUGFIX] Fix package.json preprocessor dependencies not being included in the registry. [#703](https://github.com/stefanpenner/ember-cli/pull/703)
* [BUGFIX] Update `testem` version to fix error thrown for certain assertions when running `ember test`, also fixes issue with `ember test --server` in Node 0.10. [#714](https://github.com/stefanpenner/ember-cli/pull/714)

### 0.0.27

* [BUGFIX] ` ENV.LOG_MODULE_RESOLVER` must be set pre-1.6 to get better container logging.
* [FEATURE] Added support for ember-scripts preprocessing.
* [ENHANCEMENT] Refactor `blueprint.js` to remove unnecessary variable assignment, change double iteration to simple reduce, and remove function that only swapped arguments and called through. [#537](https://github.com/stefanpenner/ember-cli/pull/537)
* [ENHANCEMENT] Refactor `test-loader.js` for readability and to prevent unnecessary iterations [#524](https://github.com/stefanpenner/ember-cli/pull/524)
* [ENHANCEMENT] Remove `Ember.setupForTesting` and
  `Router.reopen({location: 'none'});` from test helpers [#516](https://github.com/stefanpenner/ember-cli/pull/516).
* [ENHANCEMENT] Update loom-generators-ember-appkit to `^1.1.1`.
* [BUGFIX] Whitelist `ic-ajax` exports to prevent import validation warnings. [#533](https://github.com/stefanpenner/ember-cli/pull/533)
* [BUGFIX] `ember init` fails on `NULL_PROJECT` ([#546](https://github.com/stefanpenner/ember-cli/pull/546))
* [ENHANCEMENT] Files added by ember-cli should not needed to be specified in `Brocfile.js`. [#536](https://github.com/stefanpenner/ember-cli/pull/536)
* [ENHANCEMENT] Ensure minified output is using `compress` and `mangle` options with `uglify-js`. [#564](https://github.com/stefanpenner/ember-cli/pull/564)
* [BUGFIX] Update to Broccoli 0.10.0. This should resolve the primary issue `ember-cli` has on `Windows`. [#578](https://github.com/stefanpenner/ember-cli/pull/578)
* [ENHANCEMENT] Always Precompile Handlebars templates. [#574](https://github.com/stefanpenner/ember-cli/pull/574)
* [ENHANCEMENT] Update to Broccoli 0.11.0. This provides better timing information for `Watcher`. [#587](https://github.com/stefanpenner/ember-cli/pull/587)
* [ENHANCEMENT] Track rebuild timing. [#588](https://github.com/stefanpenner/ember-cli/pull/587)
* [ENHANCEMENT] Remove global defined helpers in favor of http://api.qunitjs.com/equal http://api.qunitjs.com/strictEqual/, etc. [#579](https://github.com/stefanpenner/ember-cli/pull/579)
* [BREAKING BUGFIX] No longer rely on `broccoli-bower` to automatically import vendored files. Use `app.import` to import dependencies and specify modules to whitelist. [#562](https://github.com/stefanpenner/ember-cli/pull/562)
* [ENHANCEMENT] Removed `proxy-url` and `proxy-host` parameters and introduced `proxy` param with full proxy url. ([#567](https://github.com/stefanpenner/ember-cli/pull/567))
* [BREAKING ENHANCEMENT] Update to jQuery 1.11.1. ** updates `bower.json`
* [ENHANCEMENT] When using non-NPM installed package (aka "running on master") the branch name and SHA are now printed along with the prior version number. [#634](https://github.com/stefanpenner/ember-cli/pull/634)

### 0.0.25

* [BUGFIX] The blueprinted application's `package.json` forces an older version of `ember-cli`. Fixed in [#518](https://github.com/stefanpenner/ember-cli/pull/518).

### 0.0.24

* Changes to `index.html`: Script tags were moved into body, `ENV` and the app are now defined in the same script tag.
* introduce NULL Project, to gracefully handle out-of-project
  invocations of the cli. Like new/init [fixes #502]
* pre 1.0.0 dependency are now locked down to exact versions, post 1.0.0 deps are in good faith semver locked.
* patch to quickfix some broccoli + Windows IO issues. We expect a proper solution soon, but this will hold us over (#493)[https://github.com/stefanpenner/ember-cli/pull/493]
* Add a custom watcher to make broccoli more usable on windows by catching file errors ([493](https://github.com/stefanpenner/ember-cli/pull/493)).
* Allow `ember new` and `ember init` to receive a `blueprint` argument to allow for alternative project scaffolding ([462](https://github.com/stefanpenner/ember-cli/pull/462))
* Add `ember test` with Testem integration ([388](https://github.com/stefanpenner/ember-cli/pull/388)).
* some improvements to bower dependency management, unfortunately until bower.json stabilizes broccoli-bower stability is at the whim of bower component authors.
* introduce maintainable + upgradable ember app specific brocfile filter
  ([396](https://github.com/stefanpenner/ember-cli/pull/396))
* ember cli now attempts to use the project-local ember-cli if
  available, this should help with people who have multiple versions of
  the cli installed. ([5a3c9a](https://github.com/stefanpenner/ember-cli/commit/5a3c9a97e407c128939feb5bd8cd98db2a8e3181))
* Complete restructuring of how ember-cli works internally
* `ember help` now offers nicely colored output
* Extracts shims in vendor into bower package ([#342](https://github.com/stefanpenner/ember-cli/pull/342))
  * locks it to version `0.0.1`
* Extracts initializers autoloading into bower package ([#337](https://github.com/stefanpenner/ember-cli/pull/337))
  * locks it to version `0.0.1`
* Introduces broccoli-bower ([#333](https://github.com/stefanpenner/ember-cli/pull/333))
  * locks it to version `0.2.0`
* Fix issue where app.js files are appended to tests.js ([#347](https://github.com/stefanpenner/ember-cli/pull/347))
* upgrade broccoli to `0.9.0` [v0.9.0 brocfile changes](https://gist.github.com/joliss/15630762fa0f43976418)
* Use configuration from `config/environments.js` to pass options to `Ember.Application.create`. ([#370](https://github.com/stefanpenner/ember-cli/pull/370))
* Adds `ic-ajax` to the list of ignored modules for tests([#378](https://github.com/stefanpenner/ember-cli/pull/378))
* Adds per command help output ([#376](https://github.com/stefanpenner/ember-cli/pull/376))
* Ensures that the broccoli trees are cleaned up properly. ([#444](https://github.com/stefanpenner/ember-cli/pull/444))
* Integrate leek package for ember-cli usage analytics reporting. ([#448](https://github.com/stefanpenner/ember-cli/pull/448))
* Generate current live build to `tmp/output/` when running `ember server`. This is very useful for
  debugging the current Broccoli tree without manually running `ember build`. ([#457](https://github.com/stefanpenner/ember-cli/pull/457))
* Use `tmp/output/` directory created in [#457](https://github.com/stefanpenner/ember-cli/pull/457) for Testem setup.
  This allows using the `testem` command to run Testem in server mode (allowing capturing multiple browsers and other goodies). [#463](https://github.com/stefanpenner/ember-cli/pull/463)
* Added `ember test --server` to run the `testem` command line server. `ember test --server` will automatically re-run your tests after a rebuild. [#474](https://github.com/stefanpenner/ember-cli/pull/474)
* Add JSHinting for `app/` and `test/` trees when building in development. This generates console logs as well as QUnit tests (so that `ember test` shows failures). [#482](https://github.com/stefanpenner/ember-cli/pull/482)
* Use the name specified in `package.json` while doing `ember init`. This allows you to use a different application name than your folder name. [#491](https://github.com/stefanpenner/ember-cli/pull/491)
* Allow disabling live reload via `ember server --live-reload=false`. [#510](https://github.com/stefanpenner/ember-cli/pull/510)

### 0.0.23

* Adds ES6 import validation ([#209](https://github.com/stefanpenner/ember-cli/pull/209))
* CSS broccoli fixes ([#325](https://github.com/stefanpenner/ember-cli/pull/325))
* Speed up boot ([#273](https://github.com/stefanpenner/ember-cli/pull/273))

### 0.0.22

* Makes sure that user cannot create an application named `test`([#256](https://github.com/stefanpenner/ember-cli/pull/256))
* Adds broccoli-merge-trees dependency and updates Brocfile to use it
* Locks blueprint to particular version of ember-cli, broccoli & friends:
  * ember-cli 0.0.21
  * broccoli (v0.7.2)
  * broccoli-es6-concatenator (v0.1.4)
  * broccoli-static-compiler (v0.1.4)
  * broccoli-replace version (v0.1.5)

### 0.0.21

* Use `loader.js` from `bower` ([0c1e8d28](https://github.com/stefanpenner/ember-cli/commit/0c1e8d28ca4bf6d24dc28af1fa4736690394eb5a))
* Drops implementation files ([54df0288](https://github.com/twokul/ember-cli/commit/54df0288cd456aec782f0cbda269c603fe7be005))
* Drop boilerplate tests ([c6f7475e](https://github.com/twokul/ember-cli/commit/c6f7475e0c8b3013b4af8ea5139aa25818aedeaf))
* Use named-amd version of `ic-ajax` ([#225](https://github.com/stefanpenner/ember-cli/pull/225))
* Separate `tests` and `app` code. Tests are now within 'assets/tests.js' (#220).
* Implement `--proxy-port` and `--proxy-host` parameters to `ember server` command (#40)
* Add support for `.ember-cli` file to provide default flags to commands ([7b90bd9](https://github.com/stefanpenner/ember-cli/commit/dfac84ffd27acedfd18189a0e4b0b5d3fb13bd7b))
* Ember initializers are required automatically ([#242](https://github.com/stefanpenner/ember-cli/pull/242))
* Supports alternate preprocessors (eg. broccoli-sass vs. broccoli-ruby-sass) ([59ddbd](https://github.com/stefanpenner/ember-cli/commit/59ddbdf4ce14e8f514d124e158cfdc9708026623))
* Also exposes `registerPlugin` method on preprocessor module that allows anyone to register additional plugins ([59ddbd](https://github.com/stefanpenner/ember-cli/commit/59ddbdf4ce14e8f514d124e158cfdc9708026623))

### 0.0.20

* Run tests through /tests.
* Integrate ember-qunit.
* Makes sure `livereload` reports error from `watcher` ([a1d447fe](https://github.com/stefanpenner/ember-cli/commit/a1d447fe654271f6cf4ea1e6b092a17bc6beed3a))
* Support multiple CSS Preprocessors ([LESS](http://lesscss.org/), [Sass](http://sass-lang.com/) and [Stylus](http://learnboost.github.io/stylus/))
* upgrade broccoli to 0.5.0. slight Brocfile syntax change:

  ```js
  var foo = makeTree("foo")
  // is now just
  var foo = "foo";
  ```
