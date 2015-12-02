
0.14.2 / Tue, 30 Jun 2015
=========================
  * [804b5b1](https://github.com/foreverjs/forever/commit/804b5b1) [dist] Version bump. 0.14.2 (`indexzero`)
  * [1e4953d](https://github.com/foreverjs/forever/commit/1e4953d) [fix] Do not break tests. (`indexzero`)
  * [310edd2](https://github.com/foreverjs/forever/commit/310edd2) fixes #699 (`Mike van Rossum`)
  * [19f7909](https://github.com/foreverjs/forever/commit/19f7909) Add license attribute (`Gilad Peleg`)
  * [36fbaf1](https://github.com/foreverjs/forever/commit/36fbaf1) isNaN() doesn't check if the value is numeric; it only checks if it is equal to NaN.  In particular it won't catch null. (`Craig R. Hughes`)
  * [31c2f16](https://github.com/foreverjs/forever/commit/31c2f16) Remove overwhelm, highlight forever start in doc (`Doug Carroll`)
  * [19a0de8](https://github.com/foreverjs/forever/commit/19a0de8) Update CHANGELOG.md. (`XhmikosR`)

v0.14.1 / Wed, 4 Feb 2015
=========================
  * [509eaf2](https://github.com/foreverjs/forever/commit/509eaf2) [dist] Version bump. 0.14.1 (`indexzero`)
  * [e5296a2](https://github.com/foreverjs/forever/commit/e5296a2) [minor] Small style change. (`indexzero`)
  * [c33f56e](https://github.com/foreverjs/forever/commit/c33f56e) fix critical bugs @v0.14.0 (`Tjatse`)
  * [545be49](https://github.com/foreverjs/forever/commit/545be49) [doc] Add docs badge to README (`René Föhring`)
  * [14bbf8c](https://github.com/foreverjs/forever/commit/14bbf8c) Fix links in readme (`jomo`)

v0.14.0 / Tue, 30 Dec 2014
==========================
  * [477c460](https://github.com/foreverjs/forever/commit/477c460) [dist] Version bump. 0.14.0 (`indexzero`)
  * [b1158de](https://github.com/foreverjs/forever/commit/b1158de) Fixed typos. (`Sean Hussey`)
  * [cdfa701](https://github.com/foreverjs/forever/commit/cdfa701) [refactor] Tidy the logic for handling data back from monitor processes [fix text] Assert the correct things in test/core/stopbypid-peaceful-test.js [dist minor] Correct file headers in some test files [dist minor] s/if(/if (/, s/){/) {/, and other minor whitespace (`indexzero`)
  * [b678eb7](https://github.com/foreverjs/forever/commit/b678eb7) test cases for `stop<all|bypid> peaceful` (`Tjatse`)
  * [c3baf77](https://github.com/foreverjs/forever/commit/c3baf77) if target is not a number, it could only be a script path, otherwise it is pid|uid|index|id (`Tjatse`)
  * [5c7ba63](https://github.com/foreverjs/forever/commit/5c7ba63) clean codes, and improve`findBy` performance (`Tjatse`)
  * [2c394ab](https://github.com/foreverjs/forever/commit/2c394ab) `forever stopbypid` is deprecated now, using `forever stop <pid>` instead. (`Tjatse`)
  * [d506771](https://github.com/foreverjs/forever/commit/d506771) [fix] handle monitor error, and make `forever stopall` peaceful (`Tjatse`)
  * [ab8bcb8](https://github.com/foreverjs/forever/commit/ab8bcb8) ignore .idea on MAC (`Tjatse`)
  * [1308a96](https://github.com/foreverjs/forever/commit/1308a96) [ci] try and fix build (`Jarrett Cruger`)
  * [68502f1](https://github.com/foreverjs/forever/commit/68502f1) fix missing parameter (`peecky`)
  * [b85f8e2](https://github.com/foreverjs/forever/commit/b85f8e2) [dist fix] Added CHANGELOG.md again. Fixes #630. (`indexzero`)

v0.13.0 / Tue, 4 Nov 2014
=========================
  * [8707877](https://github.com/foreverjs/forever/commit/8707877) [dist] Version bump. 0.13.0 (`indexzero`)
  * [3865596](https://github.com/foreverjs/forever/commit/3865596) [dist] Up-to-date linting with JSHint. Fixes #419. (`indexzero`)
  * [1d863ba](https://github.com/foreverjs/forever/commit/1d863ba) Renaming stoppid -> stopbypid (`Anthony Akentiev`)
  * [4adf834](https://github.com/foreverjs/forever/commit/4adf834) Little bug fix: comparing integers (`Anthony Akentiev`)
  * [c29de4b](https://github.com/foreverjs/forever/commit/c29de4b) README updated (`Anthony Akentiev`)
  * [54194df](https://github.com/foreverjs/forever/commit/54194df) stoppid command added to stop running under forever process by PID (`Anthony Akentiev`)
  * [c568f89](https://github.com/foreverjs/forever/commit/c568f89) [minor] Some small style changes to new(er) tests. (`indexzero`)
  * [8e4f1fb](https://github.com/foreverjs/forever/commit/8e4f1fb) code 0 should be treated as a Number too. (`Tjatse`)
  * [663e49a](https://github.com/foreverjs/forever/commit/663e49a) wait more... (`Tjatse`)
  * [1f184e8](https://github.com/foreverjs/forever/commit/1f184e8) test case for start/stop peaceful (`Tjatse`)
  * [8f7dfba](https://github.com/foreverjs/forever/commit/8f7dfba) [fix] relative script file should works fine, both with `start` or `stop`. (`Tjatse`)
  * [84cf5ad](https://github.com/foreverjs/forever/commit/84cf5ad) Add --workingDir option to specify the CWD of the process in which SCRIPT is run (`Myk Willis`)
  * [cb72aed](https://github.com/foreverjs/forever/commit/cb72aed) [dist] Update several dependencies to latest. (`indexzero`)

v0.12.0 / Thu, 30 Oct 2014
==========================
  * [b01eabb](https://github.com/foreverjs/forever/commit/b01eabb) [dist] Version bump. 0.12.0 (`indexzero`)
  * [9b6c8f7](https://github.com/foreverjs/forever/commit/9b6c8f7) [dist] Assign things to the author and the contributors. (`indexzero`)
  * [af8d228](https://github.com/foreverjs/forever/commit/af8d228) fixes EACCESS error with .sock (UNIX domain sockets) on Windows. Uses named pipes instead. (`Miroslav Mocek`)
  * [eecf6a2](https://github.com/foreverjs/forever/commit/eecf6a2) improved error handling (`Kevin "Schmidty" Smith`)
  * [6707a40](https://github.com/foreverjs/forever/commit/6707a40) improved error handling (`Kevin "Schmidty" Smith`)
  * [a2320aa](https://github.com/foreverjs/forever/commit/a2320aa) [minor] Do not check for a variable twice. (`indexzero`)
  * [283f210](https://github.com/foreverjs/forever/commit/283f210) [refactor] Update to `forever-monitor@1.4.0` and do not use the deprecated `.options` option. You can see why it is deprecated. (`indexzero`)
  * [73359e8](https://github.com/foreverjs/forever/commit/73359e8) [doc] Make note that the new root is actually NOT the default since it looks like it could be. (`indexzero`)
  * [a74e87c](https://github.com/foreverjs/forever/commit/a74e87c) [fix] inherits configuration from parent process when using `startDaemon` method. - make variable names camelCasing (`Tjatse`)
  * [1f9b7f7](https://github.com/foreverjs/forever/commit/1f9b7f7) Test case of startDaemon() method - configuration inheritance issue. (`Tjatse`)
  * [1102d11](https://github.com/foreverjs/forever/commit/1102d11) [fix] inherits configuration from parent process when using `startDaemon` method. (`Tjatse`)
  * [487fc54](https://github.com/foreverjs/forever/commit/487fc54) An 'error' on stopall is not actually an error (`Ryan Angilly`)
  * [250a4f8](https://github.com/foreverjs/forever/commit/250a4f8) [fix doc] More documentation around `forever.startServer`. Fixes #566. (`indexzero`)
  * [dfed754](https://github.com/foreverjs/forever/commit/dfed754) [fix] Set `forever.root` and `forever.config.get(root)` for symmetry. (`indexzero`)
  * [9eeeeb1](https://github.com/foreverjs/forever/commit/9eeeeb1) [fix doc] Update documentation. Fixes #594. (`indexzero`)
  * [35f477f](https://github.com/foreverjs/forever/commit/35f477f) [fix] Update documentation for `forever.list`. Fixes #598. (`indexzero`)
  * [c21f55d](https://github.com/foreverjs/forever/commit/c21f55d) [doc fix] Document FOREVER_ROOT environment variable. Properly respect -p. Fixes #548. Fixes #541. Fixes #568. (`indexzero`)
  * [0f227e5](https://github.com/foreverjs/forever/commit/0f227e5) [dist] Remove `foreverd` from scripts. Fixes #581. (`indexzero`)
  * [5fb6329](https://github.com/foreverjs/forever/commit/5fb6329) [dist breaking api] Remove `forever.service`. Fixes #372. (`indexzero`)
  * [938bf33](https://github.com/foreverjs/forever/commit/938bf33) [fix] Properly boolean-ize `--killTree`. Fixes #579. (`indexzero`)
  * [45f321c](https://github.com/foreverjs/forever/commit/45f321c) [fix] Actually support the documented `--uid` or `-u` CLI option. Fixes #424. (`indexzero`)
  * [3a40761](https://github.com/foreverjs/forever/commit/3a40761) Added uid information to help usage as per README (`brianmarco`)
  * [fefce03](https://github.com/foreverjs/forever/commit/fefce03) fixed wrong usage for option fifo (`lulurun`)
  * [a216e76](https://github.com/foreverjs/forever/commit/a216e76) checks proc.running and writes STOPPED instead of uptime if stopped (`smoodiver`)
  * [55141c8](https://github.com/foreverjs/forever/commit/55141c8) Adds id parameter as outlined in https://github.com/nodejitsu/forever/issues/461. (`Jackson Gariety`)
  * [99ee565](https://github.com/foreverjs/forever/commit/99ee565) [dist] Bump to `forever-monitor@1.3.0` (`indexzero`)
  * [99cddb5](https://github.com/foreverjs/forever/commit/99cddb5) [dist] Added .jshintrc (`indexzero`)
  * [16b1013](https://github.com/foreverjs/forever/commit/16b1013) use SVG to display Travis CI build testing status (`Mithgol`)
  * [b4a8135](https://github.com/foreverjs/forever/commit/b4a8135) fixing a small typo in the 'e.g.' portion of request, whoops. (`Andrew Martin`)
  * [a248968](https://github.com/foreverjs/forever/commit/a248968) updating docs with the uid flag (`Andrew Martin`)
  * [f730407](https://github.com/foreverjs/forever/commit/f730407) [dist] v0.11.1 (`Julian Duque`)
  * [23a217c](https://github.com/foreverjs/forever/commit/23a217c) Fix remark from @julianduque (`Ignat Kolesnichenko`)
  * [7b20f0f](https://github.com/foreverjs/forever/commit/7b20f0f) Slightly better English for the 'restarting' messages (`Dan Dascalescu`)
  * [af83c0e](https://github.com/foreverjs/forever/commit/af83c0e) Allow to get logFile and pidFile from config (`Ignat Kolesnichenko`)
  * [2cb60e8](https://github.com/foreverjs/forever/commit/2cb60e8) Update cli.js (`Kevin Hill`)

v0.11.0 / Thu, 10 Apr 2014
==========================
  * [09d8403](https://github.com/foreverjs/forever/commit/09d8403) [dist] Version bump. 0.11.0 (`Jarrett Cruger`)
  * [5e15626](https://github.com/foreverjs/forever/commit/5e15626) FIX: added FOREVER_ROOT variable (`srossross`)
  * [3cbabf4](https://github.com/foreverjs/forever/commit/3cbabf4) "forever start" hangs with node 0.11.9 (`jeromew`)
  * [7ff651b](https://github.com/foreverjs/forever/commit/7ff651b) Delete CHANGELOG.md (`Alexey Simonenko`)
  * [786271f](https://github.com/foreverjs/forever/commit/786271f) [dist] v0.10.11 (`Julian Duque`)
  * [7f4e4e9](https://github.com/foreverjs/forever/commit/7f4e4e9) [dist] Bump dependencies (`Julian Duque`)
  * [4822fec](https://github.com/foreverjs/forever/commit/4822fec) [fix] Trying to avoid the non-determinism in tests (`Julian Duque`)
  * [2e75aa1](https://github.com/foreverjs/forever/commit/2e75aa1) [fix] Add --killSignal to help (`Julian Duque`)
  * [b2b49d1](https://github.com/foreverjs/forever/commit/b2b49d1) [minor] Change order of option in help (`Julian Duque`)
  * [b0ec661](https://github.com/foreverjs/forever/commit/b0ec661) [dist] v0.10.10 (`Julian Duque`)
  * [bc48ca6](https://github.com/foreverjs/forever/commit/bc48ca6) [fix] Make vows happy (`Julian Duque`)
  * [2df789d](https://github.com/foreverjs/forever/commit/2df789d) Updated timespan to 2.1.0 (`Gabriel Petrovay`)
  * [70ab37e](https://github.com/foreverjs/forever/commit/70ab37e) Add --watchIgnore and colors (`Patrick Hogan`)
  * [a7d419c](https://github.com/foreverjs/forever/commit/a7d419c) Update README.md (`Jure`)
  * [2ba3158](https://github.com/foreverjs/forever/commit/2ba3158) Fixed watchIgnorePatterns assignment (`kbackowski`)
  * [acf59a7](https://github.com/foreverjs/forever/commit/acf59a7) Proper Revert "[fix] Make `-v|--version` work. Fixes #303." (`Maurycy Damian Wasilewski`)

v0.10.9 / Tue, 15 Oct 2013
==========================
  * [bc55bbf](https://github.com/foreverjs/forever/commit/bc55bbf) [dist] Bump version to 0.10.9 (`Maciej Małecki`)
  * [b4b0541](https://github.com/foreverjs/forever/commit/b4b0541) [dist] Use `forever-monitor@1.2.3` (`Maciej Małecki`)

v0.10.8 / Fri, 10 May 2013
==========================
  * [a4289d1](https://github.com/foreverjs/forever/commit/a4289d1) [dist] Bump version to 0.10.8 (`Maciej Małecki`)
  * [8afad64](https://github.com/foreverjs/forever/commit/8afad64) [ui dist] Output info about process being killed by signal (`Maciej Małecki`)

v0.10.7 / Sat, 27 Apr 2013
==========================
  * [22a3923](https://github.com/foreverjs/forever/commit/22a3923) [dist] Version bump. 0.10.7 (`indexzero`)
  * [6440b4e](https://github.com/foreverjs/forever/commit/6440b4e) [fix] remove duplicate option (`Julian Duque`)

v0.10.6 / Sun, 21 Apr 2013
==========================
  * [e8c48d4](https://github.com/foreverjs/forever/commit/e8c48d4) [dist] Version bump. 0.10.6 (`indexzero`)

v0.10.5 / Sun, 21 Apr 2013
==========================
  * [a9d7aa1](https://github.com/foreverjs/forever/commit/a9d7aa1) [dist] Version bump. 0.10.5 (`indexzero`)
  * [1a1ba32](https://github.com/foreverjs/forever/commit/1a1ba32) [fix] Make `-v|--version` work. Fixes #303. (`indexzero`)
  * [10fa40f](https://github.com/foreverjs/forever/commit/10fa40f) [fix dist] Bump to `nssocket@0.10.0` to support `node@0.10.x`. Update travis to test it. Fixes #370. Fixes #400. (`indexzero`)
  * [bd42888](https://github.com/foreverjs/forever/commit/bd42888) [fix] Manually merge #405. Fixes #405. (`indexzero`)
  * [d3675fa](https://github.com/foreverjs/forever/commit/d3675fa) process exit on error (`Noah H. Smith`)
  * [b641a4a](https://github.com/foreverjs/forever/commit/b641a4a) [minor] Style compliance for #403. Fixes #403. (`indexzero`)
  * [477082b](https://github.com/foreverjs/forever/commit/477082b) add the --watchIgnore option to be able to ignore files or directories when --watch is enabled (`Stéphane Gully`)
  * [5fa39ce](https://github.com/foreverjs/forever/commit/5fa39ce) [fix] Return `monitor` from `.startDaemon()`. Fixes #387. Fixes #389. (`indexzero`)
  * [bda8604](https://github.com/foreverjs/forever/commit/bda8604) [fix] Manually merge `plain-feature` because of trailing space noise. Fixes #381. [dist] Bump dependencies (`indexzero`)
  * [6047462](https://github.com/foreverjs/forever/commit/6047462) [fix] Added the default `dir` column which outputs the sourceDir from `forever-monitor`. Fixes #367. (`indexzero`)
  * [9cbe4cb](https://github.com/foreverjs/forever/commit/9cbe4cb) [fix dist] Update to the latest `forever-monitor`. Fixes #361. (`indexzero`)
  * [055c483](https://github.com/foreverjs/forever/commit/055c483) [fix] Warn users if `--minUptime` or `--spinSleepTime` are not specified. Fixes #344. (`indexzero`)
  * [1e4b2f6](https://github.com/foreverjs/forever/commit/1e4b2f6) added  and  cli options for streaming log output, updated README.md and tests to reflect changes (`John Lancaster`)
  * [94f61f5](https://github.com/foreverjs/forever/commit/94f61f5) removed trailing whitespace from lib/forever.js and lib/forever/cli.js ☠ (`John Lancaster`)
  * [352947e](https://github.com/foreverjs/forever/commit/352947e) Update package.json (`Thomas`)
  * [e442ea9](https://github.com/foreverjs/forever/commit/e442ea9) add no process error handling to cli.restartAll (`Evan You`)
  * [d3ff4bd](https://github.com/foreverjs/forever/commit/d3ff4bd) Add timestamp support to forever log (`Julian Duque`)
  * [b999bc2](https://github.com/foreverjs/forever/commit/b999bc2) Support exit signal customization (comes from another commit to forever-monitor) (`Alexander Makarenko`)
  * [3feef60](https://github.com/foreverjs/forever/commit/3feef60) Use `path` option as forever root if given. (`filipovskii_off`)
  * [3496b64](https://github.com/foreverjs/forever/commit/3496b64) use process.env.USERPROFILE as alternative to process.env.HOME (for windows) (`ingmr`)
  * [2b2ebbc](https://github.com/foreverjs/forever/commit/2b2ebbc) Fix uids mistakenly taken for an id (`Jazz`)
  * [e52b063](https://github.com/foreverjs/forever/commit/e52b063) wrapped fs.unlinkSync in try-catch-block (`Felix Böhm`)
  * [33dc125](https://github.com/foreverjs/forever/commit/33dc125) added a helpful error message (`Felix Böhm`)
  * [f69eb4d](https://github.com/foreverjs/forever/commit/f69eb4d) Updated flatiron dependency to 0.2.8 (`Ian Babrou`)
  * [4e7fa8f](https://github.com/foreverjs/forever/commit/4e7fa8f) pid variable not use. (`Thomas Tourlourat`)
  * [5b7f30b](https://github.com/foreverjs/forever/commit/5b7f30b) don't remove log files (`Felix Böhm`)
  * [a73eb5a](https://github.com/foreverjs/forever/commit/a73eb5a) remove pid- & logfiles on `exit` and `stop` (`Felix Böhm`)
  * [70a6acd](https://github.com/foreverjs/forever/commit/70a6acd) [api] Accept --killTree from CLI (`indexzero`)
  * [777256f](https://github.com/foreverjs/forever/commit/777256f) Update lib/forever.js (`Bram Stein`)

v0.10.1 / Sun, 8 Jul 2012
=========================
  * [4ed446f](https://github.com/foreverjs/forever/commit/4ed446f) [dist] Version bump. 0.10.1 (`indexzero`)
  * [df802d0](https://github.com/foreverjs/forever/commit/df802d0) [dist] Bump forever-monitor version (`indexzero`)

v0.10.0 / Sun, 8 Jul 2012
=========================
  * [c8afac3](https://github.com/foreverjs/forever/commit/c8afac3) [dist] Version bump. 0.10.0 (`indexzero`)
  * [c2baf66](https://github.com/foreverjs/forever/commit/c2baf66) [minor] Prefer no spaces when declaring Array instances (`indexzero`)
  * [9823d13](https://github.com/foreverjs/forever/commit/9823d13) [fix] Ensure pidFile is written to disk (and updated on restart) by bin/monitor (`indexzero`)
  * [1dfe0d0](https://github.com/foreverjs/forever/commit/1dfe0d0) [dist] Update dependencies to hard versions (`indexzero`)
  * [6921e6c](https://github.com/foreverjs/forever/commit/6921e6c) [refactor minor] Final integrations for `forever-monitor@1.0.1` (`indexzero`)
  * [f27cdaa](https://github.com/foreverjs/forever/commit/f27cdaa) [doc] Remove documenetation specific to `forever-monitor` (`indexzero`)
  * [d9e5faa](https://github.com/foreverjs/forever/commit/d9e5faa) [fix] Remove require for unused `ps-tree` (`indexzero`)
  * [14e5bda](https://github.com/foreverjs/forever/commit/14e5bda) [dist] Only support node@0.8.x (`indexzero`)
  * [91bda36](https://github.com/foreverjs/forever/commit/91bda36) [refactor] Examples are now in `forever-monitor` (`indexzero`)
  * [c1f1e6f](https://github.com/foreverjs/forever/commit/c1f1e6f) [dist] Remove outdated docco docs (`indexzero`)
  * [b5ce548](https://github.com/foreverjs/forever/commit/b5ce548) [refactor] Finish refactor of core Monitor functionality into `forever-monitor` (`indexzero`)
  * [5225d68](https://github.com/foreverjs/forever/commit/5225d68) [refactor] Moved test/core/check-process-test.js into `forever-monitor` (`indexzero`)
  * [b46c4c0](https://github.com/foreverjs/forever/commit/b46c4c0) [refactor] Remove all code in `forever-monitor` (`indexzero`)
  * [a5343df](https://github.com/foreverjs/forever/commit/a5343df) [fix] Use process.execPath for spawning. (`Charlie McConnell`)
  * [4ed1beb](https://github.com/foreverjs/forever/commit/4ed1beb) [fix] Use process.execPath instead of a hashbang. (`Charlie McConnell`)
  * [4f72f8c](https://github.com/foreverjs/forever/commit/4f72f8c) [fix] Fix bad require path. (`Charlie McConnell`)
  * [665e1ec](https://github.com/foreverjs/forever/commit/665e1ec) [test] Temporary: prevent test failure from deprecation warning in core. (`Charlie McConnell`)
  * [1e8d7ca](https://github.com/foreverjs/forever/commit/1e8d7ca) [refactor] Remove unused fork-shim (`Charlie McConnell`)
  * [a1e8f21](https://github.com/foreverjs/forever/commit/a1e8f21) [test] Only test on node 0.8.x (`Charlie McConnell`)
  * [b7c303a](https://github.com/foreverjs/forever/commit/b7c303a) [refactor] Implement silent fork via spawn stdio options. (`Charlie McConnell`)
  * [4fed919](https://github.com/foreverjs/forever/commit/4fed919) [refactor] Refactor to remove daemon.node (`Charlie McConnell`)
  * [485a18b](https://github.com/foreverjs/forever/commit/485a18b) [dist] Remove microtime dependency (`Charlie McConnell`)
  * [45a7e51](https://github.com/foreverjs/forever/commit/45a7e51) [dist] Remove `node-fork` dependency (`Maciej Małecki`)
  * [ba6b76d](https://github.com/foreverjs/forever/commit/ba6b76d) [test] Remove test for `forkShim` option (`Maciej Małecki`)
  * [16d1419](https://github.com/foreverjs/forever/commit/16d1419) [refactor api] Start using native fork (`Maciej Małecki`)
  * [d000278](https://github.com/foreverjs/forever/commit/d000278) [docs] Add Travis CI badge to README. (`Charlie McConnell`)
  * [2e2d18a](https://github.com/foreverjs/forever/commit/2e2d18a) [test] Add .travis.yml for Travis CI. (`Charlie McConnell`)

v0.9.2 / Mon, 11 Jun 2012
=========================
  * [02abd44](https://github.com/foreverjs/forever/commit/02abd44) [dist] Version bump v0.9.2 (`Charlie McConnell`)
  * [95d3e1a](https://github.com/foreverjs/forever/commit/95d3e1a) [minor] Remove unused argument. (`Charlie McConnell`)
  * [44490e6](https://github.com/foreverjs/forever/commit/44490e6) [test fix] Add missing .foreverignore test fixture. (`Charlie McConnell`)
  * [4245e54](https://github.com/foreverjs/forever/commit/4245e54) [fix] Update startOrRestart to fix bugs. (`Christian Howe`)
  * [592a1eb](https://github.com/foreverjs/forever/commit/592a1eb) Added `watchDirectory` to the list of options in README (to fulfill #271) (`Fedot Praslov`)
  * [cf5e5be](https://github.com/foreverjs/forever/commit/cf5e5be) [dist] Use daemon.node v0.5.x (`Charlie McConnell`)
  * [13ef52f](https://github.com/foreverjs/forever/commit/13ef52f) [dist] Fix maintainers field (`Christian Howe`)

v0.9.1 / Sat, 5 May 2012
========================
  * [75bfdab](https://github.com/foreverjs/forever/commit/75bfdab) [dist] Version bump v0.9.1 (`Charlie McConnell`)
  * [4116f85](https://github.com/foreverjs/forever/commit/4116f85) [fix] Pass argv options properly. (`Charlie McConnell`)
  * [44c2337](https://github.com/foreverjs/forever/commit/44c2337) closes #164 and #235 fix wrong usage of matchBase option of minimatch, use relative to watchDirectory path fore matching (`Oleg Slobodskoi`)
  * [2a7c477](https://github.com/foreverjs/forever/commit/2a7c477) Added watchDirectory to command line options (`Fedot Praslov`)
  * [8af6803](https://github.com/foreverjs/forever/commit/8af6803) [fix] Revert bad options commit. (`Charlie McConnell`)
  * [5d21f97](https://github.com/foreverjs/forever/commit/5d21f97) [fix] Fix unhandled `error` event in `forever stopall` (`Maciej Małecki`)
  * [49c2c47](https://github.com/foreverjs/forever/commit/49c2c47) [fix] Correct function name (`Maciej Małecki`)
  * [f3b119b](https://github.com/foreverjs/forever/commit/f3b119b) [dist] Version bump v0.9.0 (`Charlie McConnell`)
  * [b4798d8](https://github.com/foreverjs/forever/commit/b4798d8) [test fix] Make logger test more consistent. (`Charlie McConnell`)
  * [4848f90](https://github.com/foreverjs/forever/commit/4848f90) [test] Add test fixture for producing logging output. (`Charlie McConnell`)
  * [73b10be](https://github.com/foreverjs/forever/commit/73b10be) [test] New logging test for the new logging plugin. (`Charlie McConnell`)
  * [8ec0bce](https://github.com/foreverjs/forever/commit/8ec0bce) [fix] Restore stdout and stderr events, fix semantics of silent option. (`Charlie McConnell`)
  * [0b80e4d](https://github.com/foreverjs/forever/commit/0b80e4d) Minor wording fix (`Andrew Radev`)
  * [9c787df](https://github.com/foreverjs/forever/commit/9c787df) Stop or restart a process by its uid (`Andrew Radev`)
  * [af5b8c2](https://github.com/foreverjs/forever/commit/af5b8c2) [fix] cli pidFile text (`Bradley Meck`)
  * [1ad16b0](https://github.com/foreverjs/forever/commit/1ad16b0) [pull-request] #244, from @michaelcdillon (`Bradley Meck`)
  * [ea5317c](https://github.com/foreverjs/forever/commit/ea5317c) [doc] Remove unused `forever` option from docs (`Maciej Małecki`)
  * [8474c9c](https://github.com/foreverjs/forever/commit/8474c9c) [api] forkShim option should allow a string to say which module to use when shimming (rather than the one currently used by this process) (`Bradley Meck`)
  * [2f93ba4](https://github.com/foreverjs/forever/commit/2f93ba4) [fix] Destroy log file streams in a more intelligent way. (`Charlie McConnell`)
  * [89f3614](https://github.com/foreverjs/forever/commit/89f3614) [fix] Logging now survives child process restarts. (`Charlie McConnell`)
  * [2d7d462](https://github.com/foreverjs/forever/commit/2d7d462) [minor] Dont use optimist directly (`Joshua Holbrook`)
  * [cda371d](https://github.com/foreverjs/forever/commit/cda371d) [fix] Pass argvOptions to app.config.argv as well. (`Joshua Holbrook`)
  * [8529281](https://github.com/foreverjs/forever/commit/8529281) [refactor] Remove logging code from monitor. (`Charlie McConnell`)
  * [70ae4f4](https://github.com/foreverjs/forever/commit/70ae4f4) [refactor] Replace logging plugin. (`Charlie McConnell`)
  * [a6a1675](https://github.com/foreverjs/forever/commit/a6a1675) [fix] Remove duplicate alias. (`Charlie McConnell`)
  * [dd1508b](https://github.com/foreverjs/forever/commit/dd1508b) [fix] s/appendLog/append/g to make --append work. (`Charlie McConnell`)
  * [298ec73](https://github.com/foreverjs/forever/commit/298ec73) [fix] Restore optional logfile destination functionality. (`Charlie McConnell`)
  * [a0c9ac5](https://github.com/foreverjs/forever/commit/a0c9ac5) [fix] Restore self.warn method on monitor instance. (`Charlie McConnell`)
  * [114e378](https://github.com/foreverjs/forever/commit/114e378) [dist] Update package.json to use fork of daemon.node. (`Charlie McConnell`)
  * [8186994](https://github.com/foreverjs/forever/commit/8186994) [fix] Alter logging paths to reduce memory leakage and prevent stdio issues. (`Charlie McConnell`)
  * [5c8fcc5](https://github.com/foreverjs/forever/commit/5c8fcc5) [fix] Update forever.startDaemon to use adjusted daemon.node api. (`Charlie McConnell`)
  * [f44d5f4](https://github.com/foreverjs/forever/commit/f44d5f4) Fix worker crash from bad socket client (`Felix Geisendörfer`)
  * [b093bfc](https://github.com/foreverjs/forever/commit/b093bfc) [fix api] Expose `checkFile` and fix logical condition (`Maciej Małecki`)
  * [e154a50](https://github.com/foreverjs/forever/commit/e154a50) [fix] Pass options.argv instead of options (cli.js, line 188) (`Joshua Holbrook`)
  * [5aa16c3](https://github.com/foreverjs/forever/commit/5aa16c3) [dist] v0.8.5 (`Bradley Meck`)
  * [157ce7b](https://github.com/foreverjs/forever/commit/157ce7b) [fix] use env `bash` rather than bin/sh (`Bradley Meck`)
  * [205e6f3](https://github.com/foreverjs/forever/commit/205e6f3) [fix] EACCESS should still go to next() in `forever list` (`Bradley Meck`)

v0.8.4 / Sun, 15 Jan 2012
=========================
  * [6094c7c](https://github.com/foreverjs/forever/commit/6094c7c) [dist] Version bump. 0.8.4 (`indexzero`)
  * [1f4f5dc](https://github.com/foreverjs/forever/commit/1f4f5dc) [fix test] Make test/monitor/fork-test.js idempotent for processes created (`indexzero`)
  * [b6daac5](https://github.com/foreverjs/forever/commit/b6daac5) [dist] `node-fork@0.4.x` (`indexzero`)
  * [92d7dee](https://github.com/foreverjs/forever/commit/92d7dee) [doc] Update examples/cli-multiple-start (`indexzero`)
  * [1fa4943](https://github.com/foreverjs/forever/commit/1fa4943) [refactor] Create unique worker socket files using the `microtime` module (`indexzero`)
  * [72dac45](https://github.com/foreverjs/forever/commit/72dac45) [fix] Update bad reference variable to forever in watch plugin (`indexzero`)

v0.8.3 / Fri, 13 Jan 2012
=========================
  * [96277b7](https://github.com/foreverjs/forever/commit/96277b7) [dist] Version bump. 0.8.3. (`indexzero`)
  * [432a088](https://github.com/foreverjs/forever/commit/432a088) [fix] Allow for `forever set` to include values with `/` (i.e. directories) (`indexzero`)
  * [6bfe071](https://github.com/foreverjs/forever/commit/6bfe071) [fix test] try/catch around test/fixtures/* (`indexzero`)
  * [7064adb](https://github.com/foreverjs/forever/commit/7064adb) Show -a option when user met log file exist error. (`skyisle`)
  * [4e2ab81](https://github.com/foreverjs/forever/commit/4e2ab81) [fix] Don't leak `fs`, `path`, `nssocket`, `utile` and `forever` (`Maciej Małecki`)
  * [9b0ad25](https://github.com/foreverjs/forever/commit/9b0ad25) [fix] Don't leak `mkdirp` and `async` (`Maciej Małecki`)

v0.8.2 / Fri, 6 Jan 2012
========================
  * [6779342](https://github.com/foreverjs/forever/commit/6779342) [dist] Version bump. 0.8.2 (`indexzero`)
  * [6588f59](https://github.com/foreverjs/forever/commit/6588f59) [api test] Expose `.forkShim` for communicating between `0.6.x` and `0.4.x` processes (`indexzero`)
  * [82e2a7d](https://github.com/foreverjs/forever/commit/82e2a7d) [api refactor] Remove fork hack since we are now using `node-fork` (`indexzero`)
  * [a2c4313](https://github.com/foreverjs/forever/commit/a2c4313) [test fix] Fix test/worker/multiple-workers-test.js to pass on node@0.4.x and be idempotent (`indexzero`)
  * [63676ed](https://github.com/foreverjs/forever/commit/63676ed) [minor] Whitespace update (`indexzero`)
  * [a987826](https://github.com/foreverjs/forever/commit/a987826) [fix] Attempt to listen again if EADDRINUSE in forever.Worker (`indexzero`)
  * [d711ab8](https://github.com/foreverjs/forever/commit/d711ab8) [minor] Whitespace update (`indexzero`)

v0.8.1 / Thu, 5 Jan 2012
========================
  * [b15cd34](https://github.com/foreverjs/forever/commit/b15cd34) [dist] Version bump. 0.8.1 (`indexzero`)
  * [4e25765](https://github.com/foreverjs/forever/commit/4e25765) [fix] Print help on just `forever` (`indexzero`)
  * [0d5e893](https://github.com/foreverjs/forever/commit/0d5e893) [api] Added `forever restartall` and `forever.restartAll()`. Fixes #131 (`indexzero`)
  * [cef3435](https://github.com/foreverjs/forever/commit/cef3435) [doc fix] Update `.cleanup` to `.cleanUp`. Fixes #199 (`indexzero`)

v0.8.0 / Thu, 5 Jan 2012
========================
  * [53ba981](https://github.com/foreverjs/forever/commit/53ba981) [dist] Version bump. 0.8.0 (`indexzero`)
  * [7fc258c](https://github.com/foreverjs/forever/commit/7fc258c) [dist] Added @mmalecki to contributors (`indexzero`)
  * [93b3fd0](https://github.com/foreverjs/forever/commit/93b3fd0) [dist] Update node version to reflect backwards compatibility (`indexzero`)
  * [49de211](https://github.com/foreverjs/forever/commit/49de211) [dist test] Move test/fork-test.js to test/monitor/fork-test.js (`indexzero`)
  * [4ab4438](https://github.com/foreverjs/forever/commit/4ab4438) [fix] A couple of minor fixes to CLI edge cases (`indexzero`)
  * [b830218](https://github.com/foreverjs/forever/commit/b830218) [fix] Ensure `forever script.js` works (`indexzero`)
  * [285b659](https://github.com/foreverjs/forever/commit/285b659) [merge] Resolve bad cherry-pick from `fork` branch (`indexzero`)
  * [1f673f9](https://github.com/foreverjs/forever/commit/1f673f9) [fix] use node-fork for listing (`bradleymeck`)
  * [fa02258](https://github.com/foreverjs/forever/commit/fa02258) [fix] use node-fork so 0.6 can talk to 0.4 using the fork: true in combination with command (`bradleymeck`)
  * [b06d58b](https://github.com/foreverjs/forever/commit/b06d58b) [api] Expose `Monitor.fork` for using `child_process.fork()` (`indexzero`)
  * [2c6800a](https://github.com/foreverjs/forever/commit/2c6800a) [api] Expose `Monitor.fork` for using `child_process.fork()` (`indexzero`)
  * [7aa72c9](https://github.com/foreverjs/forever/commit/7aa72c9) [api test doc] Expose `.fork()` through forever for node-specific processes. Currently blocked by joyent/node#2454 (`indexzero`)
  * [1f78240](https://github.com/foreverjs/forever/commit/1f78240) [test minor] A couple of small updates for tests after recent API changes. Readd Worker.exitOnStop (`indexzero`)
  * [bde27e0](https://github.com/foreverjs/forever/commit/bde27e0) [refactor] Use the nssocket defined protocol for stopping and restarting worker processes (`indexzero`)
  * [dc0b457](https://github.com/foreverjs/forever/commit/dc0b457) [dist] Remove bin/forever-worker now that it `daemon.node` works again (`indexzero`)
  * [9cee338](https://github.com/foreverjs/forever/commit/9cee338) [wtf.node] BLACK VOODOO MAGIC. `daemon.node` somehow works even though libuv isnt fork(2)-safe (`indexzero`)
  * [ebd80a2](https://github.com/foreverjs/forever/commit/ebd80a2) [refactor] Attempt to spawn workers via bin/forever-worker. (`indexzero`)
  * [8f9f0ad](https://github.com/foreverjs/forever/commit/8f9f0ad) [refactor] Significant refactor to how forever works in the rewrite (`indexzero`)
  * [bca8ed9](https://github.com/foreverjs/forever/commit/bca8ed9) [test] Basic CLI test in `sh` (`Maciej Małecki`)
  * [a9247de](https://github.com/foreverjs/forever/commit/a9247de) [dist] Update `watch` to `watch@0.5` (`Maciej Małecki`)
  * [e57568b](https://github.com/foreverjs/forever/commit/e57568b) [test] Remove `cli` test (`Maciej Małecki`)
  * [9ff117d](https://github.com/foreverjs/forever/commit/9ff117d) [refactor] Move `daemon` to devDependencies on its way to deprecation (`indexzero`)
  * [84be160](https://github.com/foreverjs/forever/commit/84be160) [fix] Make logs work again (`Maciej Małecki`)
  * [d983726](https://github.com/foreverjs/forever/commit/d983726) [bin] Make `forever start` work with parameters (`Maciej Małecki`)
  * [55d96b2](https://github.com/foreverjs/forever/commit/55d96b2) [fix] Wrap parsing data from socket into `try .. catch` (`Maciej Małecki`)
  * [85c4542](https://github.com/foreverjs/forever/commit/85c4542) [minor] Remove unused `daemon` require (`Maciej Małecki`)
  * [321c182](https://github.com/foreverjs/forever/commit/321c182) [refactor] Replace `daemon.node` with `child_process.fork` (`Maciej Małecki`)
  * [df8d71d](https://github.com/foreverjs/forever/commit/df8d71d) [bin] Supress `stdout` and `stderr` when run as a fork (`Maciej Małecki`)
  * [2ead453](https://github.com/foreverjs/forever/commit/2ead453) [test] Test `kill` action (`Maciej Małecki`)
  * [a0d09d2](https://github.com/foreverjs/forever/commit/a0d09d2) [api] `kill` action for `Worker` (`Maciej Małecki`)
  * [6517f74](https://github.com/foreverjs/forever/commit/6517f74) [test] Add `MonitorMock.kill` (`Maciej Małecki`)
  * [883e712](https://github.com/foreverjs/forever/commit/883e712) [api] First pass at Worker integration (`Maciej Małecki`)
  * [bbc23e2](https://github.com/foreverjs/forever/commit/bbc23e2) [test] DRY tests a bit (`Maciej Małecki`)
  * [831f76f](https://github.com/foreverjs/forever/commit/831f76f) [api] Worker `spawn` command (`Maciej Małecki`)
  * [768f074](https://github.com/foreverjs/forever/commit/768f074) [api] If worker is a fork, notify master that it's listening (`Maciej Małecki`)
  * [cf716d5](https://github.com/foreverjs/forever/commit/cf716d5) [api] Guard for no options for Worker (`Maciej Małecki`)
  * [d174539](https://github.com/foreverjs/forever/commit/d174539) [test] Test if worker responds to `data` (`Maciej Małecki`)
  * [3059a9d](https://github.com/foreverjs/forever/commit/3059a9d) [api] Worker responds to `data` now (`Maciej Małecki`)
  * [e248716](https://github.com/foreverjs/forever/commit/e248716) [test] Add `data` property to `MonitorMock` (`Maciej Małecki`)
  * [748380b](https://github.com/foreverjs/forever/commit/748380b) [test] Don't hardcode socket path in tests (`Maciej Małecki`)
  * [d8b81dd](https://github.com/foreverjs/forever/commit/d8b81dd) [api] `Worker.start` calls back with socket path (`Maciej Małecki`)
  * [7be6917](https://github.com/foreverjs/forever/commit/7be6917) [test refactor] Restructure worker test a bit (`Maciej Małecki`)
  * [c710dc5](https://github.com/foreverjs/forever/commit/c710dc5) [test] Basic test for worker (`Maciej Małecki`)
  * [f06c345](https://github.com/foreverjs/forever/commit/f06c345) [api] Sketch of `Worker` (`Maciej Małecki`)
  * [34ccb24](https://github.com/foreverjs/forever/commit/34ccb24) [refactor] Remove watching code from `forever.Monitor` (`Maciej Małecki`)
  * [0e6ea8f](https://github.com/foreverjs/forever/commit/0e6ea8f) [test] Basic tests for `Logger` plugin (`Maciej Małecki`)
  * [f84634b](https://github.com/foreverjs/forever/commit/f84634b) [refactor] Add `Logger` plugin (`Maciej Małecki`)
  * [ab0f8e9](https://github.com/foreverjs/forever/commit/ab0f8e9) [refactor] Remove logging from `forever.Monitor` (`Maciej Małecki`)
  * [8a9af6b](https://github.com/foreverjs/forever/commit/8a9af6b) [refactor] Inherit from `broadway.App` (`Maciej Małecki`)
  * [d945bb2](https://github.com/foreverjs/forever/commit/d945bb2) [dist] Depend on `broadway` and `eventemitter2` (dev dep) (`Maciej Małecki`)
  * [cdb355f](https://github.com/foreverjs/forever/commit/cdb355f) [test] Add useful mocks (`Maciej Małecki`)
  * [ed75bd4](https://github.com/foreverjs/forever/commit/ed75bd4) [dist] Ignore vim swap files (`Maciej Małecki`)

v0.7.6 / Fri, 23 Dec 2011
=========================
  * [2ac0459](https://github.com/foreverjs/forever/commit/2ac0459) [dist] Version bump. 0.7.6. 0.4.x only. `forever >= 0.8.0` will be 0.6.x compatible (`indexzero`)
  * [88d9c20](https://github.com/foreverjs/forever/commit/88d9c20) [dist] Remove clip dependency (`indexzero`)
  * [2815f71](https://github.com/foreverjs/forever/commit/2815f71) [fix] Break apart cli.logs to support `forever logs` and `forever logs <script|index>` correctly (`indexzero`)
  * [72f4d14](https://github.com/foreverjs/forever/commit/72f4d14) [test] Update test fixture pathing mistake (`indexzero`)
  * [c6072f5](https://github.com/foreverjs/forever/commit/c6072f5) [dist] Remove console.error/log statements (`indexzero`)
  * [ed0d1e8](https://github.com/foreverjs/forever/commit/ed0d1e8) [fix minor] Fix 2 typos in forever service CLI (`Maciej Małecki`)
  * [079137c](https://github.com/foreverjs/forever/commit/079137c) [refactor] Refactor Forever service CLI (`Maciej Małecki`)
  * [c01abef](https://github.com/foreverjs/forever/commit/c01abef) [api] Export `cli.getOptions` (`Maciej Małecki`)
  * [13e8db8](https://github.com/foreverjs/forever/commit/13e8db8) [api] Expose `argvOptions` (`Maciej Małecki`)
  * [ee9f98b](https://github.com/foreverjs/forever/commit/ee9f98b) [doc fix] `--pidfile` is now called `--pidFile` (`Maciej Małecki`)
  * [1d1656c](https://github.com/foreverjs/forever/commit/1d1656c) [test refactor] `test/{helpers.js => helpers/macros.js}` (`Maciej Małecki`)
  * [ce7d5a1](https://github.com/foreverjs/forever/commit/ce7d5a1) [fix] Fix option parsing for starting actions (`Maciej Małecki`)
  * [fc4dec5](https://github.com/foreverjs/forever/commit/fc4dec5) Fixed broken link, replaced indexzero with nodejitsu in url. (`Louis Galipeau`)
  * [0812449](https://github.com/foreverjs/forever/commit/0812449) [fix] Respect `-c` on restart. Fixes #159 (`indexzero`)
  * [0e7873b](https://github.com/foreverjs/forever/commit/0e7873b) [fix] Improve the ordering of options parsing and include some options missed on the reparse. Fixes #139 (`indexzero`)

v0.7.5 / Fri, 2 Dec 2011
========================
  * [76b4d96](https://github.com/foreverjs/forever/commit/76b4d96) [dist] Version bump. 0.7.5 (`indexzero`)
  * [d6c7590](https://github.com/foreverjs/forever/commit/d6c7590) [minor] Always try to parse the response before calling next() (`indexzero`)
  * [dcbfc70](https://github.com/foreverjs/forever/commit/dcbfc70) [dist] Various small esoteric changes. Fixes #179 (`indexzero`)
  * [061d14f](https://github.com/foreverjs/forever/commit/061d14f) [fix doc] Fix README to match flatiron refactor (`Maciej Małecki`)
  * [517d31b](https://github.com/foreverjs/forever/commit/517d31b) [fix] Make option aliases work again (`Maciej Małecki`)
  * [63d91b2](https://github.com/foreverjs/forever/commit/63d91b2) [fix] Fix for pass-through parameters (`nconf@0.5`) (`Maciej Małecki`)
  * [e7e8fdf](https://github.com/foreverjs/forever/commit/e7e8fdf) prevent leading dashes in autogenerated log/pid filenames (`Brian Mount`)
  * [76bea57](https://github.com/foreverjs/forever/commit/76bea57) [fix] Fix `require`s in `foreverd` (`Maciej Małecki`)
  * [7cdca07](https://github.com/foreverjs/forever/commit/7cdca07) [fix] Make it compatible with `broadway@0.1.1` (`nconf@0.5`) (`Maciej Małecki`)
  * [791c123](https://github.com/foreverjs/forever/commit/791c123) [dist] Locked in nconf to v0.4.x. Bumped to v0.7.4. Should close #172 (`Marak Squires`)
  * [4ae63d0](https://github.com/foreverjs/forever/commit/4ae63d0) [merge] A few random missed conflicts from `git cherry-pick` on 22 commits. oops. (`indexzero`)
  * [60a576a](https://github.com/foreverjs/forever/commit/60a576a) [test fix] Since forever.kill is async, use `async.forEach`. Update test/cli-test.js to rimraf ~/.forever temporarily (`indexzero`)
  * [1a04002](https://github.com/foreverjs/forever/commit/1a04002) [fix] Make `--help` work (`Maciej Małecki`)
  * [58c251f](https://github.com/foreverjs/forever/commit/58c251f) [fix] Make column operations work (`Maciej Małecki`)
  * [b9c5f18](https://github.com/foreverjs/forever/commit/b9c5f18) [refactor minor] Code formatting, unused variable (`Maciej Małecki`)
  * [feade6c](https://github.com/foreverjs/forever/commit/feade6c) [test] Basic CLI tests with some helpers (`Maciej Małecki`)
  * [d6b6c58](https://github.com/foreverjs/forever/commit/d6b6c58) [fix] Reset system store before reparsing argv (`Maciej Małecki`)
  * [736fecb](https://github.com/foreverjs/forever/commit/736fecb) [test] Clean up after tests are done (`Maciej Małecki`)
  * [6b1a08d](https://github.com/foreverjs/forever/commit/6b1a08d) [test] Add test for option parsing (`Maciej Małecki`)
  * [a52ee8a](https://github.com/foreverjs/forever/commit/a52ee8a) [refactor] Make `forever app.js` work (`Maciej Małecki`)
  * [93359eb](https://github.com/foreverjs/forever/commit/93359eb) [refactor doc] Document `cli.startDaemon` and `cli.cleanLogs` (`Maciej Małecki`)
  * [93482cb](https://github.com/foreverjs/forever/commit/93482cb) [refactor minor] Remove unused `tty` require (`Maciej Małecki`)
  * [4d3958e](https://github.com/foreverjs/forever/commit/4d3958e) [refactor] Better option parsing (`Maciej Małecki`)
  * [dde31b7](https://github.com/foreverjs/forever/commit/dde31b7) [refactor bin] Remove options parsing from bin (`Maciej Małecki`)
  * [d793874](https://github.com/foreverjs/forever/commit/d793874) [api] Remove redudant `forever` options (`Maciej Małecki`)
  * [c9ab4f0](https://github.com/foreverjs/forever/commit/c9ab4f0) [dist] Add `flatiron` dependency (`Maciej Małecki`)
  * [8abe38d](https://github.com/foreverjs/forever/commit/8abe38d) [refactor] Implement pass-through options for child (`Maciej Małecki`)
  * [b30316e](https://github.com/foreverjs/forever/commit/b30316e) [refactor] Use `utile.randomString` (`Maciej Małecki`)
  * [dbf46c3](https://github.com/foreverjs/forever/commit/dbf46c3) [refactor fix] Pass options to `forever.start` (`Maciej Małecki`)
  * [3d262df](https://github.com/foreverjs/forever/commit/3d262df) [refactor] Add `help` command (`Maciej Małecki`)
  * [1da249c](https://github.com/foreverjs/forever/commit/1da249c) [fix] Fix `cli.start` regex to match .* instead of .+ (`Maciej Małecki`)
  * [89969ef](https://github.com/foreverjs/forever/commit/89969ef) [refactor] First pass on flatiron refactor (`Maciej Małecki`)
  * [8b05686](https://github.com/foreverjs/forever/commit/8b05686) [dist] Depend on `utile` (`Maciej Małecki`)
  * [71cf0de](https://github.com/foreverjs/forever/commit/71cf0de) [test fix] Kill child in `forever-test.js` (`Maciej Małecki`)

v0.7.3 / Thu, 17 Nov 2011
=========================
  * [865a8fd](https://github.com/foreverjs/forever/commit/865a8fd) [dist] Version bump. 0.7.3 (`indexzero`)
  * [7ab97bd](https://github.com/foreverjs/forever/commit/7ab97bd) always killTree (`Fabian Jakobs`)
  * [e4f2b09](https://github.com/foreverjs/forever/commit/e4f2b09) [dist] Update `watch` dependency. Fixes #155 (`indexzero`)
  * [5f20181](https://github.com/foreverjs/forever/commit/5f20181) [fix] give sigkills after a timeout given by options.killTTL in MS (`bradleymeck`)
  * [3f1ed35](https://github.com/foreverjs/forever/commit/3f1ed35) [test minor] Change `assert.length` to `assert.lengthOf` (`Maciej Małecki`)

v0.7.2 / Sat, 22 Oct 2011
=========================
  * [382f8e7](https://github.com/foreverjs/forever/commit/382f8e7) [dist] Version bump. 0.7.2 (`indexzero`)
  * [9131af7](https://github.com/foreverjs/forever/commit/9131af7) [fix] Return when no index or script is passed to `forever logs`. Fixes #141 (`indexzero`)
  * [8176f9f](https://github.com/foreverjs/forever/commit/8176f9f) Make sure all data is streamed before we try to parse it. (`Mariusz Nowak`)
  * [4ca2862](https://github.com/foreverjs/forever/commit/4ca2862) [dist] Remove unnecessary eyes dependency (`indexzero`)
  * [74f3140](https://github.com/foreverjs/forever/commit/74f3140) [fix] Prefer `-` to `$` in `forever.randomString` (`indexzero`)
  * [684296a](https://github.com/foreverjs/forever/commit/684296a) [test] Test `checkProcess` (`Maciej Małecki`)
  * [c17d004](https://github.com/foreverjs/forever/commit/c17d004) [refactor] Make `forever.checkProcess` synchronous (`Maciej Małecki`)
  * [f820056](https://github.com/foreverjs/forever/commit/f820056) [fix] Use `process.kill` to check if process is alive (`Maciej Małecki`)

v0.7.1 / Sun, 9 Oct 2011
========================
  * [d791422](https://github.com/foreverjs/forever/commit/d791422) [dist] Verion bump. 0.7.1 (`indexzero`)
  * [0d4f68e](https://github.com/foreverjs/forever/commit/0d4f68e) [fix] Pass proc.spawnWith to `forever.restart`. Fixes #116 (`indexzero`)

v0.7.0 / Sat, 8 Oct 2011
========================
  * [39f8b5a](https://github.com/foreverjs/forever/commit/39f8b5a) [dist] Version bump. 0.7.0 (`indexzero`)
  * [0baaccf](https://github.com/foreverjs/forever/commit/0baaccf) [dist] Updated CHANGELOG.md (`indexzero`)
  * [91dbd32](https://github.com/foreverjs/forever/commit/91dbd32) [api test] Expose `this.spawnWith` in Monitor.data (`indexzero`)
  * [14c82fd](https://github.com/foreverjs/forever/commit/14c82fd) [dist] Update daemon to >= 0.3.2 (`indexzero`)
  * [e740fb6](https://github.com/foreverjs/forever/commit/e740fb6) [doc] Update README.md for `forever logs *` commands (`indexzero`)
  * [0d6f85f](https://github.com/foreverjs/forever/commit/0d6f85f) [api test] Added `forever logs` CLI commands and `forever.tail()` method with appropriate tests. Fixes #123, #93 (`indexzero`)
  * [3d23311](https://github.com/foreverjs/forever/commit/3d23311) [minor] Minor whitespace fix (`indexzero`)
  * [02f7b0f](https://github.com/foreverjs/forever/commit/02f7b0f) [dist] Update `test` command in package.json (`indexzero`)
  * [fa03117](https://github.com/foreverjs/forever/commit/fa03117) [fix] Add the child PID to the list from `psTree` not remove it (`indexzero`)
  * [7ae3d1d](https://github.com/foreverjs/forever/commit/7ae3d1d) [doc] Updated CHANGELOG.md (`indexzero`)
  * [7c82d4b](https://github.com/foreverjs/forever/commit/7c82d4b) [dist] Update contributors in package.json (`indexzero`)
  * [067d50c](https://github.com/foreverjs/forever/commit/067d50c) [minor] Remove file headers in examples/* (`indexzero`)
  * [a942985](https://github.com/foreverjs/forever/commit/a942985) [dist] Update Copyright to Nodejitsu Inc. (`indexzero`)
  * [877ef3b](https://github.com/foreverjs/forever/commit/877ef3b) [minor] Update file headers (`indexzero`)
  * [a61e6be](https://github.com/foreverjs/forever/commit/a61e6be) [dist] Updates for JSHint in bin/* (`indexzero`)
  * [f7575f9](https://github.com/foreverjs/forever/commit/f7575f9) [dist] Update for JSHint (`indexzero`)
  * [4e27e3d](https://github.com/foreverjs/forever/commit/4e27e3d) [api] Expose `Monitor.killTree` for killing process trees for processes spawned by forever (`indexzero`)
  * [a83a1e1](https://github.com/foreverjs/forever/commit/a83a1e1) kill all children of a monitored process. (`Dominic Tarr`)
  * [89be252](https://github.com/foreverjs/forever/commit/89be252) [refactor test dist] Refactor /lib/foreverd/ into /lib/forever/service/ (`indexzero`)
  * [36e0b9b](https://github.com/foreverjs/forever/commit/36e0b9b) [minor] Updated foreverd for JSHint (`indexzero`)
  * [3525130](https://github.com/foreverjs/forever/commit/3525130) [minor] Update lib/forever* for JSHint (`indexzero`)
  * [1390910](https://github.com/foreverjs/forever/commit/1390910) [fix] forgot to add adapters (`bradleymeck`)
  * [bad47f6](https://github.com/foreverjs/forever/commit/bad47f6) [fix][WIP] basic working order, starting CLI cleanup (`bradleymeck`)
  * [6f68823](https://github.com/foreverjs/forever/commit/6f68823) [API][WIP] Moved service manager out to its own system (`bradleymeck`)
  * [61651a7](https://github.com/foreverjs/forever/commit/61651a7) [fix] daemonize ourselve on startup rather than rely on OS function (TODO exit codes) (`bradleymeck`)
  * [782cca7](https://github.com/foreverjs/forever/commit/782cca7) [fix] services should be added to run levels during install (`bradleymeck`)
  * [f2026b3](https://github.com/foreverjs/forever/commit/f2026b3) [fix] service process listing (`bradleymeck`)
  * [1bfdcdb](https://github.com/foreverjs/forever/commit/1bfdcdb) [fix] Use lsb functions for starting up a daemon (`bradleymeck`)
  * [60d4329](https://github.com/foreverjs/forever/commit/60d4329) [fix] make services use hyphenated commands (`bradleymeck`)
  * [93053d6](https://github.com/foreverjs/forever/commit/93053d6) [api] Revive the service api stubs (`bradleymeck`)

v0.6.9 / Tue, 4 Oct 2011
========================
  * [620a362](https://github.com/foreverjs/forever/commit/620a362) [dist] Version bump. 0.6.9 (`indexzero`)
  * [2b8cf71](https://github.com/foreverjs/forever/commit/2b8cf71) [doc] Add `--plain` option to README (`Maciej Małecki`)
  * [4b08542](https://github.com/foreverjs/forever/commit/4b08542) [bin] Add `--plain` option disabling CLI colors (`Maciej Małecki`)

v0.6.8 / Sat, 1 Oct 2011
========================
  * [dfb12a6](https://github.com/foreverjs/forever/commit/dfb12a6) [dist] Version bump. 0.6.8 (`indexzero`)
  * [7d7398b](https://github.com/foreverjs/forever/commit/7d7398b) [doc] Update README.md with watch file options (`indexzero`)
  * [8c8f0e0](https://github.com/foreverjs/forever/commit/8c8f0e0) [fix minor] A couple of small changes to merge in watch from @mmalecki (`indexzero`)
  * [d891990](https://github.com/foreverjs/forever/commit/d891990) [test] Add tests for watch (`Maciej Małecki`)
  * [f636447](https://github.com/foreverjs/forever/commit/f636447) [test] Add fixtures for watch test (`Maciej Małecki`)
  * [836ea31](https://github.com/foreverjs/forever/commit/836ea31) [fix minor] Use `path.join` (`Maciej Małecki`)
  * [b9b3129](https://github.com/foreverjs/forever/commit/b9b3129) [fix refactor] Use `watch.watchTree` function (`Maciej Małecki`)
  * [1b02785](https://github.com/foreverjs/forever/commit/1b02785) [fix minor] Remove stupid `options.watch || false` (`Maciej Małecki`)
  * [7ababd6](https://github.com/foreverjs/forever/commit/7ababd6) [bin] Add --watch/-w command line option (`Maciej Małecki`)
  * [e2b3565](https://github.com/foreverjs/forever/commit/e2b3565) [api] Add watchDirectory option (`Maciej Małecki`)
  * [b9d9703](https://github.com/foreverjs/forever/commit/b9d9703) [api] Complete file watching with .foreverignore (`Maciej Małecki`)
  * [28a7c16](https://github.com/foreverjs/forever/commit/28a7c16) [dist] Add minimatch dependency (`Maciej Małecki`)
  * [fff672d](https://github.com/foreverjs/forever/commit/fff672d) [api] simplest possible file watcher (ref #41) (`Maciej Małecki`)
  * [d658ee3](https://github.com/foreverjs/forever/commit/d658ee3) [dist] add watch dependency (`Maciej Małecki`)

v0.6.7 / Mon, 12 Sep 2011
=========================
  * [c87b4b3](https://github.com/foreverjs/forever/commit/c87b4b3) [dist] Version bump. 0.6.7 (`indexzero`)
  * [227b158](https://github.com/foreverjs/forever/commit/227b158) [refactor] replace sys module usages in examples with util (`Maciej Małecki`)
  * [8ae06c0](https://github.com/foreverjs/forever/commit/8ae06c0) [refactor test] replace sys module usages in tests with util (`Maciej Małecki`)
  * [72eba1f](https://github.com/foreverjs/forever/commit/72eba1f) [refactor] replace sys module usages with util (`Maciej Małecki`)
  * [00628c2](https://github.com/foreverjs/forever/commit/00628c2) [dist] Update winston version (`indexzero`)

v0.6.6 / Sun, 28 Aug 2011
=========================
  * [3f3cd17](https://github.com/foreverjs/forever/commit/3f3cd17) [dist] Version bump. 0.6.6 (`indexzero`)
  * [735fc95](https://github.com/foreverjs/forever/commit/735fc95) [minor test] Update to the `hideEnv` implementation from @bmeck. Added tests appropriately (`indexzero`)
  * [52c0529](https://github.com/foreverjs/forever/commit/52c0529) [style] cleanup unused variable (`Bradley Meck`)
  * [03daece](https://github.com/foreverjs/forever/commit/03daece) [api] Add options.hideEnv {key:boolean_hide,} to hide default env values (`Bradley Meck`)

v0.6.5 / Fri, 12 Aug 2011
=========================
  * [a3f0df5](https://github.com/foreverjs/forever/commit/a3f0df5) [dist] Version bump. 0.6.5 (`indexzero`)
  * [fdf15a0](https://github.com/foreverjs/forever/commit/fdf15a0) [api test] Update `forever.Monitor.prototype.restart()` to allow force restarting of processes in less than `.minUptime` (`indexzero`)

v0.6.4 / Thu, 11 Aug 2011
=========================
  * [f308f7a](https://github.com/foreverjs/forever/commit/f308f7a) [dist] Version bump. 0.6.4 (`indexzero`)
  * [9dc7bad](https://github.com/foreverjs/forever/commit/9dc7bad) [doc] Added example about running / listing multiple processes programmatically (`indexzero`)
  * [c3fe93a](https://github.com/foreverjs/forever/commit/c3fe93a) [fix] Update forever.startServer() to support more liberal arguments (`indexzero`)

v0.6.3 / Sat, 23 Jul 2011
=========================
  * [fa3b225](https://github.com/foreverjs/forever/commit/fa3b225) [dist] Version bump. 0.6.3 (`indexzero`)
  * [e47af9c](https://github.com/foreverjs/forever/commit/e47af9c) [fix] When stopping only respond with those processes which have been stopped. Fixes #87 (`indexzero`)
  * [e7b9e58](https://github.com/foreverjs/forever/commit/e7b9e58) [fix] Create `sockPath` if it does not exist already. Fixes #92 (`indexzero`)

v0.6.2 / Tue, 19 Jul 2011
=========================
  * [845ce2c](https://github.com/foreverjs/forever/commit/845ce2c) [dist] Version bump. 0.6.2 (`indexzero`)
  * [f756e62](https://github.com/foreverjs/forever/commit/f756e62) [fix] Display warning / error messages to the user when contacting UNIX sockets. Fixes #88 (`indexzero`)

v0.6.1 / Fri, 15 Jul 2011
=========================
  * [72f200b](https://github.com/foreverjs/forever/commit/72f200b) [dist] Version bump. 0.6.1 (`indexzero`)
  * [1c0792e](https://github.com/foreverjs/forever/commit/1c0792e) Process variables are not always available, for example if you execute forever with a different process like monit. (`Arnout Kazemier`)
  * [7ff26de](https://github.com/foreverjs/forever/commit/7ff26de) Fixed a bug where numbers in the file path caused forever to think that it should stop the script based on index instead of stopping it based on script. (`Arnout Kazemier`)

v0.6.0 / Mon, 11 Jul 2011
=========================
  * [df54bc0](https://github.com/foreverjs/forever/commit/df54bc0) [dist] Version bump. 0.6.0 (`indexzero`)
  * [8a50cf6](https://github.com/foreverjs/forever/commit/8a50cf6) [doc] Minor updates to README.md (`indexzero`)
  * [1dac9f4](https://github.com/foreverjs/forever/commit/1dac9f4) [doc] Updated README.md (`indexzero`)
  * [9d35315](https://github.com/foreverjs/forever/commit/9d35315) [fix minor] Update how forever._debug works. Use updated CLI options in `forever restart` (`indexzero`)
  * [da86724](https://github.com/foreverjs/forever/commit/da86724) [doc] Regenerate docco docs (`indexzero`)
  * [ad40a95](https://github.com/foreverjs/forever/commit/ad40a95) [doc] Added some code docs (`indexzero`)
  * [221c170](https://github.com/foreverjs/forever/commit/221c170) [doc] Update help in bin/forever (`indexzero`)
  * [091e949](https://github.com/foreverjs/forever/commit/091e949) [api] Finished fleshing out `forever columns *` commands (`indexzero`)
  * [581a132](https://github.com/foreverjs/forever/commit/581a132) [fix] Update `forever cleanlogs` for 0.6.x (`indexzero`)
  * [a39fee1](https://github.com/foreverjs/forever/commit/a39fee1) [api] Began work on `forever columns *` (`indexzero`)
  * [381ecaf](https://github.com/foreverjs/forever/commit/381ecaf) [api] Expose `forever.columns` and update `forever.format` to generate results dynamically (`indexzero`)
  * [bc8153a](https://github.com/foreverjs/forever/commit/bc8153a) [minor] Trim whitespace in lib/* (`indexzero`)
  * [2a163d3](https://github.com/foreverjs/forever/commit/2a163d3) [dist] Add `portfinder` dependency to package.json (`indexzero`)
  * [57a5600](https://github.com/foreverjs/forever/commit/57a5600) [doc] Remove references to *.fvr files in README.md (`indexzero`)
  * [ef59672](https://github.com/foreverjs/forever/commit/ef59672) [test] Updated tests for refactor in previous commit (`indexzero`)
  * [7ae870e](https://github.com/foreverjs/forever/commit/7ae870e) [refactor] **Major awesome breaking changes** Forever no longer uses *.fvr files in-favor of a TCP server in each forever process started by the CLI. Programmatic usage will require an additional call to `forever.createServer()` explicitally in order for your application to be available in `forever list` or `forever.list()` (`indexzero`)
  * [a26cf9d](https://github.com/foreverjs/forever/commit/a26cf9d) [minor] Catch `uncaughtException` slightly more intelligently (`indexzero`)
  * [4446215](https://github.com/foreverjs/forever/commit/4446215) [api] Include uids in `forever list` (`indexzero`)
  * [57bc396](https://github.com/foreverjs/forever/commit/57bc396) [minor] Create `options.uid` by default in `.startDaemon()` if it is already not provided (`indexzero`)
  * [dbf4275](https://github.com/foreverjs/forever/commit/dbf4275) [api] Default `minUptime` to 0 (`indexzero`)
  * [079ca20](https://github.com/foreverjs/forever/commit/079ca20) [doc] Small update to README.md (`indexzero`)
  * [aaefc95](https://github.com/foreverjs/forever/commit/aaefc95) [fix] use default values for log file and pid file (prevents a process from being nuked by being daemonized) (`Bradley Meck`)
  * [76be51e](https://github.com/foreverjs/forever/commit/76be51e) [fix] Quick fix for the last commit (`indexzero`)
  * [6902890](https://github.com/foreverjs/forever/commit/6902890) [api test] Added generic hooks for forever.Monitor (`indexzero`)
  * [c7ff2d9](https://github.com/foreverjs/forever/commit/c7ff2d9) [doc] Update the help in the forever CLI and README.md (`indexzero`)
  * [725d11d](https://github.com/foreverjs/forever/commit/725d11d) [doc] Update README.md (`indexzero`)
  * [5a8b32e](https://github.com/foreverjs/forever/commit/5a8b32e) [doc] Regenerated docco docs (`indexzero`)
  * [dfb54be](https://github.com/foreverjs/forever/commit/dfb54be) [api test doc] Remove deprecated `forever.Forever` from samples and tests. Added `env` and `cwd` options and associated tests. Some additional code docs and minor style changes (`indexzero`)
  * [c5c9172](https://github.com/foreverjs/forever/commit/c5c9172) [api] Update `forever list` to use cliff (`indexzero`)
  * [d2aa52b](https://github.com/foreverjs/forever/commit/d2aa52b) [dist] Drop eyes in favor of cliff (`indexzero`)
  * [bc5995f](https://github.com/foreverjs/forever/commit/bc5995f) [fix minor] Keep processes silent on `forever restart` if requested. A couple of minor log formatting updates (`indexzero`)
  * [f11610e](https://github.com/foreverjs/forever/commit/f11610e) [minor api] Update to optional debugging. Various small style updates (`indexzero`)
  * [686d009](https://github.com/foreverjs/forever/commit/686d009) [minor api] Added forever.debug for debugging purposes (`indexzero`)
  * [abed353](https://github.com/foreverjs/forever/commit/abed353) [doc] Updated README.md with newer options and events (`indexzero`)
  * [da44ad0](https://github.com/foreverjs/forever/commit/da44ad0) [doc] Kill some ancient stuff in README.md (`indexzero`)
  * [3ef90c1](https://github.com/foreverjs/forever/commit/3ef90c1) [doc] Add a little more color to documentation for `forever.load()` (`indexzero`)
  * [3d6018f](https://github.com/foreverjs/forever/commit/3d6018f) [doc] Update documentation on forever.load(). Fixes #72 (`indexzero`)
  * [3c8e6eb](https://github.com/foreverjs/forever/commit/3c8e6eb) [api fix] When executing stopall, dont kill the current process. Refactor flow-control of forever.cleanUp() (`indexzero`)
  * [d681cb7](https://github.com/foreverjs/forever/commit/d681cb7) [fix] Dont allow `-` in uuids generated by forever. Fixes #66. (`indexzero`)
  * [e0c3dcf](https://github.com/foreverjs/forever/commit/e0c3dcf) [dist] Minor style updates. Update to use pkginfo (`indexzero`)

v0.5.6 / Tue, 7 Jun 2011
========================
  * [de0d6d2](https://github.com/foreverjs/forever/commit/de0d6d2) [dist] Version bump. 0.5.6 (`indexzero`)

v0.5.5 / Tue, 31 May 2011
=========================
  * [4c5b73a](https://github.com/foreverjs/forever/commit/4c5b73a) [dist] Version bump. 0.5.5 (`indexzero`)
  * [1af1fe3](https://github.com/foreverjs/forever/commit/1af1fe3) [fix] Remove .fvr file when a forever.Monitor child exits (`indexzero`)

v0.5.4 / Mon, 30 May 2011
=========================
  * [4e84d71](https://github.com/foreverjs/forever/commit/4e84d71) [dist] Version bump. 0.5.4 (`indexzero`)
  * [5b2bf74](https://github.com/foreverjs/forever/commit/5b2bf74) [test] Update test/multiple-processes-test.js so that it doesnt leave zombie processes behind (`indexzero`)
  * [6d93dcc](https://github.com/foreverjs/forever/commit/6d93dcc) Add --spinSleepTime to throttle instead of killing spinning scripts (`Dusty Leary`)

v0.5.3 / Sun, 29 May 2011
=========================
  * [7634248](https://github.com/foreverjs/forever/commit/7634248) [dist] Version bump. 0.5.3 (`indexzero`)
  * [d6b0d0e](https://github.com/foreverjs/forever/commit/d6b0d0e) [test] Update tests to be consistent with new functionality (`indexzero`)
  * [921966a](https://github.com/foreverjs/forever/commit/921966a) [api] Improve forever when working with `-c` or `--command` (`indexzero`)
  * [349085d](https://github.com/foreverjs/forever/commit/349085d) [dist] Minor update to dependencies (`indexzero`)
  * [96c3f08](https://github.com/foreverjs/forever/commit/96c3f08) [dist] Update .gitignore for npm 1.0 (`indexzero`)
  * [f4982cd](https://github.com/foreverjs/forever/commit/f4982cd) [doc] Update README.md to still use -g (`indexzero`)
  * [3feb0bc](https://github.com/foreverjs/forever/commit/3feb0bc) [dist] Update package.json dependencies (`indexzero`)
  * [270d976](https://github.com/foreverjs/forever/commit/270d976) preferGlobal (`Dustin Diaz`)
  * [de90882](https://github.com/foreverjs/forever/commit/de90882) [doc] Update installation instructions with `-g` for npm 1.0 (`indexzero`)

v0.5.2 / Fri, 13 May 2011
=========================
  * [2c99741](https://github.com/foreverjs/forever/commit/2c99741) [dist] Version bump. 0.5.2 (`indexzero`)
  * [eab1c04](https://github.com/foreverjs/forever/commit/eab1c04) [fix] Check if processes exist before returning in `.findByScript()`. Fixes #50 (`indexzero`)
  * [e18a256](https://github.com/foreverjs/forever/commit/e18a256) [fix] Batch the cleaning of *.fvr and *.pid files to avoid file descriptor overload. Fixes #53 (`indexzero`)
  * [828cd48](https://github.com/foreverjs/forever/commit/828cd48) [minor] *print help when a valid action isn't given (`nlco`)

v0.5.1 / Sun, 1 May 2011
========================
  * [f326d20](https://github.com/foreverjs/forever/commit/f326d20) [dist] Version bump. 0.5.1. Add `eyes` dependency (`indexzero`)

v0.5.0 / Sun, 1 May 2011
========================
  * [7b451d9](https://github.com/foreverjs/forever/commit/7b451d9) [dist] Version bump. 0.5.0 (`indexzero`)
  * [1511179](https://github.com/foreverjs/forever/commit/1511179) [doc] Regenerated docco docs (`indexzero`)
  * [0fb8abe](https://github.com/foreverjs/forever/commit/0fb8abe) [minor] Small require formatting updates. Try to be more future-proof. (`indexzero`)
  * [3112380](https://github.com/foreverjs/forever/commit/3112380) [fix] Small fixes found from some upstream integrations (`indexzero`)
  * [9788748](https://github.com/foreverjs/forever/commit/9788748) [fix] Better handling of bookkeeping of *.fvr and *.pid files. Closes #47 (`indexzero`)
  * [864b1d1](https://github.com/foreverjs/forever/commit/864b1d1) [minor] Small fixes (`indexzero`)
  * [9b56c41](https://github.com/foreverjs/forever/commit/9b56c41) [api] Allow for forced exit if scripts restart in less than `minUptime` (`indexzero`)
  * [650f874](https://github.com/foreverjs/forever/commit/650f874) [minor] Add docs for `forever clear <key>` (`indexzero`)
  * [396b9a1](https://github.com/foreverjs/forever/commit/396b9a1) [doc] Regenerate docco docs (`indexzero`)
  * [a49483d](https://github.com/foreverjs/forever/commit/a49483d) [doc] Updated README.md (`indexzero`)
  * [f0ba253](https://github.com/foreverjs/forever/commit/f0ba253) [bin api minor] Update Copyright headers. Refactor bin/forever into lib/forever/cli.js. Add `forever config`, `forever set <key> <value>`, and `forever clear <key>` (`indexzero`)
  * [dffd0d1](https://github.com/foreverjs/forever/commit/dffd0d1) [minor dist api] Small updates for storing a forever global config file. Update package.json using require-analyzer (`indexzero`)
  * [6741c3a](https://github.com/foreverjs/forever/commit/6741c3a) [minor] More work for multiple processes from a single programmatic usage (`indexzero`)
  * [6e52e03](https://github.com/foreverjs/forever/commit/6e52e03) [minor test] Added tests for multiple processes from a single node process (`indexzero`)
  * [1c16e81](https://github.com/foreverjs/forever/commit/1c16e81) [api test] Update to use nconf for forever configuration. Use uids for filenames instead of forever* and forever pids (more defensive + support for multiple monitors from a single `forever` process). (`indexzero`)
  * [be6de72](https://github.com/foreverjs/forever/commit/be6de72) [minor] Small updates after merging from kpdecker (`indexzero`)
  * [95434b3](https://github.com/foreverjs/forever/commit/95434b3) Proper pid lookup in getForeverId (`kpdecker`)
  * [13bf645](https://github.com/foreverjs/forever/commit/13bf645) Add custom root directory to the initd-example (For cases where /tmp is removed) (`kpdecker`)
  * [b181dd7](https://github.com/foreverjs/forever/commit/b181dd7) Init.d Example script (`kpdecker`)
  * [51bc6c0](https://github.com/foreverjs/forever/commit/51bc6c0) Append log implementation (`kpdecker`)
  * [588b2bf](https://github.com/foreverjs/forever/commit/588b2bf) Append log CLI (`kpdecker`)
  * [ab497f4](https://github.com/foreverjs/forever/commit/ab497f4) forever.stat append flag (`kpdecker`)
  * [dca33d8](https://github.com/foreverjs/forever/commit/dca33d8) CLI pidfile argument (`kpdecker`)
  * [52184ae](https://github.com/foreverjs/forever/commit/52184ae) forever.pidFilePath implementation (`kpdecker`)
  * [e9b2cd3](https://github.com/foreverjs/forever/commit/e9b2cd3) forever.logFilePath utility. Treat paths that start with / as paths relative to the root, not the forever root. (`kpdecker`)
  * [8e323ca](https://github.com/foreverjs/forever/commit/8e323ca) Pass cwd to spawn (`kpdecker`)
  * [b29a258](https://github.com/foreverjs/forever/commit/b29a258) Return non-zero error code on tryStart failure (`kpdecker`)
  * [11ffce8](https://github.com/foreverjs/forever/commit/11ffce8) Load the forever lib relative to the binary rather than using module notation. (`kpdecker`)

v0.4.2 / Wed, 13 Apr 2011
=========================
  * [7089311](https://github.com/foreverjs/forever/commit/7089311) [dist] Version bump. 0.4.2 (`indexzero`)

v0.4.1 / Sat, 19 Feb 2011
=========================
  * [f11321f](https://github.com/foreverjs/forever/commit/f11321f) [dist] Version bump. 0.4.1 (`indexzero`)
  * [987d8ed](https://github.com/foreverjs/forever/commit/987d8ed) [fix] Update sourceDir option to check for file paths relative to root (`indexzero`)

v0.4.0 / Wed, 16 Feb 2011
=========================
  * [b870d47](https://github.com/foreverjs/forever/commit/b870d47) [dist] Version bump. 0.4.0 (`indexzero`)
  * [d9911dd](https://github.com/foreverjs/forever/commit/d9911dd) [doc] Update docs for v0.4.0 release (`indexzero`)
  * [6862ad5](https://github.com/foreverjs/forever/commit/6862ad5) [api] Expose options passed to child_process.spawn (`indexzero`)
  * [4b25241](https://github.com/foreverjs/forever/commit/4b25241) [doc] Added example for chroot (`indexzero`)
  * [9d2eefa](https://github.com/foreverjs/forever/commit/9d2eefa) [fix] Dont slice off arguments after [SCRIPT] if it is not passed to the CLI (e.g. forever list) (`indexzero`)
  * [7c0c3b8](https://github.com/foreverjs/forever/commit/7c0c3b8) [api] Refactor to use winston instead of pure sys.puts() for logging (`indexzero`)
  * [cc3d465](https://github.com/foreverjs/forever/commit/cc3d465) [api] Make forever.load() sync and not required for default configurations. Grossly simplifies saving / reloading semantics (`indexzero`)
  * [fd1b9a6](https://github.com/foreverjs/forever/commit/fd1b9a6) [api] Added `restart` command to both forever.Monitor and CLI (`indexzero`)
  * [c073c47](https://github.com/foreverjs/forever/commit/c073c47) [api] First pass at "restart" functionality, not 100% yet (`indexzero`)
  * [7b9b4be](https://github.com/foreverjs/forever/commit/7b9b4be) [docs] Updated docs from docco (`indexzero`)
  * [ea89def](https://github.com/foreverjs/forever/commit/ea89def) [minor] Small formatting update to package.json (`indexzero`)
  * [85b0a02](https://github.com/foreverjs/forever/commit/85b0a02) [api] Added ctime property to forever instances to track uptime (`indexzero`)
  * [bc07f95](https://github.com/foreverjs/forever/commit/bc07f95) [docs refactor] Refactor forever.Forever into lib/forever/monitor.js (`indexzero`)

v0.3.5 / Fri, 11 Feb 2011
=========================
  * [884037a](https://github.com/foreverjs/forever/commit/884037a) [dist] Version bump. 0.3.5. depends on daemon > 0.3.0 & node > 0.4.0 (`indexzero`)
  * [7b31da2](https://github.com/foreverjs/forever/commit/7b31da2) [api minor] Updates for daemon.node 0.2.0. Fix randomString so it doesnt generate strings with "/" (`indexzero`)
  * [a457ab7](https://github.com/foreverjs/forever/commit/a457ab7) [doc] Add docs from docco (`indexzero`)
  * [4a0ca64](https://github.com/foreverjs/forever/commit/4a0ca64) expose command to bin/forever as an option (`Adrien Friggeri`)

v0.3.1 / Fri, 24 Dec 2010
=========================
  * [3c7e4a7](https://github.com/foreverjs/forever/commit/3c7e4a7) [dist doc] Version bump 0.3.1. Added CHANGELOG.md (`indexzero`)
  * [38177c4](https://github.com/foreverjs/forever/commit/38177c4) [bin] Ensure both daemons and long running processes get the same stat checking (`indexzero`)
  * [ea6849d](https://github.com/foreverjs/forever/commit/ea6849d) [api] Make it the responsibility of the programmer to save/re-save the Forever information on start or restart events (`indexzero`)
  * [14c7aa8](https://github.com/foreverjs/forever/commit/14c7aa8) [api test bin doc] Added stop by script name feature. Improved the cleanlogs functionality. Made event emission consistent. Added to docs (`indexzero`)
  * [b7f792b](https://github.com/foreverjs/forever/commit/b7f792b) [minor] Small update to how forever works with pid files (`indexzero`)
  * [57850e9](https://github.com/foreverjs/forever/commit/57850e9) [api fix] Improved the way forever manages pid / fvr files. Added cleanlogs command line option (`indexzero`)
  * [070313e](https://github.com/foreverjs/forever/commit/070313e) [api] Push options hierarchy up one level. e.g. Forever.options.silent is now Forever.silent (`indexzero`)
  * [124cc25](https://github.com/foreverjs/forever/commit/124cc25) [fix api bin test] Check for scripts with fs.stat() before running them. Use process.kill instead of exec('kill'). Clean logs from command line. Display log file in forever list. Emit save event. (`indexzero`)
  * [57273ea](https://github.com/foreverjs/forever/commit/57273ea) updated the readme with non-node usage and an example (`James Halliday`)
  * [cc33f06](https://github.com/foreverjs/forever/commit/cc33f06) passing test for non-node array usage (`James Halliday`)
  * [761b31b](https://github.com/foreverjs/forever/commit/761b31b) file array case shortcut to set the command and options (`James Halliday`)
  * [02de53f](https://github.com/foreverjs/forever/commit/02de53f) "command" option to spawn() with, defaults to "node" (`James Halliday`)
  * [6feedc1](https://github.com/foreverjs/forever/commit/6feedc1) [minor] Remove unnecessary comma in package.json (`indexzero`)

v0.3.0 / Tue, 23 Nov 2010
=========================
  * [5d6f8da](https://github.com/foreverjs/forever/commit/5d6f8da) [dist] Version bump. 0.3.0 (`indexzero`)
  * [29bff87](https://github.com/foreverjs/forever/commit/29bff87) [doc] Updated formatting in README.md (`indexzero`)
  * [00fc643](https://github.com/foreverjs/forever/commit/00fc643) [api bin doc test] Added stop, stopall, and list command line functionality. Forever now tracks all daemons running on the system using *.fvr files (`indexzero`)
  * [d084ad1](https://github.com/foreverjs/forever/commit/d084ad1) [minor] Make samples/server.js listen on 8000 (`indexzero`)

v0.2.7 / Tue, 16 Nov 2010
=========================
  * [29bc24f](https://github.com/foreverjs/forever/commit/29bc24f) [minor] Version bump. Fix small bug in 0.2.6 (`indexzero`)

v0.2.6 / Mon, 15 Nov 2010
=========================
  * [2bcc53d](https://github.com/foreverjs/forever/commit/2bcc53d) [bin dist] Version bump. Small fixes from 0.2.5 (`indexzero`)
  * [faacc0f](https://github.com/foreverjs/forever/commit/faacc0f) [doc] Typo (`indexzero`)

v0.2.5 / Sun, 14 Nov 2010
=========================
  * [0d5a789](https://github.com/foreverjs/forever/commit/0d5a789) [dist] Version bump. (`indexzero`)
  * [04705ed](https://github.com/foreverjs/forever/commit/04705ed) [api test bin dist] Update to use daemon.node (`indexzero`)
  * [65a91fb](https://github.com/foreverjs/forever/commit/65a91fb) [minor] Added .gitignore (`indexzero`)

v0.2.0 / Mon, 27 Sep 2010
=========================
  * [f916359](https://github.com/foreverjs/forever/commit/f916359) [minor dist] Added LICENSE. Refactor forever.js to be more DRY (`indexzero`)
  * [9243dee](https://github.com/foreverjs/forever/commit/9243dee) Removed repeating function and replaced it by template generator (`Fedor Indutny`)
  * [347dcaa](https://github.com/foreverjs/forever/commit/347dcaa) [minor] Updated contributors (`indexzero`)
  * [a4f1700](https://github.com/foreverjs/forever/commit/a4f1700) [api test doc dist] Version bump. Merged from donnerjack. Added ability to log to file(s). Updated docs. (`indexzero`)
  * [d5d2f1d](https://github.com/foreverjs/forever/commit/d5d2f1d) New-line at the end of file (`Fedor Indutny`)
  * [73b52a4](https://github.com/foreverjs/forever/commit/73b52a4) Added chaining to run, simplyfied exports.run (`Fedor Indutny`)
