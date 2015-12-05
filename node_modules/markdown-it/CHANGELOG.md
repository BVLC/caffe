4.3.0 / 2015-06-29
------------------

- `linkify-it` dependency update (1.2.0). Now accepts dash at the end of links.


4.2.2 / 2015-06-10
------------------

- CM spec 0.20.
- Added support for multichar substituition in smartquites, #115.
- Fixed code block render inside blockquites, #116.
- Doc fixes.


4.2.1 / 2015-05-01
------------------

- Minor emphasis update to match CM spec 0.19.


4.2.0 / 2015-04-21
------------------

- Bumped [linkify-it](https://github.com/markdown-it/linkify-it) version to
  1.1.0. Now links with IP hosts and without protocols are not linkified by
  default, due possible collisions with some version numbers patterns (0.5.0.0).
  You still can return back old behaviour by `md.linkify.set({ fuzzyIP: true })`.


4.1.2 / 2015-04-19
------------------

- Bumped linkifier version. More strict 2-chars tald support for links without
  schema. Should not linkify things like `io.js` and `node.js`.


4.1.1 / 2015-04-15
------------------

- Improved pipe chars support in table cells, #86 (thanks to @jbt).


4.1.0 / 2015-03-31
------------------

- Security: disabled `data:` URLs by default (except some image mimes), to avoid
  possible XSS. Version bumped, because features changed (formally). If you did
  not used `data:` URLs, consider this version as 4.0.4 (no API changes).
- Simplified link validator code. Now more easy to understand and to copy
  into your projects for customization.


4.0.3 / 2015-03-25
------------------

- Updated linkifier.
- Smartquotes rule cleanup (#76).
- Fixed replacements rule bug in PhantomJS (#77).


4.0.2 / 2015-03-22
------------------

- Fixed emphasis `marker` fields in tokens (#69).
- Fixed html block tokens with numbers in name (#74).


4.0.1 / 2015-03-13
------------------

- Updated `linkify-it` version.
- Added custom container plugin demo.


4.0.0 / 2015-03-11
------------------

- Breaking internal API changes. See [v4 migration notes](https://github.com/markdown-it/markdown-it/blob/master/docs/4.0_migration.md). In usual case you will need to update plugins.
- Token internals changed
- Unified the most of renderer methods.
- Changed tokens creation - use `state.push(...)` (see sources)
- Moved `normalizeUrl()` to root class as `.normalizeLink()` &
  added `normalizeLinkText()` method.
- Moved `.validateUrl()` to root class and simplified logic - no more need to
  replace entities.
- Joined md unescape & replace entities logic to `utils.unescapeAll()`.
- Removed `replaceEntities()` in `utils`.
- `md.utils.lib` now exposes useful libs for plugins.
- Use entities data from external package.
- Fixed emphasis regression, caused by CM v0.18 spec (#65).


3.1.0 / 2015-03-05
------------------

- Significantly improved autolinking quality (use `linkify-it` package), #2.
- Rewritten links normalizer to solve different edge cases (use `mdurl`
  package), #29.
- Moved link title entities replace out of renderer.
- Fixed escaped entities in links (`foo\&amp;/bar`).
- Improved smartquotes logic, #61.
- Spec conformance update to 0.18.


3.0.7 / 2015-02-22
------------------

- Match table columns count by header.
- Added basic CLI support.
- Added \v \f to valid whitespaces.
- Use external package for unicode data (punctuation).


3.0.6 / 2015-02-12
------------------

- Fixed hang on long vertical list of links. Appeared in 3.0.5. See #54 for
  details. Thanks to @fengmk2 for report!
- Table lines now can have escaped pipe char `\|` (#5).
- Sync scroll result => source in demo.
- Moved `normalizeReference()` to utils.


3.0.5 / 2015-02-06
------------------

- Fixed link validator - could skip some kind of javascript links with uppercase
  digital entities (thanks to @opennota)
- Significantly improved tests coverage (with dead code removal and other
  related things).


3.0.4 / 2015-01-13
------------------

- Improved errors processing in url normalizer (for broken sequences).
- Improved nesting limit processing in inline parser.
- Reorganised tests & improved coverage.
- Show inline diffs for failed tests.


3.0.3 / 2015-01-11
------------------

- Fixed punctuation check in emphasis.


3.0.2 / 2015-01-09
------------------

- Allow dashes in HTML tag names (needed for custom HTML tags).


3.0.1 / 2015-01-07
------------------

- Improved link encoder - fix invalid surrogates to avoid errors.
- Added # to terminator chars.


3.0.0 / 2015-01-04
------------------

- Big split. All "rare" rules moved to external plugins (deflist, abbr, footnote,
  sub, sup, ins, mark).
- Updated CM spec conformance to v0.15 (better unicode support).
- Added `md` (parser instance) link to all state objects (instead of former
  options/parser).
- References/Footnotes/Abbrs moved to `block` chain.
- Input normalization moved to `core` chain.
- Splitted links and images to separate rules.
- Renamed some rules.
- Removed `full` preset. Not needed anymore.
- enable/disable methods now throw by default on invalid rules (exceptions can
  be disabled).
- Fixed inline html comments & cdata parse.
- Replace NULL characters with 0xFFFD instead of strip.
- Removed custom fences sugar (overcomplication).
- Rewritten link components parse helpers.
- More functions in `md.utils`.


2.2.1 / 2014-12-29
------------------

- Added development info.
- Fixed line breaks in definitions lists.
- .use() now pass any number of params to plugins.


2.2.0 / 2014-12-28
------------------

- Updated CM spec conformance to v0.13.
- API docs.
- Added 'zero' preset.
- Fixed several crashes, when some basic rules are disabled
  (block termination check, references check).


2.1.3 / 2014-12-24
------------------

- Added curring to `set`/`configure`/`enable`/`disable` methods.
- Demo rework - now can include plugins.
- Docs update.


2.1.2 / 2014-12-23
------------------

- Exposed helpers into parser instances (for plugins).
- Removed utils from global export - been in instances seems enougth.
- Refactored demo & added markdown-it-emoji to it.


2.1.1 / 2014-12-22
------------------

- Refreshed browser builds, missed in prev release.
- Minor changes.


2.1.0 / 2014-12-21
------------------

- Separated method to enable rules by whitelist (enableOnly).
- Changed second param of enable/disable ruler methods.
- Shortcuts in main class for bulk enable/disable rules.
- ASCII-friendly browserified files.
- Separate package for spec tests.


2.0.0 / 2014-12-20
------------------

- New project name & home! Now it's `markdown-it`,
- Sugar for constructor call - `new` is not mandatory now.
- Renamed presets folder (configs -> presets).
