1.2.0 / 2015-06-29
------------------

- Allow dash at the end of url, thanks to @Mumakil.


1.1.1 / 2015-06-09
------------------

- Allow ".." in link paths.


1.1.0 / 2015-04-21
------------------

- Added options to control fuzzy links recognition (`fuzzyLink: true`,
  `fuzzyEmail: true`, `fuzzyIP: false`).
- Disabled IP-links without schema prefix by default.


1.0.1 / 2015-04-19
------------------

- More strict default 2-characters tlds handle in fuzzy links, to avoid
  false positives for `node.js`, `io.js` and so on.


1.0.0 / 2015-03-25
------------------

- Version bump to 1.0.0 for semver.
- Removed `Cf` class from whitespace & punctuation sets (#10).
- API change. Exported regex names renamed to reflect changes. Update your
  custom rules if needed:
  - `src_ZPCcCf` -> `src_ZPCc`
  - `src_ZCcCf` -> `src_ZCc`


0.1.5 / 2015-03-13
------------------

- Fixed special chars handling (line breaks).
- Fixed demo permalink encode/decode.


0.1.4 / 2015-03-12
------------------

- Allow `..` and `...` inside of link paths (#9). Useful for github links with
  commit ranges.
- Added `.pretest()` method for speed optimizations.
- Autogenerate demo sample from fixtures.


0.1.3 / 2015-03-11
------------------

- Maintenance release. Deps update.


0.1.2 / 2015-02-26
------------------

- Fixed blockquoted links (some symbols exclusions), thanks to @MayhemYDG.
- Fixed demo permalinks, thanks to @MayhemYDG.


0.1.1 / 2015-02-22
------------------

- Moved unicode data to external package.
- Demo permalink improvements.
- Docs update.


0.1.0 / 2015-02-12
------------------

- First release.
