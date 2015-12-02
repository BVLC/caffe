'use strict';

var pkg = require('./package.json')
    ,gulp = require('gulp')
    ,header = require('gulp-header')
    ,uglify = require('gulp-uglify')
    ,rename = require('gulp-rename')
    ,browserify = require('browserify')
    ,notify = require("gulp-notify")
    ,source = require('vinyl-source-stream')
    ,buffer = require('vinyl-buffer')
    ;


function getBundler(options) {
  return browserify(options.browserify)
    .add('./src/index.js')
    .bundle()
    .on('error', function() {
      var args = Array.prototype.slice.call(arguments);

      // Send error to notification center with gulp-notify
      notify.onError({
        title: "Compile Error",
        message: "<%= error.message %>"
      }).apply(this, args);

      // Keep gulp from hanging on this task
      this.emit('end');
    })
    .pipe(source(options.file))
    .pipe(header('/* safe-clone-deep v${pkg.version} - https://github.com/tracker1/safe-clone-deep */\n\n', {pkg:pkg}))
    //.pipe(rename('safe-clone-deep.browser.js'))
    .pipe(gulp.dest('dist/'))
    .pipe(buffer())
    .pipe(uglify())
    .pipe(rename(options.filemin))
    .pipe(header('/* safe-clone-deep v${pkg.version} - https://github.com/tracker1/safe-clone-deep */\n\n', {pkg:pkg}))
    .pipe(gulp.dest('dist/'));
}


gulp.task('browser',function(){
  return getBundler({
    file: 'safe-clone-deep.browser.js'
    ,filemin: 'safe-clone-deep.browser.min.js'
    ,browserify: {
      standalone:'Object.safeCloneDeep'
      ,detectGlobals: false //this script detects for "Buffer" as an available type
      ,bundleExternal: false //no external dependencies/shims needed
    }
  });
});


gulp.task('amd',function(){
  return getBundler({
    file: 'safe-clone-deep.amd.js'
    ,filemin: 'safe-clone-deep.amd.min.js'
    ,browserify:{
      detectGlobals: false //this script detects for "Buffer" as an available type
      ,bundleExternal: false //no external dependencies/shims needed
    }
  });
});


gulp.task('default',['browser','amd']);
