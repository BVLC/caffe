var gulp = require('gulp'),
    qunit = require("gulp-qunit"),
    uglify = require('gulp-uglify'),
    clean = require('gulp-clean'),
    rename = require('gulp-rename'),
    SRC = 'lib/underscore.string.js',
    DEST = 'dist',
    MIN_FILE = 'underscore.string.min.js',
    TEST_SUITES = ['test/test.html', 'test/test_underscore/index.html'];

gulp.task('test', function() {
    return gulp.src(TEST_SUITES)
        .pipe(qunit());
});

gulp.task('clean', function() {
    return gulp.src(DEST)
        .pipe(clean());
});

gulp.task('build', ['test', 'clean'], function() {
    return gulp.src(SRC)
        .pipe(uglify())
        .pipe(rename(MIN_FILE))
        .pipe(gulp.dest(DEST));
});
