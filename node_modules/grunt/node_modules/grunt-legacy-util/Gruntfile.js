'use strict';

module.exports = function(grunt) {

  grunt.initConfig({
    jshint: {
      options: {
        jshintrc: '.jshintrc',
      },
      all: ['*.js', 'test/*.js'],
    },
    nodeunit: {
      util: ['test/index.js']
    },
    watch: {
      all: {
        files: ['<%= jshint.all %>'],
        tasks: ['test'],
      },
    },
  });

  grunt.loadNpmTasks('grunt-contrib-jshint');
  grunt.loadNpmTasks('grunt-contrib-nodeunit');
  grunt.loadNpmTasks('grunt-contrib-watch');

  grunt.registerTask('test', ['jshint', 'nodeunit']);
  grunt.registerTask('default', ['test', 'watch']);

};
