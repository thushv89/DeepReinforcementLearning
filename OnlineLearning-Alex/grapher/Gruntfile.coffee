module.exports = (grunt) ->
    grunt.initConfig
        nodemon:
            dev:
                script: 'index.coffee'

    grunt.loadNpmTasks('grunt-nodemon')
    grunt.registerTask 'serve', 'nodemon'
