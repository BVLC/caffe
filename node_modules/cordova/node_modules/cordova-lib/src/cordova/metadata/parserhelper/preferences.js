/**
    Licensed to the Apache Software Foundation (ASF) under one
    or more contributor license agreements.  See the NOTICE file
    distributed with this work for additional information
    regarding copyright ownership.  The ASF licenses this file
    to you under the Apache License, Version 2.0 (the
    "License"); you may not use this file except in compliance
    with the License.  You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing,
    software distributed under the License is distributed on an
    "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
    KIND, either express or implied.  See the License for the
    specific language governing permissions and limitations
    under the License.
*/

/* jshint sub:true */

'use strict';

var events = require('cordova-common').events;

var _ORIENTATION_ALL = 'all';
var _ORIENTATION_SENSOR_LANDSCAPE = 'sensorLandscape';

var _ORIENTATION_DEFAULT = 'default';
var _ORIENTATION_PORTRAIT = 'portrait';
var _ORIENTATION_LANDSCAPE = 'landscape';
var _ORIENTATION_GLOBAL_ORIENTATIONS = [ _ORIENTATION_DEFAULT, _ORIENTATION_PORTRAIT, _ORIENTATION_LANDSCAPE ];

var _ORIENTATION_SPECIFIC_TO_A_PLATFORM_ORIENTATIONS = {
    'ios' : [ _ORIENTATION_ALL ],
    'android' : [ _ORIENTATION_SENSOR_LANDSCAPE ]
};

module.exports = {

    // Specific to a platform 
    ORIENTATION_ALL: _ORIENTATION_ALL, // iOS
    ORIENTATION_SENSOR_LANDSCAPE : _ORIENTATION_SENSOR_LANDSCAPE, // Android
    
    // Global
    ORIENTATION_DEFAULT: _ORIENTATION_DEFAULT,
    ORIENTATION_PORTRAIT: _ORIENTATION_PORTRAIT,
    ORIENTATION_LANDSCAPE: _ORIENTATION_LANDSCAPE,
    
    // Collections
    ORIENTATION_GLOBAL_ORIENTATIONS: _ORIENTATION_GLOBAL_ORIENTATIONS,
    ORIENTATION_SPECIFIC_TO_A_PLATFORM_ORIENTATIONS: _ORIENTATION_SPECIFIC_TO_A_PLATFORM_ORIENTATIONS,

    /**
     * @param  {String}  orientation Orientation
     * @return {Boolean}             True if the value equals ORIENTATION_DEFAULT
     */
    isDefaultOrientation: function (orientation) {
        return orientation.toLowerCase() === this.ORIENTATION_DEFAULT.toLowerCase();
    },

    /**
     * @param  {String}  orientation Orientation
     * @return {Boolean}             True if the value equals one of _ORIENTATION_SUPPORTED_GLOBAL_ORIENTATIONS
     */
    isGlobalOrientation: function (orientation) {
        return this.ORIENTATION_GLOBAL_ORIENTATIONS.some(function (supportedOrientation) {
            return orientation.toLowerCase() === supportedOrientation.toLowerCase();
        });
    },

    /**
     * @param  {String}  orientation Orientation
     * @return {Boolean}             True if the value equals one of _ORIENTATION_PLATFORM_SPECIFIC_ORIENTATIONS
     */
    isOrientationSpecificToAPlatform: function (orientation, platform) {
        var platformArray = this.ORIENTATION_SPECIFIC_TO_A_PLATFORM_ORIENTATIONS[ platform ];
        
        if (!platformArray) {
            return false;
        } 
        
        return platformArray.some(function (supportedOrientation) {
            return orientation.toLowerCase() === supportedOrientation.toLowerCase();
        });
    },

    /**
     * Queries ConfigParser object for the orientation <preference> value.
     *
     * @param  {Object} config    ConfigParser object
     * @param  {String} [platform]  Platform name
     * @return {String}           Global/platform-specific orientation in lower-case (or empty string if both are undefined)
     */
    getOrientation: function (config, platform) {

        var platformOrientation = platform ? config.getPlatformPreference('orientation', platform) : '';
        var globalOrientation = '';

        if (platformOrientation) {
            return platformOrientation;
        }

        globalOrientation = config.getGlobalPreference('orientation');

        // Check if the given global orientation is supported, or if the orientation is supported for a platform
        if ((globalOrientation && !this.isGlobalOrientation(globalOrientation)) && !this.isOrientationSpecificToAPlatform(globalOrientation, platform)) {
            events.emit( 'warn', [ 'Unsupported global orientation:', globalOrientation ].join(' ') );
            events.emit( 'warn', [ 'Defaulting to value:', this.ORIENTATION_DEFAULT ].join(' ') );
            globalOrientation = this.ORIENTATION_DEFAULT;
        }

        return globalOrientation;
    }

};
