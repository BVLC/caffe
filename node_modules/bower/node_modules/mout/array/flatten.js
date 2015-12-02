var isArray = require('../lang/isArray');
var append = require('./append');

    /*
     * Helper function to flatten to a destination array.
     * Used to remove the need to create intermediate arrays while flattening.
     */
    function flattenTo(arr, result, level) {
        if (level === 0) {
            append(result, arr);
            return result;
        }

        var value,
            i = -1,
            len = arr.length;
        while (++i < len) {
            value = arr[i];
            if (isArray(value)) {
                flattenTo(value, result, level - 1);
            } else {
                result.push(value);
            }
        }
        return result;
    }

    /**
     * Recursively flattens an array.
     * A new array containing all the elements is returned.
     * If level is specified, it will only flatten up to that level.
     */
    function flatten(arr, level) {
        if (arr == null) {
            return [];
        }

        level = level == null ? -1 : level;
        return flattenTo(arr, [], level);
    }

    module.exports = flatten;



