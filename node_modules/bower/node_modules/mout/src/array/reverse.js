define(function () {

    /**
     * Returns a copy of the array in reversed order.
     */
    function reverse(array) {
        var copy = array.slice();
        copy.reverse();
        return copy;
    }

    return reverse;

});
