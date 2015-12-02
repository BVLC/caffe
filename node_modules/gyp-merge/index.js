module.exports = function merge(src, dest, config) {
    if (!config) config = {
        noSingletons: false
    };

    return mergeDictionary(src, dest);

    function mergeDictionary(dest, src) {
        var out = Object.create(dest);

        if (typeof src == 'undefined') {
            return out;
        }

        if (typeof src != 'object' || Array.isArray(src) || typeof dest != 'object' || Array.isArray(dest)) {
            throw new TypeError("dest and src must both be objects");
        }

        for (var k in src) {
            if (Array.isArray(src[k])) {
                var suffix = k.substr(-1);
                var k2 = (suffix != '?' && suffix != '=' && suffix != '+') ? k : k.substr(0, k.length - 1);
                out[k2] = mergeList(out[k2], src[k], suffix);
            } else if (typeof src[k] == 'object') {
                out[k] = mergeDictionary(out[k], src[k]);
            } else {
                out[k] = src[k];
            }
        }

        return out;
    }

    function mergeList(dest, src, suffix) {
        if (!dest || suffix == '=') {
            return src;
        } else if (suffix == '?') {
            return dest ? dest : src;
        } else if (suffix == '+') {
            return concat(src, dest);
        } else {
            return concat(dest, src);
        }
    }

    function concat(dest, src) {
        if (!src) return dest;
        if (!dest) return src;

        if (config.noSingletons) {
            return dest.concat(src);
        } else {
            var out = dest.slice(0);
            for (var i = 0; i < src.length; i++) {
                if (typeof src[i] != 'string' || src[i][0] == '-' || out.indexOf(src[i]) == -1) {
                    out.push(src[i]);
                }
            }

            return out;
        }
    }
};
