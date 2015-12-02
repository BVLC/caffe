exports.quote = function (xs) {
    return xs.map(function (s) {
        if (/["\s]/.test(s) && !/'/.test(s)) {
            return "'" + s.replace(/(['\\])/g, '\\$1') + "'";
        }
        else if (/["'\s]/.test(s)) {
            return '"' + s.replace(/(["\\$`(){}!#&*|])/g, '\\$1') + '"';
        }
        else {
            return s.replace(/([\\$`(){}!#&*|])/g, '\\$1');
        }
    }).join(' ');
};

exports.parse = function (s) {
    return s.match(/(['"])((\\\1|[^\1])*?)\1|(\\ |\S)+/g)
        .map(function (s) {
            if (/^'/.test(s)) {
                return s
                    .replace(/^'|'$/g, '')
                    .replace(/\\(["'\\$`(){}!#&*|])/g, '$1');
                ;
            }
            else if (/^"/.test(s)) {
                return s
                    .replace(/^"|"$/g, '')
                    .replace(/\\(["'\\$`(){}!#&*|])/g, '$1');
                ;
            }
            else return s.replace(/\\([ "'\\$`(){}!#&*|])/g, '$1');
        })
    ;
};
