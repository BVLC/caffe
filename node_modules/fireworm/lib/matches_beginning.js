var minimatch = require('minimatch')
var sep = process.platform === 'win32' ? '\\' : '/'
module.exports = function(path, pattern){
    if (pattern instanceof RegExp){
        pattern = pattern.source
        for (var i = 2; i < pattern.length + 1; i++){
            try{
                var regex = new RegExp(pattern.substring(0, i) + '$')
                if (regex.test(path)) return true
            }catch(e){
                if (e instanceof SyntaxError){ /* let syntax errors go */ }
                else throw e
            }
        }
    }else{
        var idx = pattern.length
        while (idx !== -1){
            pattern = pattern.substring(0, idx)
            if (minimatch(path, pattern)) return true
            idx = pattern.lastIndexOf(sep)
        }
    }
    return false
}