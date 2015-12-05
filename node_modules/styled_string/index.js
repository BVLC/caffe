var Attributes = {
    display: {
        reset : 0,
        bright : 1,
        dim : 2,
        underscore : 4,
        blink : 5,
        reverse : 7,
        hidden : 8
    }
    , foreground: {
        black : 30,
        red : 31,
        green : 32,
        yellow : 33,
        blue : 34,
        magenta : 35,
        cyan : 36,
        white : 37
    }
    , background: {
        black : 40,
        red : 41,
        green : 42,
        yellow : 43,
        blue : 44,
        magenta : 45,
        cyan : 46,
        white : 47
    }
}

function StyledString(str, attrs, children){
    if (this.constructor !== StyledString){
        return new StyledString(str, attrs, children)
    }
    this.str = str
    this.children = children
    this.length = str != null ? str.length : this.childrenLength()
    this.attrs = attrs || {}
}

StyledString.prototype.substring = function(){
    if (this.str != null){
        var str = this.str.substring.apply(this.str, arguments)
        return new StyledString(str, this.attrs)
    }else{
        var startIdx = arguments[0]
        var endIdx = arguments[1]
        var retval = StyledString('')
        var currIdx = 0
        this.children.forEach(function(child){
            var part
            if (startIdx < currIdx + child.length){
                part = child.substring(startIdx - currIdx)
            }
            if (endIdx !== undefined && endIdx < currIdx + child.length){
                part = (part || child).substring(0, endIdx - currIdx)
            }
            if (part){
                retval = retval.concat(part)
            }
            currIdx += child.length
        })
        return retval
    }
}

StyledString.prototype.childrenLength = function(){
    return this.children.reduce(function(curr, child){
        return curr + child.length
    }, 0)
}

StyledString.prototype.substr = StyledString.prototype.substring

StyledString.prototype.match = function(){
    return this.str.match.apply(this.str, arguments)
}

StyledString.prototype.concat = function(){
    var args = Array.prototype.slice.apply(arguments)
    if (this.children){
        return StyledString(null, null, this.children.concat(args))
    }else{
        var children = [this].concat(Array.prototype.slice.apply(args)
            .filter(function(s){
                return s.length > 0
            })
        )
        return StyledString(null, null, children)
    }
}

function copy(src){
    var dst = {}
    for (var key in src)
        dst[key] = src[key]
    return dst
}

StyledString.prototype.split = function(){
    var args = arguments
    if (this.str != null){
        var attrs = copy(this.attrs)
        return this.str.split.apply(this.str, args).map(function(s){
            return StyledString(s, attrs)
        })
    }else{
        return this.children.reduce(function(curr, child){
            var results = child.split.apply(child, args)
            if (curr.length > 0 && results.length > 0){
                var last = curr[curr.length - 1]
                curr[curr.length - 1] = last.concat(results[0])
                results = results.slice(1)
            }
            return curr.concat(results)
        }, [])
    }
}

StyledString.prototype.unstyled = function(){
    if (this.str != null){
        return this.str
    }else{
        return this.children.map(function(child){
            return child.unstyled()
        }).join('')
    }
}

StyledString.prototype.applyAttrs = function(str){
    for (var key in this.attrs){
        if (Attributes[key]){
            var code = Attributes[key][this.attrs[key]]
            if (code){
                str = '\033[' + code + 'm' + str + '\033[0m'
            }
        }
    }
    return str
}

StyledString.prototype.toString = function(){
    var str
    if (this.str != null){
        str = this.str
    }else{
        str = this.children.reduce(function(curr, child){
            if (child.length > 0)
                return curr + child.toString()
            else
                return curr
        }, '')
    }
    return this.applyAttrs(str)
}

// Non-string-like methods
StyledString.prototype.append = function(another){
    if (this.str != null){
        this.children = [StyledString(this.str, this.attrs), another]
        this.str = null
        this.attrs = {}
        this.length = this.childrenLength()
    }else{
        this.children.push(another)
    }
    return this
}

StyledString.prototype.attr = function(attrs){
    for (var key in attrs){
        this.attrs[key] = attrs[key]
    }
    return this
}

StyledString.prototype.foreground = function(color){
    this.attr({foreground: color})
    return this
}

StyledString.prototype.background = function(color){
    this.attr({background: color})
    return this
}

StyledString.prototype.display = function(value){
    this.attr({display: value})
    return this
}


StyledString.Attributes = Attributes

module.exports = StyledString