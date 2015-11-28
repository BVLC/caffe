<%comment%>
    this.<%funcName%> = function(msg, block, callback) {
        var self = this;
        this.client.httpSend(msg, block, function(err, res) {
            if (err)
                return self.sendError(err, null, msg, callback);

            var ret;
            try {
                ret = res.data;
                var contentType = res.headers["content-type"];
                if (contentType && contentType.indexOf("application/json") !== -1)
                    ret = JSON.parse(ret);
            }
            catch (ex) {
                if (callback)
                    callback(new error.InternalServerError(ex.message), res);
                return;
            }
<%afterRequest%>
            if (callback)
                callback(null, ret);
        });
    };
