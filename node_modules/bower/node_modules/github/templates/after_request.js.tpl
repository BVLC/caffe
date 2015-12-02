
            if (!ret)
                ret = {};
            if (typeof ret == "object") {
                if (!ret.meta)
                    ret.meta = {};
                [<%headers%>].forEach(function(header) {
                    if (res.headers[header])
                        ret.meta[header] = res.headers[header];
                });
            }
