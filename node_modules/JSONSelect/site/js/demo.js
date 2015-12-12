window.jsel = JSONSelect;

$(document).ready(function() {
    var theDoc = JSON.parse($("pre.doc").text());

    function highlightMatches(ar) {
        // first calculate match offsets
        var wrk = [];
        var html = $.trim(JSON.stringify(theDoc, undefined, 4));
        var ss = "<span class=\"selected\">";
        var es = "</span>";
        for (var i = 0; i < ar.length; i++) {
            var found = $.trim(JSON.stringify(ar[i], undefined, 4));
            // turn the string into a regex to handle indentation
            found = found.replace(/[-[\]{}()*+?.,\\^$|#]/g, "\\$&").replace(/\s+/gm, "\\s*");
            var re = new RegExp(found, "m");
            var m = re.exec(html);
            if (!m) continue;
            wrk.push({ off: m.index, typ: "s" });
            wrk.push({ off: m[0].length+m.index, typ: "e" });
        }
        // sort by offset
        wrk = wrk.sort(function(a,b) { return a.off - b.off; });

        // now start injecting spans into the text
        var cur = 0;
        var cons = 0;
        for (var i = 0; i < wrk.length; i++) {
            var diff = wrk[i].off - cons;
            cons = wrk[i].off;
            var tag = (wrk[i].typ == 's' ? ss : es);
            cur += diff;
            html = html.substr(0, cur) + tag + html.substr(cur);
            cur += tag.length;
        }
        return html;
    }

    // when a selector is chosen, update the text box
    $(".selectors .selector").click(function() {
        $(".current input").val($(this).text()).keyup();
    });

    var lastSel;
    $(".current input").keyup(function () {
        try {
            var sel = $(".current input").val()
            if (lastSel === $.trim(sel)) return;
            lastSel = $.trim(sel);
            var ar = jsel.match(sel, theDoc);
            $(".current .results").text(ar.length + " match" + (ar.length == 1 ? "" : "es"))
                .removeClass("error");
            $("pre.doc").html(highlightMatches(ar));
            $("pre.doc .selected").hide().fadeIn(700);
        } catch(e) {
            $(".current .results").text(e.toString()).addClass("error");
            $("pre.doc").text($.trim(JSON.stringify(theDoc, undefined, 4)));
        }
        $(".selectors .selector").removeClass("inuse");
        $(".selectors div.selector").each(function() {
            if ($(this).text() === sel) $(this).addClass("inuse");
        });
    });
});
