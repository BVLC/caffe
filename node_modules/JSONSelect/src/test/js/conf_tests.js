$(document).ready(function() {
    var tests = {};

    $("#runit").click(function() {
        for (var k in tests) {
            var obj = JSON.parse($("." + k + "_document").text());
            for (var i = 0; i < tests[k].length; i++) {
                var n = tests[k][i];
                var cl = k + "_" + n;
                var b = $("." + cl + "_output.before");
                var a = $("." + cl + "_output.after");
                var s = $("." + cl + "_selector.selector");
                try {
                    a.text("");
                    JSONSelect.forEach(s.text(), obj, function(m) {
                        a.text($.trim(a.text() + "\n" + JSON.stringify(m, null, "    ")));
                    });
                } catch(e) {
                    a.text("Error: " + e);
                }
                if (a.text() === b.text()) s.addClass("success").removeClass("failure");
                else s.addClass("failure").removeClass("success");
            }
        }
    });

    function fetchFile(p, c) {
        $.get(p, function (data) {
            $("." + c).text($.trim(data));
        });
    }

    function renderTests() {
        function setClickToggle(cTarget, node) {
            cTarget.click(function() { node.toggle("medium"); });
        }

        var c = $("<div/>");
        for (var k in tests) {
            c.append($("<h1/>").text("document: " + k));
            var cl = k + "_document";
            c.append($("<pre/>").addClass(cl).addClass("document").text("loading document..."));
            fetchFile("tests/" + k + ".json", cl);
            for (var i = 0; i < tests[k].length; i++) {
                var n = tests[k][i];
                var cl = k + "_" + n + "_selector";
                var s = $("<pre/>").addClass(cl).addClass("selector").text("loading selector...");
                c.append(s);
                fetchFile("tests/" + k + "_" + n + ".selector", cl);
                cl = k + "_" + n + "_output";
                var t = $("<table/>").append($("<tr/>").append(
                    $("<td/>").append($("<pre/>").addClass(cl).addClass("before").text("loading output..."))).append(
                    $("<td/>").append($("<pre/>").addClass(cl).addClass("after").text("... test output ..."))));

                c.append(t);
                t.hide();
                setClickToggle(s, t);
                fetchFile("tests/" + k + "_" + n + ".output", cl + ".before");
            }
        }
        c.appendTo($("#tests"));
    }

    $.get("tests/alltests.txt", function (data) {
        var lines = data.split("\n");
        for (var i = 0; i < lines.length; i++) {
            var f = $.trim(lines[i]);
            if (f.length == 0) continue;
            var m = /^([A-Za-z]+)_(.+)\.selector$/.exec(f);
            if (m) {
                if (!tests.hasOwnProperty(m[1])) tests[m[1]] = [];
                tests[m[1]].push(m[2]);
            }
        }
        renderTests();
    });
});
