$(document).ready(function() {
    var docsLoaded = false;

    $(window).hashchange(function(e){
        e.preventDefault();
        e.stopPropagation();

        if (location.hash === "#tryit") {
            $("#main > .content").hide();
            $("#tryit input").val("").keyup();
            $("#tryit").fadeIn(400, function() {
                $("#tryit input").val(".languagesSpoken .lang").keyup();
            });
        } else if (location.hash === "#cred") {
            $("#main > .content").hide();
            $("#cred").fadeIn(400);
        } else if (location.hash === '#overview' || location.hash === '') {
            $("#main > .content").hide();
            $("#splash").fadeIn(400);
        } else if (location.hash === '#code' || location.hash === '') {
            $("#main > .content").hide();
            $("#code").fadeIn(400);
        } else if (location.hash.substr(0,5) === "#docs") {
            function showIt() {
                var where = window.location.hash.substr(6);
                if (!where) {
                    $("#doc").fadeIn(400);
                } else {
                    $("#doc").show();
                    var dst = $("a[name='" + where + "']");
                    if (dst.length) {
                        $('html, body').animate({scrollTop:dst.offset().top - 100}, 500);
                    }
                }
            }
            $("#main > .content").hide();
            if (!docsLoaded) {
                $.get("JSONSelect.md").success(function(data) {
                    var converter = new Showdown.converter();
                    $("#doc").html(converter.makeHtml(data));
                    $("#doc a").each(function() {
                        var n = $(this).attr('href');
                        if (typeof n === 'string' && n.substr(0,1) === '#') {
                            $(this).attr('href', "#docs/" + n.substr(1));
                        }
                    });
                    docsLoaded = true;
                    showIt();
                }).error(function() {
                    $("#doc").text("Darnit, error fetching docs...").fadeIn(400);
                });
            } else {
                showIt();
            }
        } else {
        }
        return false;
    });

    // Trigger the event (useful on page load).
    if (window.location.hash === "")
        window.location.hash = "#overview";
    $(window).hashchange();
});
