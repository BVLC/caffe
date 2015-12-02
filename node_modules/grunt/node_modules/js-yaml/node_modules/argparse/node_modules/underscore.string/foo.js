
    function boolMatch(s, matchers) {
      var i, matcher, down = s.toLowerCase();
      matchers = [].concat(matchers);
      for (i = 0; i < matchers.length; i += 1) {
        matcher = matchers[i];
        if (matcher.test && matcher.test(s)) return true;
        if (matcher && matcher.toLowerCase() === down) return true;
      }
    }
