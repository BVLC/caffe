if !exists("*s:RunTests")
  function s:RunTests()
    let command="tmux send-keys -t bottom-left mocha " . "' '" . "tests/fs-tree-test.js $'\n'"

    echo system(command)
  endfunction
endif

nmap <buffer> T :call <SID>RunTests()<CR>
