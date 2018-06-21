/num_output:/ {
    if (conv == 1 && prune == 1) {
        old = $2
        new = int($2 * prec)
        if (new % 16 != 0) {
            new = new + 16 - (new % 16)
        }
        print "        num_output: " new " # was " old
    } else {
        print $0
    }
    num = 1
}
/layer/ {
    conv = 0
}
/name:/ {
    if ($2 == "\"res5c_branch2c\"") {
        prune = 1
    } else {
        prune = 1
    }
}
/type:/ {
    if ($2 == "\"Convolution\"") {
        conv = 1
    }
}
// {
    if (num == 0)
        print $0
    num = 0
}

