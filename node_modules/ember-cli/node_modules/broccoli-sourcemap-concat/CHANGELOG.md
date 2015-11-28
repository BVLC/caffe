# 2.0.1

 * Newer fast-sourcemap-concat for upstream bugfixes.
 * Better perf due to reduced use of `stat` calls.

# 2.0.0

  * structure of output file is now as follows: 
    1. header
    2. headerFiles
    3. inputFiles
    4. footerFiles
    5. footer

    Previous, 4 and 5 where reversed. This made wrapping in an IIFE needless
    complex

  * headerFiles & footerFiles now explicity throw if provided a glob entry
  * any inputFiles that also exist in headerFiles and footerFiles are now
    dropped from inputFiles


# beginning of time
