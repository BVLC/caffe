For details about the Microsoft COCO ("Common Objects in Context") dataset [1],
visit mscoco.org.  This README provides instructions for downloading and
installing the tools and dataset.

1) Download and extract the COCO Python tools by running:

    ./download_tools.sh

2) Install the tools, and optionally download the data by running:

    cd coco/PythonAPI
    python setup.py install  # follow prompts to download or skip data

3) Download train/val/test splits using:

    ./get_coco2014_aux.sh

(or see the COCO README (tools/README) for more information).


[1] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona,
    Deva Ramanan, Piotr Dollár, and C. Lawrence Zitnick.
    "Microsoft COCO: Common Objects in Context."
    arXiv preprint arXiv:1405.0312 (2014).
