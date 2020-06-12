Uses of Complex Wavelets in Deep Convolutional Neural Networks
========================

This repo contains the latex source and images used to generate my [PhD thesis](build/thesis.pdf). I know looking at
other's PhDs was a great help in writing my own (e.g. [Yani's thesis](https://github.com/yanii/phd-thesis)), so I hope
to pass on the favour.

My tikz pictures are nice, but please don't judge me on my tikz code! I'm no expert, and was just able to wrangle
together what I needed. I suggest to anyone who is writing their thesis up in the near future to start making their
figures early. Each figure you make gets better and better.

Building
---------
I used `latexmk` to build my thesis from the command line. Running `latexmk thesis.tex` should build the entire thesis
(note that I have a local `.latexmkrc` file). This will output the files to the `build/` directory. Similarly, you can
build individual chapters by running `latexmk compile_chX.tex`.

Copyright/License
-----------------
I maintain copyright over the content of this document, but please feel free to copy the structure of my latex
project/tikz figures. If you wish to publish large parts of this work, please contact me. Additionally, I'd appreciate
citations of the original publication if you do use any content: [citeme.bbl](citeme.bbl).
