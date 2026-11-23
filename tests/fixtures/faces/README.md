# Face fixture

`portrait.jpg`: Abraham Lincoln, photographed by Alexander Gardner on 8 November 1863.
**Public domain** (published before 1930, and the author died in 1882). Source:
[Wikimedia Commons, File:Abraham_Lincoln_O-77_matte_collodion_print.jpg](https://commons.wikimedia.org/wiki/File:Abraham_Lincoln_O-77_matte_collodion_print.jpg),
the 330 px wide thumbnail, unmodified.

`tests/test_framing.py` composites it onto a wide canvas at a known x to check that the
real face detector (YuNet) finds the face there.
