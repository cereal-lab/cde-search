#!/bin/bash
convert -delay 10 -loop 0 img/*[13579].png res1.gif
convert -delay 10 -loop 0 img/*[02468].png res2.gif
convert -delay 10 -loop 0 img/*.png res.gif