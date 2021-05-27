#!/bin/sh
convert -delay 30 -loop 0 out/model*.png out/model.gif
convert -delay 30 -loop 0 out/disc*.png out/discriminator.gif
