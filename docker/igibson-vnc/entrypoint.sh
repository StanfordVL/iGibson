#!/usr/bin/env bash
VNC_PASSWORD=${VNC_PASSWORD:-112358}

# start X server with dummy display on :0
X -config /opt/misc/x-dummy.conf > /opt/logs/x-dummy.log 2>&1  &

sleep 2

# start xcfe desktop
startxfce4 > /opt/logs/xcfe4.log 2>&1 &

# start x11VNC server. Must publish port 5900 at `docker run`
x11vnc -display :0 -noxrecord -noxfixes -noxdamage -forever -passwd $VNC_PASSWORD > /opt/logs/x11vnc.log 2>&1 &

"$@"
