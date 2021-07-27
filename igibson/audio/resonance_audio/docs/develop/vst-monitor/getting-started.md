---
title: Audio Monitor VST
weight: 160
getting_started: true
---

<img src="{{ site.baseurl }}/images/vst/vst_updated_hero.png">

Use the Resonance Audio Monitor VST plugin to preview Ambisonic soundfield
assets that your users will hear in your YouTube 360 videos, VR, AR, or
gaming experiences.

The Resonance Audio Monitor VST plugin lets you render Ambisonic content just
like it will sound on platforms powered by Resonance Audio, including YouTube
and the Resonance Audio SDKs.

This guide shows you how to install the Resonance Audio Monitor VST plugin
in a Digital Audio Workstation (DAW) to monitor Ambisonic sound
assets during post-production.


## Set up your development environment
You can use the Resonance Audio Monitor VST plugin in any product that
supports Steinberg's Virtual Studio Technology (VST).

### Digital Audio Workstation (DAW) Requirements
You'll need a DAW that:<br>

*  Can render four-channel audio files
*  [Hosts VST plugins](//en.wikipedia.org/wiki/Virtual_Studio_Technology#VST_hosts){: .external}

    <img src="{{ site.baseurl }}/images/vst/VST3_Logo.png"><br>
    VST PlugIn Technology by Steinberg Media Technologies

### Audio asset requirements
*  To use the Resonance Audio monitor plugin, your audio assets must be
   [Ambisonically-encoded](//support.google.com/youtube/answer/6395969?co=GENIE.Platform%3DDesktop&hl=en){: .external}.<br>
   Support for different Ambisonic orders varies depending on the product
   where you are using Resonance Audio.

### Download the Resonance Audio monitor plugin
1.  Download the plugin from the [releases](//github.com/resonance-audio/resonance-audio-daw-tools/releases){: .external} page on GitHub.

## Import the Resonance Audio monitor plugin
1.  Copy the plugin file that you downloaded into one of the following
    directories.<br>
    *  **MacOS:** `~/Library/Audio/Plug-ins`
    *  **Windows:** `/Program Files(x86)/Common Files/VST3/`


## Integrate the plugin into your DAW project

### About this tutorial
For tutorial purposes, this guide uses [REAPER](//reaper.fm){: .external}, a DAW that hosts [Virtual Studio Technology (VST)](//en.wikipedia.org/wiki/Virtual_Studio_Technology){: .external}
plugins and can monitor and output Ambisonic audio files.

If you want to use REAPER, you can download and install the latest Windows or MacOS version
from the [REAPER download page](//reaper.fm/download.php){: .external}.

If you are not using REAPER, you can use any DAW that supports four-channel
audio files and hosts VST plugins.

Note that if you are using a different DAW, some instructions and interface
details might vary from those described here.


### Configure binaural stereo monitoring track to preview Ambisonic tracks

With the Resonance Audio Monitor VST plugin, Ambisonic audio sent to your
monitor is decoded and rendered using [Head Related Transfer Functions (HRTFs)]({{ site.baseurl }}/discover/concepts#simulating-sound-waves-interacting-with-human-ears).

1.  Open REAPER and select **File** > **New project**.

1.  Select **View** > **Monitoring FX** to create a monitoring FX track.

1.  Add the **ResonanceAudioMonitor** VST plugin to your monitoring FX:

    <img src="{{ site.baseurl }}/images/vst/resonance_audio_monitor_fx.png">

1.  If you are using other plugins, make sure that the
    Resonance Audio Monitor VST is the last plugin on the monitor track.

### Add an Ambisonic audio track

1.  Select **Track** > **Insert new track** to create a new audio track.

1.  Select **Insert** > **Media file...** and choose an ambiX file. You can download
    and use the `ResonanceAudioMonitorVst_*_samples.zip` file from the plugin
    [releases](//github.com/resonance-audio/resonance-audio-daw-tools/releases/){: .external} page on GitHub.

### Configure your MASTER bus routing
To be able to binaurally render your Ambisonic audio in the Monitor FX chain,
the MASTER bus routing needs to be set to 4 channels:

1.  In the **Mixer** view for your source track, click **Routing**.

1.  Set **Track channels** to **4**.

1.  In **Hardware**, set output to **Multichannel source** > **4 channels** > **1-4**.

    <img src="{{ site.baseurl }}/images/vst/master_bus_settings.png">

1.  Press **Play**. You should hear your Ambisonic audio rendered using HRTFs.

### (Optional) Add head rotation
You can opt to include head rotation support on your monitor track. This allows
Resonance Audio to [maintain sound source locations around the user in response
to their head movements]({{ site.baseurl }}/discover/concepts#simulating-sound-wave-interactions-with-their-environment).

1.  Download the [ambiX Ambisonic plugin suite](http://www.matthiaskronlachner.com/?p=2015){: .external}.

1.  Copy the plugin file that you downloaded into one of the following
    directories.<br>
    *  **MacOS:** `~/Library/Audio/Plug-ins`
    *  **Windows:** `/Program Files(x86)/Common Files/VST3/`

1.  In REAPER, select **View** > **Monitoring FX**.

1.  Add the **ambix_rotator_o1** VST plugin to your monitoring FX.

1.  Make sure the **ambix_rotator_o1** plugin, and any other plugins, are
    positioned **before** the **ResonanceAudioMonitor** plugin on the monitor
    track.

    <img src="{{ site.baseurl }}/images/vst/resonance_audio_monitor_rotator_fx.png">


## Next steps
To learn more about YouTube 360/VR video requirements, see [this support page](//support.google.com/youtube/answer/6395969){: .external}.

