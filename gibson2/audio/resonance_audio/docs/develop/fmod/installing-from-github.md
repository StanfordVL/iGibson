---
title: Installing the Latest Resonance Audio Plugin Version From GitHub
weight: 122
exclude_from_menu: true
---

If you want to try the latest Resonance Audio features that might not be in
FMOD Studio yet, you can **get the latest Resonance Audio FMOD plugin
binaries** directly from GitHub.

If you are new to the Resonance Audio plugin for FMOD,
see [Get Started with the Resonance Audio Plugin for FMOD]({{ site.baseurl }}/develop/fmod/getting-started).

## Use the binaries in FMOD Studio

You can access the latest binaries from the [Resonance Audio SDK for FMOD repo on GitHub](https://github.com/resonance-audio/resonance-audio-fmod-sdk){: .external}.

Once you have cloned or downloaded the **resonance-audio-fmod-sdk**, you can find
all of the platform-specific binaries under <code>resonance-audio-fmod-sdk-master/Plugins/&lt;platform&gt;</code>.

To use the latest version of the plugin in FMOD Studio:

1.  Copy the Resonance Audio plugin dynamic library to the corresponding
    location for Windows or OSX:

    **Windows**

     * Copy <code>Plugins/Win/&lt;architecture&gt;/resonanceaudio.dll</code>
       into <code>Program Files\FMOD SoundSystem\FMOD Studio &lt;version&gt;\plugins</code>
     * Copy the <code>Plugins/RA_logo_small.png</code> and
        <code>Plugins/resonanceaudio.plugin.js</code> files into <code>Program Files\FMOD SoundSystem\FMOD     Studio &lt;version&gt;\plugins</code>.

    **OSX**

     *  Locate the `FMOD Studio.app` (usually in <code>/Applications/FMOD Studio/</code>).
     *  Right-click the `FMOD Studio.app` and select **Show package contents**.
     *  Copy <code>Plugins/Mac/resonanceaudio.dylib</code> into <code>FMOD Studio.app/Contents/Plugins</code>.
     *  Copy the <code>Plugins/RA_logo_small.png</code> and
        <code>Plugins/resonanceaudio.plugin.js</code> files into <code>FMOD             Studio.app/Contents/Plugins</code>.

## Use the binaries in game engine integrations

This guide assumes that you have already imported the newest FMOD Studio integration
into either Unity or Unreal Engine. If not, see
[Game Engine Integration]({{ site.baseurl }}/develop/fmod/game-engine-integration)
for instructions.

1.  After importing the FMOD Studio integration into your game engine, copy all of
    the platform-specific binaries that you need.

1.  After switching to a newer version of Resonance Audio for FMOD in
    Unity or Unreal Engine, make sure to rebuild all banks for your project
    with the new binaries.

### Unity

1.  Open your Unity project in the Unity **Editor** and navigate to
    **Assets** > **Plugins**.

1.  Depending on your operating system, make the following update for each
    platform that you are building for:

    **OSX**

     * Replace <code>resonanceaudio.bundle</code> with the
      <code>/resonanceaudio.dylib</code> file that you downloaded or cloned from
       GitHub.
     * **Rename this file** to <code>resonanceaudio.bundle</code>.


    **Windows**

     * Replace <code>Win/resonanceaudio.dll</code> with the corresponding file
       that you downloaded or cloned from GitHub.

1. Move the <code>resonance-audio-fmod-sdk-master/UnityIntegration/Assets/ResonanceAudio</code>
   folder that you downloaded or cloned from GitHub into your Unity project's
   **Assets** folder. Make sure to replace any existing **ResonanceAudio**
   folder there.

### Unreal Engine

1.  Make sure that Unreal Engine is closed.
1.  Replace each of the existing Resonance Audio plugin binaries in <code>&lt;YourProject&gt;/Plugins/FMODStudio/Binaries/&lt;Platform&gt;</code>
    with the corresponding Resonance Audio binaries from the folder that you
    downloaded or cloned from GitHiub.

