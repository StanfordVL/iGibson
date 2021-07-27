---
title: Wwise
weight: 180
getting_started: true
---

<img srcset="{{ site.baseurl }}/images/wwise/wwise_hero2.png 2x">

Get started using the Resonance Audio plugin for Wwise.

The Resonance Audio Wwise integration incldues plugins for
spatial audio on the following platforms.

**Desktop**<br>

*  MacOS
*  Windows
*  Linux

**Mobile**<br>

*  Android
*  iOS


## Set up your development environment
*  You'll need Wwise version 2017.1.0.6302 or later.<br>
   You can download the latest version of the Wwise launcher [here](https://www.audiokinetic.com/download/){: .external}.

*  Download the [Resonance Audio for Wwise plugin package](https://github.com/resonance-audio/resonance-audio-wwise-sdk/releases){: .external}.


## Install the plugin package

1.   Extract and copy the Resonance Audio plugin package into your Wwise
     installation folder. Follow [Wwise's instructions on importing plugins](https://www.audiokinetic.com/library/edge/?source=SDK&id=source__control__install.html){: .external}.

### About the Resonance Audio WWise plugin package
The package has the following plugins.
<table>
  <tr>
    <th scope="col">Plugin</th>
    <th scope="col">Description</th>
  </tr>
  <tr>
    <td><b>Resonance Audio Renderer</b>
    </td>
    <td>
      Renders spatialized sound sources binaurally using
        <a href="https://info.audiokinetic.com/Ambisonics-in-wwise-overview">Wwise's Ambisonic pipeline</a>
    </td>
   </tr>
   <tr>
    <td><b>Resonance Audio Room Effects</b>
    </td>
    <td>
     Simulates room acoustics for each sound source in the environment.<br>
      The simulation uses early reflections and late reverberation calculated
      from room model properties.
    </td>
  </tr>
</table>



## Using the Resonance Audio Renderer plugin

1.   Create a new project in Wwise Authoring.

1.   Create a new **Audio Bus** under the master audio bus and set its configuration
     to one of the Ambisonics channel configurations. Increasing the Ambisonic
     order produces higher fidelity output. For example, use **Ambisonics-3-3**
     (Third-order Ambisonics) for the highest fidelity.

     <img srcset="{{ site.baseurl }}/images/wwise/01-renderer.png 2x">

1.   Add the **Resonance Audio Renderer** as an effect to the Ambisonic **Audio Bus**.

     <img srcset="{{ site.baseurl }}/images/wwise/02-renderer.png 2x">

1.   Import an audio file to the project.

1.   Create a sound source using the audio file.

1.   In the **Positioning** tab, set the sound source to **3D**.
     You can also opt to try different source
     positions by setting **Position Source** to **User-defined**.

     <img srcset="{{ site.baseurl }}/images/wwise/03-renderer.png 2x">

1.   Configure the sound source **Output Bus** to route it to the **Audio
     Bus**.

     <img srcset="{{ site.baseurl }}/images/wwise/04-renderer.png 2x">

1.   Make sure to wear headphones and press **Play**.
     You should hear the sound binaurally rendered according to the sound
     position.



## Using the Resonance Audio Room Effects plugin

1.   Follow the instructions in this guide for [setting up the Resonance Audio renderer plugin](#using-the-resonance-audio-renderer-plugin).

1.   Create a new **Audio Bus** under the master audio bus and set its channel
     configuration to **Stereo**.

     <img src="{{ site.baseurl }}/images/wwise/05-room.png" width="200">

1.   Add **Resonance Audio Room Effects** as a mixer plugin to the **Room Effects Bus**.

     <img srcset="{{ site.baseurl }}/images/wwise/06-room.png 2x">

     The **Master Bus** hierarchy should now have the corresponding buses for
     the plugins.

     <img srcset="{{ site.baseurl }}/images/wwise/07-room.png 2x">

 1.   Add an **Auxiliary Bus** under the **Audio Bus** to access the direct
      sound of each source alongside the room effects mix using the Auxiliary
      Sends.

      <img src="{{ site.baseurl }}/images/wwise/08-room.png" width="200">


1.   Select your 3D sound and route its output to the **Room Effects Bus**.
     Add the Ambisonic auxiliary mix bus as one of the **Auxiliary Sends**.

      <img srcset="{{ site.baseurl }}/images/wwise/09-room.png 2x">

1.   Switch to the **Mixer plug-in** tab to make sure that routing has been set
     up correctly. You should be able to see the Room Effects plugin UI here.

      <img srcset="{{ site.baseurl }}/images/wwise/10-room.png 2x">


Note: If room properties are not configured, the room effects bus outputs silence.<br><br>
       Set room properties using the [`AK::SoundEngine::SendPluginCustomGameData`](https://www.audiokinetic.com/library/edge/?source=SDK&id=namespace_a_k_1_1_sound_engine_abeb321ed5095bfedba3c1ab0a1878815.html){: .external}
       method in your project's game engine integration.

## Using multiple Ambisonic pipelines
You can optionally mix different types of Ambisonic inputs into one
renderer to improve performance.

This can be useful when you are rendering Ambisonic sounds with
different orders rather than point sources directly in the same pipeline.

To set up a multiple Ambisonic pipeline:

1.  Extend the Ambisonic **Auxiliary Bus** approach for each Ambisonic order.
    To do this, follow the instructions in this guide for creating auxiliary
    Ambisonic bus [using the Resonance Audio Room Effects plugin](#using-the-resonance-audio-room-effects-plugin).

1.  Create an auxiliary bus for each Ambisonic order by selecting the
    corresponding channel configuration for that bus under the main
    audio bus.

1.  Change the auxiliary send for the sound to the Ambisonic order that you want.


### Example
As an example, you can create an **Ambisonic_FOA_Mix** auxiliary bus with an
**Ambisonic 1-1** channel configuration.

This bus renders the routed sounds in First-order Ambisonics (FOA)
regardless of the binaural renderer order in the main audio bus.

<img src="{{ site.baseurl }}/images/wwise/11-room.png" width="200">

## Next Steps
*  See [Game Engine Integration](game-engine-integration)
   for details on integrating the Resonance Audio plugins for Wwise into Unity
   or Unreal projects.

