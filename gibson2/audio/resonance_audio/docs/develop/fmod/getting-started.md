---
title: FMOD
weight: 120
getting_started: true
---

<img src="{{ site.baseurl }}/images/fmod/fmod_hero.png">

Get started using Resonance Audio plugins for FMOD. This guide shows you how
to add and configure the plugins in your FMOD Studio projects.

## Set up your development environment
* Download and install the latest version of [FMOD Studio](//www.fmod.com/download){: .external}.

### About the Resonance Audio FMOD plugin package
The Resonance FMOD plugin package is included by default in FMOD Studio
versions 1.10.02 and later.

The package has the following FMOD::DSP audio plugins.
<table>
  <tr>
    <th scope="col">Plugin</th>
    <th scope="col">Description</th>
  </tr>
  <tr>
    <td><b>Resonance Audio Listener</b>
    </td>
    <td>
        <ul>
        <li> Enables binaural spatialization of sound sources created using the
           <b>Resonance Audio Source</b> or <b>Resonance Audio Soundfield</b> plugins.</li>
        <li>Additionally handles <a href="{{ site.baseurl }}/develop/fmod/game-engine-integration">room effects simulation</a>.</li>
      </ul>
    </td>
   </tr>
   <tr>
    <td><b>Resonance Audio Source</b>
    </td>
    <td>
      <ul>
      <li>Spatialize mono sound sources anywhere around a listener.</li>
       <li> Apply effects such as distance attenuation and
        directivity patterns.</li>
        <li> Requires <b>Resonance Audio Listener</b></li>
      </ul>
    </td>
   </tr>
   <tr>
    <td><b>Resonance Audio Soundfield</b>
    </td>
    <td>
      <ul>
        <li>Play first-order Ambisonic soundfields that react to a user's
          head rotation.</li>
        <li>Requires <b>Resonance Audio Listener</b></li>
      </ul>
    </td>
  </tr>
</table>


## Add the Resonance Audio Listener plugin to your project
The Resonance Audio Listener is required for using the
Resonance Audio Source and Resonance Audio Soundfield plugins in your project.

1.   Create a new empty project in FMOD Studio.

1.   Open the **Mixer** window using **ctrl** + **2** (Windows) or **cmd** + **2** (MacOS).

1.   In the **Mixer** window, select the Master Bus.

1.   Right-click on the deck and select **Add Effect** > **Plug-in Effects** >
     **Google** > **Resonance Audio Listener** to place the plugin on the Master Bus.

     You can also place the plugin on an event group's Mixer deck to spatialize
     specific sources. See [Where to place the Resonance Audio Listener](#where-to-place-the-resonance-audio-listener)
     for details.

1.   Position the Resonance Audio Listener plugin. You can position the
     Resonance Audio Listener before the fader so that the fader controls the
     plugin output.

     In rare cases where you need to bypass the fader, you can position the
     Resonance Audio Listener after the fader.



### Plugin features
The Resonance Audio Listener plugin has the following features.
<table>
  <tr>
    <th scope="col">Feature</th>
    <th scope="col">Description</th>
  </tr>
  <tr>
    <td>
      <b>Global Gain</b>
    </td>
    <td>Controls the gain applied to all Resonance Audio sources in your project.
    </td>
  </tr>
  <tr>
    <td>
      <b>Binaural Level Meter</b>
    </td>
    <td>Indicates the output levels of your binaural mix.
    </td>
  </tr>
</table>

### Where to place the Resonance Audio Listener
The Resonance Audio Listener plugin can be placed on any mixer. The Resonance
Audio Listener mixes binaurally rendered spatialized sources with the incoming
**stereo** input of the corresponding mixer.

**Spatialize all sources**<br>
You can place the Resonance Audio Listener plugin on the Master Bus deck.

**Spatialize specific sources**<br>
You can also opt to spatialize specific sources in your project.
To do this, create an event group and place the plugin on the event group's
mixer deck.


## Add a Resonance Audio Source to an event
After [adding the Resonance Audio Listener](#add-the-resonance-audio-listener-plugin-to-your-project)
to your project, you can add a Resonance Audio Source to an event.

1.   Right-click in the **Events** pane to create a new event.
1.   Select the event's master track and delete the FMOD default spatializer
     from the master track deck.
     <img src="{{ site.baseurl }}/images/fmod/fmod-setup-remove.png">
1.   Right-click in the master track deck.<br>
1.   Select **Add Effect** > **Plug-in Effects** > **Google** >
     **Resonance Audio Source** to place the plugin on the master track deck.

      Typically, you place the plugin on the master track deck.
     See [Where to place the Resonance Audio Source](#where-to-place-the-resonance-audio-source)
     for details on spatializing a specific track in an event instead.


### Plugin features

The Resonance Audio Source plugin has the following features.
<table>
  <tr>
    <th scope="col"></th>
    <th scope="col">Feature</th>
    <th scope="col">Description</th>
  </tr>
  <tr>
    <th scope="row"><b>1</b></th>
    <td><b>Distance attenuation curve</b>
    </td>
    <td>Shows a distance
      attenuation curve shape as a source moves towards or away from the
      user.
    </td>
  </tr>
  <tr>
    <th scope="row"><b>2</b></th>
    <td><b>Curve Type Selector</b>
    </td>
    <td>Choose between a linear distance attenuation curve,
      a logarithmic attenuation curve, a custom distance attenuation curve,
      a linear squared curve, or a logarithmic tapered distance attenuation curve.
    </td>
  </tr>
    <tr>
    <th scope="row"><b>3</b></th>
    <td><b>Gain</b>
    </td>
    <td>Gain applied to the given Resonance Audio Source.
    </td>
  </tr>
    <tr>
    <th scope="row"><b>4</b></th>
    <td><b>Source Directivity Curve</b>
    </td>
    <td>Indicates the degree to which a source emits
        sound in different directions.<br>
        <ul>
        <li><b>Omnidirectional curve:</b> Source emits sound uniformly in all directions.</li>
        <li><b>Cardioid curve:</b> Source emits sound mostly from the front and,
            to a lesser extent, to the sides, above and below. Source emits no sound from the back.</li>
        <li><b>Figure-eight curve:</b> Source emits sound from the front and back only.</li>
      </ul>
    </td>
  </tr>
      <tr>
        <th scope="row"><b>5</b></th>
    <td><b>Minimum and Maximum Distance</b>
    </td>
    <td>Set the distances at which the source is heard at full volume
      (minimum distance) and at which the attenuation stops and the source
      reaches its minimum volume (maximum distance). The curve shown indicates a
      volume falloff rate with distance given these two parameters.
    </td>
  </tr>
   <tr>
    <th scope="row"><b>6</b></th>
    <td><b>Bypass Room Effects</b>
    </td>
    <td>If you applied Resonance Audio room effects to your project,
        use this switch to opt out of applying room effects to a
        particular source.
    </td>
  </tr>
   <tr>
    <th scope="row"><b>7</b></th>
    <td><b>Spread</b>
    </td>
    <td>Approximate width of the given Resonance Audio Source.
        Higher values correspond to wider, less pointed, sources.
    </td>
  </tr>
    <tr>
    <th scope="row"><b>8</b></th>
    <td><b>Occlusion</b>
    </td>
    <td> Level of occlusion by virtual objects between the source and user.
         Increasing this parameter attenuates the source by removing
         high frequencies from it. The occlusion scale is exponential to allow
         the user finer control over the degree of filtering applied.
    </td>
  </tr>
   <tr>
    <th scope="row"><b>9</b></th>
    <td><b>Directivity</b>
    </td>
    <td>As this parameter increases, the directivity curve evolves according to
        the following patterns:
      <li><b>Omnidirectional</b> = 0</li>
      <li><b>Cardioid</b> = 0.5</li>
      <li><b>Figure-eight</b> = 1.0</li>
    </td>
  </tr>
     <tr>
     <th scope="row"><b>10</b></th>
    <td> <b>Directivity Sharpness</b>
       </td>
    <td>As this parameter increases, the directivity curve lobe width decreases.
        This creates narrower emission patterns for your sound sources.
    </td>
  </tr>
</table>

<img src="{{ site.baseurl }}/images/fmod/fmod-source-features.png">


### Where to place the Resonance Audio Source
**Spatialize all tracks**<br>
In most cases, you should place the Resonance Audio Source plugin in
an event's master track deck.

**Spatialize a particular track**<br>
You can also opt to spatialize a **particular** audio track for an
event. This directs the track's audio into the Resonance Audio
system and away from the events master track.

To do this, place a Resonance Audio Source plugin in the deck for the audio
track.


## Add the Resonance Audio Soundfield plugin to an audio track
After [adding the Resonance Audio Listener](#add-the-resonance-audio-listener-plugin-to-your-project)
to your project, you can use the Resonance Audio Soundfield plugin to play
first-order Ambisonic (FOA) soundfields.

The Resonance Audio Soundfield plugin supports FOA encoded in the AmbiX (ACN-SN3D)
format. The same format is [supported by YouTube](//support.google.com/jump/answer/6399746?hl=en){: .external}.

1.  Select the audio track(s) where you want to add your four channel FOA
    ambisonic soundfield audio file(s), and add the files accordingly.

1.  Use one of the [following options](#where-to-add-the-resonance-audio-soundfield-plugin)
    to add the Resonance Audio Soundfield plugin.

### Where to add the Resonance Audio Soundfield Plugin

Using either of these methods, you can audition soundfield rotation using
the 3D preview in the **Overview** pane.

#### Option 1: Place the plugin on the event master track

Using this option you can place multiple ambisonic soundfields on separate event
tracks within the same event.

* Right-click in the event's deck and select **Add Effect** >
  **Plug-in Effects** > **Google** > **Resonance Audio Soundfield** to place
  the plugin on the audio event deck.

<img src="{{ site.baseurl }}/images/fmod/AmbisonicMaster.png">


#### Option 2: Place the plugin and audio file on the same audio track

You can place the Resonance Audio Soundfield plugin on the same audio
track as the audio file and not on the event master track.

* Right-click in the track's deck and select **Add Effect** >
  **Plug-in Effects** > **Google** > **Resonance Audio Soundfield** to place
  the plugin on the audio track deck.

<img src="{{ site.baseurl }}/images/fmod/Ambisonic4ch.png">



## Next Steps
* See [Game engine integration]({{ site.baseurl }}/develop/fmod/game-engine-integration)
  to learn more about using the Resonance Audio FMOD plugin in Unity and Unreal.

