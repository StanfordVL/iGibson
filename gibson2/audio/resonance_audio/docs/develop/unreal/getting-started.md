---
title: Unreal
weight: 150
getting_started: true
---

<img src="{{ site.baseurl }}/images/unreal/unreal-hero-image.png">

This guide shows you how to set up your development environment and create a
spatialized sound effect with the Resonance Audio plugin for Unreal.

## Prerequisites
You'll need headphones to experience the spatial audio effects in this guide.

## Download and build Unreal Engine with Resonance Audio plugin from source
1.  [Gain access](https://www.unrealengine.com/ue4-on-github){: .external} to Epic's GitHub repository.
1.  Download or clone the 4.19-resonance-audio branch on the [repository](//github.com/resonance-audio-unreal/UnrealEngine){: .external}.<br>
    Note that this page displays a `404` error if you have not yet gained access
    to Epic's GitHub repository.

1.  Follow Epic's instructions on [Building Unreal Engine from Source](//docs.unrealengine.com/latest/INT/Programming/Development/BuildingUnrealEngine/index.html){: .external}.

    Warning: Avoid overwriting Resonance Audio binaries when running the `setup`
    script in a terminal window. If you see:
    <pre>
    The following file(s) have been modified:
      Engine/Plugins/Runtime/ResonanceAudio/...
      Engine/Source/ThirdParty/ResonanceAudioApi/...
    Would you like to overwrite your changes (y/n)?
    </pre>
    Type `n` to opt out of overwriting. This prevents the script from replacing
    Resonance Audio binaries with an older version.

## Create a new UE4 project with the Resonance Audio plugin

1.  Open the UE4Editor and create a new Blueprint/C++ project.

1.  Select **First Person**. Leave all other default settings
    in place.

1.  The Resonance Audio plugin should be enabled by default. To check its status,
    go to **Edit** > **Plugins** and type "Resonance Audio".

    Make sure that the **Resonance Audio** box is checked.


##  Set up plugins for selected platforms
In Unreal, you can select plugin components to use for the following platforms:

*  Android
*  iOS
*  Linux
*  Mac
*  Windows

Note: Currently, the **Spatialization** and **Reverb** plugins must be enabled
for the Resonance Audio plugin to work properly.

To set up the plugins:

1.  Open **Settings** and select **Project Settings...**.

1.  Scroll down to **Platforms** and select a supported platform.

1.  In the **Audio** section for all **Spatialization Plugin**,
    **Reverb Plugin** and **Occlusion Plugin** select
    **Resonance Audio** from the drop-down list.

1.  (Optional) Repeat the above steps for other platforms.


##  Configure plugin rendering quality settings

1.  Open **Settings** and select **Project Settings...**.

1.  Scroll down to **Plugins** > **Resonance Audio**.

1.  Open the **General** settings tab and find the **Quality Mode** drop-down.

1.  By default, **Binaural High Quality** is the selected mode. Select a
    different mode if needed.

Note: In all binaural modes, you can render each sound source using head-related
  transfer functions (HRTF) or stereo panning. HRTF is the default option.<br><br>
  If you select the **Stereo Panning** quality mode, you cannot change the
  rendering mode for individual sound sources to **HRTF** (binaural).

See [Spatialize sounds with the Resonance Audio plugin](#spatialize-sounds-with-the-resonance-audio-plugin)
for more details.

## Create a new object with an Audio Component

1.  Open the **Content Browser** and select **Content** > **StarterContent** >
    **Blueprints**.

1.  Locate the `Blueprint_Effect_Fire` blueprint.

1.  Drag and drop the blueprint into your level.
    <img src="{{ site.baseurl }}/images/unreal/unreal_drag_drop_blueprint.png">

Note: You can also create an empty sound source. To do this, create a new
'Ambient Sound' by dragging and dropping it into the editor window.

### Preview the sound effect

1.  Click **Play** to preview the game.

    You should see something like this and hear a fire sound effect.
    <img src="{{ site.baseurl }}/images/unreal/unreal_fire_sound_effect.png">

    The sound effect is rendered using standard UE4 stereo panning. In the next
    section, we'll add HRTF spatialization to the sound effect.


## Spatialize sounds with the Resonance Audio plugin

### Configure the spatialization method

1.  Select the `Blueprint_Effect_Fire` object and open the **Details** panel.

    Note that `Blueprint_Effect_Fire` has two components:

    *  `P_Fire` (Particle System Component)
    *  `Fire Audio` (Audio Component).

1.  Select `Fire Audio` and navigate to **Attenuation**.

1.  Select **Override Attenuation**.

1.  Scroll up to **Attenuation Spatialization**. Make sure that **Enable
    Spatialization** is selected.

1.  In the **Spatialization Method** drop-down list, change **Panning** to **Binaural**.

    <img src="{{ site.baseurl }}/images/unreal/unreal_attenuation_settings_window.png">

### Create new spatialization plugin settings

1.  Scroll down to **Attenuation Plugin Settings**.

1.  In the **Plugin Settings** > **Spatialization Plugin Settings** click **+**
    to add a new array element.

1.  Open the **Spatialization Plugin Settings** tab and click the drop-down
    currently set to **none**.

1.  Under **Create New Asset**, select **Resonance Audio Spatialization Source Settings**.

1.  Provide a name for the new settings. For example, `FireSoundSpatializationSettings`.

    <img src="{{ site.baseurl }}/images/unreal/unreal_attenuation_plugin_settings_window.png">

1.  You should now see a new icon in the **Content**
    section.

1.  Double-click the new icon to open `FireSoundSpatializationSettings`.

    <img src="{{ site.baseurl }}/images/unreal/resonance_audio_spatialization_settings.png">

    This menu lets you configure spatialization settings for sound sources.
    You can also [control certain parameters in real time]({{ site.baseurl }}/develop/unreal/developer-guide).
    For purposes of this tutorial, you can use the default settings.

    Note: The same spatialization settings apply to all sound sources that share
    the same **Resonance Audio Spatialization Source Settings** asset.


###  Play the sound effect with Resonance Audio spatialization
1.  Make sure to wear headphones to experience the spatialized audio.

1.  Go back to the main editor. Click **Play**
    to hear the sound effect with Resonance Audio spatialization.

## Ambisonic playback

The Resonance Audio plugin lets you binaurally decode First Order Ambisonic
soundfield files that you import into your project. Your soundfields are decoded
and filtered with the same [Head Related Transfer Functions (HRTFs)]({{ site.baseurl }}/discover/concepts#simulating-sound-waves-interacting-with-human-ears)
used to create the fire sound spatialization effect.

The Resonance Audio plugin supports Ambisonic audio encoded in the **AmbiX** format
(ACN channel sequence and SN3D normalization) and it uses the same binaural
rendering pipeline as, for example,
[YouTube VR/360](//support.google.com/youtube/answer/6395969?co=GENIE.Platform%3DDesktop&hl=en){: .external}.

When you play the Ambisonic sound field in your game, the Resonance Audio
decoder automatically detects your player's orientation so that sounds always
come from the correct direction.

### Import an AmbiX file into your project

1.  Prepare your Ambisonic asset by appending `_ambix` to its file name.
    This is required in order for Unreal to interpret the Ambisonic file correctly.

    For example, you can download and use the `voice_o1_ambix.wav` or
    `noise_o1_ambix.wav` test files from `ResonanceAudioMonitorVst_*_samples.zip`
    on [GitHub](//github.com/resonance-audio/resonance-audio-daw-tools/releases/){: .external}.

1.  In the **Content Browser**, select **Import** and choose the AmbiX file.

1.  Double click on your imported asset and verify that Unreal interpreted the
    Ambisonic file correctly. Make sure that in the **Sound** tab, the **Is
    Ambisonics** box is checked.

1.  Go back to the **Content Browser** and play your asset. You should now hear
    your Ambisonic sound field decoded binaurally using the Resonance Audio
    decoder.

Note: When importing Ambisonic audio into Unreal Engine, make sure to append
`_ambix` at the end of the file name. Otherwise, the audio engine will **not**
mark the asset as `Ambisonic` and will **not** route it to the Resonance Audio
Ambisonic binaural decoder.

## Next steps
See the [Resonance Audio plugin for Unreal Developer Guide]({{ site.baseurl }}/develop/unreal/developer-guide)
to learn about advanced configurations for:

   *  Spatialization parameters
   *  Room effects
   *  Occlusion

