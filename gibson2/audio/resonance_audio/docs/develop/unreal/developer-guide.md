---
title: Developer Guide for Resonance Audio for Unreal
weight: 151
exclude_from_menu: true
---


This guide shows you how to configure and control spatialization, room effects,
and occlusion settings in real time using the Resonance Audio plugin for Unreal.

The steps here build on the completed steps in [Getting Started with the Resonance Audio Plugin for Unreal]({{ site.baseurl }}/develop/unreal/getting-started). If you are new to working with the Resonance Audio plugin for Unreal or have not yet set up
your development environment, start there.


## Prerequisites
This guide uses the `Blueprint_Effect_Fire` object that you [set up in the Getting Started guide]({{ site.baseurl }}/develop/unreal/getting-started#create-a-new-object-using-a-blueprint).

To follow the steps included here, you'll need the same `Blueprint_Effect_Fire`
object.

## Configure Spatialization plugin settings

### Directivity

1.  In the Unreal Engine editor, to go **Content**, double-click on the
    `FireSoundSpatializationSettings` asset that you created earlier. Expand the
    **Directivity** tab.

1.  In the **Pattern** field type `0.5` and in the **Sharpness** field type
    `2.0`.

    These values cause sound to be emitted from the front of the source and
    severely attenuated from the back.

1.  Check **Toggle Visualization**. This option lets you display the
    visualization of sound source directivity patterns directly in the
    viewport.

    You can apply a scale factor to your visualization. This does not affect
    the sound properties of your sound sources.

    By default, directivity visualization rendering is enabled only in the editor
    and not in the actual game.

Note: Directivity visualization meshes are created for all sound sources in
your project using the current **Resonance Audio Spatialization Source Settings** asset.

### Source spread (width)
This parameter lets you control how wide the source appears and can be used to
simulate volumetric sound sources as opposed to point-like sources.

For purposes of this tutorial, leave the value at the default `0.0`, which
corresponds to the narrowest possible source.

### Near-field effect

This parameter lets you simulate the behavior of a sound source at a short
distance, such as within one meter, from your head.

Set **Gain** to its maximum value of 9 in order to get the most physically
correct amplitude boost (including low-frequency boost) of a near-field source.

Make sure to avoid clipping when using normalized audio samples because the
output can be amplified by up to 25dB, depending on the frequency content.

### Distance attenuation

To enable Resonance Audio **Distance attenuation**:

1.  In your Audio Component **Details** panel, disable UE4 distance attenuation
    by unchecking **Enable Volume Attenuation** in the **Attenuation Distance**
    tab.

1.  Go back to `FireSoundSpatializationSettings` and select an attenuation
    curve in the **Distance attenuation** section. For example, select
    **Logarithmic**.

1.  Leave other parameters unchanged.

1.  Save your settings and close `FireSoundSpatializationSettings`.

## Control spatialization in real time using Blueprints
You can control directivity and spread parameters of your sound sources in
real time using Blueprints.

1.  In the editor, select your `Blueprint_Effect_Fire`
    object and go to the **Details** tab.

1.  In the blueprint editor, go to **Event Graph**.

1.  Right-click anywhere in the blueprint editor window to bring up a helper
    window with possible actions.

1.  Uncheck **Context Sensitive**.

1.  Type "Resonance Audio" to filter the available actions.<br>
    You should now be able to find functions for controlling Resonance Audio
    **Spatialization Source Settings**.

1.  Click **Set Sound Source Spread** to bring this function into the blueprint
    editor.

1.  (Optional) You can create a **Set Sound Source Directivity** or
    **Set Sound Source Near Field Effect Gain** functions in the same way.


### Create a Resonance Audio Spatialization Source Settings reference
The above functions require a **Target**. In our case, these are
**Resonance Audio Spatialization Source Settings** assets.

1.  In the **My Blueprint** panel on the left, select **Variables** and
    click **+** to create a new variable.

1.  Your object type should be **Resonance Audio Spatialization Source Settings**.
    Type this name and the relevant options should appear.

1.  Select an **Object Reference** as a variable type and provide a name for it.
    For example, you can use "FireSpatializationSettings".

1.  Drag and drop the reference into the blueprint editor.<br>
    When prompted, select **Get**.

1.  Click **Compile**.

### Set the variable's default value

1.  Click the `FireSpatializationSettings` reference that you just created.<br>

1.  In the **Details** window on the right, find the **Default Value** setting.

1.  Set the default value to the name of the [plugin settings asset that you created earlier](#create-a-spatialization-plugin-settings-asset).
    This asset's name is `FireSoundSpatializationSettings`.

1.  You can now use this reference as a target for **Set Sound Source Spread** and
    **Set Sound Source Directivity** functions.

1.  Click **Compile**.

### Review your results
You should now have something like this:

<img src="{{ site.baseurl }}/images/unreal/unreal_result_image.png">

The green inlets in the functions indicate that they expect float values. These
values are set whenever the functions are executed (gray triangles).

There are several ways in which you can interact with the functions. You can
also configure where the float data comes from in the game.

As an example, the following blueprint uses numeric keys +/- to change spread by
increments of five degrees and outputs debug messages to the screen:

<img src="{{ site.baseurl }}/images/unreal/unreal_numeric_keys_new.png">

## Using the Reverb plugin
Resonance Audio lets you add and control Room Effects in your game using either
a *Global Reverb* or by attaching *Resonance Audio Reverb Plugin Effects*
to Unreal's Audio Volumes. You can also use both methods simultaneously.

### Set up the Global Reverb Preset

To set up the Global Reverb Preset in your project:

1.  Open **Settings** and select **Project Settings...**.

1.  Scroll down to **Plugins** > **Resonance Audio**.

1.  Open the **General** settings tab and find the **Global Reverb Preset** drop-down.

1.  Click the drop-down and from **Create New Asset** select **Resonance Audio Reverb Plugin Preset**.

1.  Save the new asset as **GlobalReverbPreset**.

Now you can use the above preset to enable room effects in your scene.

1. Double-click the **GlobalReverbPreset** to open the settings for editing.

   Note: You can also access this preset in **Content Browser** > **Content**.

1. Select **Enable Room Effects**.

1. In **Dimensions** enter `4000`, `3000`, and `2000`. For now, these
   are arbitrary values, but we will change them later in this tutorial.

1. From the **Acoustic Materials** drop-down lists, choose **Uniform** material
   for each surface.

   Note: <b>Uniform</b> material ensures that reverb decay time and reflections strength
   are the same for all frequencies. You can select different materials as needed
   to better suit the actual environment you are modelling.

1. Leave all other settings unmodified.

1. Save your settings and click **Play**. The spatialized sound effect now plays
   with Room Effects applied.

### Add and control Room Effects using Audio Volumes
You can use the Resonance Audio reverb plugin with
[Unreal Engine's Audio Volumes](//docs.unrealengine.com/latest/INT/Engine/Actors/Volumes/#audiovolume){: .external}.

The reverb plugin enables multiple room effect volumes, sometimes called "reverb
zones", in a single level. This provides an intuitive visual interface
for creating multiple reverberant environments.

Audio Volumes in Unreal Engine pass their room transform properties (position,
rotation and dimensions) to the reverb plugin, where room effect parameters are
updated accordingly.

In this part of the tutorial, you'll create an Audio Volume in your project and
assign a Resonance Audio Reverb Plugin Preset to it. This enables reverberation
when the player is inside the Audio Volume.


### Create an Audio Volume
1.  Go to **Modes** and find **Volumes** > **Audio Volume**.

1.  Drag and drop the Audio Volume into the editor window. Make sure that the
    Audio Volume encompasses your sound source.

1.  (Optional) Go to **Details** > **Transform** > **Scale** to increase the
    Audio Volume scale. For example, increase the X & Y scale to 2.0.


### Add a Resonance Audio Reverb Plugin Preset to the Audio Volume

1.  Select the Audio Volume. In its **Details** panel, open the **Reverb** tab.

1.  In the **Reverb Plugin Effect** drop-down list, locate the **Create New
    Asset** section and select **Resonance Audio Reverb Plugin Preset**.

1.  Save the newly created asset as "AudioVolumeReverbPreset".

1.  Leave all other parameters in the Audio Volume as default.

1.  Double-click on the `AudioVolumeReverbPreset` asset to open it for
    editing.

1.  Select **Enable Room Effects**.

1.  Select **Get Transform from Audio Volume**. This enables the preset to use
    the Audio Volume's room transform for reverberation when the listener is
    inside the Audio Volume.

1.  From the **Acoustic Materials** drop-down lists, select the **Uniform**
    material for each surface.

1.  Leave all other settings unmodified.

1.  (Optional): To change the room transform (for example, location, rotation,
    dimensions of the room) modify the Audio Volume's transform directly.

1.  Click **Play**. Verify that a different reverb effect is applied to your
    sound source whenever the player is inside the audio volume.

Note: When the player leaves the Audio Volume, Resonance Audio
switches back to the `GlobalReverbPreset`. To disable room effects
altogether when the player is outside the Audio Volume(s), remove the
`GlobalReverbPreset` from **Project Settings...** > **Plugins** >
**Resonance Audio**. To do this, click the drop-down in **Global Reverb
Settings** and select **Clear** from the **Current Asset** section."


### Creating additional rooms
When creating additional rooms, you can use the same or different reverb
properties. Use one of the following options:

#### Use the same reverb properties
You can create additional rooms using the same reverb properties. To do this,
duplicate the existing Audio Volume.

If you change the transform of the new Audio Volume, the Resonance Audio plugin
detects and updates the transform when you switch rooms. However, changing other
room properties, such as reverb gain or materials, affects all Audio Volumes
that share the same preset.

#### Use different reverb properties

1.  Create a new Audio Volume.

1.  In the Audio Volume's **Details** panel, go to **Reverb** > **Reverb Plugin
    Effect**.

1.  Select **Create New Asset** > **Resonance Audio Reverb Plugin Preset** from
    the drop-down list.

1.  Provide a descriptive name for your new preset. For example, use
    "WoodenRoomReverbPreset" to name a room with wooden materials.

1.  Click **Save**.

1.  You can now modify the new preset to match the sound properties that you
    need for your space.


### Control Room Effects using Blueprints
You can control room effects parameters in real time using Blueprints. This is
similar to [controlling spatialization parameters in real time using Blueprints](#control_spatialization_in_real_time_using_blueprints)
with the exception that now you should use
**Resonance Audio Reverb Plugin Preset** as your object reference.

Within a blueprint, you can use the following functions to control Room Effects:
<img src="{{ site.baseurl }}/images/unreal/resonance_audio_blueprint_functions.png" width="400">

### Add or remove Room Effects for individual sound sources
You can choose which sound sources are affected by room effects. To do this:

1.  Select the `Blueprint_Effect_Fire` object and open the **Details** panel.

1.  Select `Fire Audio` and navigate to **Attenuation Reverb Send**.

1.  Select **Enable Reverb Send**.

1.  Scroll down to **Attenuation Plugin Settings**.

1.  In **Plugin Settings** > **Reverb Plugin Settings** click **+** to add a new
    array element.

1.  Open the **Reverb Plugin Settings** tab and click the drop-down currently
    set to **none**.

1.  Under **Create New Asset**, select **Resonance Audio Reverb Source Settings**.

1.  Save the newly created asset as "FireSoundReverbSettings".

1.  Open `FireSoundReverbSettings` asset for editing and in the **Reverb
    Settings** section, toggle **Bypass Room Effect** on.

1.  Click **Play** and verify that the sound is no longer affected by the room
    effects.

## Using the Occlusion plugin
Use the Resonance Audio Occlusion plugin to apply frequency-dependent
attenuation to the direct sound of your sound source whenever a visibility check
indicates that your sound source is occluded by another game object. This plugin
lets you ensure that only the direct sound path is affected, while early
reflections and reverb still render correctly.

To enable the occlusion effect for your sound source:

1.  Select the `Blueprint_Effect_Fire` object and open the **Details** panel.

1.  Select `Fire Audio` and navigate to **Attenuation Occlusion**.

1.  Select **Enable Occlusion**.

1.  Scroll down to **Attenuation Plugin Settings**.

1.  In **Plugin Settings** > **Occlusion Plugin Settings** click **+** to add a
    new array element.

1.  Open the **Occlusion Plugin Settings** tab and click the drop-down currently
    set to **none**.

1.  Under **Create New Asset**, select **Resonance Audio Occlusion Source Settings**.

1.  Save the newly created asset as "FireSoundOcclusionSettings".

1.  Open the `FireSoundOcclusionSettings` asset for editing and in the
    **Occlusion Settings** sections, increase the **Occlusion Intensity** to
    `1.0`.

1.  Click **Play**. Move behind the nearest cube object in the level. Verify
    that the fire sound is attenuated. It should sound muffled.

Note: Higher **Occlusion Intensity** values cause more frequencies to be
absorbed so that the sound becomes more low-pass filtered.


### Control reverb & occlusion source settings using Blueprints
You can control reverb & occlusion source settings in real time using
Blueprints. This is similar to
[controlling spatialization parameters in real time using Blueprints](#control_spatialization_in_real_time_using_blueprints)
with the exception that now you should use
**Resonance Audio Reverb Source Settings** &
**Resonance Audio Occlusion Source Settings** as your object references.


## Troubleshooting
To troubleshoot, you can inspect Resonance Audio `debug` messages.

1.  Go to **Window** > **Developer Tools** > **Output Log**.<br>
    This window is usually at the bottom of your screen next to the
    **Content Browser**.

1.  Type "ResonanceAudio" to filter the `debug` messages.

