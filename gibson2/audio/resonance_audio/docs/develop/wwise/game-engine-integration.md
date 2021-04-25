---
title: Wwise plugin Game Engine Integration
weight: 181
exclude_from_menu: true
---
# Resonance Audio Game Engine Integration


This guide shows you how to integrate the Resonance Audio plugins for Wwise into
your projects.

If you are new to the Resonance Audio plugins for Wwise, see
[Getting Started with the Resonance Audio Plugin for Wwise]({{ site.baseurl }}/develop/wwise/getting-started).


## About the Resonance Audio Wwise integration
The Resonance Audio Wwise integration incldues plugins for
spatial audio on the following platforms.

**Desktop**

*  MacOS
*  Windows
*  Linux

**Mobile**

*  Android
*  iOS

The Resonance Audio plugins follow the [standard Wwise plugin integration
process.](https://www.audiokinetic.com/library/edge/?source=SDK&id=integrating__elements__plugins.html){: .external}


## Room effects game engine integration
The Resonance Audio Room Effects plugin uses the plugin custom game data
structure in the Wwise SDK to set room properties in the environment.

Note: If room properties are not configured, the room effects bus outputs silence.<br><br>
      If you are using the Resonance Audio Room Effects Plugin, **make sure to
      set room properties** using the [`AK::SoundEngine::SendPluginCustomGameData`](https://www.audiokinetic.com/library/edge/?source=SDK&id=namespace_a_k_1_1_sound_engine_abeb321ed5095bfedba3c1ab0a1878815.html){: .external}
      method in your project's game engine integration. Resonance Audio provides
      reference scripts for you to check how this function can be used in Unity.

For more information, see [Room effects in Unity](#room-effects-in-unity) in
this guide.


## Unity integration

### Software requirements
You'll need Unity 5.6 or later to use the Resonance Audio plugin
package.

### Load the Resonance Audio plugins
Use the provided dynamic libraries to set up the Wwise integration with
Resonance Audio plugins. Using the dynamic libraries lets you avoid having to
recompile any code during installation.

1.  Download and install the Wwise Unity plugin. Follow the
    [Wwise Unity Integration guide](https://www.audiokinetic.com/library/edge/?source=Unity&id=main.html){: .external}.

1.  Set up your Unity project and link it to your Wwise project.

1.  Close the Unity Editor.

1.  Copy the Resonance Audio plugin dynamic libraries for each platform into
    the corresponding DSP folders in the Wwise integration.
1.  You can now access the plugins within Unity. See the following instructions
    for including the [room effects in Unity]({{ site.baseurl }}/develop/unity/developer-guide#room-effects-in-unity) using the
    provided scripts.

Caution:  Make sure to copy the **GeneratedSoundbanks** folder from your Wwise
          project into the **StreamingAssets** folder in your Unity project so that [Wwise
          Unity integration works as expected on deployed
          platforms](https://www.audiokinetic.com/library/edge/?source=Unity&id=pg__deploy.html){: .external}.

### Room effects in Unity
The Resonance Audio plugin package includes an example wrapper implementation of
room detection and multiple room management scripts for the Unity integration.
You can find the corresponding scripts in the **UnityIntegration** folder.

The **UnityIntegration** folder also includes the `WwiseResonanceAudioRoom` wrapper
script. This script encapsulates an audio room component that you can attach to
any game object in your Unity scene.

The `WwiseResonanceAudioRoom` component is based on the **ResonanceAudioRoom**
component in the [Resonance Audio SDK for Unity package](https://github.com/resonance-audio/resonance-audio-unity-sdk).

To use the included audio room component:

1.  Add the ResonanceAudio folder into the **Assets** folder of your Unity project.

1.  Attach the `WwiseResonanceAudioRoom` component to a `GameObject` in your
    scene.

1.  Adjust the `WwiseResonanceAudioRoom` properties as needed. Make sure that
    the audio listener and the audio sources are inside the room boundaries in
    order for room effects to work.

1.  In the Unity **Inspector** window, make sure that your **Room Effects Bus**
    `Name` field matches the name of the Audio Bus in your Wwise project where
    you attached the Resonance Audio Room Effects plugin. Typically, this is
    the **Room Effects Bus**. Update the `Name` field if it does not match.

    This step is required only once per scene. Additional
    `WwiseResonanceAudioRoom` components added to the scene are updated
    automatically with the corresponding plugin name.

See ["Room Effects in Unity" in the Unity Developer Guide]({{ site.baseurl }}/develop/unity/developer-guide#room-effects-in-unity)
for more details on working with audio rooms in your Unity scenes.

## Unreal integration

### Software requirements
To use the Resonance Audio plugin package, you'll need:

*  Unreal Engine version 4.16 or later
*  Wwise 2017.1.0 build 6302 or later


### Load the Resonance Audio plugins
Use the provided static libraries to recompile the
Wwise integration code with the Resonance Audio plugins. The Wwise integration
for Unreal does not currently support using dynamic libraries to load plugins.

1.  Download and install the Wwise Unreal plugin. Follow the
    [Wwise Unreal Integration guide](https://www.audiokinetic.com/library/edge/?source=UE4&id=installation.html){: .external}.

1.  Add the Resonance Audio plugin static runtime libraries to your project:

    *  In `AkAudioDevice.cpp` add `#include <AK/Plugin/ResonanceAudioFXFactory.h>`
    *  In `AkAudio.Build.cs` add `AddWwiseLib(Target, "ResonanceAudioFX");`

1.  Rebuild the integration code to complete plugin registration.

