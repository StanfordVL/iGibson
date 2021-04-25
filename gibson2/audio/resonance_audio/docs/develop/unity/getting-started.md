---
title: Unity
weight: 140
getting_started: true
---

<img srcset="{{ site.baseurl }}/images/unity/unity_hero_dark.png 2x">

Get started using Resonance Audio in Unity. This guide shows you how to:

*  Set up Unity for development with Resonance Audio.
*  Explore how Resonance Audio components render realistic spatial audio in a
   demo scene.
*  Add Resonance Audio components to your scenes.
*  [Upgrade projects](#upgrading-existing-projects-from-google-vr-audio-components) with Google VR SDK audio components to Resonance Audio.

## Set up your development environment

Software requirements:

*   Install [Unity 2017.1](//unity3d.com/get-unity/download){: .external} or
    newer.

*   Download the latest `ResonanceAudioForUnity_*.unitypackage` from the
    [releases](//github.com/resonance-audio/resonance-audio-unity-sdk/releases){: .external}
    page.

    The download includes a demo scene that you will explore in this guide.


## Create a new Unity project and import the SDK
1.  Open Unity and create a new **3D** project.

1.  Select **Assets** > **Import Package** > **Custom Package**.

1.  Select the `ResonanceAudioForUnity_*.unitypackage` file that you downloaded.

1.  In the **Importing Package** dialog, click **Import**.
    Accept any [API upgrades](//docs.unity3d.com/Manual/APIUpdater.html){: .external} if prompted.


## Configure your Unity project to use Resonance Audio

1.  Use **Edit** > **Project Settings** > **Audio** to open the AudioManager
    settings.

1.  Select **Resonance Audio** as the **Spatializer Plugin**.

1.  Select **Resonance Audio** as the **Ambisonic Decoder Plugin**.


## Try out the Resonance Audio demo
The Unity software package includes a simple demo scene in which you look for
a cube that moves around the scene when you click on it.

1.  In the Unity **Project** window, go to **Assets** > **ResonanceAudio** >
    **Demos** > **Scenes** and double-click `ResonanceAudioDemo`.

1.  Make sure to wear headphones to experience the spatialized audio. Click
    **Play** in the Unity Editor. You should hear the cube sound played back in 
    the scene.

1.  Interact with the scene using your mouse:

    *  Left-click in the **Game** view and move the mouse to look around the
       scene. Listen to the audio clip from different directions.
    *  In the Game view, left click on the cube to "teleport" it around the scene.


## Resonance Audio components
The Resonance Audio SDK for Unity includes the following components for rendering
spatial audio.

<table>
    <tr>
      <th scope="col" style="white-space: nowrap;">Component / Prefab name</th>
      <th scope="col">Description</th>
  </tr>
  <tr>
    <td><b>ResonanceAudioListener</b>
    </td>
     <td>
       <ul>
         <li>Enhances Unity's <b>AudioListener</b> features by introducing additional optional parameters,
   such as global gain and source occlusion masks.</li>
     <li>Includes an Ambisonic soundfield recorder component that allows baking
           spatial audio sources in the scene into an Ambisonic soundfield.</li>
 <li>
   Requires a Unity <b>AudioListener</b> in the same game object.</li>
 </ul>
    </td>
  </tr>
    <tr>
    <td><b>ResonanceAudioSource</b>
    </td>
   <td>
     <ul>
       <li>Enhances Unity's <b>AudioSource</b> features by introducing additional optional parameters
       such as directivity patterns, occlusion, and rendering quality.</li>
<li>
  Requires a Unity <b>AudioSource</b> in the same game object.</li>
</ul>
    </td>
  </tr>
      <tr>
    <td><b>ResonanceAudioSoundfield</b>
    </td>
   <td>
     <ul>
       <li>Represents full 360° spatial audio by encoding sound waves on a
         virtual sphere around a listener.
       </li>
   </ul>
    </td>
  </tr>
    <tr>
    <td>
      <b>ResonanceAudioRoom</b>
    </td>
   <td>
     <ul>
       <li>Simulates room effects for a particular space by introducing dynamic
         early reflections and late reverberation.</li>
<li>Uses the <b>Transform</b> properties of the attached game object and applies
   room effects accordingly.</li>
       <li>The corresponding room effects are enabled whenever the <b>AudioListener</b> is
   inside the specified boundaries of the room model.</li>
   </ul>
    </td>
  </tr>
     <tr>
       <td><b>ResonanceAudioReverbProbe</b>
    </td>
   <td>
     <ul>
       <li>Offers an advanced option for
finer modeling of spaces and more nuanced reverb effects.</li>
   </ul>
    </td>
  </tr>
  </table>

Learn more about adding these components to your projects in the
[Developer Guide for Resonance Audio for Unity]({{ site.baseurl }}/develop/unity/developer-guide).



## Upgrading existing projects from Google VR audio components
Upgrade existing projects from Google VR audio components to Resonance Audio
(Requires [Unity 2017.1](//unity3d.com/get-unity/download){: .external} or
later).

The new Resonance Audio SDK components have similar functionality and
configuration properties to those in the Google VR SDK.

Upgrading to Resonance Audio requires updating the Unity project's audio
settings, replacing the Google VR SDK audio components with corresponding
Resonance Audio components, and then copying component properties accordingly.


### Update the project audio settings and replace Google VR components
1.  Use **Edit** > **Project Settings** > **Audio** to open the AudioManager
    settings.

1.  Select **Resonance Audio** as the **Spatializer Plugin**.

1.  Select **Resonance Audio** as the **Ambisonic Decoder Plugin**.


### `GvrAudioListener`

1.  Add a new `ResonanceAudioListener` component attached to the Main Camera.

1.  Copy the `GvrAudioListener` settings, such as the global gain and occlusion
    mask, to the newly created `ResonanceAudioListener`.
    Note that the global quality option is now replaced by a real-time source
    based rendering quality setting in `ResonanceAudioSource` component.

    Additionally, a runtime setting to enable the stereo speaker mode option
    has been added. This option allows all sources to be rendered with
    stereo panning.

1.  Remove the `GvrAudioListener` component from the game object.

1.  For additional details, review [Add an audio listener to your scene]({{ site.baseurl }}/develop/unity/developer-guide#add-an-audio-listener-to-your-scene) in the Developer Guide for Resonance Audio for Unity for more details.

### `GvrAudioSource`

1.  Add a new `ResonanceAudioSource` component to the game object that contains
    the `GvrAudioSource` component you want to replace.

    Unity automatically adds a standard **AudioSource** component to the game
    object if one does not already exist.

1.  Copy the `GvrAudioSource` properties:

    *  Copy the standard audio source properties, such as mute or volume, to the
       `AudioSource` component.

    *  Copy values specific to spatial audio, such as directivity,
       to the `ResonanceAudioSource`.<br>

    *  Note that the `GvrAudioSource`'s **Enable HRTF** option is now replaced by
       source based rendering **Quality** in Resonance Audio.

1.  Remove the `GvrAudioSource` component from the game object.

1.  For additional details, review [Add a sound source to your scene]({{ site.baseurl }}/develop/unity/developer-guide#add-a-sound-source-to-your-scene)
    in the Developer Guide for Resonance Audio for Unity.

### `GvrAudioSoundfield`
Resonance Audio uses the same `ResonanceAudioSource` component for sound sources
and Ambisonic soundfields.

Note that with the introduction of Ambisonic decoder plugins in Unity,
it is no longer necessary to provide separate stereo tracks as audio clips.

1.  Add a new `ResonanceAudioSource` component to the game object that contains
    the `GvrAudioSoundfield` component you want to replace.

    Unity automatically adds a standard **AudioSource** component to the game
    object if one does not already exist.

1.  Copy the `GvrAudioSoundfield` properties to the newly created `AudioSource`
    and `ResonanceAudioSource` components.

1.  Remove the `GvrAudioSoundfield` component from the game object.

1.  For additional details, review [Add an Ambisonic soundfield to your scene]({{ site.baseurl }}/develop/unity/developer-guide#add-an-ambisonic-soundfield-to-your-scene) in the Developer Guide for
    Resonance Audio for Unity.


### `GvrAudioRoom`

1.  Add a new `ResonanceAudioRoom` component to the game object that contains
    the `GvrAudioRoom` component you want to replace.

1.  Copy the `GvrAudioRoom` properties to the newly created `ResonanceAudioRoom`
    component.

1.  Remove the `GvrAudioRoom` component from the game object.

1.  For additional details, review [Add an audio room to your scene]({{ site.baseurl }}/develop/unity/developer-guide#add-an-audio-room-to-your-scene)
    in the Developer Guide for Resonance Audio for Unity.


### Delete Google VR audio assets and libraries

1.  Delete the Google VR audio libraries:

        Assets/GoogleVR/Plugins/Android/libs/armeabi-v7a/libaudioplugingvrunity.so
        Assets/GoogleVR/Plugins/Android/libs/x86/libaudioplugingvrunity.so
        Assets/GoogleVR/Plugins/iOS/libaudioplugingvrunity.a
        Assets/GoogleVR/Plugins/x86/audioplugingvrunity.dll
        Assets/GoogleVR/Plugins/x86_64/audioplugingvrunity.bundle
        Assets/GoogleVR/Plugins/x86_64/audioplugingvrunity.dll
        Assets/GoogleVR/Plugins/x86_64/libaudioplugingvrunity.so

1.  Delete the Google VR audio Objective-C (iOS) assets:

        Assets/GoogleVR/Plugins/iOS/GvrAudioAppController.h
        Assets/GoogleVR/Plugins/iOS/GvrAudioAppController.mm

1.  Delete the Google VR audio mixer asset:

        Assets/GoogleVR/Legacy/Resources/GvrAudioMixer.mixer

1.  Verify that your **scenes** do not make use of any Google VR audio
    components.

    For each scene in your project, try each of the following searches in the
    Hierarchy view's search field:

    ```t:GvrAudioListener```

    ```t:GvrAudioRoom```

    ```t:GvrAudioSoundfield```

    ```t:GvrAudioSource```

1.  Verify that your **prefabs** do not make use of any Google VR audio
    components:

    * Use `t:prefab` in the Project view's search field to identify all prefabs
      in your project

    * Examine each prefab to ensure it doesn't make use of any Google VR audio
      components.

1.  Verify that your **scripts** do not make use of any Google VR audio
    components.

1.  (Optional) Delete unused Google VR audio prefabs:

        Assets/GoogleVR/Legacy/prefabs/Audio/GvrAudioRoom.prefab
        Assets/GoogleVR/Legacy/prefabs/Audio/GvrAudioSoundfield.prefab
        Assets/GoogleVR/Legacy/prefabs/Audio/GvrAudioSource.prefab

1.  (Optional) Delete unused Google VR audio scripts:

        Assets/GoogleVR/Legacy/Editor/Audio/GvrAudioListenerEditor.cs
        Assets/GoogleVR/Legacy/Editor/Audio/GvrAudioRoomEditor.cs
        Assets/GoogleVR/Legacy/Editor/Audio/GvrAudioSoundfieldEditor.cs
        Assets/GoogleVR/Legacy/Editor/Audio/GvrAudioSourceEditor.cs
        Assets/GoogleVR/Legacy/Scripts/Audio/GvrAudio.cs
        Assets/GoogleVR/Legacy/Scripts/Audio/GvrAudioListener.cs
        Assets/GoogleVR/Legacy/Scripts/Audio/GvrAudioRoom.cs
        Assets/GoogleVR/Legacy/Scripts/Audio/GvrAudioSoundfield.cs
        Assets/GoogleVR/Legacy/Scripts/Audio/GvrAudioSource.cs


## Next steps

*  See the [Resonance Audio for Unity Developer Guide]({{ site.baseurl }}/develop/unity/developer-guide) to
   learn about using Resonance Audio Room Effects and Reverb Probes for realistic
   environmental audio.

*  Get [design tips for working with Resonance Audio]({{ site.baseurl }}/develop/design-tips).

*  See the [Resonance Audio SDK for Unity API Reference]({{ site.baseurl }}/reference/unity) for complete details
   on components and scripts in the SDK.



