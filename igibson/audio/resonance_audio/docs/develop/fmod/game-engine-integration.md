---
title: Game Engine Integration
weight: 121
exclude_from_menu: true
---

This guide shows you how to integrate the Resonance Audio plugin for FMOD into
Unity or Unreal projects.

If you are new to the Resonance Audio plugin for FMOD,
see [Get Started with the Resonance Audio Plugin for FMOD]({{ site.baseurl }}/develop/fmod/getting-started).


## Unity integration
You can use the Resonance Audio plugin as part of any FMOD Unity project.

### Add the Resonance Audio plugin into your FMOD Unity project
1.  Download and import the [FMOD Studio Unity Integration package](http://www.fmod.org/download/){: .external}
    into your Unity project. This package contains scripts needed for accessing
    the Resonance Audio plugins.
1.  Open Unity and navigate to **FMOD** > **Edit Settings**.
1.  Select your FMOD Studio project and click **Add Plugin**.
1.  Type `resonanceaudio` in the text field.


### Room effects in Unity FMOD projects
[Resonance Audio provides room effects in Unity]({{ site.baseurl }}/develop/unity/developer-guide#room-effects-in-unity)
that let you:

*  Control how sounds react to the size and surface properties of the
   rooms you create.

*  React to the movements of the listener and the sound sources within
   your scene.

If you are using Resonance Audio room effects, first remove the FMOD Reverb
return effect from the master:

1.  In FMOD Studio, navigate to the **Mixer** window using **Ctrl** + **2** (Windows)
    or **Cmd** + **2** (MacOS).
1.  Select the reverb return from the **Routing** menu on the far left.
1.  Right-click on the return and select **Delete** to remove
    it.

#### Add room effects
Once you have sound sources using Resonance Audio spatialization within your project, you
can add room effects to your scenes.

The following C# scripts let you control room effects in your
Unity project.

*  `FmodResonanceAudio.cs`
*  `FmodResonanceAudioRoom.cs`
*  `FmodResonanceAudioRoomEditor.cs`

To add room effects to a scene:

1.  Add the `ResonanceAudio` folder to the `($UnityProject)/Assets/` folder.
1.  Attach the `FmodResonanceAudioRoom` component to a `GameObject` in your scene.
1.  Adjust the room properties as needed. Make sure
    that the Resonance Audio Listener and your Resonance Audio Sources are inside the selected
    room boundaries for the corresponding room effects to be applied.

      A [Unity Gizmo](//docs.unity3d.com/Manual/GizmosMenu.html){: .external}
      appears in the **Scene** view to show the spatial boundaries of your
      `ResonanceAudioRoom`.


You can add any number of Resonance Audio Rooms to a scene. As the game object with the
FMOD listener script attached moves around the scene, room effects react
smoothly in real time. The room effects update according to the room, listener,
and any Resonance Audio Sources that are currently within the room.


## Deploying the plugins on iOS using Unity
To deploy on iOS:

1.  After completing the previous steps, modify
    `($UnityProject)/Assets/Plugins/FMOD/fmodplugins.cpp` to uncomment the
    following declaration at the top of the file:

        FMOD_DSP_DESCRIPTION* FMOD_ResonanceAudioListener_GetDSPDescription();
        FMOD_DSP_DESCRIPTION* FMOD_ResonanceAudioSoundfield_GetDSPDescription();
        FMOD_DSP_DESCRIPTION* FMOD_ResonanceAudioSource_GetDSPDescription();

1.  In the same file, uncomment the following code provided in the `FmodUnityNativePluginInit()` method:

        uint32_t result = 0;
        result = FMOD5_System_RegisterDSP(system, FMOD_ResonanceAudioListener_GetDSPDescription(), nullptr);
        if (result != 0)
        {
          return result;
        }
        result = FMOD5_System_RegisterDSP(system, FMOD_ResonanceAudioSoundfield_GetDSPDescription(), nullptr);
        if (result != 0)
        {
          return result;
        }
        result = FMOD5_System_RegisterDSP(system, FMOD_ResonanceAudioSource_GetDSPDescription(), nullptr);
        if (result != 0)
        {
          return result;
        }

## Unreal Integration

To use the Resonance Audio FMOD plugins in Unreal Engine:

1.   Move the Resonance Audio binaries for the relevant platforms into one of the following directories:

     *  Your Unreal project's <code>Plugins/FMODStudio/Binaries/&lt;platform&gt;</code> directory.
          <aside class="caution"> <b>Caution:</b> Use this option if you
          are deploying your project on Android. The FMOD plugins will not load
          during deployment if they are not in the project folder.</aside>

     *  **MacOS:** <code>/Users/Shared/UnrealEngine/&lt;version&gt;/Engine/Plugins/FMODStudio/Binaries/&lt;platform&gt;</code>

     *  **Windows:** <code> &lt;UnrealEngineRootDir&gt;/Engine/Plugins/FMODStudio/Binaries/&lt;platform&gt;</code>


1.   Within your FMOD Studio project, make sure that you are building your banks into
     your Unreal Engine project’s `Content/FMOD` directory.

1.  Under **Edit** > **Project Settings** > **Packaging**, confirm that
    **FMOD** was added to **Additional Non-Asset Directories to Package**.

1.   Within UE4 go to **Edit** > **Project Settings...**. Scroll to the
     **FMOD Studio** settings at the lower left and select “FMOD Studio”.

1.   Go to the **Advanced** tab and add a plugin file named `resonanceaudio`.

1.   You should now be able to load and play FMOD events utilizing the Resonance Audio
     plugins.

     <aside class="caution"> <b>Caution:</b> Unreal does not automatically deploy
      FMOD plugins. To ensure that your plugins are deployed follow <a href="https://www.fmod.com/resources/documentation-api?page=content/generated/engine_ue4/getting_started.html" class="external">Unreal's guidelines</a>
    </aside>

1.   Create a file named `plugins.txt` within your game folder under
     <code>Plugins/FMODStudio/Binaries/&lt;platform&gt;</code> for each platform
     to which you are deploying. Each file should contain the name of the
     plugins you are using. To use the Resonance Audio plugins the file should include
     only the word `resonanceaudio`.

### Deploying the plugins on Android using Unreal
To deploy on Android:

1.  After completing the previous steps, add a file named
    `resonanceaudio_APL.xml` to `Plugins/FMODStudios/Binaries/Android/`.

1.  Add the following content to the file:

        <?xml version="1.0" encoding="utf-8"?>
        <!--Plugin additions-->
        <root xmlns:android="http://schemas.android.com/apk/res/android">
          <!-- init section is always evaluated once per architecture -->
          <init>
            <log text="resonanceaudio APL init"/> </init>

        <!-- optional files or directories to copy to Intermediate/Android/APK -->
        <resourceCopies>
          <log text="resonanceaudio APL copying files for $S(Architecture)/"/>
          <copyFile src="$S(PluginDir)/$S(Architecture)/libresonanceaudio.so"
                    dst="$S(BuildDir)/libs/$S(Architecture)/libresonanceaudio.so" />
        </resourceCopies>

        <!-- optional libraries to load in GameActivity.java before libUE4.so -->
        <soLoadLibrary>
          <log text="resonanceaudio APL adding loadLibrary references"/>
          <loadLibrary name="resonanceaudio" failmsg="resonanceaudio not loaded and required!" />
        </soLoadLibrary> </root>


## Room effects for the FMOD plugin in Unreal Engine or native code
To control Resonance Audio room effects directly from your C++ source code, pass the
plugin a `RoomProperties` struct containing your room parameters.

To do this, place a call to the [`FMOD::DSP::setParameterData` method](http://www.fmod.org/documentation/#content/generated/FMOD_DSP_SetParameterData.html){: .external} in your project's native code.

Pass in the data as a void pointer:

    FMOD_RESULT DSP::setParameterData(
      int index,
      void *data,
      unsigned int length
    );

To set the `RoomProperties` struct to control room effects, pass a value of `1` to
the first `index` parameter. For example:

    setParameterData( 1 /* Room properties index */, room_properties_ptr, length_bytes);

The `RoomProperties` struct header [is available on github](//github.com/resonance-audio/resonance-audio-fmod-sdk/tree/master/Plugins/include){: .external}.

