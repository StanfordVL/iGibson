---
title: Resonance Audio SDK for iOS
weight: 130
getting_started: true
---

<img src="{{ site.baseurl }}/images/ios/ios_hero_image.png">

Get started with Resonance Audio SDK for iOS. This guide shows you how to:

*  Set up your development environment
*  Try out a sample app
*  Explore the app source code to learn about the Resonance Audio SDK.

Note: Resonance Audio is packaged as "Google VR Audio" in the Google VR SDK for
iOS. Access Resonance Audio capabilities using `GVRAudio` components, as shown
in this guide.

## Set up your development environment

Hardware requirements:

*   You'll need an iPhone running iOS 7 or higher and a [Cardboard viewer](//vr.google.com/cardboard/get-cardboard/).


Software requirements:

*   Install [Xcode 7.1](//developer.apple.com/xcode){: .external} or higher.

*   Install [CocoaPods](//cocoapods.org){: .external}.


## Download and build the demo app

1.  Clone the Google VR SDK and the Treasure Hunt demo app from GitHub by
    running this command:

    <pre class="devsite-terminal devsite-click-to-copy">git clone https://github.com/googlevr/gvr-ios-sdk.git</pre>

2.  In a Terminal window, navigate to the `Samples/TreasureHunt` folder, and
    update the CocoaPod dependencies by running this command:

    <pre class="devsite-terminal devsite-click-to-copy">pod update</pre>

3.  In Xcode, open the **TreasureHunt** _workspace_
    (`Samples/TreasureHunt/TreasureHunt.xcworkspace`),
    and click **Run**.

    <img src="{{ site.baseurl }}/images/ios/xcode_2.png">


## Try out the app

In the Treasure Hunt sample app, you look for and collect cubes in 3D space.
Make sure to wear headphones to experience the game's spatial audio.

1.  Move your head in any direction until you see a cube.
    <img src="{{ site.baseurl }}/images/ios/green_cubes.png">
2.  Look directly at the cube. This causes it to turn orange.
    <img src="{{ site.baseurl }}/images/ios/orange_cubes.png">
3.  Press the Cardboard viewer button to collect the cube.


## How Resonance Audio works in the sample app
The Treasure Hunt sample app uses Resonance Audio to render realistic
spatial audio.

Walk through the Treasure Hunt code in the sample app to
explore the SDK.

### Open the source code in XCode
1.  To start exploring the code, navigate to
    **sdk-treasure-hunt** > **Samples** > **TreasureHunt** >
    `TreasureHuntRenderer.m`

### Importing the audio engine
`TreasureHuntRenderer` imports the GVR Audio engine SDK.

     `#import "GVRAudioEngine.h"`

### Constructing the audio engine
Sound files are declared for two sound assets and the GVR Audio engine
object is declared.

      // Sample sound file names.
      static NSString *const kObjectSoundFile = @"cube_sound.wav";
      static NSString *const kSuccessSoundFile = @"success.wav";

      @implementation TreasureHuntRenderer {
      [...]
      GVRAudioEngine *_gvr_audio_engine;
      int _sound_object_id;
      int _success_source_id;
      bool _is_cube_focused;
    }

If ARC is not enabled, a call to the `dealloc` method must be made. See the
Example Usage snippet below. In the Treasure Hunt app, the `stopSound` and `stop`
audio engine methods are called to stop audio playback.

    - (void)dealloc {
      [_gvr_audio_engine stopSound:_sound_object_id];
      [_gvr_audio_engine stop];
    }


### Initializing the GVR Audio engine
`TreasureHuntRenderer` initializes the GVR Audio engine, preloads two
sound files, and starts audio playback.

    - (void)initializeGl {
    [...]
      // Initialize the GVRAudioEngine with high binaural rendering quality.
      // High binaural quality is the default rendering mode.
      _gvr_audio_engine =
          [[GVRAudioEngine alloc] initWithRenderingMode:kRenderingModeBinauralHighQuality];
      // Preload sound files.
      // This method can be called on mono and multi-channel Ambisonic sound files.
      [_gvr_audio_engine preloadSoundFile:kObjectSoundFile];
      [_gvr_audio_engine preloadSoundFile:kSuccessSoundFile];
      //Start audio playback
      [_gvr_audio_engine start];

By default, the audio engine initializes with high binaural rendering quality.
See the API reference for additional [rendering quality options](//developers.google.com/vr/android/reference/com/google/vr/sdk/audio/GvrAudioEngine.RenderingMode).


### Creating a sound object
A sound object is created to represent the cube's audio. Sound objects take
in audio and spatialize it for playback.

    - (void)initializeGl {
        [...]
        // Create the first sound object
        _sound_object_id = [_gvr_audio_engine createSoundObject:kObjectSoundFile];

There are several methods for setting sound object properties, including:

*  Position
*  Distance rolloff
*  Volume

You can also check whether a sound object is currently playing audio.

See the API reference for further details.

#### Handling sound object removal
Removing the sound object is handled in the `clearGl` method.

    - (void)clearGl {
        // On removing the sound object, it is destroyed and its reference
        // becomes invalid. If the sound object has looping audio, the current
        // playback loop completes before the object is destroyed.
        [_gvr_audio_engine stopSound:_sound_object_id];
        [_gvr_audio_engine stop];

        [super clearGl];
      }



### Attaching the sound object to a game object
Once the sound object is created, `TreasureHuntRenderer` creates a new cube game
object and sets the sound object's position to match it.

    // Spawns the next cube at a new position.
    - (void)spawnCube {
      // Set the new position and restart the playback.
      [self setRandomCubePosition:kMinCubeDistance maxLimit:kMaxCubeDistance];
      [_gvr_audio_engine setSoundObjectPosition:_sound_object_id
                                              x:_cube_position[0]
                                              y:_cube_position[1]
                                              z:_cube_position[2]];
      [_gvr_audio_engine playSound:_sound_object_id loopingEnabled:true];
    }


### Making the sound object respond to user head movements
On each update, `TreasureHuntRenderer` gets the latest user head position
quaternion and passes it to the audio engine for rendering the sound
object's spatial audio.

    - (void)update:(GVRHeadPose *)headPose {
      // Update audio listener's head rotation.
      const GLKQuaternion head_rotation =
          GLKQuaternionMakeWithMatrix4(GLKMatrix4Transpose([headPose headTransform]));
      [_gvr_audio_engine setHeadRotation:head_rotation.q[0]
                                       y:head_rotation.q[1]
                                       z:head_rotation.q[2]
                                       w:head_rotation.q[3]];
      // Update the audio engine.
      [_gvr_audio_engine update];



### Playing audio on user interactions
If the user presses the Cardboard button to collect the cube,
`TreasureHuntRenderer` plays a "success" sound effect.

    - (void)handleTrigger {
      NSLog(@"User performed trigger action");
      // Check whether the user located the cube.
      if (_is_cube_focused) {
        // Initialize audio to indicate that the user successfully collected the cube.
         _success_source_id = [_gvr_audio_engine createStereoSound:kSuccessSoundFile];
        // Play the audio for successful cube collection.
        [_gvr_audio_engine playSound:_success_source_id loopingEnabled:false];
        // Vibrate the device to show successful cube collection.
        AudioServicesPlaySystemSound(kSystemSoundID_Vibrate);
        // Generate the next cube for the user to find.
        [self spawnCube];
      }
    }


### Pausing audio
`TreasureHuntRenderer` pauses the GVR Audio engine on app pause events.

    - (void)pause:(BOOL)pause {
      [super pause:pause];

      if (pause) {
        [_gvr_audio_engine pauseSound:_sound_object_id];
      } else {
        [_gvr_audio_engine resumeSound:_sound_object_id];
      }
    }


## Add Ambisonic soundfields and room effects

### Ambisonic soundfields
Resonance Audio lets you add Ambisonic soundfields to your apps. Ambisonic
soundfields are captured or pre-rendered 360&deg; recordings, similar
to 360&deg; video.

Use soundfields for:<br>

*  Accompanying 360&deg; video playback
*  Introducing background or environmental effects like rain or crowd noise
*  Prebaking 3D audio to reduce rendering costs.

Soundfields surround the user and respond to the user's
head rotation.


The `GVRAudioEngine` supports full 3D First-Order Ambisonic recordings
using ACN channel ordering and SN3D normalization. For more information, see our
[Spatial Audio specification](//github.com/google/spatial-media/blob/master/docs/spatial-audio-rfc.md#semantics){: .external}.

#### Creating Ambisonic soundfields
Ambisonic soundfield can be rendered from Ambix encoded (ACN/SND3d) multi-channel
audio assets.

When you construct an Ambisonic soundfield, you get a soundfield `ID` that you
can use to begin playback, adjust volume, or stop playback and remove the
soundfield.


### Room effects
The `GVRAudioEngine` lets you simulate room effects centered around the user.

A `GVRAudioRoom` has walls, ceiling, and a floor. When you enable room effects,
you can specify different [materials](//developers.google.com/vr/android/reference/com/google/vr/sdk/audio/GvrAudioEngine.MaterialName) for each of the room surfaces.

### Next Steps
*  See the [API Reference](//developers.google.com/vr/ios/reference/interface_g_v_r_audio_engine) for more details.

*  Get [design tips for working with Resonance Audio]({{ site.baseurl }}/develop/design-tips).


