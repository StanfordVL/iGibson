---
title: Android Studio
weight: 110
getting_started: true
---

<img srcset="{{ site.baseurl }}/images/android/android_hero_image.png 2x">

Get started adding Resonance Audio to your Daydream and Cardboard apps
in Android Studio.

This guide helps you set up your development environment and tour the
Resonance Audio features used in a sample virtual reality app.

Note: Resonance Audio is packaged as "Google VR Audio" in the Google VR SDK for
Android. Access Resonance Audio capabilities using `GvrAudio`
components, as shown in this guide.

## Set up your development environment
Hardware requirements:

*   **Daydream:** You'll need a [Daydream-ready phone](//vr.google.com/daydream/smartphonevr/phones/){: .external}
    and a [Daydream View](//madeby.google.com/vr/){: .external}.

*   **Cardboard:** You'll need an Android device running Android 4.4 'Kit Kat'
    (API level 19) or higher and a [Cardboard viewer](//vr.google.com/cardboard/get-cardboard/){: .external}.

*   You'll need headphones to experience the sample app's spatial audio.

Software requirements:

*   [Android Studio](//developer.android.com/studio/index.html){: .external}
    version 2.3.3 or higher.

*   Android SDK 7.1.1 'Nougat' (API level 25) or higher.<br>In Android Studio, go to
    **Preferences** > **Appearance and Behavior ** > **System Settings** >
    **Android SDK** to review or update installed SDKs.

*   [Google VR SDK for Android](//github.com/googlevr/gvr-android-sdk/releases){: .external} version 1.80.0 or higher.<br>


## Open the Google VR SDK project in Android Studio
1.  Extract the downloaded Google VR SDK into a convenient location.

1.  Open Android Studio and select **Open an existing Android Studio project**.<br>
    Select the directory where you extracted the Google VR SDK.

1.  In the **Project** window, find the **sdk-treasurehunt** module in
    **gvr-android-sdk** > **samples**.

     <figure>
       <img src="{{ site.baseurl }}/images/android-treasure-hunt-sample.png"
          width="400">
     </figure>

## Build and run the sample app

1.  Connect your phone to your machine using a USB cable.

1.  In Android Studio, select **Run** > **Run...** and select the
    **samples-sdk-treasurehunt** target.
    Android Studio compiles and runs the application on your phone.

1.  Put your phone into your viewer and use the app.
    Make sure to wear headphones to experience the app’s spatial audio.
    *  Look around for the large cube.
    *  **Daydream:** Point the controller at the cube and press the touchpad
        button to collect it.<br>
       **Cardboard:** Look at the cube and press the Cardboard button to
        collect it.
    *   The cube moves to a new location after a button press.

## How Resonance Audio works in the sample app
The Treasure Hunt app uses Resonance Audio to render realistic
spatial audio.

You can walk through the `TreasureHuntActivity` code in the sample app to
explore the Resonance Audio API.

### Open the source code in Android Studio
1.  To start exploring the code, navigate to
    **sdk-treasure-hunt** > **src** > **main** > **java** >
    **com.google.vr.sdk.samples.treasurehunt** > `TreasureHuntActivity`

### Initializing the audio engine
The Resonance Audio engine is initialized in the `TreasureHuntActivity`
`onCreate()` method.

    gvrAudioEngine =
        new GvrAudioEngine(this, GvrAudioEngine.RenderingMode.BINAURAL_HIGH_QUALITY);

Note that the [audio engine constructor](https://developers.google.com/vr/android/reference/com/google/vr/sdk/audio/GvrAudioEngine.html#GvrAudioEngine(android.content.Context, int)) lets you specify a rendering quality mode.

### Pausing and resuming audio
`TreasureHuntActivity` handles app pause or resume
events in `onPause()` and `onResume`. Within these methods,
the Resonance Audio engine also pauses and resumes.


### Preloading sound files
Sound files can be streamed during playback or preloaded into memory before
playback.

Perform preloading in a separate thread in order to avoid blocking the main
thread.

    new Thread(
            new Runnable() {
              @Override
              public void run() {
                // Preload the sound file
                gvrAudioEngine.preloadSoundFile(SOUND_FILE);
              }
            })
        .start();


### Creating, positioning, and playing audio for sound objects
In `TreasureHuntActivity`, a sound object represents spatial audio for the
moving cube that users look for and collect.

The same thread used for preloading the sound file in
`TreasureHuntActivity` is used for setting up the sound object:


    public void onSurfaceCreated(EGLConfig config) {
    [...]
        new Thread(
            new Runnable() {
              @Override
              public void run() {
                // Preload the sound file
                gvrAudioEngine.preloadSoundFile(SOUND_FILE);
                // Create the 'sourceId' sound object.
                // You can create multiple sound objects using the same sound file.
                sourceId = gvrAudioEngine.createSoundObject(SOUND_FILE);
                // Set the sound object position relative to the room model.
                gvrAudioEngine.setSoundObjectPosition(
                sourceId, modelPosition[0], modelPosition[1], modelPosition[2]);
                // Start audio playback of the sound object sound file at the
                // cube model position. You can update the sound object position
                // whenever the cube moves during run time.
                gvrAudioEngine.playSound(sourceId, true /* looped playback */);
              }
            })
        .start();
    }

### Making the sound object respond to model and user head movements
In `TreasureHuntActivity`, when the cube moves during run time, the
sound object that represents the cube audio moves with it.

    protected void updateModelPosition() {
      [...]
      // Update the sound location to match the new cube position.
      if (sourceId != GvrAudioEngine.INVALID_ID) {
        gvrAudioEngine.setSoundObjectPosition(
            sourceId, modelPosition[0], modelPosition[1], modelPosition[2]);
    }

On each frame update, `TreasureHuntActivity` gets the latest user head position
quaternion and passes it to the audio engine for rendering the sound
object's spatial audio.

    public void onNewFrame(HeadTransform headTransform) {
      [...]
      // Update the 3d audio engine with the most recent head rotation.
      headTransform.getQuaternion(headRotation, 0);
      gvrAudioEngine.setHeadRotation(
          headRotation[0], headRotation[1], headRotation[2], headRotation[3]);
      //Update the GVR Audio engine on each frame update.
      gvrAudioEngine.update()
    }


## Using the Resonance Audio SDK in your own projects
To use Resonance Audio in your own projects, set up dependencies.

If you are using [ProGuard](https://developer.android.com/studio/build/shrink-code.html){: .external}
in your app, add rules to ensure that it does not obfuscate any SDK code.

### Setting up Resonance Audio dependencies
1.   Configure your project level *build.gradle* file:
     *  Make sure that the default `jcenter()` repository location is declared.
     *  Declare an Android Gradle plugin dependency:<br>
        **Google VR SDK projects**: Use `gradle:2.3.3` or higher.<br>
        **Google VR NDK projects**: Use `gradle-experimental:0.9.3` or higher.

              allprojects {
                repositories {
                    google()
                    jcenter()
                }
              }

              dependencies {
                // The Google VR SDK requires version 2.3.3 or higher.
                classpath 'com.android.tools.build:gradle:2.3.3'

                // The Google VR NDK requires experimental version 0.9.3 or higher.
                // classpath 'com.android.tools.build:gradle-experimental:0.9.3'
              }


1.   Add Resonance Audio to other dependencies in your module level *build.gradle*
     files.
     As an example, review the `dependencies` declared for the Treasure Hunt app
     in **gvr-android-sdk** > **samples** > **sdk-treasurehunt** > *build.gradle*.

          dependencies {
             // Adds Google VR spatial audio support
             compile 'com.google.vr:sdk-audio:1.80.0'

             // Required for all Google VR apps
             compile 'com.google.vr:sdk-base:1.80.0'
          }

     For more information, see [Add Build Dependencies](//developer.android.com/studio/build/dependencies.html){: .external}
     in the Android Studio guide.

### Configure ProGuard
If you are using [ProGuard](https://developer.android.com/studio/build/shrink-code.html){: .external} to minimize your app's APK file,
make sure that ProGuard does not obfuscate any Google VR SDK or NDK code.
This makes it easier to debug stack traces in release builds.

Add the Google VR ProGuard [proguard-gvr.txt](https://github.com/googlevr/gvr-android-sdk/blob/master/proguard-gvr.txt){: .external}
rules to your module level *build.gradle* file:

    android {
        ...
        buildTypes {
            release {
                minifyEnabled true
                proguardFiles.add(file('../../proguard-gvr.txt'))
            }
        }
    }


## Next Steps
*  Review the [Resonance Audio API reference documentation](https://developers.google.com/vr/android/reference/com/google/vr/sdk/audio/package-summary)
   for complete details on Resonance Audio capabilities that you can use in your
   own projects.

*  Get [design tips for working with Resonance Audio]({{ site.baseurl }}/develop/design-tips).

