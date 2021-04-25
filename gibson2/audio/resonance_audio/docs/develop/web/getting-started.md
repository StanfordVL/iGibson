---
title: Resonance Audio SDK for Web
weight: 170
getting_started: true
---

<img srcset="{{ site.baseurl }}/images/web/web_hero_image.png 2x">

Get started using the Resonance Audio SDK for Web.

Resonance Audio is a real-time JavaScript SDK that lets you encode spatial
audio dynamically into a scalable Ambisonic soundfield for [Web Audio](//developer.mozilla.org/en-US/docs/Web/API/Web_Audio_API){: .external} applications.

This guide shows you how to install the Resonance Audio SDK and use it to
create an example scene.

## Set up your web project
Install the Resonance Audio SDK and include it in your project.

### Install the SDK
1.  Use [`npm`](https://www.npmjs.com/){: .external} to install the SDK in your web project:

        npm install resonance-audio

    You can also clone [this repository](https://github.com/resonance-audio/resonance-audio-web-sdk){: .external} to use
    the SDK file.

### Include the SDK in your project
1.  Include the SDK file in an HTML document using one of the following options.

    *  If you do not plan to modify the Resonance Audio source code, use the CDN:<br>
       <pre class="devsite-click-to-copy">&lt;script src="https://cdn.jsdelivr.net/npm/resonance-audio/build/resonance-audio.min.js">&lt;/script></pre>

    *  Developers planning to extend Resonance Audio source code can use the installed `node_modules`:<br>
       <pre class="devsite-click-to-copy">&lt;script src="node_modules/resonance-audio/build/resonance-audio.min.js">&lt;/script></pre>


## Build an example scene
The following steps show you how to build an example scene with audio output.
You can try out a live demo of the example [here](//cdn.rawgit.com/resonance-audio/resonance-audio-web-sdk/master/examples/hello-world.html){: .external}.<br>
Make sure to wear headphones to experience the scene's spatial audio.

Note: It is recommended that you use mono (single channel) audio assets for
spatialization in your projects. Otherwise, in some web browsers, spatial audio
effects may not be applied correctly.

### Create a Resonance Audio scene

1.  Create an `AudioContext` and Resonance Audio scene.

        // Create an AudioContext
        let audioContext = new AudioContext();

        // Create a (first-order Ambisonic) Resonance Audio scene and pass it
        // the AudioContext.
        let resonanceAudioScene = new ResonanceAudio(audioContext);

1.  Configure the scene's audio output.

        // Connect the scene’s binaural output to stereo out.
        resonanceAudioScene.output.connect(audioContext.destination);

### Add a room to the scene
Resonance Audio room modelling helps you create realistic spatial
audio reflections and reverberation for your scene.

1.  Start by defining room dimensions in meters.

        // Define room dimensions.
        // By default, room dimensions are undefined (0m x 0m x 0m).
        let roomDimensions = {
          width: 3.1,
          height: 2.5,
          depth: 3.4,
        };


1.  Define room materials for each of the room's six surfaces
    (four walls, a ceiling, and a floor).

        // Define materials for each of the room’s six surfaces.
        // Room materials have different acoustic reflectivity.
        let roomMaterials = {
          // Room wall materials
          left: 'brick-bare',
          right: 'curtain-heavy',
          front: 'marble',
          back: 'glass-thin',
          // Room floor
          down: 'grass',
          // Room ceiling
          up: 'transparent',
        };


1.  Add the room definition to the scene.

        // Add the room definition to the scene.
        resonanceAudioScene.setRoomProperties(roomDimensions, roomMaterials);


### Add an audio input source to the scene

1.  Create an `AudioElement` and load a source audio file into it.

        // Create an AudioElement.
        let audioElement = document.createElement('audio');

        // Load an audio file into the AudioElement.
        audioElement.src = 'resources/SpeechSample.wav';

1.  Generate a `MediaElementSource` using the `AudioElement`.

        // Generate a MediaElementSource from the AudioElement.
        let audioElementSource = audioContext.createMediaElementSource(audioElement);


1.  Add the new `MediaElementSource` to your scene as an audio input.

        // Add the MediaElementSource to the scene as an audio input source.
        let source = resonanceAudioScene.createSource();
        audioElementSource.connect(source.input);


### Position the source and render the scene

1.  To render the scene binaurally, position your source relative to the origin
    and play the audio.

        // Set the source position relative to the room center (source default position).
        source.setPosition(-0.707, -0.707, 0);

        // Play the audio.
        audioElement.play();


## Next Steps

*  See the [Resonance Audio Developer Guide]({{ site.baseurl }}/develop/web/developer-guide) for
   details on migrating existing web projects to Resonance Audio or working with the Resonance Audio SDK.

*  Try out more Resonance Audio demos:

    *  [vs. PannerNode](//cdn.rawgit.com/resonance-audio/resonance-audio-web-sdk/master/examples/vs-pannernode.html){: .external}
    *  [Room Models](//cdn.rawgit.com/resonance-audio/resonance-audio-web-sdk/master/examples/room-models.html){: .external}
    *  [Flock of Birds](//cdn.rawgit.com/resonance-audio/resonance-audio-web-sdk/master/examples/birds.html){: .external}
    *  [Treasure Hunt](//cdn.rawgit.com/resonance-audio/resonance-audio-web-sdk/master/examples/treasure-hunt.html){: .external}

*  Learn more about [Omnitone](//googlechrome.github.io/omnitone/#home){: .external},
   an Ambisonic decoder that Resonance Audio uses internally to render binaural Ambisonic audio output.

