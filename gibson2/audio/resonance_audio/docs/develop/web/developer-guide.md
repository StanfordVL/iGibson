---
title: Developing for the Web
weight: 171
exclude_from_menu: true
---

This guide provides developer information on:

*  Browser support
*  Migrating projects from PannerNode to Resonance Audio
*  Extending the Resonance Audio SDK for Web
*  Managing your Resonance Audio SDK package


If you are new to the Resonance Audio SDK for Web, see [Getting Started with Resonance Audio for the Web]({{ site.baseurl }}/develop/web/getting-started) to learn how to install the SDK and use it to create an example scene.


## Browser Support
The Resonance Audio SDK for Web supports the latest versions of the following
browsers.

*  Chrome
*  Firefox
*  Edge
*  Opera
*  Android
*  Safari and iOS


## Migrating projects using WebAudio's PannerNode to Resonance Audio

### Developer advantages
In addition to several [general developer advantages]({{ site.baseurl }}/develop/overview#developer-advantages),
migrating to Resonance Audio has specific benefits for web developers.

Resonance Audio was designed to support Ambisonics and to scale well with a
large number of sound sources, making your web application's audio
scalable and VR-ready. Resonance Audio also offers you direct access to Ambisonic
channels.

Resonance Audio provides the following performance benefits over PannerNode:

<img srcset="{{ site.baseurl }}/images/web/web_chart.png 2x">

You can try out a demo comparing Resonance Audio to PannerNode [here](//cdn.rawgit.com/resonance-audio/resonance-audio-web-sdk/master/examples/vs-pannernode.html){: .external}.


### How to migrate from PannerNode to Resonance Audio
To simplify the migration from WebAudio projects, the Resonance Audio Web SDK
was designed with an API similar to [WebAudio's Panner Node](//developer.mozilla.org/en-US/docs/Web/API/PannerNode){: .external}.

For example, here is the same configuration in PannerNode and Resonance Audio:

**PannerNode**

    // Create a PannerNode Panner object.
    let panner = audioContext.createPanner();

    // Initialize the panner properties.
    panner.panningModel = 'HRTF';

    // Connect an input to the panner.
    audioElementSource.connect(panner);

    // Connect the panner to audio output.
    panner.connect(audioContext.destination);

    // Set the panner and listener positions.
    panner.setPosition(x, y, z);
    audioContext.listener.setPosition(x, y, z);


**Resonance Audio**

    // Create a Resonance Audio Source object with properties.
    let source = resonanceAudio.createSource({
    });

    // Connect an input to the source.
    audioElementSource.connect(source.input);

    // Connect Resonance Audio’s output to audio output.
    resonanceAudio.output.connect(audioContext.destination);

    // Set the source and listener positions.
    source.setPosition(x, y, z);
    resonanceAudio.setListenerPosition(x, y, z);

To start migrating to Resonance Audio, see the [Getting Started with Resonance Audio]({{ site.baseurl }}/develop/web/getting-started) instructions for [installing]({{ site.baseurl }}/develop/web/getting-started#install-the-sdk) and [including]({{ site.baseurl }}/develop/web/getting-started#include-the-sdk-in-your-project) the Resonance Audio SDK in your projects.


## Extending the Resonance Audio SDK for Web
Developers are encouraged to extend Resonance Audio and to contribute to the
open source Resonance Audio SDK for Web project. See the [CONTRIBUTING.md on the
Resonance Audio GitHub repo](https://cdn.jsdelivr.net/npm/resonance-audio/CONTRIBUTING.md){: .external} for more information.


## Managing your Resonance Audio SDK software package
The Resonance Audio SDK uses [webpack](https://webpack.js.org/concepts/){: .external} to
build the minified library and to manage dependencies.

You can use the following [npm](https://www.npmjs.com/){: .external} commands to manage your
software package.

      npm install         # install dependencies.
      npm run build       # build a non-minified library.
      npm run watch       # recompile whenever any source file changes.
      npm run build-all   # build a minified library and copy static resources.
      npm test            # launch the test runner
      npm run eslint      # lint code for ES6 compatibility.
      npm run build-doc   # generate documentation.

## API Reference
*  See the [Resonance Audio SDK for Web API Reference](/resonance-audio/reference/web) for
   complete details.


## Additional Resources
*  Learn more about [Omnitone](https://googlechrome.github.io/omnitone/#home){: .external},
   an Ambisonic decoder that Resonance Audio uses internally to render binaural
   Ambisonic audio output.

*  Report any issues with the Resonance Audio SDK [here](https://github.com/resonance-audio/resonance-audio-sdk-web/issues){: .external}.


