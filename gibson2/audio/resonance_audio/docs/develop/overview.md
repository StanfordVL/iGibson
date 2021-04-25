---
title: Developing with Resonance Audio
weight: 100
---


<img srcset="{{ site.baseurl }}/images/hero_images/RA_Develop.png 2x">

Get started developing with Resonance Audio.

## Developer advantages
Resonance Audio is powerful spatial audio technology optimized for performance
and multi-platform support.

### Advanced audio engineering features
Resonance Audio goes beyond basic 3D spatialization, providing powerful tools
for accurately modeling complex sound environments.

The SDK enables:

*  Sound source directivity customization
*  Near-field effects
*  Sound source spread
*  Geometry-based reverb
*  Occlusions
*  Recording of Ambisonic audio files


### Cross-platform support
Resonance Audio SDKs integrate seamlessly with the most popular game engines,
audio engines, and digital audio workstations (DAWs),
letting you focus on creating more immersive audio.

### Performance
Resonance Audio delivers high fidelity spatial audio at scale on mobile
and desktop.

#### Cost
Resonance Audio internally projects all sound sources into a global high-order
Ambisonic soundfield. This allows head-related transfer functions ([HRTFs](//en.wikipedia.org/wiki/Head-related_transfer_function){: .external})
to be applied just once to the soundfield rather than to individual sound sources
within it.

This optimization keeps the CPU costs per sound source at a minimum,
allowing playback of many more simultaneous sources than most traditional
per-sound-source spatialization techniques.

#### Quality
Ambisonic order in Resonance Audio is adjustable, letting you control
spatial resolution. Using higher-order Ambisonics gives you higher fidelity
output and better direct source localization.

Resonance Audio's digital signal processing algorithms are optimized to
spatialize hundreds of simultaneous 3D sound sources without compromising audio
quality, even on mobile.


## Get Started with Resonance Audio
Select a platform at left to get started with:

*  Installing Resonance Audio software and adding it to your projects
*  Developer guidance for fine-tuning configurations
*  Game engine integration for plugins

You can also get [design tips]({{ site.baseurl }}/develop/design-tips) on
making the most impact with Resonance Audio.

