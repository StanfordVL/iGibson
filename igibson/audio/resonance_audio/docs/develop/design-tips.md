---
title: Resonance Audio Design Tips
weight: 101
---

Resonance Audio dramatically increases immersion in VR, AR, media, and gaming. It
lets developers render audio from every direction, simulating real world sound
just like your ears were meant to hear it.

Learn how to achieve the most impact with Resonance Audio.

## Sound sources

Resonance Audio enables the spatialization of virtual sound sources and
Ambisonic soundfields.

Sound sources (also often referred as point sources) define an
audio source at a given point within a virtual environment.

Each sound source is rendered from a mono-audio stream and can
be configured with a directivity pattern and distance attenuation curves.

For example, if you attach a sound source to a bird, the user hears
the sound of the bird's wings or call changing naturally as it flies closer
or father away.

<img src="{{ site.baseurl }}/images/concepts/unity-add-bird-sound.png">

You can distribute sound sources throughout an environment to create
a general ambience.

When creating sound sources, use monophonic sound files that are
dry and do not include any reverb.

## Ambisonic soundfields
Like light probes for visual rendering, Ambisonic soundfields 
represent full 360 &deg; spatial audio by encoding sound waves on a virtual 
sphere around a listener.

Resonance Audio supports the decoding of up to third-order Ambisonics. Each
Ambisonic soundfield is rendered from a multi-channel input audio stream. The stream
is always rendered at the listener's location.

As Ambisonic files respond only to head rotation, soundfields
work best to represent sounds in the distance.

<img src="{{ site.baseurl }}/images/concepts/unity-atmosphere.png">

## Audio Room

Depending on the environment you are creating, you might need to use an audio
room in it.

Audio rooms provide early reflections and reverb. This helps make
sounds more realistic when there are walls or structures near the user's
position in a scene.

Configure the reverb and room material surfaces to match the audio effects of
the environment you are building.

Audio rooms are most useful when your scene takes place
inside a room. For realistic outdoor scenes, an audio room can feel less
natural. This is because outdoors, the ground might be your scene's only
reflective surface.

<img src="{{ site.baseurl }}/images/concepts/unity-use-room-model.png">

## Sound design tips

### Animate a sound source

If you want the user to notice a sound outside of their view, you can
animate the position of the sound. This helps the user to locate the sound more
quickly.

<img src="{{ site.baseurl }}/images/concepts/animate-sound.png">

### Repeat a sound

To help the user locate a sound, play it more than once.

Repeated sounds help users to recognize and locate them. For example, phones ring
multiple times to help us notice an incoming call and find the phone to answer
it.

You can achieve the same effect by using sounds that comprise many distinct
elements.

<img src="{{ site.baseurl }}/images/concepts/soundwaves.png">

### Use complex sounds
Craft sounds that have:

*  Sufficient volume levels
*  A full spectrum of frequencies
*  Complexity

Avoid using sounds that are too quiet, lack high frequencies, or are too simple,
such a as sine wave beep.

### Creating Ambisonic files

With [digital audio workstation](//en.wikipedia.org/wiki/Digital_audio_workstation){: .external}
(DAW) software and a plugin such as Ambix, you can create Ambisonic files two ways:

1.  Using monophonic files, place sounds on a virtual sphere around the user.
    You can move the sounds around and add effects to them.

    <img src="{{ site.baseurl }}/images/concepts/ambix-encoder.png">

2.  Use an Ambisonic microphone like the SoundField ST450, TetraMic, or Zoom H2n
    to capture the sound of an environment in 3D. You can load the captured
    sound into the Ambix plugin and run effects on it, rotate it if needed, and
    then mix.

For more information, see [YouTube's Ambisonic spatial audio
support](//support.google.com/youtube/answer/6395969){: .external}.


## User experience and quality tips

### Sync up visual and audio experiences
Make sure that what users see in a scene matches what they hear.

For example, if you hear ocean waves crashing on the shore but the ocean in
your scene looks motionless, the scene feels less realistic.

### Check your work
Check your work throughout the development process to make sure that your sound 
sources and configurations are delivering optimal
audio experiences.

### Test sounds throughout your environments
Visit all of the places that your users can explore to make sure
that environments sound natural throughout. Typically, when users can roam
freely in an environment, they move right up to sound sources.

### Ensure sound quality, volume, and responsiveness
Make sure that your sounds are of high quality, are at a clear but comfortable
volume, and adjust realistically to movement.

### Use headphones for testing
Headphones are necessary to experience spatial audio fully. Make sure to test
your sounds on a variety of headphones. Do not use computer or other speakers
for testing.

## Additional resources

[YouTube: Spatial Audio presentation from Google I/O 2016](//youtu.be/Na4DYI-WjlI){: .external}

[YouTube: Spatial Audio engine demo](//youtu.be/I9zf4hCjRg0){: .external}
