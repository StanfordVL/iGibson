---
title: Fundamental Concepts
weight: 2
---


Resonance Audio gives developers and sound designers powerful technology
for delivering high fidelity spatial audio at scale, to users across the
top mobile and desktop platforms.

Resonance Audio has cost efficiency and quality advantages, of particular
benefit to mobile platforms with limited computational resouces.


## How does Resonance Audio work?
Resonance Audio replicates how real sound waves interact with human ears and
with the environment.

### Simulating sound waves interacting with human ears
Resonance Audio simulates how sound waves interact with human ears.
In the real world, we use these sound wave interactions to determine a sound's
horizontal location and its elevation. Resonance Audio replicates these
interactions to create the illusion of sounds coming from specific locations in
a virtual world.


#### Interaural time differences
The time difference between sound wave arrival at our left and
right ears helps us determine the horizontal location of low-frequency sounds.

<img src="{{ site.baseurl }}/images/concepts/soundwave-to-ears3.gif">

This **interaural time difference** (ITD) depends on a sound source's
horizontal position relative to you. The farther that a source is to the left
or right of your head, the greater the time difference.


#### Interaural level differences
Humans cannot use interaural time differences to locate higher-frequency sounds.

Instead, we determine their horizontal location using **interaural level differences**,
(ILD). These are differences in
loudness and frequency distribution between our right and left ears, caused by
the human head's acoustic shadow.

#### Spectral effects
While time and level differences help us to locate a sound horizontally, other
sound wave interactions help us determine a sound's elevation.

Sounds coming from different directions bounce off of the
inside of our outer ears in different ways.


<img src="{{ site.baseurl }}/images/concepts/sound-hits-ears.png" width="400px">
Image: Sebastian Kaulitzki/Getty

Humans use these changes in frequency, or **spectral effects**,
to determine the vertical location of a sound source.<br>


#### Simulating audio cues with head-related transfer functions (HRTFs)
To simulate real sound wave interactions with our ears, Resonance Audio uses
[head-related transfer functions](//en.wikipedia.org/wiki/Head-related_transfer_function){: .external} (HRTFs).

HRTFs include effects for the time and level differences and the spectral
effects that we use to determine a sound's location.

Hearing audio processed with HRTFs over headphones gives users the illusion that
sounds have a specific location in the virtual world around them.


### Simulating sound waves interacting with their environment
In addition to simulating sound wave interactions with our ears, Resonance Audio
simulates sound wave interactions with their environment.


#### Head movements and sound position
Moving our heads helps us to perceive relative changes in audio location.
Resonance Audio responds to these head movements, maintaining a source's location
in the sphere of sound.

<img src="{{ site.baseurl }}/images/concepts/spatial-rotate.gif">

Head-mounted displays track user head movements. Resonance Audio uses this
information to rotate the sphere of sound in the opposite direction of the user's
head movement. In this way, virtual sounds maintain their position relative to
the user.


#### Early reflections and reverb
In the real world, as sound waves travel through the air, they bounce off of
every surface in our environment. This creates a complex mix of reflections.

Resonance Audio separates this mix of sound waves into the following
three components.

**Direct sound**<br>
The first wave that hits our ears is **direct sound** that travels
directly from the source to us.

As a sound source's distance from us increases, its energy decreases. This is
why sounds farther away from us have a lower volume than sounds closer to us.

<img src="{{ site.baseurl }}/images/concepts/distance-from-speaker.png">

**Early reflections**<br>
The first few reflected waves that arrive at our ears are known as **early
reflections**. These reflections give us an impression of the size and shape of
the room in which we are located.

<img src="{{ site.baseurl }}/images/concepts/reflections.png">

Resonance Audio spatializes early reflections in real time and renders simulated
sources for each reflection.

**Late reverb**<br>
Over time, the density of reflections arriving at our ears builds more and
more until individual sound waves are indistinguishable. This phenomenon is
called **late reverb**.

Resonance Audio has a powerful built-in reverb engine that can match the sound
of real rooms closely.

If you change the size of the room or the surface materials of its walls,
the reverb engine reacts in real time and adjusts the sound waves to match the
new conditions.

#### Occlusion
To add further realism, Resonance Audio can also simulate how real sound waves
traveling between a source and listener are blocked by objects between them.

Resonance Audio simulates these environmental **occlusion** effects by treating
high and low frequency components differently. High frequencies are blocked
more than low frequencies, mimicking what happens in the real world.

<img src="{{ site.baseurl }}/images/concepts/sound-occluded.png">

#### Directivity
A sound source's **directivity pattern** is related closely to occlusion.

A directivity pattern or shape represents the way in which sound emanates from a
source in different directions. We hear a source's sound differently depending
on its directivity pattern and our location relative to the source.

As an example, you might walk in a circle around someone playing a guitar.
The guitar sounds much louder from the front, where its strings and sound hole
are located.

When you are behind the guitar, the guitar and player's body occlude the sound
coming from the strings.

<img src="{{ site.baseurl }}/images/concepts/directivity2.gif">

You can use Resonance Audio to change the directivity pattern for a source and
mimic the natural, non-uniform, ways in which real sources emit sound.

There are two directivity parameters that you can configure:

*   **Alpha**: Represents pattern shape. You can use cardiod, circular, or
               figure-eight shapes.
*   **Sharpness**: Represents pattern width.

### Ambisonics

Resonance Audio uses a technology called [Ambisonics](//en.wikipedia.org/wiki/Ambisonics){: .external}
to envelop the user in a sphere of sound.
<img src="{{ site.baseurl }}/images/concepts/speakers-surround-head-3d.png">

As Ambisonic order increases, sound wave simulation becomes more accurate.


## Next Steps

**Get started**

*  See the developer overview to [learn about developer advantages and start using Resonance Audio in
   your platform of choice]({{ site.baseurl }}/develop/overview).

*  Get [Design tips]({{ site.baseurl }}/develop/design-tips) for achieving the
   most impact with Resonance Audio.

**Learn more about Google's spatial audio technology**

*  [YouTube video: Spatial Audio presentation from Google I/O 2016](https://youtu.be/Na4DYI-WjlI){: .external}

*  [YouTube video: Spatial Audio engine demo](https://youtu.be/I9zf4hCjRg0){: .external}


