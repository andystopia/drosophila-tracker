# Multi Object Tracker Designed To Track Small Colliding Objects.

## Motivation

In efforts to track small flies moving around in a container, when existing tracking solutions fail, it can be necessary to write your own software. 

## Technology

Most of this is possible thanks to the hard work of the contributors of the OpenCV image processing library creators.
We shall use their tools extensively instead of attempting to partake in the tedious task of writing our own.

## Approaches to tracking

### OpenCV CRST
Issues with CRST tracking. CRST tracking, while very impressive, tends to prefer strictly "stable" tracking options.
Because the tracker maintains object persistence across frames, it does make for a relatively nice tracking option.
However, because the CRST prefers tracking the "nearest" object, if two flies approach each other and then one jumps, 
the two CRST trackers will begin tracking the same fly instead of the original two flies. Not what we want at all!
We want to maintain object persistence, so that exactly one object gets tracked per CRST tracker. 

#### Benefits of CRST Tracking
 * Persistent across frames
 * Does an amazing job at not getting "lost" often.
 * Is relatively predictable
 * If two flies are close together, it keeps two trackers for that one area.
#### Drawbacks of CRST Tracking
 * Does get "lost" occasionally, where it can't find the fly from the last frame.
 * Is slow 
   - ~15ms per frame per tracker on my Mac M1 Pro
 * Tends to allow tracking one fly as if it were two flies, if two flies get close together, leaving one fly untracked, and one fly double tracked.


### OpenCV SimpleBlobDetector

SimpleBlobDetector just detects blobs of color within an image. 

### Benefits
 * Is often pessimistic about blob matching in my experience
 * Can find flies when CRST tracking can't
 * Can be used to initialize CRST trackers.
 * Is relatively fast ~5ms globally per frame
### Drawbacks
 * Does not have persistence
 * Recognizes two nearby flies as one fly, leading to ambiguous situations for any software which stitches frames together.
 * Missing real flies on frames due to pessimism can lead to very hairy permissive logic to compensate

## A New Approach

The nice thing is that the areas which both trackers are weak don't overlap very well is pretty small. Which means that if we try both 
these approaches throughout the video we can gain a few advantages.

 * Persistent Tracking
 * Auto Fly Detection
 * Demerging Logic is simple enough to implement, more later.

### Demerging
Demerging nearby flies is the really hard problem to solve, but since SimpleBlobDetector can tell me that two blobs are near each other, 
and I can write logic to see if there are two nearby trackers and "steal" one of those, since it's clear that they're tracking the same blob.

Stealing trackers can be done on a variety of conditions, but what I've found to be sufficient is the measure of distance and the measure of change in motion over the previous frame. By "stealing" the tracker that is a balance of the closest and the lowest in motion, you can get realy good scores where the flies aren't being confused with each  other. 

### Correcting For Lost Trackers

Sometimes, if a a fly moves too much over the course of a frame, it's CRST tracker may get "lost" as in, it can't find the fly on the next frame. This is obviously a massive issue, because it ruins the tracking for that fly for the rest of time. Luckily there is a way to correct for this as well, and that is, if a tracker becomes lost, the first thing that you need to do is to see if you can find a blob within some distance that does not have a pre-existing CRST tracker. Note that this may correct a tracker onto a "noise" pattern if there is noise in the input video, but hopefully the noise's regions of high intensity are small enough that they are rejected most of the time by the SimpleBlobDetector. By auto-correcting "lost" trackers, the process of fixing trackers can be greatly accelerated.

## Approach

The first thing that needs to be done is "foreground extraction", in our case, we're extracting the flies. Right now, we clone stamp out all the flies in a single frame and then subtract that frame from all other frames, this approach yields a nice clear model where flies are visible which we can then threshold to remove noise from. 

When the program starts, it uses a simple blob detector to spot all the flies and then calculates their bounding box and then it hands that bounding box off to the CRST tracker to continue tracking. 

For each frame, 
we detect all the blobs in the frame and then we do the CRST tracking step, doing the above corrections to ensure that the model stays stable and correctly tracks flies.

## For Future Bike Shedding

 * Currently the coordinates are exported as according to the CRST tracker. However, the CRST tracker does not care about where the pattern (fly) is in it's bounding box, as long as it's in there, this leads to some noise. An alternative would be to export the nearest blob detector blob, because that centers "tightly" around the fly, but will yield some noise when two flies get close to each other. However, the CRST tracker also creates a decent amount of noise when two flies are near each other. 
## Road map

- [x] Implement the two above steps
  - [x] Parallelize the CRST tracking
- [ ] Upgrade to an object orientated model
   * Helps with parrelizatio
   * Extensibility
   * Support for subtracting the difference map in this tracker rather than in video edition software.
- [ ] Allow adding new flies later so that if two flies 
are next to each other when the video starts they can be regressively detected and have their data preserved.




