## Inspiration

The challenge was to complete a predefined course with a Comma Body in 2 days.  I wanted to achieve that goal while also adding some extra functionality (ie. recording you while you preform a task like juggling a soccer ball, going for a hike, or touring an apartment).

## What it does

Leon follows you and watches you, wherever you may go, without getting too close.  With his built-in camera and record button on his screen, he makes for a great camera man.

## How we built it

Leon uses Yassine's [YoLo](https://github.com/YassineYousfi) model to detect people.  He locks on to the person who takes up the biggest part of his field of view and rotates his body to always look their way.  If you get too close he backs up to give you space and if you walk too far away he tries to catch up... until he looses you :'(

## Challenges we ran into

- Communication between WSL2 & the Comma
- Installing a custom Python library (face_recognition) on the Comma

## Accomplishments that we're proud of

- Got a deliverable result that completes the objective (& more) despite having no prior ML, CV, or hackathon experience

## What we learned

- Deeper familiarity with openpilot & openCV
- What it's like to participate in a hackathon

## What's next for Nepo Leon

- The ability to stream audio & video through a commonly-used service (ie. Instagram, YouTube, Twitch, Zoom, Google Meet, etc)
- Face detection in order to lock on to a specific companion
- Understanding hand gestures for extra control
- Motion smoothing when transitioning between states
- Collision avoidance

## Documentation

- https://github.com/MikeBusuttil/openpilot/tree/local-buddy#nepo-leon---body-buddy
