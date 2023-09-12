# Overview


Shazam's and other music similar service's inability to reliably identify artists in classical and jazz genres, especially in live performance, is the motivation for this project. Professional musicians often can identify and distinguish between different individuals playing after hearing a short sample, this is not limited to recorded examples but extends to any circumstances where music is heard. Services like Shazam use a database and a lookup method, which is very good at identifying most music, but if something is not in the database the results are usually not consistent.

 In my experience, live recordings of jazz and classical music are usually not able to be identified unless it has since been put onto spotify or something like that. An example of the shortcomings of Shazam; [Drip by Tigran Hamasyan](https://open.spotify.com/track/56aT6RHzJGhsnprd9bFSvP?si=ca0ddaba4f514712) when on spotify can be identified by Shazam very quickly. [This live recording](https://www.youtube.com/watch?v=z7j7bdEPSd0&ab_channel=BerkleeCollegeofMusic) of the same song is not identified at all by Shazam.This live version is clearly the same song, the only differences being expanded instrumentation and form adjustments. To a human this is very easy to identify, but is not to Shazam. The goal of this project is to fill in these gaps with a neural net. 
 
 Furthermore, in classical music I want to be able to ID the orchestra from a sound clip. Trained ears can tell the difference between American and European, English and German, East coast and West coast etc. orchestras with ease, but Shazam has a very hard time even IDing the piece being played let alone which orchestra is playing it, if it is not already in their database, which many live recordings are not. This is another goal of this project which might have to be its own model. 

I am starting with a nn trained to identify classical pianists. It is trained on many hours of solo classical piano albums downloaded from youtube. Raw mp3 file turned into CQT arrays with librosa. This is a very basic model with no real rhyme or reason to layer selection and overall architecture. In its current form it has very high test acc but only correctly IDs 1/5 unseen examples. 

I am working on an alternative approach to leverage pre-built and pre-trained models by turing the CQT arrays into images and doing image classification.  





# Things to add
* One hot encoding to classes
* fix dimensional issues 
* Image classification
* Audio Spectrogram Transformer
