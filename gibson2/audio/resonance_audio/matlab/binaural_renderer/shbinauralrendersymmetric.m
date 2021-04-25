%{
Copyright 2018 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS-IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
%}

function audioOut = shbinauralrendersymmetric(audioIn, shHrirs)
%SHBINAURALRENDERSYMMETRIC Binaurally renders an Ambisonic sound field.
%   Binaurally renders an Ambisonic sound field of an arbitrary order by
%   performing convolution in the spherical harmonics domain. The order
%   (hence the number of channels) of the input must match the order (hence
%   the number of channels) of the spherical harmonic-encoded HRIRs
%   |shHrirs|. Assumes that HRIRs are symmetric with respect to the
%   sagittal plane.
%
%   inputs:
%
%   audioIn - Ambisonic sound field input with channels in colums and audio
%             data in rows. Expected channel sequence is ACN.
%   shHrirs - Matrix of spherical harmonics-encoded Head Related Impulse
%             Responses (HRIRs) with channels in columns and audio data in
%             rows. Expected channel sequence is ACN.
%
%   output:
%
%   audioOut  - Binaural 2-channel output.

% Import required ambisonic functions.
addpath( '../ambisonics/ambix/');

if nargin ~= 2
    error('Number of arguments must be exactly 2.');
end

% Check if the |audioIn| and |shHrirs| have the same number of channels.
numAudioInChannels = size(audioIn, 2);
numShHrirsChannels = size(shHrirs, 2);
if numAudioInChannels ~= numShHrirsChannels
    error('Number of channels in input signal and HRIRs must be the same.');
end

audioInLength = size(audioIn, 1);
shHrirLength = size(shHrirs, 1);
audioOutLength = audioInLength + shHrirLength - 1;

% Pre-allocate data matrices for speed.
audioOut = single(zeros(audioOutLength, 2));

% Zero-pad |audioIn| and |shHrirs| matrices for frequency domain convolution.
audioIn = [audioIn; zeros(shHrirLength - 1, numAudioInChannels)];
shHrirs = [shHrirs; zeros(audioInLength - 1, numShHrirsChannels)];

% Main processing loop.
for channel = 1:numShHrirsChannels
    harmonic = channel - 1; % because of 1-based indexing used in MATLAB.
    [~, m] = getnm(harmonic);
    filteredAudioIn = fftfilt(audioIn(:, channel), shHrirs(:, channel));
    if m < 0
        % Antisymmetric spherical harmonic case (wrt the median plane).
        audioOut(:, 1) = audioOut(:, 1) + filteredAudioIn;
        audioOut(:, 2) = audioOut(:, 2) - filteredAudioIn;
    else
        % Symmetric spherical harmonic case (wrt the median plane).
        audioOut(:, 1) = audioOut(:, 1) + filteredAudioIn;
        audioOut(:, 2) = audioOut(:, 2) + filteredAudioIn;
    end
end
end
