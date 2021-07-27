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

function audioOut = binauralrendersymmetric(audioIn, hrirs)
%BINAURALRENDER Convolves the virtual speaker feeds with symmetric HRIRs.
%
%   inputs:
%
%   audioIn - Decoded, multi-channel audio signal (speaker feeds). Time
%             samples should be as rows, and channels should be as columns.
%   hrirs   - HRIR cell array of matrices with sample data as rows and
%             channels as colums. Two column matrices represent a symmetric
%             pair. One column matrices represent a centered self-symmetric
%             loudspeaker.
%
%   output:
%
%   audioOut - Binaural 2-channel output.

if nargin ~= 2
    error('Number of arguments must be 2');
end

audioInLength = size(audioIn, 1);
numChannels = size(audioIn, 2);

hrirLength = size(hrirs{1}, 1);
audioOutLength = (hrirLength + audioInLength - 1);

% Zero-pad the input for the FFT filtering.
% For HRIRs, the padding will be done inside the loop.
audioInPad = zeros(audioOutLength - audioInLength, numChannels);
audioIn = [audioIn; audioInPad];

% Initialize audio output.
audioOut = zeros(audioInLength + hrirLength - 1, 2);

audioInChannel = 1;
for i = 1:length(hrirs)
    hrir = hrirs{i};
    hrirPad = zeros(audioOutLength - hrirLength, size(hrir, 2));
    hrir = [hrir; hrirPad];

    if size(hrir, 2) == 1
      leftContrib = fftfilt(audioIn(:, audioInChannel), hrir);
      rightContrib = leftContrib;
      audioInChannel = audioInChannel + 1;
    else
      audioInPair = audioIn(:, audioInChannel:audioInChannel + 1);
      chSum = sum(audioInPair, 2) / 2;
      chDiff = diff(audioInPair, 1, 2) / 2;

      convSum = fftfilt(chSum, sum(hrir, 2));
      convDiff = fftfilt(chDiff, diff(hrir, 1, 2));

      leftContrib = convSum + convDiff;
      rightContrib = convSum - convDiff;
      audioInChannel = audioInChannel + 2;
    end

    audioOut(:, 1) = audioOut(:, 1) + leftContrib;
    audioOut(:, 2) = audioOut(:, 2) + rightContrib;
end
