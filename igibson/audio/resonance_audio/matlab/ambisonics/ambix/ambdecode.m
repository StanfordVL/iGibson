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

function [audioOut] = ambdecode(audioIn, speakerAzRad, speakerElRad)
%AMBDECODE Decodes Ambisonic input for a given 3D loudspeaker array.
%
%   inputs:
%
%   audioIn      - AmbiX input signal. Must have sample data as rows and
%                  channels as columns.
%   speakerAzRad - Speaker horizontal angles in radians. Must be a vector.
%   speakerElRad - Speaker vertical angles in radians. Must be a vector.
%
%   outputs:
%
%   audioOut - Decoded Ambisonic signal.

if nargin < 2
    error('Number of arguments must be at least 2');
end

if nargin == 3
    if (length(speakerAzRad) ~= length(speakerElRad))
        error('Horizontal and vertical angle vectors must have the same length');
    end
end

numChannels = size(audioIn, 2);

% 2-D case.
if nargin == 2
    % Determine the Ambisonic order from the channel count.
    order = (numChannels - 1) / 2;

    % Compute the decoder matrix for the specified speaker configuration.
    D = ambdecodematrix(order, speakerAzRad);
end

% 3-D case.
if nargin == 3
    if length(speakerAzRad) ~= length(speakerElRad)
        error('Speaker angle vectors must be the same length');
    end

    % Determine the Ambisonic order from the channel count.
    order = sqrt(numChannels) - 1;

    % Compute the decoder matrix for the specified speaker configuration.
    D = ambdecodematrix(order, speakerAzRad, speakerElRad);
end

audioOut = audioIn * D;
