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

function [audioOut] = ambencode(audioIn, order, sourceAzRad, sourceElRad)
%AMBENCODE Encodes a single mono audio input into AmbiX Ambisonics output.
%
%   audioIn     - A mono audio input to be encoded (a single column
%                 vector).
%   order       - Ambisonic order.
%   sourceAzRad - Sound source horizontal angle in radians, [-pi:pi].
%   sourceElRad - Sound source vertical angle in radians, [-pi/2:pi/2].

if nargin < 3
    error('Number of arguments must be at least 3.');
end

numChannels = size(audioIn, 2);

if numChannels > 1
    error('Only mono input audio is supported');
end

switch nargin
    case 3
        % 2-D case.
        ambisonicCoefficients = ambencodecoeff(order, sourceAzRad);
    case 4
        % 3-D case.
        if length(sourceAzRad) ~= length(sourceElRad)
            error('Source angle vectors must be the same length');
        end
        ambisonicCoefficients = ambencodecoeff(order, sourceAzRad, ...
            sourceElRad);
    otherwise
        error('Too many input arguments.');
end

audioOut = audioIn(:, 1) * ambisonicCoefficients;
