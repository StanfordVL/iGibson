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

function [ soundfieldOutput ] = ambishelffilter(soundfieldInput, sampleRate)
%AMBISHELFFILTER Filters Ambisonic input with a pair of shelf-filters.

%   Filters Ambisonic input with a pair of shelf-filters and applies MaxRe
%   correction weightings to the high-passed portion of the spectrum.
%   Since the filters operate 'per-order', it is agnostic to the channel
%   sequence convention used. Expected normalization is SN3D. At the
%   moment it supports 1st to 5th Order Ambisonics, however it is easily
%   extensible to even higher orders by simply providing required
%   cross-over frequencies and MaxRe values.
%
%   This function is used to perform pre-filtering of the Spherical
%   Harmonic-encoded HRIRs.
%
%   input:
%
%   soundfieldInput - Sound field input singnal with audio sample data as
%                     rows and channels as columns. Expected normalization
%                     is SN3D.
%   sampleRate      - Sampling frequency in [Hz].
%
%   output:
%
%   soundfieldOutput - Shelf-filtered sound field output signal.

if nargin < 2
    error('Number of arguments must be 2');
end

% Maximum supported order.
MAX_SUPPORTED_ORDER = 5;

% Filter cross-over frequencies defined for 1st - 5th order sound fields.
% Can be modified if different values are required.
% Extending support to orders beyond MAX_SUPPORTED_ORDER consists in
% supplying additional cross-over values to this vector.
CROSSOVER_FREQUENCIES = [690, 1250, 1831, 2423, 3022];

% MaxRe (energy optimization) values, equivalent to the highest roots of
% the associated Legendre polynomials of degree n + 1. These values can be
% found in the following table: http://goo.gl/4PXhVj. Extending support to
% Ambisonic orders beyond MAX_SUPPORTED_ORDER consists in supplying
% additional MaxRe values to this vector.
MAX_RE_VALUES = [0.5774, 0.7746, 0.8611, 0.9062, 0.9325];

% Determine the Ambisonic order from the number of channels.
numInputChannels = size(soundfieldInput, 2);
order = sqrt(numInputChannels) - 1;

if order > MAX_SUPPORTED_ORDER || order < 1 || order ~= floor(order)
    error('Unsupported Ambisonic order');
end

% Compute raw coefficients (channel ratios).
coeffsRaw = ones(order + 1, 1);
for n = 1:order
    coeffsRaw(n + 1) = pnm(n, 0, MAX_RE_VALUES(order));
end

% Compute gain-corrected coefficients (channel gains).
numerator = numInputChannels;
denominator = 1;
for n = 1:order
    denominator = denominator + ((n + 1) ^ 2 - n ^ 2) * ...
        coeffsRaw(n + 1) ^ 2;
end
% Change of sign (-1) is necessary because high-passed and low-passed
% signals are out-of-phase after filtering.
correction = -sqrt(numerator / denominator);
coeffsGainCorrected = coeffsRaw * correction;

% Filter sound field input signal.
[bLp, bHp, a] = bandsplittingfilterscoeffs(CROSSOVER_FREQUENCIES(order),...
    sampleRate);
loPassSoundfield = zeros(size(soundfieldInput));
hiPassSoundfield = zeros(size(soundfieldInput));
soundfieldOutput = zeros(size(soundfieldInput));
for channel = 1:numInputChannels
    loPassSoundfield(:, channel) = filter(bLp, a, ...
        soundfieldInput(:, channel));
    hiPassSoundfield(:, channel) = filter(bHp, a, ...
        soundfieldInput(:, channel));

    % Get the Ambisonic order of the current channel.
    currentOrder = floor(sqrt(channel - 1));
    % Apply gain correction to the velocity components and combine
    % low-passed and (processed) high-passed buffers to produce the output.
    soundfieldOutput(:, channel) =  ...
        coeffsGainCorrected(currentOrder + 1) * ...
        hiPassSoundfield(:, channel) + loPassSoundfield(:, channel);
end
