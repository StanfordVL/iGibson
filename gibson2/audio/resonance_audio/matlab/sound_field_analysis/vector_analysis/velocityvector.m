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

function [velocityVec] = velocityvector(gainVec, speakerAzVec, speakerElVec)
%VELOCITYVECTOR Computes a 3D velocity vector of a decoded sound field.
%  This funcion is spatialization method agnostic, e.g. it can be applied
%  equally well to First and Higher Order Ambisonics (FOA & HOA), Ambisonic
%  Equivalent Panning (AEP), amplitude/intensity panning, etc.
%  It analyzes (virtual) loudspeaker gains in order to determine
%  instantaneous particle velocity of a sound field.
%
%  input:
%
%  gainVec      - NxM matrix containing N gain samples for M loudspeakers.
%  speakerAzVec - Vector containing loudspeaker azimuth angles in radians
%                 (must be the same length as the gainVec)
%  speakerElVec - Vector containing loudspeaker elevation angles in radians
%                 (must be the same length as the gainVec)
%  outuput:
%
%  velocityVec  - Instantaneous velocity vector of a decoded sound field.

narginchk(3, 3);

% Check if the input arguments are vectors.
if ~ismatrix(gainVec)
  error('gainVec must be a matrix.');
end

if ~isvector(speakerAzVec) || ~isvector(speakerElVec)
    error('speakerAzVec and speakerElVec must be vectors.');
end

if iscolumn(speakerAzVec)
    speakerAzVec = speakerAzVec';
end

if iscolumn(speakerElVec)
    speakerElVec = speakerElVec';
end

% Check if loudspeaker gain and angle vectors are the same length.
numSamples = size(gainVec, 1);
numLoudspeakers = size(gainVec, 2);
if length(speakerAzVec) ~= numLoudspeakers ||  ...
        length(speakerElVec) ~= numLoudspeakers
    error('Loudspeaker angle vectors must match gainVec height.');
end

% If the input is very small, return a matrix of zeros.
totalGain = sum(gainVec, 2);

% Return a row of zeros for any samples where total gain is very small.
nonzeroRows = abs(totalGain) >= 1e-16;
velocityVec = zeros(numSamples, 3);

gainVec = gainVec(nonzeroRows, :);
totalGain = totalGain(nonzeroRows, :);

% Compute x, y and z components of the velocity vector velocityVec.
rowwise_multiply = @(x, y) bsxfun(@times, x, y);
vx = sum(rowwise_multiply(...
    gainVec, cos(speakerElVec) .* cos(speakerAzVec)), 2);
vy = sum(rowwise_multiply(...
    gainVec, cos(speakerElVec) .* sin(speakerAzVec)), 2);
vz = sum(rowwise_multiply(gainVec, sin(speakerElVec)), 2);
velocityVec(nonzeroRows, :) = [vx vy vz] ./ repmat(totalGain, 1, 3);
end
