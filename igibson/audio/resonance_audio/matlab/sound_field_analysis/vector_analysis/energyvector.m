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

function [energyVec] = energyvector(gainVec, speakerAzVec, speakerElVec)
%ENERGYVECTOR Computes a 3D energy vector of a decoded sound field.
%  This funcion is spatialization method agnostic, e.g. it can be applied
%  equally well to First and Higher Order Ambisonics (FOA & HOA), Ambisonic
%  Equivalent Panning (AEP), amplitude/intensity panning, etc.
%  It analyzes (virtual) loudspeaker gains in order to determine
%  instantaneous direction of max. energy concentration in a sound field.
%
%  input:
%
%   gainVec      - NxM matrix containing N gain samples for M loudspeakers.
%   speakerAzVec - Vector containing loudspeaker azimuth angles in radians
%                  (must be the same length as the gainVec).
%   speakerElVec - Vector containing loudspeaker elevation angles in radians
%                  (must be the same length as the gainVec).
%   outuput:
%
%   energyVec      - Instantaneous energy vector of a decoded sound field.

narginchk(3, 3);

% Check that inputs have the correct shape.
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

signalEnergy = gainVec .^ 2;
totalEnergy = sum(signalEnergy, 2);

% Return a row of zeros for any samples where total energy is very small.
nonzeroRows = abs(totalEnergy) >= 1e-16;
energyVec = zeros(numSamples, 3);

signalEnergy = signalEnergy(nonzeroRows, :);
totalEnergy = totalEnergy(nonzeroRows, :);

% Compute x, y and z components of the energy vector energyVec.
rowwise_multiply = @(x, y) bsxfun(@times, x, y);
ex = sum(rowwise_multiply(...
    signalEnergy, cos(speakerElVec) .* cos(speakerAzVec)), 2);
ey = sum(rowwise_multiply(...
    signalEnergy, cos(speakerElVec) .* sin(speakerAzVec)), 2);
ez = sum(rowwise_multiply(signalEnergy, sin(speakerElVec)), 2);
energyVec(nonzeroRows, :) = [ex ey ez] ./ repmat(totalEnergy, 1, 3);
end
