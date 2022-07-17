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

function [energySph, velocitySph, energyCart, velocityCart] = ...
    vectoranalysis(soundField)
%VECTORANALYSIS Analyzes B-Format sound field using energy/velocity vectors.
%   This function analyzes a B-Format sound field (up to and including 5th
%   order) in terms of its energy and velocity vectors. In each case sound
%   field is first decoded using a regular 50-point Lebedev grid which
%   provides smooth energy and velocity vector distributions across the unit
%   sphere.
%
%  input:
%
%  soundField      - B-Format sound field using ACN/SN3D convention. Channel data
%                    is expected as columns while time sample date is expected as
%                    rows. Only 1st - 5th order sound fields are supported.
%  output:
%
%  energySph       - Instantaneous energy vectors of a B-Format sound field.
%                    Vector az, el, r components are in columns while time sample
%                    data is in rows.
%  velocitySph     - Instantaneous velocity vectors of a B-Format sound field.
%                    Vector az, el, r components are in columns while time sample
%                    data is in rows.
%  energyCart      - Same values as energySph, but in cartesian coordinates
%                    x, y, z.
%  velocityCart    - Same values as velocitySph, but in cartesian coordinates
%                    x, y, z.

% Import ambisonic functions.
thisScriptPath = fileparts(mfilename('fullpath'));
addpath(fullfile(thisScriptPath, '../../ambisonics/ambix'));

numChannels = size(soundField, 2);
ambOrder = sqrt(numChannels) - 1;

if ambOrder > 5
    error('Currently only up to 5th order sound fields are supported');
end

% Load virtual loudspeaker locations for the decoder.
load('lebedev011.mat');
speakerAzVec = lebedev011(:, 1) * pi / 180;
speakerElVec = lebedev011(:, 2) * pi / 180;
clear lebedev11

% Decode the sound field.
decodedSoundfield = ambdecode(soundField, speakerAzVec, speakerElVec);

% Analyze instantaneous sound field in terms of its energy and velocity vectors.
energyCart = energyvector(decodedSoundfield, speakerAzVec, speakerElVec);
velocityCart = velocityvector(decodedSoundfield, speakerAzVec, speakerElVec);

energySph = cart2sphdeg(energyCart);
velocitySph = cart2sphdeg(velocityCart);

end

function [sphericalVec] = cart2sphdeg(cartesianVec)
  [azRad, elevRad, radius] = ...
      cart2sph(cartesianVec(:, 1), cartesianVec(:, 2), cartesianVec(:, 3));

  sphericalVec = [rad2deg(azRad), rad2deg(elevRad), radius];
end
