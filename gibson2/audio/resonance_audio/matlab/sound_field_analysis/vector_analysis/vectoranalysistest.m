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

% This script tests vectoranalysis function. In particular, it tests
% whether vectoranalysis correctly predicts the direction of a
% broadband sound source in a synthesized B-Format sound field. It also
% checks whether energy vector magnitudes grow as the order of a sound
% field increases.

clearvars
close all
clc

% Import ambisonic functions.
thisScriptPath = fileparts(mfilename('fullpath'));
addpath(fullfile(thisScriptPath, '../../ambisonics/ambix'));

% Tolerated error margin.
EPSILON = 1e-8;

% The highest sound field orders.
MAX_ORDER = 5;

% Arbitrary azimuth and elevation angles [deg] of a sound source to be encoded
% into a B-Format sound field.
SOURCE_AZIM = [0, 33, -45, 157];
SOURCE_ELEV = [0, -45, 53, -80];

% Lenght of the B-Format singal in samples.
SOUND_FIELD_LENGTH = length(SOURCE_AZIM);

% Initialize energy and velocity vector direction containers.
energySph = zeros(SOUND_FIELD_LENGTH, MAX_ORDER, 3);
velocitySph = zeros(SOUND_FIELD_LENGTH, MAX_ORDER, 3);

% Encode the sound source into sound fields of orders 1-5 and compute their
% energy and velocity vector directions.
for order = 1:MAX_ORDER
    for sample = 1:SOUND_FIELD_LENGTH
        soundField = ambencode(1, order, ...
            SOURCE_AZIM(sample) * pi / 180, SOURCE_ELEV(sample) * pi / 180);

        [energySph(sample, order, :), velocitySph(sample, order, :)] = ...
            vectoranalysis(soundField);
    end
end

[energyVecAzim, energyVecElev, energyVecMag] = ...
    deal(energySph(:, :, 1), energySph(:, :, 2), energySph(:, :, 3));
[velocityVecAzim, velocityVecElev, velocityVecMag] = ...
    deal(velocitySph(:, :, 1), velocitySph(:, :, 2), velocitySph(:, :, 3));

% Check if all the directions (for all the sound field orders) are predicted
% correctly by both energy and velocity vector directions.
for order = 1:MAX_ORDER
    for sample = 1:SOUND_FIELD_LENGTH
        assert(abs(energyVecAzim(sample, order) - ...
            SOURCE_AZIM(sample)) < EPSILON);
        assert(abs(energyVecElev(sample, order) - ...
            SOURCE_ELEV(sample)) < EPSILON);
        assert(abs(velocityVecAzim(sample, order) - ...
            SOURCE_AZIM(sample)) < EPSILON);
        assert(abs(velocityVecElev(sample, order) - ...
            SOURCE_ELEV(sample)) < EPSILON);
    end
end

% Check if energy vector magniuteds grow as the sound field order increases.
for order = 1:MAX_ORDER - 1
    for sample = 1:SOUND_FIELD_LENGTH
        assert(energyVecMag(sample, order) < energyVecMag(sample, order + 1));
    end
end

% Check if all the velocity vectors have unity magnitude.
for order = 1:MAX_ORDER
    for sample = 1:SOUND_FIELD_LENGTH
        assert(abs(velocityVecMag(sample, order) - 1) < EPSILON);
    end
end
