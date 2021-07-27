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

% Tests maxrespread function.

clearvars
close all
clc

% Import Ambisonic functions.
thisScriptPath = fileparts(mfilename('fullpath'));
addpath(fullfile(thisScriptPath, '../ambix/'));
addpath(fullfile(thisScriptPath, '../../binaural_renderer/'));
addpath(fullfile(thisScriptPath, ...
    '../../sound_field_analysis/vector_analysis/'))

% Tolerated error margin in decibels. We want to make sure that the
% original source and spread-controlled source do not differ in terms of
% loudness.
EPSILON = 1;

MAX_ORDER_1 = 1;
MAX_ORDER_3 = 3;
TESTED_SPREADS = [30:5:360] * pi / 180;

% At MAX_ORDER_1, the narrowest source is approx. 54 degrees. That is why,
% up until 53 degrees we expect all the spread coefficients to be at unity.
for spread = linspace(0, 53 * pi / 180, 100)
    maxReSpreadCoeffs = maxrespread(spread, MAX_ORDER_1);
    for coeff = maxReSpreadCoeffs'
        assert(coeff == 1);
    end
end

% At MAX_PRDER_1, beyond the spread of 54 degrees, the pressure
% coefficients should start growing monotonically, while the velocity
% coefficients  should monotonically decrease.
previousPressureCoeff = 1;
previousVelocityCoeff = 1;
for spread = linspace(54 * pi / 180, 360 * pi / 180, 100)
    maxReSpreadCoeffs = maxrespread(spread, MAX_ORDER_1);
    assert(maxReSpreadCoeffs(1) > previousPressureCoeff);
    assert(maxReSpreadCoeffs(2) < previousVelocityCoeff);
    previousPressureCoeff = maxReSpreadCoeffs(1);
    previousVelocityCoeff = maxReSpreadCoeffs(2);
end

% At MAX_ORDER_3, the narrowest source is approx. 30 degrees. That is why,
% up until 30 degrees we expect all the spread coefficient to be at unity.
% Please note, since the relationship between channels at higher ambisonic
% orders is not as trivial as in the FOA case, we cannot simply check if
% the coefficients are either monotonically growing or decaying with
% increasing the source spread.
for spread = linspace(0, 30 * pi / 180, 100)
    maxReSpreadCoeffs = maxrespread(spread, MAX_ORDER_1);
    for coeff = maxReSpreadCoeffs'
        assert(coeff == 1);
    end
end

% Using energy vector analysis, for arbitrary source spreads check if the
% energy vector magnitudes decrease monotonically with increasing the
% spread.
input = ambencode(1, MAX_ORDER_3, 0, 0);
inputMaxRe = zeros(size(input));
output = zeros(size(input));
maxReCoeffs = maxrecoeffs(MAX_ORDER_3);

% Apply MaxRe coefficients to the input soundfield signal.
for channel = 1:size(input, 2)
    [n, m] = getnm(channel - 1);
    inputMaxRe(:, channel) = input(:, channel) .* maxReCoeffs(n + 1);
end

PreviousEnergyVectorMagnitude = 1;
for spread = TESTED_SPREADS
    maxReSpreadCoeffs = maxrespread(spread, MAX_ORDER_3);
    for channel = 1:size(input, 2)
        [n, m] = getnm(channel - 1);
        output(:, channel) = inputMaxRe(:, channel) .* ...
            maxReSpreadCoeffs(n + 1);
    end
    % Check if the magnitude of the energy vector decreases
    % monotonically with increasing the source spread.
    energyVectorSpherical = vectoranalysis(output);
    currentEnergyVectorMagnitude = energyVectorSpherical(3);
    assert(PreviousEnergyVectorMagnitude > currentEnergyVectorMagnitude);
    PreviousEnergyVectorMagnitude = currentEnergyVectorMagnitude;
end

% Check if the total energy of the input soundfield is
% preserved after applying the MaxRe spread correction gains.
input = ambencode(1, 3, pi / 3, pi / 4);
inputMaxRe = zeros(size(input));
output = zeros(size(input));
maxReCoeffs = maxrecoeffs(MAX_ORDER_3);
% Apply MaxRe coefficients to the input soundfield signal.
for channel = 1:size(input, 2)
    [n, m] = getnm(channel - 1);
    inputMaxRe(:, channel) = input(:, channel) .* maxReCoeffs(n + 1);
end

% Virtual speaker horizontal and vertical angles obtained from HRIRs.
HRIR_PATH = '../../hrtf_data/sadie/sadie_subject_002_symmetric_lebedev26/';
[~, ~, speakerAngles, ~] = loadhrirssymmetric(HRIR_PATH);

% Decode original MaxRe input for the above virtual speaker array.
decodedInputMaxRe = ambdecode(inputMaxRe, speakerAngles(:, 1), ...
    speakerAngles(:, 2));
decodedInputMaxReTotalEnergy = db(sum(decodedInputMaxRe.^2), 'power');
for spread = TESTED_SPREADS
    maxReSpreadCoeffs = maxrespread(spread, MAX_ORDER_3);
    for channel = 1:size(input, 2)
        [n, m] = getnm(channel - 1);
        output(:, channel) = inputMaxRe(:, channel) .* ...
            maxReSpreadCoeffs(n + 1);
    end
    % Check if the output energy of the ambisonic source signal is
    % preserved.
    decodedOutput = ambdecode(output, speakerAngles(:, 1), ...
        speakerAngles(:, 2));
    decodedOutputTotalEnergy = db(sum(decodedOutput.^2), 'power');
    assert(abs(decodedOutputTotalEnergy - decodedInputMaxReTotalEnergy) ...
        < EPSILON);
end
