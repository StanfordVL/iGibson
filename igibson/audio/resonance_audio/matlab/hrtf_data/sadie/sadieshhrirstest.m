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

% Tests if the encoded SADIE SH HRIRs decode an Ambisonic input correctly.

clearvars
close all
clc

% Import required functions.
addpath('../../binaural_renderer/');

% Tolerated error margin equivalent to -60dB;
ERROR_MARGIN = 0.001;
% Target sampling rate.
SAMPLING_RATE = 48000;
INPUT = [1; zeros(255, 1)];
SOURCE_AZIMUTH_RAD = pi / 3;
SOURCE_ELEVATION_RAD =  pi / 4;

for ambisonicOrder = 1:5

    % Paths to directories containing standard symmetric SADIE HRIRs.
    switch ambisonicOrder
        case 1
            hrirDir = 'sadie_subject_002_symmetric_cube';
        case 2
            hrirDir = 'sadie_subject_002_symmetric_dodecahedron_faces';
        case  3
            hrirDir = 'sadie_subject_002_symmetric_lebedev26';
        case 4
            hrirDir = 'sadie_subject_002_symmetric_pentakis_dodecahedron';
        case 5
            hrirDir = ...
                'sadie_subject_002_symmetric_pentakis_icosidodecahedron';
        otherwise
            error('Unsupported Ambisonic order');
    end

    % Get the standard HRIRs for the virtual loudspeaker decode.
    [ hrirMatrix, ~, hrirAngles ] = loadhrirssymmetric( hrirDir );

    % Get the SH-encoded HRIRs for the fast decode.
    [shHrirs, shHrirFs] = ...
        audioread(['sadie_002_symmetric_sh_hrir_o_', ...
        num2str(ambisonicOrder), '/sh_hrir_order_', ...
        num2str(ambisonicOrder), '.wav']);
    assert(shHrirFs == SAMPLING_RATE);

    % Encode the Ambisonic sound source.
    encodedInput = ambencode(INPUT, ambisonicOrder, SOURCE_AZIMUTH_RAD, ...
        SOURCE_ELEVATION_RAD);

    % Apply shelf-filters.
    encodedInputSf = ambishelffilter(encodedInput, SAMPLING_RATE);

    % Decode input for the virtual loudspeaker arrays.
    decodedInputSf = ambdecode(encodedInputSf, hrirAngles(:, 1), ...
        hrirAngles(:, 2));

    % Render binaurally using the 'standard' decoder method.
    binauralOutputOldSf = binauralrendersymmetric(decodedInputSf, ...
        hrirMatrix);

    % Render binaurally using SH-encoded HRIRs method.
    binauralOutputNewSf = shbinauralrendersymmetric(encodedInput, ...
        shHrirs);

    % Check whether the binaural outputs using both methods are the same.
    assert(size(binauralOutputNewSf, 1) == size(binauralOutputOldSf, 1));

    % Check if both versions are sample-accurate.
    for sample = 1:size(binauralOutputNewSf, 1)
        assert((binauralOutputNewSf(sample, 1) - ...
            binauralOutputOldSf(sample, 1)) < ERROR_MARGIN);
        assert((binauralOutputNewSf(sample, 2) - ...
            binauralOutputOldSf(sample, 2)) < ERROR_MARGIN);
    end

    disp(['Spherical Harmonic encoding of Ambisonic order ', ...
        num2str(ambisonicOrder), ' HRIRs was successful!']);
end
