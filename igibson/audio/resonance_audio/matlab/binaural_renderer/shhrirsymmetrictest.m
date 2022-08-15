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

% Tests the spherical harmonics HRIR encoding function (shhrirsymmetric) as well
% as the binaural rendering function shbinauralrendersymmetric by comparing the
% binaural stereo output to the 'virtual loudspeakers' binaural decoding method.

clearvars
close all
clc

% Import required ambisonic functions.
addpath('../ambisonics/ambix/');
addpath('../ambisonics/shelf_filters/');

% Tolerated error margin equivalent to -60dB.
ERROR_MARGIN = 0.001;
SAMPLING_RATE = 48000;
INPUT = [1; zeros(511, 1)];
SOURCE_AZIMUTH_RAD = pi * (2 * rand(1) - 1);
SOURCE_ELEVATION_RAD =  pi * (rand(1) - 0.5);
MAX_AMBISONIC_ORDER = 5;

% Paths to HRIRs for each binarual decoder
HRIR_PATH = '../hrtf_data/sadie/';
CONFIGS = {'symmetric_cube', ...
           'symmetric_dodecahedron_faces', ...
           'symmetric_lebedev26', ...
           'symmetric_pentakis_dodecahedron', ...
           'symmetric_pentakis_icosidodecahedron'};

for ambOrder = 1:MAX_AMBISONIC_ORDER
    for shelfFilter = 0:1 % when 1, use shelf-filters.
        
        % Get the HRIRs for the required Ambisonic binaural decoder.
        currentHrirPath = [HRIR_PATH, 'sadie_subject_002_', CONFIGS{ambOrder}];
        [hrirMatrix, ~, hrirAngles] = loadhrirssymmetric(currentHrirPath);
        
        % Encode the sound source.
        encodedInput = ambencode(INPUT, ambOrder, SOURCE_AZIMUTH_RAD, ...
            SOURCE_ELEVATION_RAD);
        
        if shelfFilter == 1
            % Shelf-filter the encoded input.
            encodedInputSf = ambishelffilter(encodedInput, SAMPLING_RATE);
            decodedInput = ambdecode(encodedInputSf, hrirAngles(:, 1), ...
                hrirAngles(:, 2));
            
        else 
            decodedInput = ambdecode(encodedInput, hrirAngles(:, 1), ...
                hrirAngles(:, 2));
        end
        
        % Render binaurally using the 'virtual loudspeakers' method.
        binauralOutputOld = binauralrendersymmetric(decodedInput, hrirMatrix);
        
        % Render binaurally using SH-encoded HRIRs method.
        shHrirs = shhrirsymmetric(currentHrirPath, ambOrder, shelfFilter);
        binauralOutputNew = shbinauralrendersymmetric(encodedInput, shHrirs);
        
        % Check whether the binaural outputs using both methods are the same.
        assert(size(binauralOutputNew, 1) == size(binauralOutputOld, 1));
        for sample = 1:size(binauralOutputNew, 1)
            assert((binauralOutputNew(sample, 1) - ...
                binauralOutputOld(sample, 1)) < ERROR_MARGIN);
            assert((binauralOutputNew(sample, 2) - ...
                binauralOutputOld(sample, 2)) < ERROR_MARGIN);
        end
        disp(['Result for Ambisonic order ', num2str(ambOrder), ...
              ', shelf-filter = ', num2str(shelfFilter), ': OK!']);
    end
end
