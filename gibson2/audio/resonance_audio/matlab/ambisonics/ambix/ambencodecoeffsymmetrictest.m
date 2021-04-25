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

% Tests whether ambencodecoeffsymmetric function returns the same
% coefficients as the reference function ambencodecoeffs.

clearvars
close all
clc

% Ambisonic order.
ORDER = 3;

% Tolerated error margin.
EPSILON = 1e-12;

% Pre-compute AmbiX coefficient table for angles 0 - 90 degrees.
ENCODER_TABLE = getencodertable(ORDER);
SYMMETRY_MATRIX = getshsymmetries(ORDER);

for PHI = [-158 -52 0 41 169]
    for THETA = [-88 -21 0  42 90]
        coeffsTested = ambencodecoeffsymmetric(ENCODER_TABLE, ...
            SYMMETRY_MATRIX, PHI, THETA);
        coeffsReference =  ambencodecoeff(ORDER, PHI * pi / 180, ...
            THETA * pi / 180);
        % Check if the coefficients are identical.
        for i = 1:length(coeffsTested)
            assert(abs(coeffsReference(i) - coeffsTested(i)) < EPSILON)
        end
    end
end
