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

% Tests if the maxrecoeffs function returns expected (pre-computed)
% coefficients.

clearvars
close all
clc

EPSILON = 1e-6;
AMBISONIC_ORDERS = [0, 1, 2, 3, 4];

MaxRe0 = 1;
MaxRe1 = sqrt(4 / 2) * [1 0.577];
MaxRe2 = sqrt(9 / 3.6) * [1 0.775 0.4];
MaxRe3 = sqrt(16 / 5.75) * [1 0.861 0.612 0.305];
MaxRe4 = sqrt(25 / 8.441) * [1 0.906 0.732 0.501 0.246];
EXPECTED_COEFFS = [MaxRe0 0 0 0 0; MaxRe1 0 0 0; MaxRe2 0 0; MaxRe3 0; ...
    MaxRe4];
NUM_COEFFS = size(EXPECTED_COEFFS, 2);

for i = 1:length(AMBISONIC_ORDERS)
    computedMaxReCoeffs = maxrecoeffs(AMBISONIC_ORDERS(i));
    for j = 1:NUM_COEFFS
        assert(abs(EXPECTED_COEFFS(i, j) - computedMaxReCoeffs(j)) ...
            < EPSILON);
    end
end

% Also, check if the MaxRe coefficient values monotonically increase with
% the (fractional) order.
NUM_MEASUREMENTS = 1000;
MAX_ORDER = AMBISONIC_ORDERS(end);
previousCoeffs = zeros(1, NUM_COEFFS);
for order = linspace(0, MAX_ORDER, NUM_MEASUREMENTS)
    currentCoeffs =  maxrecoeffs(order);
    for i = 1:NUM_COEFFS
        assert(currentCoeffs(j) >= previousCoeffs(j));
    end
    previousCoeffs = currentCoeffs;
end
