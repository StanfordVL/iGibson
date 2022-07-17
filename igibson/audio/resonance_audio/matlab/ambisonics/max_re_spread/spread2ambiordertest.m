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

% Checks if the spread2ambiorder function returns expected ambisonic orders
% for known spread values.

clearvars
close all
clc

EPSILON = 0.05;
TESTED_SPREADS = [2 * pi, 0.955745543485583, 0.6841, 0.5336, 0.4371];
EXPECTED_ORDERS = [0, 1, 2, 3, 4];

for i = 1:length(TESTED_SPREADS)
    assert(abs(spread2ambiorder(TESTED_SPREADS(i)) - ...
        EXPECTED_ORDERS(i)) < EPSILON);
end

% Also, check if the fractional order is monotonically growing with
% decreasing the spread.
NUM_MEASUREMENTS = 1000;
previousOrder = -1;
for spread = linspace(2 * pi, 0, NUM_MEASUREMENTS)
    currentOrder = spread2ambiorder(spread);
    assert(currentOrder > previousOrder)
    previousOrder = currentOrder;
end
