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

function [ encoderTable ] = getencodertable(order)
%GETENCODERTABLE Returns a look-up table with AmbiX encoder coefficients.
%
%   Helper function for the fast symmetric Ambisonic AmbiX encoder.
%   The table is generated for the top left quadrant of the sphere (e.g.
%   0 - 90 degrees azimuth and elevation angles).
%
%   Note: Skips processing of the trivial 0th Ambisonic order channel case.
%
%   order - Ambisonic order.

% Use 1-degree azimuth/elevation resolution by default.
NUM_ANGLES = 91;
NUM_CHANNELS = (order + 1)^2 - 1;

encoderTable = zeros(NUM_ANGLES, NUM_ANGLES, NUM_CHANNELS);

% Compute spherical harmonics coefficients for the top-left quadrant.
phi = linspace(0, pi / 2, NUM_ANGLES);
theta = phi;

for i = 1:length(phi)
    for j = 1:length(theta)
        for n = 1:order
            for m = -n:n
                acn = n * n + n + m; % Ambisonic Channel Number.
                encoderTable(i, j, acn) = ynm(n, m, phi(i), theta(j));
            end
        end
    end
end
end
