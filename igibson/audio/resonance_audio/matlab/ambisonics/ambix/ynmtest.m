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

% Tests spherical harmonic coeffients computed using ynm function against
% precomputed Ambisonic equations available e.g. at:
% http://ambisonics.ch/standards/channels/index.

clearvars
close all
clc

% Azimuth and elvation angles used.
AZIMUTH = linspace(0, 2 * pi, 10);
ELEVATION = linspace(-pi / 2, pi / 2, 10);

% Tolerated error margin.
EPSILON = 1e-15;

for a = AZIMUTH
    for e = ELEVATION
        assert(abs(ynm(0, 0, a, e) - 1) < EPSILON);
        assert(abs(ynm(1, -1, a, e) - cos(e) * sin(a)) < EPSILON);
        assert(abs(ynm(1, 0, a, e) - sin(e)) < EPSILON);
        assert(abs(ynm(1, 1, a, e) - cos(e) * cos(a)) < EPSILON);
        assert(abs(ynm(2, -2, a, e) - sqrt(3) / 2 * cos(e) * cos(e) * ...
            sin(2 * a)) < EPSILON);
        assert(abs(ynm(2, -1, a, e) - sqrt(3) / 2 * sin(2 * e) * sin(a))...
            < EPSILON);
        assert(abs(ynm(2, 0, a, e) - 0.5 * (3 * sin(e) * sin(e) - 1)) ...
            < EPSILON);
        assert(abs(ynm(2, 1, a, e) - sqrt(3) / 2 * sin(2 * e) * cos(a)) ...
            < EPSILON);
        assert(abs(ynm(2, 2, a, e) - sqrt(3) / 2 * cos(e) * cos(e) * ...
            cos(2 * a)) < EPSILON);
        assert(abs(ynm(3, -3, a, e) - sqrt(5 / 8) * (cos(e)).^3 * ...
            sin(3 * a)) < EPSILON);
        assert(abs(ynm(3, -2, a, e) - sqrt(15) / 2 * sin(e) * cos(e) * ...
            cos(e) * sin(2 * a)) < EPSILON);
        assert(abs(ynm(3, -1, a, e) - sqrt(3 / 8) * cos(e) * (5 * sin(e)...
            * sin(e) - 1) * sin(a)) < EPSILON);
        assert(abs(ynm(3, 0, a, e) - 0.5 * sin(e) * (5 * sin(e) * ...
            sin(e) - 3)) < EPSILON);
        assert(abs(ynm(3, 1, a, e) - sqrt(3 / 8) * cos(e) * (5 * sin(e) ...
            * sin(e) - 1) * cos(a)) < EPSILON);
        assert(abs(ynm(3, 2, a, e) - sqrt(15) / 2 * sin(e) * cos(e) * ...
            cos(e) * cos(2 * a)) < EPSILON);
        assert(abs(ynm(3, 3, a, e) - sqrt(5 / 8) * (cos(e)).^3 * ...
            cos(3 * a)) < EPSILON);
    end
end
