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

function [symmetryMatrix] = getshsymmetries(order)
%GETSHSYMMETRIES Returns spherical harmonics symmetry matrix.
%
%   Helper function for the fast symmetric ambisonic AmbiX encoder.
%   For each ACN channel (columns) returns information whether the current
%   spherical harmonic is symmetric (+1) or asymmetric (-1) with respect to
%   the y (left-right), z (up-down), and x (front-back) axis (rows).
%
%   Note: Skips processing of the trivial 0th order channel case.
%
%   input:
%
%   order - Ambisonic order.
%
%   output:
%
%   symmetryMatrix - Spherical harmonics symmetry matrix.

numChannels = (order + 1)^2 - 1;

y = zeros(1, numChannels);
z = zeros(1, numChannels);
x = zeros(1, numChannels);

% Compute symmetry information with respect to each axis.
for acn = 1:numChannels
    [n, m] = getnm(acn);
    if m < 0
       y(acn) = -1;
       x(acn) = -((-1)^abs(m));
    else
        y(acn) = 1;
        x(acn) = (-1)^m;
    end
    z(acn) = (-1)^(n + m);
end
symmetryMatrix = [y; z; x];
end
