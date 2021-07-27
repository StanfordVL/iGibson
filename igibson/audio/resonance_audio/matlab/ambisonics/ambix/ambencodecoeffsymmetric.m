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

function y = ambencodecoeffsymmetric(encoderTable, symmetryMatrix, phi, theta)
%AMBENCODECOEFFSYMMETRIC Fast AmbiX encoding coefficients from a look-up table.
%
%   Computes full 3D AmbiX encoding coefficients from a limited (one
%   sphere quadrant) look-up coefficient table and a SH symmetry matrix.
%
%   inputs:
%
%   encoderTable   - Look-up table with one quadrant of spherical harmonic
%                    (Ambisonic encoding) coefficients. Please see the
%                    getencodertable function for more information.
%   symmetryMatrix - Matrix containing information about symmetries of
%                    spherical harmonic coefficients with respect to y, z,
%                    and x axis respectively.
%  phi             - Horizontal source angle in degrees.
%  theta           - Vertical source angle in degrees.
%
%  output:
%
%  y - Ambisonic AmbiX encoding coefficients.

numChannels = size(encoderTable, 3);

% Initialize vector which will contain information about required phase
% flip for spherical harmonic coefficients.
flip = ones(1, numChannels);

% Determine the source quadrant and use symmetryMatrix to determine if the
% phase flip is required.

% Source in the right hemisphere: use y (left-right) symmetry.
if phi < 0
    flip = symmetryMatrix(1, :);
end

% Source in the bottom hemisphere: use z (up-down) symmetry.
if theta < 0
    flip = flip .* symmetryMatrix(2, :);
end

% Source in the rear hemisphere: use x (front-back) symmetry.
if abs(phi) > 90
    flip = flip .* symmetryMatrix(3, :);
    phi = 180 - abs(phi);
end

% Use angle + 1 as index because indexing in Matlab is 1-based.
y = [1 squeeze(encoderTable(abs(round(phi)) + 1, ...
    abs(round(theta)) + 1, :))' .* flip];
end
