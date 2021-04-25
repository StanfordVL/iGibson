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

function [coeffs] = generatecoefftable(minSpreadDeg, maxAmbiOrder)
%GENERATECOEFFTABLE Generates a lookup table of MaxRe spread coefficients.
%   Also, writes the coefficients into a C++ style header file.
%
%   inputs:
%
%   minSpreadDeg - minimum required angular source spread in degrees. Must
%                  be within the range 0 - 360 degrees. The following
%                  values are recommended:
%
%                  1st Order Ambisonics: 54 degrees
%                  2nd Order Ambisonics: 40 degrees
%                  3rd Order Ambisonics: 31 degrees
%
%   maxAmbiOrder - Maximum Ambisonic order for which to generate
%                  coefficients. Determines the number of coefficients
%                  (Coefficients are generated per order).
%
%  output:
%
%  coeffs - Coefficients, by which to multiply AmbiX channels of a given
%           Ambisonic order. MaxRe decoder is assumed.

if nargin ~= 2
    error('Number of arguments must be 2');
end

if minSpreadDeg < 0 || minSpreadDeg > 360
    error('Mimimum spread must be within the range 0 - 360 degrees');
end

% For example, the smallest possible source spread at first order is 53
% degrees. That is why we would like to store coefficients from 54 degrees
% onwards, etc.
spreads = [minSpreadDeg:1:360] * pi / 180;

coeffs = [];
for spread = spreads
    coeffs = [coeffs; maxrespread(spread, maxAmbiOrder)];
end
coeffs = coeffs';
assert(length(coeffs) == (maxAmbiOrder + 1) * length(spreads));

% Write coefficients to a C++ style lookup table.
fd=fopen(['kAmbiSpreadCoeffsOrder', num2str(maxAmbiOrder), '.txt'],'wt');
fprintf(fd, ['static const float kSpreadCoeffsOrder', ...
    num2str(maxAmbiOrder), '[%d] = {%.6g'], length(coeffs), coeffs(1));
fprintf(fd,'f,\n %.6g', coeffs(2:end));
fprintf(fd,'f};\n');
fclose(fd);
end
