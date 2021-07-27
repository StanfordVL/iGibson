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

function y = ambencodecoeff(n, phi, theta)
%AMBENCODECOEFF Computes Ambisonic encoding coefficients of arbitrary order.
%
%   inputs:
%
%   n     - Ambisonic order (aka spherical harmonic degree).
%   phi   - Horizontal angle in radians.
%   theta - Vertical angle in radians.
%
%   output:
%
%   y - Ambisonic AmbiX encoding coefficients.

if nargin < 2
    error('Number of arguments must be at least 2');
end

% 2D case:
if nargin == 2
    y0 = 1;
    y1 = zeros(1, 2 * n);
    for i = 1:n
        y1(1, 2 * i - 1) = sin(i * phi);
        y1(1, 2 * i) = cos(i * phi);
    end
    y = [y0 y1];
end

% 3D case:
if nargin == 3
    l_max = (n+1)^2-1;
    y = zeros(length(phi), l_max);
    for l = 0:l_max
        [n,m] = getnm(l);
        y(:, l+1) = ynm(n,m,phi,theta);
    end
end
