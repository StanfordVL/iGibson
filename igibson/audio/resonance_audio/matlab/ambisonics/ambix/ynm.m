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

function ynm = ynm(n, m, phi, theta)
%YNM Computes SN3D-normalized Spherical Harmonics of degree n and order m.
%
%  inputs:
%
%  n     - Spherical harmonic degree (aka Ambisonic order).
%  m     - Spherical harmonic order (aka Ambisonic degree).
%  phi   - Horizontal angle in radians. Can be a vector.
%  theta - Vertical angle in radians. Can be a vector.
%
%  output:
%
%  ynm - Spherical harmonic coefficients.

if nargin < 4
    error('Number of arguments must be 4');
end

if m < 0
    ynm = nnm(n,abs(m)) .* pnm(n, abs(m), sin(theta)) .* sin(abs(m) * phi);
else
    ynm = nnm(n,abs(m)) .* pnm(n, abs(m), sin(theta)) .* cos(abs(m) * phi);
end
