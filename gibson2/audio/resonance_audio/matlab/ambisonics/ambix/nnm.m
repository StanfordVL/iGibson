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

function nnm = nnm(n, m)
%NNM Computes SH normalization factor according to the SN3D scheme.
%
%  input:
%
%  n - Spherical harmonic degree (aka Ambisonic order).
%  m - Spherical harmonic order (aka Ambisonic degree).
%
%  output:
%
%  nm - AmbiX normalization factor (SN3D).

if nargin < 2
    error('Number of arguments must be 2');
end

% The following 'if' statement is used instead of the Kronecker Delta.
if m == 0
    nnm = 1;
else
    nnm = sqrt(2 * factorial(n - abs(m)) ./ (factorial(n + abs(m))));
end
