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

function pnm = pnm(n, m, x)
%PNM Computes Associated Legendre polynomial of degree n and order m.
%
%   Evaluates Associated Legendre polynomial (ALP) of degree n and order m
%   at a given x.
%
%   inputs:
%
%   n   - Degree of the ALP
%   m   - Order of the ALP.
%   x   - Input variable ALP is evaluated at.
%
%   output:
%
%   pnm - ALP evaluated at x.

if nargin < 3
    error('Number of arguments must be 3');
end

p_all = legendre(n, x);
if n == 0
    pnm = p_all;
else
    pnm = squeeze(p_all(abs(m) + 1, :, :));
end

% Undo Condon-Shortely phase (this is the convention used in AmbiX).
pnm = (-1)^m * pnm;

% Always return the same shape we were given.
pnm = reshape(pnm, size(x));
