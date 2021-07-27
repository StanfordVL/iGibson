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

function [maxReSpreadCoeffs] = maxrespread(spread, maxSourceOrder)
%MAXRESPREAD Returns MaxRe correction gains to control source spread.
%  Returns scaling coefficients which needs to be applied to MaxRe input
%  soundfield signal in order to obtain a source with a specified angular
%  spread.
%
%   input:
%
%   spread - Required source angular spread in radians. Note: If the spread
%            is infeasible for the specified maxSourceOrder, a vector of
%            ones is returned (e.g. no scaling = min. source spread).
%
%   maxSourceOrder - Maximum supported source Ambisonic order. Determines
%                    the min. spread as well as the number of scaling
%                    coefficients to be returned.
%
%   output:
%
%   maxReSpreadCoeffs - MaxRe spread-controlling scaling coefficients.

if nargin ~= 2
    error('Number of arguments must be 2');
end

currentSourceOrder = spread2ambiorder(spread);

% Make sure that maxSourceOrder is never exceeded.
currentSourceOrder = min(currentSourceOrder, maxSourceOrder);
currentCoeffsRaw = maxrecoeffs(currentSourceOrder);
maxCoeffsRaw = maxrecoeffs(maxSourceOrder);

% Select only the required number of coefficients for maxSourceOrder. For
% example, for maxSourceOrder 3, we need 4 coefficients (for 0th, 1st, 2nd,
% and 3rd order components).
NUM_REQUIRED_COEFFS = maxSourceOrder + 1;
currentCoeffsRaw = currentCoeffsRaw(1:NUM_REQUIRED_COEFFS, :);
maxCoeffsRaw = maxCoeffsRaw(1:NUM_REQUIRED_COEFFS);

% Compensate energy of the coefficients.
currentSourceEnergy = zeros(1, size(currentCoeffsRaw, 2));
maxSourceEnergy = 0;
for n = 0:maxSourceOrder
    currentSourceEnergy = currentSourceEnergy + (2 .* n + 1) .* ...
        currentCoeffsRaw(n + 1, :).^2;
    maxSourceEnergy = maxSourceEnergy + (2 .* n + 1) .* ...
        maxCoeffsRaw(n + 1).^2 ;
end
compensationGain = sqrt(maxSourceEnergy ./ currentSourceEnergy);
maxReCoeffsEnergyPreserving = diag(compensationGain) * currentCoeffsRaw';

% Normalize coefficients.
maxReSpreadCoeffs = (maxReCoeffsEnergyPreserving / diag(maxCoeffsRaw))';
