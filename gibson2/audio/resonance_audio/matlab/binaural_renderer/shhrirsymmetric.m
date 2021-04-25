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

function shHrir = shhrirsymmetric(hrirPath, order, dualBand)
%SHHRIRSYMMETRIC Returns Spherical Harmonics representation of HRIRs.
%
%  inputs:
%
%  hrirPath - Path to a symmetric HRIR set saved using vr-audio convention.
%             See loadhrirs.m for convention details.
%  order    - Required order of spherical harmonic encoding.
%  dualBand - Flag indicating whether to use (true) or not to use (false)
%             ambisonic shelf-filters.
%  output:
%
%  shHrir   - Spherical harmonic encoded HRIRs with spherical harmonic
%             components saved as channels. Should be the same length as
%             original HRIRs.

% Import required ambisonic functions.
addpath( '../ambisonics/ambix/');
addpath( '../ambisonics/shelf_filters/');

if nargin < 2 || nargin > 3
  error('Number of arguments must be 2 or 3');
end

if nargin == 2
  dualBand = 0;
end

[ hrirMatrix, fs, angles ] = loadhrirssymmetric(hrirPath);
hrirMatrixFull = [];
anglesSymmetric = angles(angles(:, 1) >= 0, :);

for angle = 1:size(anglesSymmetric, 1)
  % If HRIR is lateral.
  if size(hrirMatrix{angle}, 2) == 2
    hrirMatrixFull = [hrirMatrixFull hrirMatrix{angle}(:, 1) ...
      hrirMatrix{angle}(:, 2)];
    % If HRIR is on the sagittal plane.
  elseif size(hrirMatrix{angle}, 2) == 1
    hrirMatrixFull = [hrirMatrixFull hrirMatrix{angle}];
  else
    error('Incorrect number of channels in HRIR.');
  end
end

decodingMatrix = ambdecodematrix(order, angles(:, 1), angles(:, 2));
shHrir = hrirMatrixFull * decodingMatrix';

% Perform dual-band MaxRe pre-filtering, if required.
if dualBand == 1
  shHrir = ambishelffilter(shHrir, fs);
end
