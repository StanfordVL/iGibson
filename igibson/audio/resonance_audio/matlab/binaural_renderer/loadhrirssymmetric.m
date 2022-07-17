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

function [hrirs, fs, angles, distance] = loadhrirssymmetric(hrirPath)
%LOADHRIRSSYMMETRIC Loads symmetric HRIRs from a specified path.
%
% The loaded HRIRS can be used by binauralrendersymmetric to render
% Ambisonic audio signal binaurally.
%
%   input:
%
%   hrirPath - Path to the symmetric HRIR saved using the following
%              convention: EXX.XX_AYY.YY_DZZ.ZZ.wav, where:
%
%               XX.YY  - Elevation angle in degrees (up to 4 significant
%                        digits allowed).
%               YYY.YY - Azimuth angle in degrees (up to 5 significant
%                        digits allowed).
%               ZZ.ZZ  - Measurement distance in meters (up to 3
%                        significant digits allowed).
%
%   outputs:
%
%   hrirs  - HRIR cell array of matrices with sample data as rows and
%            channels as colums. Two column matrices represent a symmetric
%            pair. One column matrices represent a centered self-symmetric
%            loudspeaker.
%   fs     - Sampling frequency of the HRIR data in [Hz].
%   angles - Nx2 matrix of N (azimuth, elevation) angle pairs for N
%            loudspeakers (radians).
%   distance - Loudspeaker distance (meters).

if nargin ~= 1
    error('Number of arguments must be 1');
end

hrirPattern = 'E*_A*_D*.wav';
wavPathsInfo = dir(fullfile(hrirPath, hrirPattern));

if isempty(wavPathsInfo)
  error('No hrirs found in %s', hrirPath);
end

fss = zeros(length(wavPathsInfo), 1);
distances = zeros(length(wavPathsInfo), 1);
symmetricHrirs = {};
symmetricAngles = [];
selfSymmetricHrirs = {};
selfSymmetricAngles = [];

for i = 1:length(wavPathsInfo)
  wavFileName = wavPathsInfo(i).name;
  wavPath = fullfile(hrirPath, wavFileName);
  % Parse the metadata from the path.
  [hrir, fss(i)] = audioread(wavPath);
  scanned = sscanf(wavPathsInfo(i).name, 'E%g_A%g_D%g.wav');
  E = scanned(1); A = scanned(2);
  distances(i) = scanned(3);

  if size(hrir, 2) == 1
    % A single channel implies self-symmetric.
    assert(A == 0 || A == 180);
    selfSymmetricHrirs{end + 1} = hrir;
    selfSymmetricAngles(end + 1, :) = [A, E];
  else
    assert(size(hrir, 2) == 2);

    symmetricHrirs{end + 1} = hrir;
    symmetricAngles(end + 1, :) = [A, E];
    symmetricAngles(end + 1, :) = [-A, E];
  end
end

hrirs = horzcat(symmetricHrirs, selfSymmetricHrirs);

assert(all(fss == fss(1)));
fs = fss(1);

angles = deg2rad(vertcat(symmetricAngles, selfSymmetricAngles));

assert(all(distances == distances(1)));
distance = distances(1);
