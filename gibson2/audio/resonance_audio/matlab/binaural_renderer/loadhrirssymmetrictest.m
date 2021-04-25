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

% Tests if the loadhrirssymmetric function loads the HRIR data correctly.

clearvars
close all
clc

EPSILON = 1e-8;

HRIR_PATH = '../hrtf_data/sadie/';

CONFIGS = {'symmetric_cube';
    'symmetric_dodecahedron_faces';
    'symmetric_lebedev26';
    'symmetric_pentakis_dodecahedron';
    'symmetric_pentakis_icosidodecahedron'};

EXPECTED_ANGLES = {
% Cube
[135,-35;-135,-35;45,-35;-45,-35;135,35;-135,35;45,35;-45,35];
% Dodecahedron (faces)
[62,-16;-62,-16;130,-46;-130,-46;118,16;-118,16;50,46;-50,46;0,-63;0,0; ...
 180,0;180,63];
% Lebedev (26-point)
[135,-35;-135,-35;45,-35;-45,-35;90,-45;-90,-45;135,0;-135,0;45,0;-45,0;...
 90,0;-90,0;135,35;-135,35;45,35;-45,35;90,45;-90,45;0,-45;180,-45; ...
 0,-90;0,0;180,0;0,45;180,45;0,90];
 % Pentakis Dodecahedron
[90,-32;-90,-32;135,-35;-135,-35;45,-35;-45,-35;90,-69;-90,-69;111,0; ...
 -111,0;148,0;-148,0;32,0;-32,0;69,0;-69,0;90,32;-90,32;135,35;-135,35; ...
 45,35;-45,35;90,69;-90,69;0,-21;180,-21;0,-58;180,-58;0,21;180,21;0,58; ...
 180,58];
 % Pentakis Icosidodecahedron
 [122,-18;-122,-18;58,-18;-58,-18;159,-30;-159,-30;21,-30;-21,-30; ...
  90,-32;-90,-32;122,-54;-122,-54;58,-54;-58,-54;148,0;-148,0;32,0; ...
  -32,0;90,0;-90,0;122,18;-122,18;58,18;-58,18;159,30;-159,30;21,30; ...
  -21,30;90,32;-90,32;122,54;-122,54;58,54;-58,54;0,-58;180,-58;0,-90; ...
  0,0;180,0;0,58;180,58;0,90]};

EXPECTED_DISTANCE = 1;

for i = 1:size(CONFIGS)
    [hrirMatrix, fs, readAngles, readDistance] = ...
        loadhrirssymmetric([HRIR_PATH, 'sadie_subject_002_', CONFIGS{i}]);
    readAngles = readAngles * 180/pi;

    assert(readDistance == EXPECTED_DISTANCE);
    assert(all(abs(readAngles(:, 1) - EXPECTED_ANGLES{i}(:,1)) < EPSILON));
    assert(all(abs(readAngles(:, 2) - EXPECTED_ANGLES{i}(:,2)) < EPSILON));
    assert(fs == 48000);

    if i == 1
        for j = 1:length(hrirMatrix)
            subplot(length(hrirMatrix) / 4, 4, j);
            plot(hrirMatrix{j});
            title(['Loaded time-domain HRIR no. ', num2str(j)]);
            xlabel('time [samples]');
            ylabel('normalized amplitude');
            grid on
        end
    end
end
