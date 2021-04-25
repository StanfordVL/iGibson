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

% Tests the ambrotationmatrix for different orders and rotation angles.
% First, a source is encoded into AmbiX format (1st, 3rd and 5th order)
% and positioned directly in front on the x axis (0 azimuth, 0 elevation).
% Then, each sound field is rotated using the values specified in roll,
% pitch and yaw variables and the new source location is verified using
% vector analysis. For example, if we rotate the sound field against the z
% axis by +44 degrees (look left), it means that the sound source travels
% -44 degrees (in the opposite direction). If we rotate the sound field
% against the y axis by +33 degrees (look down), the source will travel +33
% degrees in elevation, etc.

clearvars
close all
clc

% Import sound field analysis functions.
thisScriptPath = fileparts(mfilename('fullpath'));
addpath(fullfile(thisScriptPath, ...
    '../../sound_field_analysis/vector_analysis/'));

% Tolerated error margin.
EPSILON = 1e-8;

% Rotation angles (in radians) to be tested.
roll  = [22 0 55] * pi / 180;
pitch = [0 33 0 ] * pi / 180;
yaw   = [0 0 44] * pi / 180;

% Initial sound source azimuth and elevation angles.
sourceAzRad = 0;
sourceElRad = 0;

%% First order rotation test.
% Spherical harmonics sampling points.
speakerAzRad1 = [45 135 225 315 45 135 225 315] * pi / 180;
speakerElRad1 = [35.26 35.26  35.26  35.26 -35.26 -35.26 -35.26 -35.26] ...
    * pi / 180;

% Generate Ambix-encoded First Order Ambisonic signal.
ambix1 = ambencode(1, 1, sourceAzRad, sourceElRad);

for i = 1:length(roll)
    % Compute the rotation matrices.
    rotationMatrix1= ambrotationmatrix(1, speakerAzRad1, ...
        speakerElRad1, roll(i), pitch(i), yaw(i));
    % Rotate the first order sound field against each axis.
    ambix1Rotated = (squeeze(rotationMatrix1) * ambix1')';
    energyVecSph1 = vectoranalysis(ambix1Rotated);
    % Check if the sound source location in the rotated sound field
    % corresponds to the rotation angles. Note that the first set of angles
    % contains only rotation against the x axis. The source is in the front
    % so its location in the sound field should not be affected.
    assert(abs(pitch(i) - energyVecSph1(2) * pi / 180) < EPSILON);
    assert(abs(yaw(i) + energyVecSph1(1) * pi / 180) < EPSILON);
end

%% Third order rotation test.
% Spherical harmonics sampling points.
speakerAzRad3 = [0 45 90 135 180 -135 -90 -45 ...
             0 90 180 -90 45 135 -135 -45] * pi / 180;
speakerElRad3 = [-13.6 13.6 -13.6 13.6 -13.6 13.6 -13.6 13.6 ...
                51.5 51.5 51.5 51.5 -51.5 -51.5 -51.5 -51.5 ] * pi / 180;

% Generate Ambix-encoded third order signal.
ambix3 = ambencode(1, 3, sourceAzRad, sourceElRad);

for i = 1:length(roll)
    % Compute the rotation matrices.
    rotationMatrix3 = ambrotationmatrix(3, speakerAzRad3, ...
        speakerElRad3, roll(i), pitch(i), yaw(i));
    % Rotate the third order sound field against each axis.
    ambix3Rotated = (squeeze(rotationMatrix3) * ambix3')';
    energyVecSph3 = vectoranalysis(ambix3Rotated);
    % Check if the sound source location in the rotated sound field
    % corresponds to the rotation angles. Note that the first set of angles
    % contains only rotation against the x axis. The source is in the front
    % so its location in the sound field should not be affected.
    assert(abs(pitch(i) - energyVecSph3(2) * pi / 180) < EPSILON);
    assert(abs(yaw(i) + energyVecSph3(1) * pi / 180) < EPSILON);
end

%% Fifth order rotation test.
% Spherical harmonics sampling points.
load('lebedev011.mat');
speakerAzRad5 = (lebedev011(:, 1) * pi / 180)';
speakerElRad5 = (lebedev011(:, 2) * pi / 180)';
clear lebedev011

% Generate Ambix-encoded fifth order signal.
ambix5 = ambencode(1, 5, sourceAzRad, sourceElRad);

for i = 1:length(roll)
    % Compute the rotation matrices.
    rotationMatrix5 = ambrotationmatrix(5, speakerAzRad5, speakerElRad5,...
        roll(i), pitch(i), yaw(i));
    % Rotate the fifth order sound field against each axis.
    ambix5Rotated = (squeeze(rotationMatrix5) * ambix5')';
    energyVecSph5 = vectoranalysis(ambix5Rotated);
    % Check if the sound source location in the rotated sound field
    % corresponds to the rotation angles. Note that the first set of angles
    % contains only rotation against the x axis. The source is in the front
    % so its location in the sound field should not be affected.
    assert(abs(pitch(i) - energyVecSph5(2) * pi / 180) < EPSILON);
    assert(abs(yaw(i) + energyVecSph5(1) * pi / 180) < EPSILON);
end
