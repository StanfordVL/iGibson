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

% Test ambdecode function.

clearvars
close all
clc

%% 2-D case.
% 3-sample long input buffer simulating a source at 0, 45 and 90 degree
% angles.
audioIn2d = [1 0 1; 1 .7071 .7071; 1 1 0];

% Quad loudspeaker array.
speakerAz2d = [45 135 -135 -45] * pi / 180;

audioOut2d = ambdecode(audioIn2d, speakerAz2d);

%% 3-D case.
% 3-sample long input buffer simulating a source at [0, 0], [45, 35.26] and
% [90, 0] degree angles.
audioIn = [1 0 0 1; 1 .7071 .7071 .7071; 1 1 0 0];

% CUBE loudspeaker array.
speakerAz = [135 -135 135 -135 45 -45 45 -45] * pi / 180;
speakerEl = [35.26 35.26 -35.26 -35.26 35.26 35.26 -35.26 -35.26] * ...
    pi / 180;

audioOut = ambdecode(audioIn, speakerAz, speakerEl);
