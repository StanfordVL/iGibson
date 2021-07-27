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

% Tests ambencode function.

clearvars
close all
clc

ORDER = 3;

% 4-sample long input audio buffer.
audioIn = [0.1 0.2 0.3 0.4]';

%% 2-D case.
% Desired source's horizontal angle.
sourceAzRad = 45 * pi / 180;

audioOut2d = ambencode(audioIn, ORDER, sourceAzRad);

% Check the output.
for i = 1:length(audioIn)
    assert(sum(audioOut2d(i, :) == audioIn(i) .* ...
        ambencodecoeff(ORDER, sourceAzRad)) == 2 * ORDER + 1);
end

%% 3-D case.
% Desired source's vertical angle.
sourceElRad = 90 * pi/180;

audioOut = ambencode(audioIn, ORDER, sourceAzRad, sourceElRad);

% Check the output.
for i = 1:length(audioIn)
    assert(sum(audioOut(i, :) == audioIn(i) .* ...
        ambencodecoeff(ORDER, sourceAzRad, sourceElRad)) == ...
        (ORDER + 1).^2);
end
