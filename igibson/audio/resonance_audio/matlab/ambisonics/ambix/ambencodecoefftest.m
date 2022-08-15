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

% Plot ambencodecoeff function.

clearvars
close all
clc

% Ambisonic order.
ORDER = 3;

%% 2D case:
for phi = 1:360
   coefficients2d(phi, :) = ambencodecoeff(ORDER, phi * pi / 180);
end
figure(1)
plot(coefficients2d)

%% 3D case:
theta = 0;
for phi = 1:360
   coefficients3d(phi, :) = ambencodecoeff(ORDER, phi * pi / 180, theta);
end
figure(2)
plot(coefficients3d)
