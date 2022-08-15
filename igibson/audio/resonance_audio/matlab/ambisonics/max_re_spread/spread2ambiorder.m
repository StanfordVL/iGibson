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

function [ambisonicOrder] = spread2ambiorder(spreadRad)
%SPREAD2AMBIORDER Returns required Ambisonic order for a given spread.
%   Returns required MaxRe Ambisonic order (fractional) that is required to
%   produce a sound source with required angular spread. Uses the following
%   numerical approximation:
%
%   y = 13.14992555 * exp(-2.741424056 * x);  (R^2 = 0.9994828308)
%
%   input:
%
%   spreadRad - Sound source angular spread in radians.
%
%   output:
%
%   ambisonicOrder - Fractional Ambisonic order.

% Constants.
A = 13.14992555;
B = -2.741424056;

ambisonicOrder = A * exp(B * spreadRad);
end
