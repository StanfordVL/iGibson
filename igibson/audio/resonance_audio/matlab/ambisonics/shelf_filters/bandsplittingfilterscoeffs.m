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

function [bLp, bHp, a] = bandsplittingfilterscoeffs(cutoff, sampleRate)
%BANDSPLITTINGFILTERSCOEFFS Computes coefficients for Lp/Hp filters.
% The filters can be used as Ambisonic shelf-filters. 'a' coefficients are
% shared across the filters. The filters are phase matched. For more
% information, please see the following paper:
% http://www.ai.sri.com/ajh/ambisonics/BLaH3.pdf.
%
% intput:
%
%   cutoff     - cut-off frequency in [Hz]
%   sampleRate - sampling frequency in [Hz]
%
% output:
%
%   bLp - 'b' coefficients of the low-pass filter
%   bHp - 'b' coefficients of the high-pass filter
%   a   - 'a' coefficients of the low-pass and high-pass filters (shared)

k = tan((pi * cutoff) ./ (sampleRate));
k2 = 2 * k;

a0 = 1;
denominator = (k^2 + k2 + 1);
a1 = (2 * (k^2 - 1)) ./ denominator;
a2 = (k^2 - k2 + 1) ./ denominator;

% Since Resonance Audio uses single precission floats, it is desired to
% simulate the behavior of the filter here in the same way.
a = single([a0 a1 a2]);

bLp0 = k^2 ./ denominator;
bLp1 = 2 * bLp0;
bLp2 = bLp0;

bLp = single([bLp0 bLp1 bLp2]);

bHp0 = 1 ./ denominator;
bHp1 = -2 * bHp0;
bHp2 = bHp0;

bHp = single([bHp0 bHp1 bHp2]);
end
