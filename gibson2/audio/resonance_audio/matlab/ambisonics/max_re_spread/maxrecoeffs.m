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

function [maxReCoeffs] = maxrecoeffs(order)
%MAXRECOEFFS Returns MaxRe coefficients for a given Ambisonic order.
%   Supports fractional Ambisonic orders up to and including 4th order.
%
%   input:
%
%   order - Ambisonic order; can be fractional; can be a row-vector.
%
%   output:
%
%   maxReCoeffs - MaxRe coefficients for a given Amisonic order.

MAX_SUPPORTED_ORDER = 4;

if order > MAX_SUPPORTED_ORDER
    error(['The maximum supported ambisonic order is currently ', ...
        num2str(MAX_SUPPORTED_ORDER)]);
end

% If multiple orders are supplied, make sure they are in a row-vector.
if size(order, 2) < size(order, 1)
    order = order';
end

maxReCoeffs = zeros(MAX_SUPPORTED_ORDER + 1, length(order));

maxReCoeffs(1, :) = ones(1, length(order)) + 0.6240159 .* order - ...
    0.2630076 .* order.^2 + 0.05826829 .* order.^3 - ...
    0.005063062 .* order.^4;
maxReCoeffs(2, :) = 1.113054 .* order - 0.3520493 .* order.^2 + ...
    0.059059 .* order.^3 - 0.004062522 .* order.^4;
maxReCoeffs(3, :) = max(0, -0.9709362 + 1.171662 .* order - ...
    0.2164685 .* order.^2 + 0.0157427 .* order.^3);
maxReCoeffs(4, :) = max(0, -1.483585 + 0.8971373 .* order - ...
    0.07767243 .* order.^2);
maxReCoeffs(5, :) = max(0, 0.4233584 .* order - 1.270075);
end
