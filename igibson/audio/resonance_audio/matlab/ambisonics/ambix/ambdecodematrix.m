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

function [decodingMatrix, reencodingMatrix, conditionNumber] = ...
    ambdecodematrix(order, speakerAzRad, speakerElRad)
%AMBDECODEMATRIX Computes Ambisonic decoder matrix of arbitrary order.
%
%   inputs:
%
%   order        - Ambisonic order. Must be an integer value.
%   speakerAzRad - Speaker horizontal angles in radians. Must be a vector.
%   speakerElRad - Speaker vertical angles in radians. Must be a vector.
%
%   outputs:
%
%   decodingMatrix   - AmbiX decoding matrix.
%   reencodingMatrix - Speaker re-encoding matrix.
%   conditionNumber  - Condition number of the speaker re-encoding matrix.
%
%   For example, for the 'cube' loudspeaker configuration:
%
%   order = 1;
%   speakerAzRad = [45 135 225 315 45 135 225 315] * pi/180;
%   speakerElRad = ...
%         [35.26 35.26  35.26  35.26 -35.26 -35.26 -35.26 -35.26] * pi/180;
%   [decodingMatrix, reencodingMatrix, conditionNumber] = ...
%    ambdecodematrix(order, speakerAzRad, speakerElRad);

if nargin < 2
    error('Number of arguments must be at least 2');
end

assert(order >= 0, 'Ambisonic Order cannot be negative');

% 2-D case.
if nargin == 2
    % Intialize the re-encoding matrix.
    reencodingMatrix = zeros(length(speakerAzRad), 2 * order + 1);

    for i = 1:length(speakerAzRad)
        reencodingMatrix(i, :) =  ambencodecoeff(order, speakerAzRad(i));
    end
end

% 3-D case.
if nargin == 3
    if length(speakerAzRad) ~= length(speakerElRad)
        error('Loudspekaer angle vectors must be the same length');
    end

    % Initialize the re-encoding matrix.
    reencodingMatrix = zeros(length(speakerAzRad), (order + 1).^2);

    % Iterate through the number of reproducing channels.
    for i = 1:length(speakerAzRad)
        reencodingMatrix(i, :) =  ambencodecoeff(order, speakerAzRad(i),...
            speakerElRad(i));
    end

end

% Create decoding matrix.
decodingMatrix = pinv(reencodingMatrix);

% Compute the matrix condition number.
conditionNumber = cond(reencodingMatrix);
