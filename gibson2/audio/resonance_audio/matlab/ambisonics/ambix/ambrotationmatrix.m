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

function [rotationMatrix] = ambrotationmatrix(order, phiVector, ...
    thetaVector, roll, pitch, yaw )
%AMBROTATIONMATRIX Computes AmbiX 3D rotation matrix of arbitrary order.
%
% inputs:
%
% order          - Maximum order of the spherical harmonics to be rotated.
% phiVector      - Vector of azimuth direction (in radians) of the
%                  spherical harmonic sampling points.
%
% thetaVector    - Vector of elevation direction (in radians) of the
%                  spherical harmonic sampling points.
%
%                  phiVector and thetaVector are used in order to compute
%                  spherical harmonics re-encoding matrix and its inverse
%                  (the decoding matrix). At the moment it is user's
%                  responsibility to ensure that the re-encoding matrix is
%                  well-conditioned (e.g. spherical harmonic sampling
%                  points are distributed regularly. Otherwise, a warning
%                  is given that the resultant rotation matrix may not be
%                  accurate.
%
% roll           - Rotation against the x axis (front) in radians.
%
% pitch          - Rotation against the y axis (left) in radians.
%
% yaw            - Rotation against the z axis (up) in radians.
%
% output:
%
% rotationMatrix - N x N rotation matrix for a 3D sound field in the Ambix
%                  format, where N is the number of spherical harmonics and
%                  can be calculated as (order + 1)^2.

narginchk(6, 6);

if length(phiVector) ~= length(thetaVector)
    error('Angle vectors must be the same length');
end

% Compute the 3D transform matrix.
transformMatrix = ...
    makehgtform('xrotate',roll,'yrotate',pitch,'zrotate',yaw);

% Convert SH sampling angles to Cartesian coordinates.
[x, y, z] = sph2cart(phiVector, thetaVector, 1);

% Rotate the SH sampling angles. We don't need the scale.
rotatedSamplingPoints = (transformMatrix(1:3, 1:3) * [x; y; z])';

% Convert SH sampling angles back to spherical.
[phiVectorRot, thetaVectorRot] = cart2sph(rotatedSamplingPoints(:,1), ...
    rotatedSamplingPoints(:,2), rotatedSamplingPoints(:,3));

% Compute the SH rotation matrix.
[decodingMatrix, ~] = ambdecodematrix(order, phiVector, thetaVector);
[~, reencodingMatrix, conditionNumber] = ambdecodematrix(order, ...
    phiVectorRot, thetaVectorRot);

% Check if the matrix is well-conditioned and display a warning, if not.
if conditionNumber > 1 / (max(size(reencodingMatrix)) * eps)
    disp('Warning: Re-encoding matrix is ill-conditioned. The resultant rotation matrix may not be accurate');
end
rotationMatrix = decodingMatrix * reencodingMatrix;
end
