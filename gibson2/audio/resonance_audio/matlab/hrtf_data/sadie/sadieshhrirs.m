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

function [] = sadieshhrirs(ambisonicOrder, shelfFilter)
%SADIESHHRIRS Encodes SADIE HRIRs into Spherical Harmonic representation.
% Generates Spherical Harmonic encoded HRIRs from the relevant directory of
% regular SADIE HRIRs. Please note: the shhrirsymmetric function is fully
% tested in the /binaural_renderer/ directory.
%
% To verify the generated SH HRIRs are correct, run 'sadieshhrirstest'.
%
%   input:
%
%   ambisonicOrder - Required Ambisonic order of the encoded SH-HRIRs.
%   shelfFilter    - Whether to use or not Ambisonic shelf-filters (bool).

% Import required functions.
addpath( '../../binaural_renderer/');

% Target sampling rate.
TARGET_SAMPLE_RATE = 48000;

switch ambisonicOrder
    case 1
        hrirDir = 'sadie_subject_002_symmetric_cube';
    case 2
        hrirDir = 'sadie_subject_002_symmetric_dodecahedron_faces';
    case 3
        hrirDir = 'sadie_subject_002_symmetric_lebedev26';
    case 4
        hrirDir = 'sadie_subject_002_symmetric_pentakis_dodecahedron';
    case 5
        hrirDir = 'sadie_subject_002_symmetric_pentakis_icosidodecahedron';
    otherwise
        error('Unsupported Ambisonic order');
end

% Output path for the SH HRIR WAV.
savedir = ['sadie_002_symmetric_sh_hrir_o_', num2str(ambisonicOrder), '/'];

if (exist(savedir, 'dir') == 0)
    mkdir(savedir);
end

% Perform SH encoding.
shHrirs = shhrirsymmetric(hrirDir, ambisonicOrder, shelfFilter);

% Write SH HRIRs as a multi-channel WAV file.
audiowrite([savedir, ...
    ['sh_hrir_order_', num2str(ambisonicOrder), '.wav']], shHrirs, ...
    TARGET_SAMPLE_RATE);
