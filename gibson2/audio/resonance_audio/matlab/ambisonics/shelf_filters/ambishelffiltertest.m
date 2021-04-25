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

% Tests ambisonic shelf-filters on an arbitrary 3rd order Ambisonic input signal
% by checking if the HOA sound field components are attenuated as expected.
% Also checks if the total energy of the input sound field is preserved.

clearvars
close all
clc

% Tolerated error margin in [dB].
ERROR_MARGIN_DB = 0.1;

% Sampling frequency.
FS = 48000;

% Number of channels of the input sound field signal.
NUM_CHANNELS = 16;

% Frequencies of the components of the test signal.
F1 = 100;
F2 = 8000;

% Generate sine tones with different frequencies
T = 1 / FS;        % Sample period
L = 1024;          % Length of signal
t = (0:L - 1) * T; % Time vector
s1 = 0.5 * sin(2 * pi * F1 * t);
s2 = 0.5 * sin(2 * pi * F2 * t);
s = s1 + s2;

% We want a sound field input with the same singal in each channel.
soundfieldInput = zeros(L, NUM_CHANNELS);
for channel = 1:NUM_CHANNELS
    soundfieldInput(:, channel) = s;
end

% Plot frequency sepctrum of the W component.
L = length(soundfieldInput(:, 1));
NFFT = 2 ^ nextpow2(L);
W = fft(soundfieldInput(:, 1), NFFT) / L;
f = FS / 2 * linspace(0, 1, NFFT / 2 + 1);

% Plot single-sided amplitude spectrum.
figure(1)
semilogx(f,2 * abs(W(1:NFFT / 2 + 1)))
title('Single-Sided Amplitude Spectrum of W(t)')
xlabel('Frequency (Hz)')
ylabel('|W(f)|')
grid on

% Shelf-filter the input sound field signal.
soundfieldOutput = ambishelffilter(soundfieldInput, FS);

% Check if the total energy of the input signal is preserved.
totalInputEnergy = db(sum(sum(soundfieldInput.^2, 1)), 'power');
totalOutputEnergy = db(sum(sum(soundfieldOutput.^2, 1)), 'power');

assert(abs(totalInputEnergy - totalOutputEnergy) < ERROR_MARGIN_DB);

% Check if the energy in HOA channels after filtering diminishes with the
% Ambisonic order, as expected.
for channel = 1:NUM_CHANNELS - 1
    % Get the order of the current channel.
    currentOrder = floor(sqrt(channel - 1));
    % For the input sound field signal, RMS in each channel should be the
    % same.
    assert(rms(soundfieldInput(:, channel)) == ...
        rms(soundfieldInput(:, channel)));
    % For the output sound field signal, RMS should be the same for the
    % same order signals, and should diminish with the order.
    if currentOrder == floor(sqrt(channel))
        assert(rms(soundfieldOutput(:, channel)) == ...
            rms(soundfieldOutput(:, channel + 1)));
    else
        assert(rms(soundfieldOutput(:, channel)) > ...
            rms(soundfieldOutput(:, channel + 1)));
    end
end

% Plot the time domain input and output signals, to verify the above
% channel amplitude relationships visually.
figure(1)
plot(soundfieldInput)
title('Sound field input signal');
xlabel('time [samples]')
ylabel('normalized amplitude');
grid on

figure(2)
plot(soundfieldOutput)
title('Sound field output signal');
xlabel('time [samples]')
ylabel('normalized amplitude');
grid on

% Check if the energy of the 'basic' and 'maxRe' vectors are the same at
% all Ambisonic orders from (1 to 5).
for order = 1:5
    % Generate high-frequency dummy soundfield input.
    numChannels = (order + 1).^2;
    basic = zeros(16, numChannels);
    basic(2, :) = ones(1, numChannels);
    % Perform shelf-filtering.
    maxRe = ambishelffilter(basic, FS);
    basicTotalEnergy = db(sum(sum(basic.^2, 2)), 'power');
    maxReTotalEnergy = db(sum(sum(maxRe.^2, 2)), 'power');
    assert(abs(basicTotalEnergy - maxReTotalEnergy) < ERROR_MARGIN_DB);
end
