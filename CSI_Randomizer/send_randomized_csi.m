%
% Copyright 2020 Francesco Gringoli
% Copyright 2020 Marco Cominelli
%
% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
%
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
%
% You should have received a copy of the GNU General Public License
% along with this program.  If not, see <https://www.gnu.org/licenses/>.
%

MCS = 0;
TCPClient = tcpip('127.0.0.1', 12345, 'NetworkRole', 'client');
set(TCPClient, 'OutputBufferSize', 40000);
set(TCPClient, 'Timeout', 10);
fopen (TCPClient);

for mcsjj = 1:length(MCS),
  mcs = MCS(mcsjj);

  BW = 80;

  vhtCfg = wlanVHTConfig;             % Create packet configuration
  vhtCfg.ChannelBandwidth = sprintf('CBW%d', BW); % 80 MHz channel bandwidth
  vhtCfg.NumTransmitAntennas = 1;     % 1 transmit antenna
  vhtCfg.NumSpaceTimeStreams = 1;     % 1 space-time stream
  vhtCfg.GuardInterval = 'Long';
  vhtCfg.MCS = mcs;             % Modulation: QPSK Rate: 1/2
  
  idleTime = 20e-6; % 20 microseconds idle period after packet
  
  % Initialize the scrambler with a random integer for each packet
  % scramblerInitialization = randi([1 127], 1, 1);
  scramblerInitialization = 5;
  
  % Create frame configuration
  macCfg = wlanMACFrameConfig('FrameType', 'QoS Data');
  macCfg.FrameFormat = 'VHT';     % Frame format
  macCfg.MSDUAggregation = false; % Form A-MSDUs internally
  macCfg.MPDUAggregation = false;
  
  payload = {'aaaa00000000080000112233445566778899'}; 
  
  seqn = 0;
  txed = 0;
  tottxed = 0;
  while 1,
      tottxed = tottxed + 1;
      macCfg.SequenceNumber = mod(seqn, 4096);
      seqn = seqn + 1;
      [macFrame, frameLength] = wlanMACFrame(payload, macCfg, vhtCfg);
      vhtCfg.APEPLength = frameLength;
      
      decimalBytes = hex2dec(macFrame);
      bitsPerByte = 8;
      frameBits = reshape(de2bi(decimalBytes, bitsPerByte)', [], 1);
      
      % Set the APEP length in the VHT configuration
      % vhtCfg.APEPLength = apepLength;
      
      data = frameBits;
      
      % Generate baseband VHT packets
      txWaveform = wlanWaveformGenerator(data, vhtCfg, ...
          'NumPackets', 1,'IdleTime', idleTime, ...
          'ScramblerInitialization', scramblerInitialization);
  
      % split spectrum into two halves
      txWaveform_left = [];
      txWaveform_rigt = [];
  
      % header is composed by a sequence of pieces
      % 1           2           3           4      5        6             7                      8    
      % L-STF (8us) L-LTF (8us) L-SIG (4us) VHT-SIG-A (8us) VHT-STF (4us) VHT-LTF (4us * stream) VHT-SIG-B (4us)

      header_durations = [8 8 4 4 4 4 4 4];
      header_tochanged = [0 0 1 1 1 0 1 1];
      Fs = 80e6;
      Tc = 1 / Fs;
      
      cursample = 0;
      
      % Generate a new set of notch filters every X trasmitted frames
      % X is drawn from an exponential distribution
      if txed == 0,
        mask = ones([256 1]);
        % Create a random number of spikes with fixed amplitudes
        for i = 1:randi([5 10])
          bin = randi([1 248]);
          mask(bin:bin+randi([2 6])) = 8;
        end
        while 1
          txed = ceil(-250 * log(1-rand()));
          if txed < 500
            break;
          end
        end
      end;
      txed = txed -1;
      
      for headjj = 1:length(header_durations),
        duration = header_durations(headjj);
        Nsamples = duration * 1e-6 / Tc;
        s = txWaveform(cursample + (1:Nsamples));
  
        if headjj == 1,
          % keep copy in the middle as it is perfectly symmetrical
          % s1 = s(1:256);
          s1 = s(257:512);
          S1 = fftshift(fft(s1));
          S1 = S1 .* mask;
          s1 = ifft(fftshift(S1));
          s(1:256) = s1;
          s(257:512) = s1;
          s(513:end) = s1(1:128);
        elseif headjj == 2,
          gi = s(1:128);
          s1 = s(129:384);
          s2 = s(385:640);
          S1 = fftshift(fft(s1));
          S1 = S1 .* mask;
          S2 = fftshift(fft(s2));
          S2 = S2 .* mask;
          s1 = ifft(fftshift(S1));
          s2 = ifft(fftshift(S2));
          gi = s2(end - 127:end);
          s = [gi; s1; s2];
        elseif ismember (headjj, [3 4 5 7 8]),
          gi = s(1:64);
          s1 = s(65:end);
          S1 = fftshift(fft(s1));
          S1 = S1 .* mask;
          s1 = ifft(fftshift(S1));
          gi = s1(end - 63:end);
          s = [gi; s1];
        % here we don't have a copy in the middle... so keep it like this (can be improved)
        elseif headjj == 6,
          s1 = s(1:256);
          S1 = fftshift(fft(s1));
          S1 = S1 .* mask;
          s1 = ifft(fftshift(S1));
          s(1:256) = s1;
          s(257:end) = s1(1:64);
        end;

        txWaveform(cursample + (1:Nsamples)) = s;
        cursample = cursample + Nsamples;
      end;
      
      datasamples = (length(txWaveform) - cursample);
      
      if strcmp(vhtCfg.GuardInterval, 'Long'),
        Nsamples_data = 4e-6 / Tc;
      elseif strcmp(vhtCfg.GuardInterval, 'Short'),
        Nsamples_data = 3.6e-6 / Tc;
      else
        disp 'Abort';
        return;
      end;
      
      if mod(datasamples, Nsamples_data) > 0,
        disp 'Invalid number of samples';
        return;
      end;
      
      datasymbols = datasamples / Nsamples_data;
      
      for datajj = 1:datasymbols,
        s = txWaveform(cursample + (1:Nsamples_data));
        gi = s(1:64);
        s1 = s(65:end);
        S1 = fftshift(fft(s1));
	S1 = S1 .* mask;
        s1 = ifft(fftshift(S1));
        gi = s1(end - 63:end);
        s = [gi; s1];
        txWaveform(cursample + (1:Nsamples_data)) = s;
        cursample = cursample + Nsamples_data;
      end;
  
      if BW == 20,
        txWaveform = [zeros([50 1]); txWaveform; zeros([10 1])];
      elseif BW == 40,
        txWaveform = [zeros([100 1]); txWaveform; zeros([20 1])];
      elseif BW == 80,
        txWaveform = [zeros([200 1]); txWaveform; zeros([40 1])];
      else
        disp 'Invalid BW';
        keyboard
      end;
      
      fs = wlanSampleRate(vhtCfg);
      txWaveform125 = resample(txWaveform, 125, fs / 1e6);
      txWaveform125 = 12000 * txWaveform125 / max(abs(txWaveform125));
  
      vhtdata = [real(txWaveform125) imag(txWaveform125)];
      vhtdata = reshape(vhtdata', 1, numel(vhtdata));
      vhtdata = int16 (vhtdata);
      vhtdata = [length(txWaveform125) vhtdata];
  
      message = typecast (vhtdata, 'uint8');
      fwrite (TCPClient, message);
      flushoutput (TCPClient);
  
      disp (sprintf('%d', seqn));
  
  end;
end;

fclose (TCPClient);
