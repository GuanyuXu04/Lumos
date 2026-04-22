%---------------------------------------------------------------
%  Continuous logger for the ADPD2211 * 5 LED
%  Line format: tick_ms,val0,val1,…  (DATA_LENGTH ints)
%---------------------------------------------------------------
clear

%% ---------- Global -------------------------------------------
NUM_LEDs = 30;
NUM_PDs  = 6;
FRAMEs   = 10000; % number of frames to collect
DATA_LENGTH = (NUM_LEDs + 1) * (NUM_PDs + 1) + 1;  % includes tick_ms

%% ---------- Serial port --------------------------------------
serialObj = serialport("COM6", 2000000);     % adjust COM & baud if needed
configureTerminator(serialObj, "LF")
flush(serialObj);

%% ---------- Output file --------------------------------------
timestamp   = datestr(now, 'yyyymmdd_HHMMSS');
outputFile  = fullfile(pwd, ['test_sample_rate_5_' timestamp '.csv']);
fid         = fopen(outputFile, 'w');

% CSV header: time_ms,L0P0, L0P1, ..., L7P6
fprintf(fid, 'time_ms');
for led = 0 : NUM_LEDs
    for pd = 0 : NUM_PDs
        fprintf(fid, ',L%dP%d', led, pd);
    end
end
fprintf(fid, '\n');

disp('Logging data …  -- press Ctrl-C to stop');

%% ---------- Timestamp unwrap state ----------------------------
base_tick = [];     % tick value at first valid frame (for starting at 0)
prev_tick = [];     % previous raw tick_ms from device
tick_offset = 0;    % added when device tick resets/wraps

%% ---------- Main acquisition loop ----------------------------
count = 0;
fmt = [repmat('%d,', 1, DATA_LENGTH-1) '%d'];  % robust sscanf format

try
    while count < FRAMEs
        line = readline(serialObj);                 % blocking read
        vals = sscanf(line, fmt);                   % DATA_LENGTH ints

        if numel(vals) ~= DATA_LENGTH
            warning("Malformed line skipped: %s", line);
            continue
        end

        curr_tick = double(vals(1));  % device tick in ms (raw)

        % init base / prev
        if isempty(base_tick)
            base_tick = curr_tick;
            prev_tick = curr_tick;
        else
            % if device tick goes backward (reset or wrap), unwrap it
            if curr_tick < prev_tick
                % make the next time continue from prev_tick
                tick_offset = tick_offset + (prev_tick + 1);
            end
            prev_tick = curr_tick;
        end

        % monotonic timestamp starting from 0 (ms)
        time_ms = curr_tick + tick_offset - base_tick;

        % Write CSV: first col is time_ms, rest are the sensor values
        fprintf(fid, '%d', round(time_ms));
        fprintf(fid, ',%d', vals(2:end));
        fprintf(fid, '\n');

        % (optional) show raw line (or comment out to speed up)
        disp(line);

        count = count + 1;
    end

catch ME
    disp('Logging stopped');
    disp(ME.message);
end

%% ---------- Tidy up ------------------------------------------
fclose(fid);
delete(serialObj);
clear serialObj
