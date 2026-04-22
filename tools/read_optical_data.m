%---------------------------------------------------------------
%  Continuous logger for the ADPD2211 * 5 LED
%  Line format: tick_ms,val0,val1,…,val11  (13 ints)
%---------------------------------------------------------------
clear
%col_idx = [1 2 3 6 8 11 13 16 18 21];   % 1-based columns to keep   
%% ---------- Global -------------------------------------------
NUM_LEDs = 7;
NUM_PDs = 6;
FRAMEs = 3000000; % number of frames to collect
DATA_LENGTH = (NUM_LEDs + 1) * (NUM_PDs + 1) + 1;

%% ---------- Serial port --------------------------------------
serialObj = serialport("COM6", 2000000);     % adjust COM & baud if needed
configureTerminator(serialObj, "LF")
flush(serialObj);

%% ---------- Output file --------------------------------------
timestamp   = datestr(now, 'yyyymmdd_HHMMSS');
outputFile  = fullfile(pwd, ['test_0_70_mm_' timestamp '.csv']);
fid         = fopen(outputFile, 'w');

% CSV he  ader:   time_ms,L0P0, L0P1, ..., L30P6
fprintf(fid, 'time_ms');
for led = 0 : NUM_LEDs
    for pd = 0 : NUM_PDs
        fprintf(fid,',L%dP%d', led, pd);
    end
end
%fprintf(fid, 'TIME,EXT,L1P1,L1P4,L2P1,L2P4,L3P1,L3P4,P1O,P4O');
fprintf(fid, '\n'); 

disp('Logging data …  -- press Ctrl-C to stop');
count = 0;
%% ---------- Main acquisition loop ----------------------------
try
    while count <= FRAMEs
        line = readline(serialObj);                 % blocking read
        vals = sscanf(line, repmat('%d,',1, DATA_LENGTH - 1) + "%d");  % 21 ints, comma-separated

        if numel(vals) ~= DATA_LENGTH
            warning("Malformed line skipped: %s", line);
            continue
        end

        % Select requested columns and write as CSV
        sel = vals;

        % (optional) show what we wrote
        disp(line);

        fprintf(fid, '%d',  sel(1));
        fprintf(fid, ',%d', sel(2:end-1));
        fprintf(fid, ',%d\n', sel(end));
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